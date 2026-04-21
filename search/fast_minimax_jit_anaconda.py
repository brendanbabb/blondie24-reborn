"""
Numba-JIT alpha-beta search for the 2001 Anaconda network.

Parallels search/fast_minimax_jit.py but the leaf evaluator runs the
92 -> 40 -> 10 -> 1 MLP fronted by the 91-filter spatial preprocessor.

The dense (91, 32) preprocessor weight matrix is built once in Python
from the flat 854-weight vector via `unpack_weights_anaconda` — the JIT
receives an already-scattered ndarray so the hot path stays branch-free.

Zobrist table is shared with the 1999 JIT module so position hashes
collide across architectures, which is fine because the TT is cleared
at every search call.
"""

import numpy as np
from numba import njit

from checkers.board import Board
from checkers.fast_board import (
    get_legal_moves_fast,
    apply_move_fast,
    NEIGHBORS,
    JUMP_TARGETS,
    DIR_DR,
    MOVE_SLOTS,
)
from neural.fast_eval_anaconda import unpack_weights_anaconda, TOTAL as ANACONDA_TOTAL
from search.fast_minimax import _py_move_key, _fast_move_key
from search.fast_minimax_jit import ZOBRIST_PIECES, ZOBRIST_SIDE
from search._order_helpers import order_moves, update_killers


INF32 = np.float32(np.inf)

_TT_EXACT = np.int8(0)
_TT_LOWER = np.int8(1)
_TT_UPPER = np.int8(2)
_TT_SIZE = 1 << 20
_TT_MASK = np.int64(_TT_SIZE - 1)

_PVS_EPS = np.float32(1e-6)
_ASP_DELTA = np.float32(0.1)


@njit(cache=True, inline='always')
def _hash_position_ana(squares, player):
    h = np.int64(0)
    for sq in range(32):
        idx = np.int64(squares[sq]) + 2
        h ^= ZOBRIST_PIECES[sq, idx]
    if player == -1:
        h ^= ZOBRIST_SIDE
    return h


@njit(cache=True)
def _eval_position_anaconda(
    squares, player,
    W_pp, b_pp, W1, W2, b2, W3, b3, piece_diff, king_weight,
    root_player,
):
    """
    Anaconda forward pass on a single position. Returns a float32 score
    from root_player's perspective.
    """
    x = np.empty(32, dtype=np.float32)
    pd_sum = np.float32(0.0)
    for i in range(32):
        sq = squares[i]
        if sq == 0:
            v = np.float32(0.0)
        elif sq == player:
            v = np.float32(1.0)
        elif sq == -player:
            v = np.float32(-1.0)
        elif sq == 2 * player:
            v = np.float32(king_weight)
        else:
            v = np.float32(-king_weight)
        x[i] = v
        pd_sum += v

    # Spatial preprocessor: 91 filters + constant-1 bias channel for fc1.
    filters = np.empty(92, dtype=np.float32)
    for f in range(91):
        s = b_pp[f]
        for k in range(32):
            s += W_pp[f, k] * x[k]
        filters[f] = np.float32(np.tanh(s))
    filters[91] = np.float32(1.0)

    # fc1: 92 -> 40 (no explicit bias; the constant-1 channel carries it)
    h1 = np.empty(40, dtype=np.float32)
    for j in range(40):
        s = np.float32(0.0)
        for i in range(92):
            s += W1[j, i] * filters[i]
        h1[j] = np.float32(np.tanh(s))

    # fc2: 40 -> 10
    h2 = np.empty(10, dtype=np.float32)
    for j in range(10):
        s = b2[j]
        for i in range(40):
            s += W2[j, i] * h1[i]
        h2[j] = np.float32(np.tanh(s))

    # fc3: 10 -> 1
    s3 = b3[0]
    for i in range(10):
        s3 += W3[0, i] * h2[i]

    score = np.float32(np.tanh(s3 + pd_sum * piece_diff))
    if player != root_player:
        score = -score
    return score


@njit(cache=True)
def _alpha_beta_anaconda(
    squares, player, depth, ply, alpha, beta, maximizing, root_player,
    W_pp, b_pp, W1, W2, b2, W3, b3, piece_diff, king_weight,
    tt_keys, tt_value, tt_depth, tt_flag, tt_best_idx,
    killers, killers_valid,
    move_bufs,
):
    buf = move_bufs[depth]
    n_moves = get_legal_moves_fast(
        squares, player, NEIGHBORS, JUMP_TARGETS, DIR_DR, buf,
    )
    if n_moves == 0:
        winner = -player
        if winner == root_player:
            return INF32
        else:
            return -INF32

    if depth == 1:
        next_player = -player
        if maximizing:
            best = -INF32
        else:
            best = INF32
        for i in range(n_moves):
            new_sq = apply_move_fast(squares, buf[i])
            gc_buf = move_bufs[0]
            gc_count = get_legal_moves_fast(
                new_sq, next_player, NEIGHBORS, JUMP_TARGETS, DIR_DR, gc_buf,
            )
            if gc_count == 0:
                winner = player
                if winner == root_player:
                    child_score = INF32
                else:
                    child_score = -INF32
            else:
                child_score = _eval_position_anaconda(
                    new_sq, next_player,
                    W_pp, b_pp, W1, W2, b2, W3, b3, piece_diff, king_weight,
                    root_player,
                )
            if maximizing:
                if child_score > best:
                    best = child_score
            else:
                if child_score < best:
                    best = child_score
        return best

    key = _hash_position_ana(squares, player)
    slot = np.int64(key) & _TT_MASK
    tt_move = np.int8(-1)
    if tt_keys[slot] == key:
        tt_move = tt_best_idx[slot]
        if tt_depth[slot] >= depth:
            v = tt_value[slot]
            f = tt_flag[slot]
            if f == _TT_EXACT:
                return v
            if f == _TT_LOWER and v >= beta:
                return v
            if f == _TT_UPPER and v <= alpha:
                return v

    canonical_of = np.empty(64, dtype=np.int8)
    for i in range(n_moves):
        canonical_of[i] = np.int8(i)
    order_moves(
        buf, n_moves, np.int64(tt_move),
        killers, killers_valid, ply,
        canonical_of,
    )

    alpha0 = alpha
    beta0 = beta
    next_player = -player
    best_idx = np.int8(0)

    if maximizing:
        value = -INF32
        for i in range(n_moves):
            new_sq = apply_move_fast(squares, buf[i])
            if i == 0:
                child_v = _alpha_beta_anaconda(
                    new_sq, next_player, depth - 1, ply + 1,
                    alpha, beta, False, root_player,
                    W_pp, b_pp, W1, W2, b2, W3, b3, piece_diff, king_weight,
                    tt_keys, tt_value, tt_depth, tt_flag, tt_best_idx,
                    killers, killers_valid,
                    move_bufs,
                )
            else:
                child_v = _alpha_beta_anaconda(
                    new_sq, next_player, depth - 1, ply + 1,
                    alpha, alpha + _PVS_EPS, False, root_player,
                    W_pp, b_pp, W1, W2, b2, W3, b3, piece_diff, king_weight,
                    tt_keys, tt_value, tt_depth, tt_flag, tt_best_idx,
                    killers, killers_valid,
                    move_bufs,
                )
                if child_v > alpha and child_v < beta:
                    child_v = _alpha_beta_anaconda(
                        new_sq, next_player, depth - 1, ply + 1,
                        child_v, beta, False, root_player,
                        W_pp, b_pp, W1, W2, b2, W3, b3, piece_diff, king_weight,
                        tt_keys, tt_value, tt_depth, tt_flag, tt_best_idx,
                        killers, killers_valid,
                        move_bufs,
                    )
            if child_v > value:
                value = child_v
                best_idx = np.int8(i)
            if value > alpha:
                alpha = value
            if alpha >= beta:
                update_killers(killers, killers_valid, ply, buf, i)
                break
    else:
        value = INF32
        for i in range(n_moves):
            new_sq = apply_move_fast(squares, buf[i])
            if i == 0:
                child_v = _alpha_beta_anaconda(
                    new_sq, next_player, depth - 1, ply + 1,
                    alpha, beta, True, root_player,
                    W_pp, b_pp, W1, W2, b2, W3, b3, piece_diff, king_weight,
                    tt_keys, tt_value, tt_depth, tt_flag, tt_best_idx,
                    killers, killers_valid,
                    move_bufs,
                )
            else:
                child_v = _alpha_beta_anaconda(
                    new_sq, next_player, depth - 1, ply + 1,
                    beta - _PVS_EPS, beta, True, root_player,
                    W_pp, b_pp, W1, W2, b2, W3, b3, piece_diff, king_weight,
                    tt_keys, tt_value, tt_depth, tt_flag, tt_best_idx,
                    killers, killers_valid,
                    move_bufs,
                )
                if child_v < beta and child_v > alpha:
                    child_v = _alpha_beta_anaconda(
                        new_sq, next_player, depth - 1, ply + 1,
                        alpha, child_v, True, root_player,
                        W_pp, b_pp, W1, W2, b2, W3, b3, piece_diff, king_weight,
                        tt_keys, tt_value, tt_depth, tt_flag, tt_best_idx,
                        killers, killers_valid,
                        move_bufs,
                    )
            if child_v < value:
                value = child_v
                best_idx = np.int8(i)
            if value < beta:
                beta = value
            if alpha >= beta:
                update_killers(killers, killers_valid, ply, buf, i)
                break

    if value <= alpha0:
        flag = _TT_UPPER
    elif value >= beta0:
        flag = _TT_LOWER
    else:
        flag = _TT_EXACT
    tt_keys[slot] = key
    tt_value[slot] = value
    tt_depth[slot] = depth
    tt_flag[slot] = flag
    tt_best_idx[slot] = canonical_of[best_idx]

    return value


class FastAgentJitAnaconda:
    """
    JIT'd alpha-beta agent for the 2001 Anaconda network. Same public
    interface as FastAgentJit.
    """

    def __init__(self, weights: np.ndarray, depth: int):
        self.depth = depth
        self.weights = np.asarray(weights, dtype=np.float32)
        (W_pp, b_pp, W1, W2, b2, W3, b3, piece_diff, king_weight) = \
            unpack_weights_anaconda(self.weights)
        self.W_pp = np.ascontiguousarray(W_pp, dtype=np.float32)
        self.b_pp = np.ascontiguousarray(b_pp, dtype=np.float32)
        self.W1 = np.ascontiguousarray(W1, dtype=np.float32)
        self.W2 = np.ascontiguousarray(W2, dtype=np.float32)
        self.b2 = np.ascontiguousarray(b2, dtype=np.float32)
        self.W3 = np.ascontiguousarray(W3, dtype=np.float32)
        self.b3 = np.ascontiguousarray(b3, dtype=np.float32)
        self.piece_diff = np.float32(piece_diff)
        self.king_weight = np.float32(king_weight)

        self.move_bufs = np.zeros((depth + 1, 64, MOVE_SLOTS), dtype=np.int8)

        self.tt_keys     = np.zeros(_TT_SIZE, dtype=np.int64)
        self.tt_value    = np.zeros(_TT_SIZE, dtype=np.float32)
        self.tt_depth    = np.full(_TT_SIZE, -1, dtype=np.int8)
        self.tt_flag     = np.zeros(_TT_SIZE, dtype=np.int8)
        self.tt_best_idx = np.full(_TT_SIZE, -1, dtype=np.int8)

        self.killers       = np.zeros((depth + 1, 2, MOVE_SLOTS), dtype=np.int8)
        self.killers_valid = np.zeros((depth + 1, 2), dtype=np.int8)

        self.nodes_evaluated = 0

    def search(self, board: Board):
        py_moves = board.get_legal_moves()
        if not py_moves:
            return None, -float("inf")
        if len(py_moves) == 1:
            return py_moves[0], 0.0

        self.tt_depth.fill(-1)
        self.killers_valid.fill(0)

        root_buf = np.zeros((64, MOVE_SLOTS), dtype=np.int8)
        n_root = get_legal_moves_fast(
            board.squares, board.current_player,
            NEIGHBORS, JUMP_TARGETS, DIR_DR, root_buf,
        )
        py_by_key = {_py_move_key(m): m for m in py_moves}
        root_py = [py_by_key[_fast_move_key(root_buf[i])] for i in range(n_root)]

        root_player = np.int8(board.current_player)
        next_player = np.int8(-board.current_player)

        best_move = root_py[0]
        best_score = -INF32
        prev_best_idx = 0
        prev_score_set = False
        prev_score = np.float32(0.0)

        start_depth = 2 if self.depth >= 2 else self.depth
        for current_depth in range(start_depth, self.depth + 1):
            if prev_best_idx > 0:
                tmp_row = root_buf[0].copy()
                root_buf[0] = root_buf[prev_best_idx]
                root_buf[prev_best_idx] = tmp_row
                root_py[0], root_py[prev_best_idx] = (
                    root_py[prev_best_idx], root_py[0]
                )

            if prev_score_set:
                alpha0 = prev_score - _ASP_DELTA
                beta0 = prev_score + _ASP_DELTA
            else:
                alpha0 = -INF32
                beta0 = INF32

            while True:
                iter_best_score = -INF32
                iter_best_idx = 0
                iter_best_move = root_py[0]
                alpha = alpha0
                beta = beta0

                for i in range(n_root):
                    fast_move = root_buf[i].copy()
                    new_sq = apply_move_fast(board.squares, fast_move)
                    if i == 0:
                        score = _alpha_beta_anaconda(
                            new_sq, next_player,
                            np.int64(current_depth - 1), np.int64(1),
                            alpha, beta, False, root_player,
                            self.W_pp, self.b_pp, self.W1, self.W2, self.b2, self.W3, self.b3,
                            self.piece_diff, self.king_weight,
                            self.tt_keys, self.tt_value, self.tt_depth, self.tt_flag,
                            self.tt_best_idx,
                            self.killers, self.killers_valid,
                            self.move_bufs,
                        )
                    else:
                        zw_hi = alpha + _PVS_EPS
                        if zw_hi > beta:
                            zw_hi = beta
                        score = _alpha_beta_anaconda(
                            new_sq, next_player,
                            np.int64(current_depth - 1), np.int64(1),
                            alpha, zw_hi, False, root_player,
                            self.W_pp, self.b_pp, self.W1, self.W2, self.b2, self.W3, self.b3,
                            self.piece_diff, self.king_weight,
                            self.tt_keys, self.tt_value, self.tt_depth, self.tt_flag,
                            self.tt_best_idx,
                            self.killers, self.killers_valid,
                            self.move_bufs,
                        )
                        if score > alpha and score < beta:
                            score = _alpha_beta_anaconda(
                                new_sq, next_player,
                                np.int64(current_depth - 1), np.int64(1),
                                score, beta, False, root_player,
                                self.W_pp, self.b_pp, self.W1, self.W2, self.b2, self.W3, self.b3,
                                self.piece_diff, self.king_weight,
                                self.tt_keys, self.tt_value, self.tt_depth, self.tt_flag,
                                self.tt_best_idx,
                                self.killers, self.killers_valid,
                                self.move_bufs,
                            )
                    if score > iter_best_score:
                        iter_best_score = score
                        iter_best_move = root_py[i]
                        iter_best_idx = i
                    if iter_best_score >= beta0:
                        break
                    if iter_best_score > alpha:
                        alpha = iter_best_score

                if iter_best_score <= alpha0 and alpha0 > -INF32:
                    alpha0 = -INF32
                    continue
                if iter_best_score >= beta0 and beta0 < INF32:
                    beta0 = INF32
                    continue
                break

            best_move = iter_best_move
            best_score = iter_best_score
            prev_best_idx = iter_best_idx
            prev_score = iter_best_score
            prev_score_set = True

        return best_move, float(best_score)

    def __call__(self, board: Board):
        move, _ = self.search(board)
        return move


def warmup_anaconda():
    """Compile the Anaconda JIT alpha-beta function with a tiny run."""
    weights = np.zeros(ANACONDA_TOTAL, dtype=np.float32)
    weights[-2] = 0.5   # piece_diff
    weights[-1] = 2.0   # king_weight
    agent = FastAgentJitAnaconda(weights, depth=3)
    agent.search(Board())
