"""
Numba-JIT alpha-beta search for Blondie24.

Replaces FastAgent's Python recursion with a fully-compiled version. Move gen,
position encoding, forward pass, TT probe/store, and alpha-beta recursion all
run as native code — no Python interpreter overhead inside the tree.

Profile after FastAgent+TT showed Python recursion self-time was the dominant
remaining cost at depth 8. This module eliminates it.
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
from neural.fast_eval import unpack_weights
from search.fast_minimax import _py_move_key, _fast_move_key


INF32 = np.float32(np.inf)

_TT_EXACT = np.int8(0)
_TT_LOWER = np.int8(1)
_TT_UPPER = np.int8(2)
_TT_SIZE = 1 << 20
_TT_MASK = np.int64(_TT_SIZE - 1)


def _build_zobrist():
    """
    Zobrist table: 32 squares x 5 piece types (indexed by piece_value + 2).
    Index 2 (empty) is zero so empty squares contribute nothing.
    """
    rng = np.random.default_rng(0xB10ED124)
    tbl = rng.integers(1, 1 << 62, size=(32, 5), dtype=np.int64)
    tbl[:, 2] = 0  # empty
    side = np.int64(rng.integers(1, 1 << 62))
    return tbl, side


ZOBRIST_PIECES, ZOBRIST_SIDE = _build_zobrist()


@njit(cache=False, inline='always')
def _hash_position(squares, player):
    h = np.int64(0)
    for sq in range(32):
        idx = np.int64(squares[sq]) + 2
        h ^= ZOBRIST_PIECES[sq, idx]
    if player == -1:
        h ^= ZOBRIST_SIDE
    return h


@njit(cache=False)
def _eval_position(
    squares, player,
    W1, b1, W2, b2, W3, b3, piece_diff, king_weight,
    root_player,
):
    """
    Forward pass on a single position. Returns a float32 score from
    root_player's perspective.
    """
    # Viewpoint encoding: own pieces positive, opponent negative.
    # Pieces -> +/-1, kings -> +/-king_weight.
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

    # Layer 1: 32 -> 40 + tanh
    h1 = np.empty(40, dtype=np.float32)
    for j in range(40):
        s = b1[j]
        for i in range(32):
            s += W1[j, i] * x[i]
        h1[j] = np.float32(np.tanh(s))

    # Layer 2: 40 -> 10 + tanh
    h2 = np.empty(10, dtype=np.float32)
    for j in range(10):
        s = b2[j]
        for i in range(40):
            s += W2[j, i] * h1[i]
        h2[j] = np.float32(np.tanh(s))

    # Layer 3: 10 -> 1
    s3 = b3[0]
    for i in range(10):
        s3 += W3[0, i] * h2[i]

    score = np.float32(np.tanh(s3 + pd_sum * piece_diff))

    if player != root_player:
        score = -score
    return score


@njit(cache=False)
def _alpha_beta(
    squares, player, depth, alpha, beta, maximizing, root_player,
    W1, b1, W2, b2, W3, b3, piece_diff, king_weight,
    tt_keys, tt_value, tt_depth, tt_flag,
    move_bufs,
):
    buf = move_bufs[depth]
    n_moves = get_legal_moves_fast(
        squares, player, NEIGHBORS, JUMP_TARGETS, DIR_DR, buf
    )
    if n_moves == 0:
        # Current player has no legal moves -> loses
        winner = -player
        if winner == root_player:
            return INF32
        else:
            return -INF32

    # ── Leaf level: expand one more ply and eval each child ──
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
                new_sq, next_player, NEIGHBORS, JUMP_TARGETS, DIR_DR, gc_buf
            )
            if gc_count == 0:
                # Child (next_player) has no moves -> player wins
                winner = player
                if winner == root_player:
                    child_score = INF32
                else:
                    child_score = -INF32
            else:
                child_score = _eval_position(
                    new_sq, next_player,
                    W1, b1, W2, b2, W3, b3, piece_diff, king_weight,
                    root_player,
                )
            if maximizing:
                if child_score > best:
                    best = child_score
            else:
                if child_score < best:
                    best = child_score
        return best

    # ── Interior node: TT probe ──
    key = _hash_position(squares, player)
    slot = np.int64(key) & _TT_MASK
    if tt_keys[slot] == key and tt_depth[slot] >= depth:
        v = tt_value[slot]
        f = tt_flag[slot]
        if f == _TT_EXACT:
            return v
        if f == _TT_LOWER and v >= beta:
            return v
        if f == _TT_UPPER and v <= alpha:
            return v

    alpha0 = alpha
    beta0 = beta
    next_player = -player

    if maximizing:
        value = -INF32
        for i in range(n_moves):
            new_sq = apply_move_fast(squares, buf[i])
            child_v = _alpha_beta(
                new_sq, next_player, depth - 1,
                alpha, beta, False, root_player,
                W1, b1, W2, b2, W3, b3, piece_diff, king_weight,
                tt_keys, tt_value, tt_depth, tt_flag,
                move_bufs,
            )
            if child_v > value:
                value = child_v
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break
    else:
        value = INF32
        for i in range(n_moves):
            new_sq = apply_move_fast(squares, buf[i])
            child_v = _alpha_beta(
                new_sq, next_player, depth - 1,
                alpha, beta, True, root_player,
                W1, b1, W2, b2, W3, b3, piece_diff, king_weight,
                tt_keys, tt_value, tt_depth, tt_flag,
                move_bufs,
            )
            if child_v < value:
                value = child_v
            if value < beta:
                beta = value
            if alpha >= beta:
                break

    # ── TT store ──
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

    return value


class FastAgentJit:
    """
    Fully JIT'd alpha-beta agent. Same interface as FastAgent.
    """

    def __init__(self, weights: np.ndarray, depth: int):
        self.depth = depth
        self.weights = np.asarray(weights, dtype=np.float32)
        (W1, b1, W2, b2, W3, b3, piece_diff, king_weight) = unpack_weights(self.weights)
        self.W1 = np.ascontiguousarray(W1, dtype=np.float32)
        self.b1 = np.ascontiguousarray(b1, dtype=np.float32)
        self.W2 = np.ascontiguousarray(W2, dtype=np.float32)
        self.b2 = np.ascontiguousarray(b2, dtype=np.float32)
        self.W3 = np.ascontiguousarray(W3, dtype=np.float32)
        self.b3 = np.ascontiguousarray(b3, dtype=np.float32)
        self.piece_diff = np.float32(piece_diff)
        self.king_weight = np.float32(king_weight)

        # Move buffers: slot 0 is scratch for terminal checks at the leaf
        # level; slots 1..depth-1 are the per-depth buffers used during
        # recursion. We never call _alpha_beta at depth >= self.depth
        # (root calls with depth = self.depth - 1), but allocate self.depth
        # slots to cover it safely.
        self.move_bufs = np.zeros((depth + 1, 64, MOVE_SLOTS), dtype=np.int8)

        # Transposition table
        self.tt_keys  = np.zeros(_TT_SIZE, dtype=np.int64)
        self.tt_value = np.zeros(_TT_SIZE, dtype=np.float32)
        self.tt_depth = np.full(_TT_SIZE, -1, dtype=np.int8)
        self.tt_flag  = np.zeros(_TT_SIZE, dtype=np.int8)

        self.nodes_evaluated = 0  # kept for interface compatibility

    def search(self, board: Board):
        py_moves = board.get_legal_moves()
        if not py_moves:
            return None, -float("inf")
        if len(py_moves) == 1:
            return py_moves[0], 0.0

        # Invalidate TT each search: entries are keyed on (squares, player)
        # but values are relative to root_player, which flips across calls.
        self.tt_depth.fill(-1)

        # Get fast moves for the root
        root_buf = np.zeros((64, MOVE_SLOTS), dtype=np.int8)
        n_root = get_legal_moves_fast(
            board.squares, board.current_player,
            NEIGHBORS, JUMP_TARGETS, DIR_DR, root_buf,
        )
        py_by_key = {_py_move_key(m): m for m in py_moves}

        root_player = np.int8(board.current_player)
        next_player = np.int8(-board.current_player)

        best_move = py_moves[0]
        best_score = -INF32
        alpha = -INF32
        beta = INF32

        for i in range(n_root):
            fast_move = root_buf[i].copy()
            key = _fast_move_key(fast_move)
            py_move = py_by_key[key]

            new_sq = apply_move_fast(board.squares, fast_move)
            score = _alpha_beta(
                new_sq, next_player, np.int64(self.depth - 1),
                alpha, beta, False, root_player,
                self.W1, self.b1, self.W2, self.b2, self.W3, self.b3,
                self.piece_diff, self.king_weight,
                self.tt_keys, self.tt_value, self.tt_depth, self.tt_flag,
                self.move_bufs,
            )
            if score > best_score:
                best_score = score
                best_move = py_move
            if best_score > alpha:
                alpha = best_score

        return best_move, float(best_score)

    def __call__(self, board: Board):
        move, _ = self.search(board)
        return move


def warmup():
    """Compile the JIT alpha-beta function with a tiny run."""
    weights = np.zeros(1743, dtype=np.float32)
    weights[0] = 0.5   # piece_diff
    weights[1] = 2.0   # king_weight
    agent = FastAgentJit(weights, depth=3)
    agent.search(Board())
