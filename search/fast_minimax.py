"""
Fast CPU minimax agent using numpy forward pass + Numba move generation.

No PyTorch in the hot loop. Alpha-beta pruning stays in Python (the profile
showed recursion self-time is <3% of total), but leaf eval and move-gen are
the hot paths that get JIT'd.

At depth=1 we batch-evaluate all children in a single numpy forward pass
— same idea as the GPU batched path in search/minimax.py, but numpy is
fast enough on a 1,743-param net that launch overhead is irrelevant.

Interface: FastAgent is callable with a Board instance so play_game can
use it as a drop-in replacement for MinimaxAgent. The hot inner loop
works on raw (squares, player) pairs — only the outermost call crosses
the Board boundary.
"""

import numpy as np
from checkers.board import Board, BLACK, WHITE
from checkers.fast_board import (
    get_legal_moves_fast,
    apply_move_fast,
    NEIGHBORS,
    JUMP_TARGETS,
    DIR_DR,
    MOVE_SLOTS,
)
from neural.fast_eval import unpack_weights, forward_batch, encode_boards


INF = float("inf")

# Transposition table flags
_TT_EXACT = 0
_TT_LOWER = 1
_TT_UPPER = 2
_TT_SIZE = 1 << 20  # 1M slots ≈ 20 MB per agent
_TT_MASK = _TT_SIZE - 1
_SIDE_XOR = 0x9E3779B97F4A7C15  # golden-ratio constant, xor'd when player == -1


def _pos_key(squares, player):
    """Fast position hash. Python's bytes hash is ~300ns and well-distributed."""
    h = hash(squares.tobytes())
    if player == -1:
        h ^= _SIDE_XOR
    return h & 0x7FFFFFFFFFFFFFFF  # clamp to positive int64 range


class FastAgent:
    """
    CPU-only minimax agent. Takes a flat weight vector (what Individuals
    already store) — no CheckersNet object required.
    """

    def __init__(self, weights: np.ndarray, depth: int, use_tt: bool = True):
        self.depth = depth
        self.use_tt = use_tt
        self.weights = np.asarray(weights, dtype=np.float32)
        (self.W1, self.b1,
         self.W2, self.b2,
         self.W3, self.b3,
         self.piece_diff,
         self.king_weight) = unpack_weights(self.weights)

        # Reusable move-gen buffer (one per depth level to avoid aliasing
        # during recursion).
        self._move_bufs = [np.zeros((64, MOVE_SLOTS), dtype=np.int8)
                           for _ in range(depth + 2)]

        # Transposition table (open-addressing, replace-always).
        # Arrays live on the agent so each worker/game gets its own TT.
        self.tt_keys  = np.zeros(_TT_SIZE, dtype=np.int64)
        self.tt_value = np.zeros(_TT_SIZE, dtype=np.float32)
        self.tt_depth = np.full(_TT_SIZE, -1, dtype=np.int8)
        self.tt_flag  = np.zeros(_TT_SIZE, dtype=np.int8)

        self.nodes_evaluated = 0
        self.tt_hits = 0
        self.tt_stores = 0

    # ---------- Evaluation ----------

    def _eval_batch(self, squares_list, players, root_player):
        """Batch evaluate a list of (squares, player) pairs."""
        n = len(squares_list)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        squares_batch = np.stack(squares_list)
        player_batch = np.asarray(players, dtype=np.int8)
        x = encode_boards(squares_batch, player_batch, self.king_weight)
        scores = forward_batch(x, self.W1, self.b1, self.W2, self.b2,
                                self.W3, self.b3, self.piece_diff)
        # Flip sign for boards whose current player differs from root
        signs = np.where(player_batch == root_player, 1.0, -1.0).astype(np.float32)
        self.nodes_evaluated += n
        return scores * signs

    # ---------- Alpha-beta ----------

    def _alpha_beta(self, squares, player, depth, alpha, beta,
                    maximizing, root_player):
        """
        Standard alpha-beta with batched leaf expansion at depth == 1.

        Returns the score from root_player's perspective.
        """
        buf = self._move_bufs[depth]
        n_moves = get_legal_moves_fast(
            squares, player, NEIGHBORS, JUMP_TARGETS, DIR_DR, buf
        )
        if n_moves == 0:
            # Current player has no moves → they lose
            winner = -player
            if winner == root_player:
                return INF
            else:
                return -INF

        # ── Batched leaf level ──
        if depth == 1:
            next_player = -player
            child_squares = []
            child_players = []
            terminal_scores = {}   # idx -> score
            for i in range(n_moves):
                move = buf[i]
                new_sq = apply_move_fast(squares, move)
                # Terminal check for the child
                gc_buf = self._move_bufs[0]
                gc_count = get_legal_moves_fast(
                    new_sq, next_player, NEIGHBORS, JUMP_TARGETS, DIR_DR, gc_buf
                )
                if gc_count == 0:
                    winner = -next_player  # i.e. player
                    if winner == root_player:
                        terminal_scores[i] = INF
                    else:
                        terminal_scores[i] = -INF
                else:
                    child_squares.append(new_sq)
                    child_players.append(next_player)
                    terminal_scores[i] = None  # pending

            nt_indices = [i for i in range(n_moves) if terminal_scores[i] is None]
            nt_scores = self._eval_batch(child_squares, child_players, root_player)

            scores = np.empty(n_moves, dtype=np.float32)
            j = 0
            for i in range(n_moves):
                if terminal_scores[i] is None:
                    scores[i] = nt_scores[j]
                    j += 1
                else:
                    scores[i] = terminal_scores[i]

            if maximizing:
                return float(scores.max())
            else:
                return float(scores.min())

        # ── Interior node: TT probe + recurse with pruning ──
        next_player = -player

        if self.use_tt:
            key = _pos_key(squares, player)
            slot = key & _TT_MASK
            if self.tt_keys[slot] == key and self.tt_depth[slot] >= depth:
                v = float(self.tt_value[slot])
                f = self.tt_flag[slot]
                if f == _TT_EXACT:
                    self.tt_hits += 1
                    return v
                if f == _TT_LOWER and v >= beta:
                    self.tt_hits += 1
                    return v
                if f == _TT_UPPER and v <= alpha:
                    self.tt_hits += 1
                    return v

        alpha0, beta0 = alpha, beta  # original window for flag decision

        if maximizing:
            value = -INF
            for i in range(n_moves):
                move = buf[i]
                new_sq = apply_move_fast(squares, move)
                child_v = self._alpha_beta(
                    new_sq, next_player, depth - 1,
                    alpha, beta, False, root_player,
                )
                if child_v > value:
                    value = child_v
                if value > alpha:
                    alpha = value
                if alpha >= beta:
                    break
        else:
            value = INF
            for i in range(n_moves):
                move = buf[i]
                new_sq = apply_move_fast(squares, move)
                child_v = self._alpha_beta(
                    new_sq, next_player, depth - 1,
                    alpha, beta, True, root_player,
                )
                if child_v < value:
                    value = child_v
                if value < beta:
                    beta = value
                if alpha >= beta:
                    break

        if self.use_tt:
            if value <= alpha0:
                flag = _TT_UPPER
            elif value >= beta0:
                flag = _TT_LOWER
            else:
                flag = _TT_EXACT
            self.tt_keys[slot]  = key
            self.tt_value[slot] = value
            self.tt_depth[slot] = depth
            self.tt_flag[slot]  = flag
            self.tt_stores += 1
        return value

    # ---------- Root search ----------

    def search(self, board: Board):
        """
        Find the best move. Returns (python_move, score) where python_move
        is in the Board.apply_move format.
        """
        self.nodes_evaluated = 0
        self.tt_hits = 0
        self.tt_stores = 0
        if self.use_tt:
            # Invalidate TT: entries are keyed on (squares, player) but their
            # value is from root_player's perspective, which flips each move.
            self.tt_depth.fill(-1)
        py_moves = board.get_legal_moves()
        if not py_moves:
            return None, -INF
        if len(py_moves) == 1:
            return py_moves[0], 0.0

        # Build fast moves from the same Board so we can pair py_move ↔ fast_move.
        root_buf = self._move_bufs[self.depth + 1]
        n_root = get_legal_moves_fast(
            board.squares, board.current_player,
            NEIGHBORS, JUMP_TARGETS, DIR_DR, root_buf,
        )
        # The two move lists must be the same set; we just need a consistent
        # ordering. Match by re-converting each fast move into a hashable form
        # and looking up against the py_moves list.
        py_by_key = {_py_move_key(m): m for m in py_moves}

        root_player = board.current_player
        next_player = -root_player

        best_move = py_moves[0]
        best_score = -INF
        alpha, beta = -INF, INF

        for i in range(n_root):
            fast_move = root_buf[i].copy()  # copy because inner search reuses bufs
            key = _fast_move_key(fast_move)
            py_move = py_by_key[key]

            new_sq = apply_move_fast(board.squares, fast_move)
            score = self._alpha_beta(
                new_sq, next_player, self.depth - 1,
                alpha, beta, False, root_player,
            )
            if score > best_score:
                best_score = score
                best_move = py_move
            if best_score > alpha:
                alpha = best_score

        return best_move, best_score

    def __call__(self, board: Board):
        move, _ = self.search(board)
        return move


# ---------- Move key helpers ----------

def _py_move_key(move):
    """Convert a Python Board move into a hashable canonical form."""
    if isinstance(move, tuple):
        return ("s", int(move[0]), int(move[1]))
    return ("j",) + tuple(int(x) for x in move)


def _fast_move_key(move):
    """Convert a fast-move row (len-17 int8) into the same canonical form."""
    kind = int(move[0])
    if kind == 0:
        return ("s", int(move[2]), int(move[3]))
    n_caps = int(move[1])
    path = [int(move[2])]
    for k in range(n_caps):
        path.append(int(move[3 + 2 * k]))
        path.append(int(move[4 + 2 * k]))
    return ("j",) + tuple(path)
