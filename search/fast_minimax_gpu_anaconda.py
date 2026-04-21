"""
Fixed-depth, no-pruning minimax for the 2001 Anaconda network that batches
*all* leaf evaluations into a single GPU forward pass per search.

The CPU JIT path (FastAgentJitAnaconda) wins by 100-220x on small populations
because the scheduler in search/parallel_search.py iterates per-tick in
Python — batches end up small and kernel launch overhead dominates. This
module flips the trade-off: no alpha-beta (so a wider tree), but exactly
one forward pass — no per-tick Python overhead, no per-network grouping.

The tree is enumerated on CPU with the JIT move-gen helpers
(get_legal_moves_fast + apply_move_fast) and stored as flat parallel arrays.
After expansion the non-terminal leaves are encoded once, shoved through
AnacondaNet.forward as a single batched call, and back-propagated in a
Python pass.

Only useful at larger search depths where the leaf count dwarfs CPU JIT
wall time. Benchmark before using.
"""

import numpy as np
import torch

from checkers.board import Board
from checkers.fast_board import (
    get_legal_moves_fast,
    apply_move_fast,
    NEIGHBORS,
    JUMP_TARGETS,
    DIR_DR,
    MOVE_SLOTS,
)
from neural.anaconda_network import AnacondaNet
from neural.fast_eval_anaconda import TOTAL as ANACONDA_TOTAL
from search.fast_minimax import _py_move_key, _fast_move_key


INF = float("inf")


def _encode_ana(squares: np.ndarray, player: int, king_weight: float) -> np.ndarray:
    """Anaconda 32-vector input from side-to-move perspective. Mirrors
    _eval_position_anaconda in fast_minimax_jit_anaconda.py. Kept for tests."""
    sp = squares.astype(np.int8) * np.int8(player)
    abs_sp = np.abs(sp).astype(np.float32)
    mag = np.where(abs_sp <= 1.0, abs_sp, np.float32(king_weight))
    return (np.sign(sp).astype(np.float32) * mag).astype(np.float32)


def _encode_ana_batch(
    squares_batch: np.ndarray,   # (N, 32) int8
    players: np.ndarray,         # (N,) int8
    king_weight: float,
) -> np.ndarray:
    """Vectorized encoding across many leaves. (N, 32) float32 output."""
    sp = squares_batch.astype(np.int8) * players[:, None].astype(np.int8)
    abs_sp = np.abs(sp).astype(np.float32)
    mag = np.where(abs_sp <= 1.0, abs_sp, np.float32(king_weight))
    return np.sign(sp).astype(np.float32) * mag


class BatchGpuAgentAnaconda:
    """
    Fixed-depth minimax (no alpha-beta). Expands the full tree on CPU,
    evaluates every non-terminal leaf in ONE GPU forward pass, then backs
    up scores with a single linear-time pass.

    Constructor takes a flat weight vector (same contract as
    FastAgentJitAnaconda) so it slots into the same benches.
    """

    def __init__(self, weights: np.ndarray, depth: int, device: str | torch.device):
        if len(weights) != ANACONDA_TOTAL:
            raise ValueError(
                f"BatchGpuAgentAnaconda expects {ANACONDA_TOTAL}-weight vector, "
                f"got {len(weights)}"
            )
        self.depth = int(depth)
        self.device = torch.device(device)
        self.net = AnacondaNet()
        self.net.set_weight_vector(np.asarray(weights, dtype=np.float32))
        self.net = self.net.to(self.device).eval()
        self.king_weight = float(self.net.king_weight.item())

        self.nodes_evaluated = 0

    @torch.no_grad()
    def search(self, board: Board):
        py_moves = board.get_legal_moves()
        if not py_moves:
            return None, -INF
        if len(py_moves) == 1:
            return py_moves[0], 0.0

        root_player = int(board.current_player)

        squares_list: list[np.ndarray] = [
            np.asarray(board.squares, dtype=np.int8).copy()
        ]
        player_list: list[int] = [root_player]
        parent: list[int] = [-1]
        is_max: list[bool] = [True]
        terminal: list[bool] = [False]
        term_score: list[float] = [0.0]
        children: list[list[int]] = [[]]

        buf = np.zeros((64, MOVE_SLOTS), dtype=np.int8)

        frontier = [0]
        for _d in range(self.depth):
            next_frontier: list[int] = []
            for node_idx in frontier:
                if terminal[node_idx]:
                    continue
                sq = squares_list[node_idx]
                p = player_list[node_idx]
                n_moves = get_legal_moves_fast(
                    sq, p, NEIGHBORS, JUMP_TARGETS, DIR_DR, buf,
                )
                if n_moves == 0:
                    terminal[node_idx] = True
                    winner = -p
                    term_score[node_idx] = INF if winner == root_player else -INF
                    continue
                for mi in range(n_moves):
                    move = buf[mi]
                    new_sq = apply_move_fast(sq, move)
                    ci = len(squares_list)
                    squares_list.append(new_sq)
                    player_list.append(-p)
                    parent.append(node_idx)
                    is_max.append((-p) == root_player)
                    terminal.append(False)
                    term_score.append(0.0)
                    children.append([])
                    children[node_idx].append(ci)
                    next_frontier.append(ci)
            frontier = next_frontier

        total = len(squares_list)
        nonterm_leaves = [
            i for i in range(total) if not terminal[i] and not children[i]
        ]
        self.nodes_evaluated = total

        scores = np.zeros(total, dtype=np.float32)
        if nonterm_leaves:
            sq_stack = np.stack([squares_list[i] for i in nonterm_leaves])
            pl_arr = np.fromiter(
                (player_list[i] for i in nonterm_leaves),
                dtype=np.int8, count=len(nonterm_leaves),
            )
            xs = _encode_ana_batch(sq_stack, pl_arr, self.king_weight)
            signs = np.where(pl_arr == np.int8(root_player), 1.0, -1.0).astype(np.float32)

            x_t = torch.from_numpy(xs).to(self.device)
            out = self.net.forward(x_t).view(-1).cpu().numpy()
            scored = out * signs
            for k, i in enumerate(nonterm_leaves):
                scores[i] = scored[k]

        for i in range(total):
            if terminal[i]:
                scores[i] = term_score[i]

        # Bottom-up pass. parent[i] < i, so walking in reverse index order
        # evaluates children before their parent.
        for i in range(total - 1, -1, -1):
            if terminal[i] or not children[i]:
                continue
            cs = children[i]
            if is_max[i]:
                best = -INF
                for c in cs:
                    s = scores[c]
                    if s > best:
                        best = s
                scores[i] = best
            else:
                best = INF
                for c in cs:
                    s = scores[c]
                    if s < best:
                        best = s
                scores[i] = best

        root_fast = np.zeros((64, MOVE_SLOTS), dtype=np.int8)
        n_root = get_legal_moves_fast(
            np.asarray(board.squares, dtype=np.int8),
            root_player, NEIGHBORS, JUMP_TARGETS, DIR_DR, root_fast,
        )
        if n_root != len(children[0]):
            raise RuntimeError(
                f"root expansion mismatch: fast move-gen says {n_root}, "
                f"tree has {len(children[0])} direct children"
            )
        py_by_key = {_py_move_key(m): m for m in py_moves}

        best_py = py_moves[0]
        best_score = -INF
        for mi, ci in enumerate(children[0]):
            s = scores[ci]
            if s > best_score:
                best_score = s
                best_py = py_by_key[_fast_move_key(root_fast[mi])]

        return best_py, float(best_score)

    def __call__(self, board: Board):
        move, _ = self.search(board)
        return move
