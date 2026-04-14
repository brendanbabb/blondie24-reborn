"""
Minimax search with alpha-beta pruning and batched GPU evaluation.

The evaluation function is a neural network — this is the core of Blondie24.
Instead of hand-crafted heuristics, the network learns its own evaluation
through evolution.

GPU OPTIMIZATION STRATEGY:
The naive approach evaluates one board per forward pass — terrible for GPU
utilization since the network is tiny (1742 params) and GPU kernel launch
overhead dominates. Instead we batch evaluations:

  When depth == 1 (one level above leaves), generate ALL child boards,
  encode them into a single tensor, run one GPU forward pass, then do
  min/max selection over the results.

For depth-4 search with ~7 moves per position, alpha-beta typically visits
~500-2000 leaf nodes per root move. Batching these into groups of 7-15
(one per depth-1 node's children) gives 3-5x speedup over one-at-a-time.
At depth-6 on an RTX 5060, this is the difference between minutes and
seconds per generation.
"""

import torch
import numpy as np
from checkers.board import Board, BLACK, WHITE, EMPTY, BLACK_PIECE, BLACK_KING, WHITE_PIECE, WHITE_KING
from neural.network import CheckersNet
from config import SearchConfig

# Large values for terminal states
INF = float("inf")


def _encode_board_fast(board: Board, king_weight: float) -> np.ndarray:
    """
    Fast board encoding using vectorized numpy operations.
    Returns a numpy array (not a torch tensor) to minimize overhead.
    """
    squares = board.squares
    player = board.current_player
    encoding = np.zeros(32, dtype=np.float32)

    if player == BLACK:
        mask_bp = squares == BLACK_PIECE
        mask_bk = squares == BLACK_KING
        mask_wp = squares == WHITE_PIECE
        mask_wk = squares == WHITE_KING
        encoding[mask_bp] = 1.0
        encoding[mask_bk] = king_weight
        encoding[mask_wp] = -1.0
        encoding[mask_wk] = -king_weight
    else:
        mask_wp = squares == WHITE_PIECE
        mask_wk = squares == WHITE_KING
        mask_bp = squares == BLACK_PIECE
        mask_bk = squares == BLACK_KING
        encoding[mask_wp] = 1.0
        encoding[mask_wk] = king_weight
        encoding[mask_bp] = -1.0
        encoding[mask_bk] = -king_weight

    return encoding


class MinimaxAgent:
    """
    Minimax agent with GPU-optimized neural network evaluation.

    Keeps the network on GPU and reuses tensors to minimize overhead.
    Uses batched evaluation at the penultimate search level.
    """

    def __init__(self, network: CheckersNet, config: SearchConfig, device: torch.device):
        self.network = network.to(device)
        self.network.eval()
        self.config = config
        self.device = device
        self.king_weight = network.king_weight.item()

        # Stats for performance monitoring
        self.nodes_evaluated = 0
        self.batch_calls = 0

        # Pre-allocate a reusable tensor for single evaluations
        self._single_buf = torch.zeros(1, 32, dtype=torch.float32, device=device)

        # Pre-allocate batch buffer (grows as needed)
        self._batch_size = 256
        self._batch_buf = torch.zeros(self._batch_size, 32, dtype=torch.float32, device=device)

    def _evaluate_single(self, board: Board, root_player: int) -> float:
        """Evaluate a single board position on GPU."""
        self.nodes_evaluated += 1
        encoding = _encode_board_fast(board, self.king_weight)
        self._single_buf[0] = torch.from_numpy(encoding)
        score = self.network(self._single_buf)[0, 0].item()

        if board.current_player != root_player:
            score = -score
        return score

    def _evaluate_batch(self, boards: list[Board], root_player: int) -> list[float]:
        """
        Evaluate multiple board positions in a single GPU forward pass.
        This is where the RTX 5060 earns its keep.
        """
        n = len(boards)
        if n == 0:
            return []

        self.nodes_evaluated += n
        self.batch_calls += 1

        # Grow batch buffer if needed
        if n > self._batch_size:
            self._batch_size = n + 64
            self._batch_buf = torch.zeros(self._batch_size, 32, dtype=torch.float32, device=self.device)

        # Encode all boards into a single stacked numpy array, then do ONE
        # host→device transfer. Avoids N tiny per-board GPU copies.
        stacked = np.empty((n, 32), dtype=np.float32)
        signs = np.empty(n, dtype=np.float32)
        for i, board in enumerate(boards):
            stacked[i] = _encode_board_fast(board, self.king_weight)
            signs[i] = -1.0 if board.current_player != root_player else 1.0

        self._batch_buf[:n].copy_(torch.from_numpy(stacked), non_blocking=True)

        # Single forward pass + sign-flip on-device before the sync
        scores = self.network(self._batch_buf[:n]).squeeze(-1)
        signs_t = torch.from_numpy(signs).to(self.device, non_blocking=True)
        scores = scores * signs_t

        return scores.cpu().tolist()

    def _alpha_beta(self, board: Board, depth: int, alpha: float, beta: float,
                    maximizing: bool, root_player: int) -> float:
        """Standard alpha-beta with single-board evaluation at leaves."""
        game_over, winner = board.is_game_over()
        if game_over:
            if winner == root_player:
                return INF
            elif winner is None:
                return 0.0
            else:
                return -INF

        if depth == 0:
            return self._evaluate_single(board, root_player)

        moves = board.get_legal_moves()

        if maximizing:
            value = -INF
            for move in moves:
                child = board.apply_move(move)
                value = max(value, self._alpha_beta(child, depth - 1, alpha, beta, False, root_player))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = INF
            for move in moves:
                child = board.apply_move(move)
                value = min(value, self._alpha_beta(child, depth - 1, alpha, beta, True, root_player))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    def _alpha_beta_batched(self, board: Board, depth: int, alpha: float, beta: float,
                            maximizing: bool, root_player: int) -> float:
        """
        Alpha-beta variant that batches leaf evaluations when depth == 1.

        When one level above leaves, generates ALL children, batch-evaluates
        them in a single GPU pass, then does min/max selection. Trades a small
        amount of pruning efficiency for massive GPU throughput gains.
        """
        game_over, winner = board.is_game_over()
        if game_over:
            if winner == root_player:
                return INF
            elif winner is None:
                return 0.0
            else:
                return -INF

        # === TWO-LEVEL BATCHED EXPANSION (depth == 2) ===
        # Expand both children and grandchildren, then evaluate ALL non-terminal
        # grandchildren in ONE forward pass. Gives up alpha-beta pruning at the
        # last two ply in exchange for ~7× larger GPU batches (typically ~49
        # boards vs ~7). Only worth it when the batch-size win exceeds the
        # extra-nodes cost — tune via the benchmark.
        if depth == 2:
            moves = board.get_legal_moves()
            children = [board.apply_move(m) for m in moves]
            child_is_max = not maximizing
            child_scores: list = [None] * len(children)

            nonterminal_gcs: list = []
            child_nt_indices: list = [[] for _ in children]
            child_terminal_gc_scores: list = [[] for _ in children]

            for ci, child in enumerate(children):
                go, w = child.is_game_over()
                if go:
                    if w == root_player:
                        child_scores[ci] = INF
                    elif w is None:
                        child_scores[ci] = 0.0
                    else:
                        child_scores[ci] = -INF
                    continue

                for gc_move in child.get_legal_moves():
                    gc = child.apply_move(gc_move)
                    go2, w2 = gc.is_game_over()
                    if go2:
                        if w2 == root_player:
                            child_terminal_gc_scores[ci].append(INF)
                        elif w2 is None:
                            child_terminal_gc_scores[ci].append(0.0)
                        else:
                            child_terminal_gc_scores[ci].append(-INF)
                    else:
                        child_nt_indices[ci].append(len(nonterminal_gcs))
                        nonterminal_gcs.append(gc)

            nt_scores = (self._evaluate_batch(nonterminal_gcs, root_player)
                         if nonterminal_gcs else [])

            for ci in range(len(children)):
                if child_scores[ci] is not None:
                    continue
                gc_scores = [nt_scores[idx] for idx in child_nt_indices[ci]]
                gc_scores.extend(child_terminal_gc_scores[ci])
                if not gc_scores:
                    child_scores[ci] = self._evaluate_single(children[ci], root_player)
                    continue
                child_scores[ci] = max(gc_scores) if child_is_max else min(gc_scores)

            return max(child_scores) if maximizing else min(child_scores)

        moves = board.get_legal_moves()

        # === BATCHED LEAF LEVEL (depth == 1, fallback for shallow searches) ===
        if depth == 1:
            children = [board.apply_move(m) for m in moves]

            # Separate terminal from non-terminal children
            terminal_scores = {}
            non_terminal_boards = []
            non_terminal_indices = []

            for i, child in enumerate(children):
                go, w = child.is_game_over()
                if go:
                    if w == root_player:
                        terminal_scores[i] = INF
                    elif w is None:
                        terminal_scores[i] = 0.0
                    else:
                        terminal_scores[i] = -INF
                else:
                    non_terminal_boards.append(child)
                    non_terminal_indices.append(i)

            # Batch evaluate non-terminal children
            batch_scores = self._evaluate_batch(non_terminal_boards, root_player) if non_terminal_boards else []

            # Reconstruct full score list
            scores = [0.0] * len(children)
            for idx, score in terminal_scores.items():
                scores[idx] = score
            for list_idx, orig_idx in enumerate(non_terminal_indices):
                scores[orig_idx] = batch_scores[list_idx]

            if maximizing:
                return max(scores)
            else:
                return min(scores)

        # Leaf node fallback (shouldn't normally reach here)
        if depth == 0:
            return self._evaluate_single(board, root_player)

        # === STANDARD ALPHA-BETA FOR INTERIOR NODES ===
        if maximizing:
            value = -INF
            for move in moves:
                child = board.apply_move(move)
                value = max(value, self._alpha_beta_batched(
                    child, depth - 1, alpha, beta, False, root_player))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = INF
            for move in moves:
                child = board.apply_move(move)
                value = min(value, self._alpha_beta_batched(
                    child, depth - 1, alpha, beta, True, root_player))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    def search(self, board: Board) -> tuple:
        """
        Find the best move for the current player.

        Uses batched leaf evaluation when on GPU and depth >= 3.

        Returns:
            (best_move, best_score)
        """
        self.nodes_evaluated = 0
        self.batch_calls = 0

        moves = board.get_legal_moves()
        if not moves:
            return None, -INF
        if len(moves) == 1:
            return moves[0], 0.0

        root_player = board.current_player
        use_batched = self.config.depth >= 2 and self.device.type != "cpu"

        best_move = moves[0]
        best_score = -INF

        for move in moves:
            child = board.apply_move(move)
            if use_batched:
                score = self._alpha_beta_batched(
                    child, self.config.depth - 1, -INF, INF, False, root_player)
            else:
                score = self._alpha_beta(
                    child, self.config.depth - 1, -INF, INF, False, root_player)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move, best_score

    def __call__(self, board: Board):
        """Make the agent callable for use with play_game."""
        move, _ = self.search(board)
        return move


# === Backward-compatible API ===

def minimax_search(board, network, config=SearchConfig(), device="cpu"):
    """Legacy API — wraps MinimaxAgent for backward compatibility."""
    if isinstance(device, str):
        device = torch.device(device)
    agent = MinimaxAgent(network, config, device)
    return agent.search(board)


def make_agent(network, config=SearchConfig(), device="cpu"):
    """Create a callable agent from a network + search config."""
    if isinstance(device, str):
        device = torch.device(device)
    return MinimaxAgent(network, config, device)
