"""
Correctness + speedup test for the transposition table in FastAgent.

For each random position, compare (best_move, best_score) from a no-TT agent
against a with-TT agent at the same depth. They must agree.
"""
import time
import numpy as np
from checkers.board import Board
from search.fast_minimax import FastAgent
N_WEIGHTS = 1743  # matches CheckersNet parameter count

DEPTHS = [4, 6, 8]
N_POSITIONS = 15
SEED = 12345


def random_position(rng, max_ply=12):
    """Play max_ply random moves from the start position."""
    board = Board()
    for _ in range(max_ply):
        moves = board.get_legal_moves()
        if not moves or board.is_game_over()[0]:
            break
        board.apply_move(moves[rng.integers(len(moves))])
    return board


def main():
    rng = np.random.default_rng(SEED)
    # Random weights with the piece-diff and king-weight slots set to
    # sensible values (otherwise the eval is useless).
    weights = rng.normal(0, 0.2, N_WEIGHTS).astype(np.float32)
    weights[0] = 0.5   # piece_diff
    weights[1] = 2.0   # king_weight

    positions = [random_position(rng) for _ in range(N_POSITIONS)]

    for depth in DEPTHS:
        print(f"\n=== depth {depth} ===")
        no_tt = FastAgent(weights, depth=depth, use_tt=False)
        with_tt = FastAgent(weights, depth=depth, use_tt=True)

        disagreements = 0
        t_no_tt = 0.0
        t_with_tt = 0.0
        total_nodes_no_tt = 0
        total_nodes_tt = 0
        total_tt_hits = 0

        for i, board in enumerate(positions):
            if board.is_game_over()[0] or not board.get_legal_moves():
                continue

            t0 = time.perf_counter()
            m1, s1 = no_tt.search(board)
            t_no_tt += time.perf_counter() - t0
            n1 = no_tt.nodes_evaluated

            t0 = time.perf_counter()
            m2, s2 = with_tt.search(board)
            t_with_tt += time.perf_counter() - t0
            n2 = with_tt.nodes_evaluated

            total_nodes_no_tt += n1
            total_nodes_tt += n2
            total_tt_hits += with_tt.tt_hits

            # Scores must match exactly (alpha-beta with/without TT is exact).
            score_match = abs(s1 - s2) < 1e-5
            # Moves may differ on ties, but the score should still match.
            if not score_match:
                disagreements += 1
                print(f"  pos {i}: MISMATCH score {s1:.4f} vs {s2:.4f} "
                      f"move {m1} vs {m2}")

        speedup = t_no_tt / t_with_tt if t_with_tt > 0 else float('nan')
        node_ratio = total_nodes_no_tt / max(1, total_nodes_tt)
        print(f"  disagreements: {disagreements}/{len(positions)}")
        print(f"  no-TT:   {t_no_tt:6.2f}s  ({total_nodes_no_tt:>10d} leaf evals)")
        print(f"  with-TT: {t_with_tt:6.2f}s  ({total_nodes_tt:>10d} leaf evals, "
              f"{total_tt_hits} TT hits)")
        print(f"  speedup: {speedup:.2f}x   leaf-eval ratio: {node_ratio:.2f}x")


if __name__ == "__main__":
    main()
