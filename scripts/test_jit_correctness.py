"""
Correctness + speedup test for FastAgentJit vs FastAgent.
Scores should be bit-similar (same alpha-beta, same TT, JIT math).
"""
import time
import numpy as np
from checkers.board import Board
from search.fast_minimax import FastAgent
from search.fast_minimax_jit import FastAgentJit, warmup

N_WEIGHTS = 1743
DEPTHS = [4, 6, 8]
N_POSITIONS = 15
SEED = 12345


def random_position(rng, max_ply=12):
    board = Board()
    for _ in range(max_ply):
        moves = board.get_legal_moves()
        if not moves or board.is_game_over()[0]:
            break
        board = board.apply_move(moves[rng.integers(len(moves))])
    return board


def main():
    print("Warming JIT...")
    warmup()
    print("Ready.\n")

    rng = np.random.default_rng(SEED)
    weights = rng.normal(0, 0.2, N_WEIGHTS).astype(np.float32)
    weights[0] = 0.5
    weights[1] = 2.0

    positions = [random_position(rng) for _ in range(N_POSITIONS)]

    for depth in DEPTHS:
        print(f"=== depth {depth} ===")
        py_agent = FastAgent(weights, depth=depth, use_tt=True)
        # Quiescence off: reference FastAgent has no quiescence, so keeping it on
        # here would produce expected score divergence and mask real bugs.
        jit_agent = FastAgentJit(weights, depth=depth, use_quiescence=False)

        disagreements = 0
        t_py = 0.0
        t_jit = 0.0

        for i, board in enumerate(positions):
            if board.is_game_over()[0] or not board.get_legal_moves():
                continue

            t0 = time.perf_counter()
            m_py, s_py = py_agent.search(board)
            t_py += time.perf_counter() - t0

            t0 = time.perf_counter()
            m_jit, s_jit = jit_agent.search(board)
            t_jit += time.perf_counter() - t0

            # Scores should match to within float32 rounding of the MLP math.
            # Allow a small epsilon because tanh is computed differently between
            # numpy (batched) and jit (scalar loops).
            if abs(s_py - s_jit) > 1e-3:
                disagreements += 1
                print(f"  pos {i}: score mismatch py={s_py:.5f} jit={s_jit:.5f} "
                      f"move py={m_py} jit={m_jit}")

        speedup = t_py / t_jit if t_jit > 0 else float('nan')
        print(f"  disagreements: {disagreements}/{len(positions)}")
        print(f"  python (TT):   {t_py:7.2f}s")
        print(f"  jit (TT):      {t_jit:7.2f}s")
        print(f"  speedup:       {speedup:.2f}x\n")


if __name__ == "__main__":
    main()
