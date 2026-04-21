"""Verify FastAgentJitAnaconda (IDDFS+killers) scores match the
unpruned-minimax GPU reference on random positions at depth 4.

Alpha-beta with any move ordering returns the same minimax value as a
full no-pruning search — so the IDDFS+killers JIT score should equal the
BatchGpuAgentAnaconda score for the best root move to within
float32 rounding.
"""

import sys, time
import numpy as np
import torch

sys.path.insert(0, ".")
from checkers.board import Board
from neural.fast_eval_anaconda import TOTAL as ANACONDA_TOTAL
from search.fast_minimax_jit_anaconda import FastAgentJitAnaconda, warmup_anaconda
from search.fast_minimax_gpu_anaconda import BatchGpuAgentAnaconda

DEPTH = 4
N_POS = 6
SEED = 42


def random_position(rng, max_ply=10):
    board = Board()
    for _ in range(max_ply):
        moves = board.get_legal_moves()
        if not moves or board.is_game_over()[0]:
            break
        board = board.apply_move(moves[rng.integers(len(moves))])
    return board


def main():
    print("warming JIT...", flush=True)
    warmup_anaconda()
    print("ready\n", flush=True)

    rng = np.random.default_rng(SEED)
    w = rng.normal(0, 0.2, ANACONDA_TOTAL).astype(np.float32)
    w[-2] = 0.5
    w[-1] = 2.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU ref device: {device}", flush=True)

    # Quiescence off: GPU reference does unpruned minimax with no quiescence,
    # so keeping it on here would produce expected score divergence.
    jit_agent = FastAgentJitAnaconda(w, depth=DEPTH, use_quiescence=False)
    gpu_agent = BatchGpuAgentAnaconda(w, depth=DEPTH, device=device)

    positions = [random_position(rng) for _ in range(N_POS)]

    disagreements = 0
    for i, b in enumerate(positions):
        if b.is_game_over()[0] or not b.get_legal_moves():
            continue
        m_jit, s_jit = jit_agent.search(b)
        m_gpu, s_gpu = gpu_agent.search(b)
        same_move = m_jit == m_gpu
        score_diff = abs(s_jit - s_gpu)
        tag = "OK" if score_diff < 1e-3 else "DIFF"
        if score_diff >= 1e-3:
            disagreements += 1
        print(f"  pos {i}: turn={b.current_player:+d} squares={b.squares.tolist()[:8]}... "
              f"jit=({m_jit}, {s_jit:+.5f}) gpu=({m_gpu}, {s_gpu:+.5f}) "
              f"delta={score_diff:.2e} same_move={same_move} [{tag}]", flush=True)

    print(f"\nscore disagreements: {disagreements}/{len(positions)}")


if __name__ == "__main__":
    main()
