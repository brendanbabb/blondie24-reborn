"""Head-to-head: CPU JIT alpha-beta vs GPU-batched fixed-depth minimax.

Usage: python bench_gpu_minimax.py [depth]

Both agents use the SAME Anaconda weight vector, search the SAME position,
to the SAME depth. We measure wall-clock per single root search (not a whole
tournament, since the scheduler-based GPU path is a different trade-off).

For each depth:
  CPU JIT  : FastAgentJitAnaconda (alpha-beta + TT)
  GPU batch: BatchGpuAgentAnaconda (no pruning, one forward pass)

The GPU variant has exponentially more leaves (no pruning) but pays ONE
kernel launch, period. Whether it wins depends on branching factor and depth.
"""

import sys
import time
import numpy as np
import torch

from checkers.board import Board
from neural.fast_eval_anaconda import TOTAL as ANACONDA_TOTAL
from search.fast_minimax_jit_anaconda import FastAgentJitAnaconda, warmup_anaconda
from search.fast_minimax_gpu_anaconda import BatchGpuAgentAnaconda


def make_weights(seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    w = rng.normal(0.0, 0.1, size=ANACONDA_TOTAL).astype(np.float32)
    w[-2] = 0.5    # piece_diff
    w[-1] = 2.0    # king_weight
    return w


def time_search(agent, board, reps: int = 3) -> tuple[float, float]:
    """Returns (mean, min) seconds across reps. Discards first run as warmup."""
    # Warm-up (important for GPU: CUDA graph compilation on first call).
    agent.search(board)
    if hasattr(torch.cuda, "synchronize") and torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        move, score = agent.search(board)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.min(times)), move, score


def main():
    depths = [int(a) for a in sys.argv[1:]] or [4, 5, 6]

    print("Warming Numba JIT...", flush=True)
    warmup_anaconda()

    w = make_weights()
    board = Board()  # opening position; 7 legal moves for black

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Weights: {ANACONDA_TOTAL} floats, seed=42")
    print()
    print(f"{'depth':>5}  {'CPU JIT mean':>13}  {'CPU min':>9}  "
          f"{'GPU mean':>10}  {'GPU min':>9}  {'CPU leaves':>10}  {'GPU leaves':>10}  {'ratio':>7}")
    print("-" * 100)

    for d in depths:
        cpu_agent = FastAgentJitAnaconda(w, depth=d)
        gpu_agent = BatchGpuAgentAnaconda(w, depth=d, device=device)

        cpu_mean, cpu_min, cpu_move, cpu_score = time_search(cpu_agent, board)
        gpu_mean, gpu_min, gpu_move, gpu_score = time_search(gpu_agent, board)

        cpu_leaves = "n/a"  # JIT doesn't expose a counter
        gpu_leaves = f"{gpu_agent.nodes_evaluated:,}"

        if cpu_mean < gpu_mean:
            ratio = f"CPU {gpu_mean/cpu_mean:.1f}x"
        else:
            ratio = f"GPU {cpu_mean/gpu_mean:.1f}x"

        same_move = (str(cpu_move) == str(gpu_move))
        match_flag = "=" if same_move else "!"
        print(f"{d:>5}  {cpu_mean*1000:>11.2f}ms  {cpu_min*1000:>7.2f}ms  "
              f"{gpu_mean*1000:>8.2f}ms  {gpu_min*1000:>7.2f}ms  "
              f"{cpu_leaves:>10}  {gpu_leaves:>10}  {ratio:>7}  "
              f"move{match_flag}")

    print()
    print("Notes:")
    print("  - CPU JIT uses alpha-beta + transposition table.")
    print("  - GPU batch uses no pruning; every leaf evaluated once.")
    print("  - move= indicates both agents picked the same root move")
    print("    (not guaranteed: ties can break differently).")


if __name__ == "__main__":
    main()
