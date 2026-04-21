"""Per-position timing bench for IDDFS+killers JIT vs depths.

Warms the JIT once, then times a sweep of (network, depth, position) with
a fresh agent per (network, depth). Records first-call + steady-state
times averaged over N_POSITIONS random mid-game boards.
"""

import sys, time
import numpy as np

sys.path.insert(0, ".")
from checkers.board import Board
from neural.fast_eval_anaconda import TOTAL as ANACONDA_TOTAL
from search.fast_minimax_jit import FastAgentJit, warmup as warmup_1999
from search.fast_minimax_jit_anaconda import FastAgentJitAnaconda, warmup_anaconda

N_POSITIONS = 8
SEED = 2026


def random_position(rng, max_ply=8):
    b = Board()
    for _ in range(max_ply):
        moves = b.get_legal_moves()
        if not moves or b.is_game_over()[0]:
            break
        b = b.apply_move(moves[rng.integers(len(moves))])
    return b


def bench_agent(agent_factory, positions, depth):
    agent = agent_factory(depth)
    # First call — each agent has its own compile warmth but TT is cold.
    t0 = time.perf_counter()
    agent.search(positions[0])
    first = (time.perf_counter() - t0) * 1000

    # Steady state: mean over the rest.
    times = []
    for b in positions[1:]:
        t0 = time.perf_counter()
        agent.search(b)
        times.append((time.perf_counter() - t0) * 1000)
    return first, float(np.mean(times)), float(np.median(times))


def main():
    print("warming JIT (both nets)...", flush=True)
    warmup_1999()
    warmup_anaconda()
    print("ready\n", flush=True)

    rng = np.random.default_rng(SEED)
    positions = [random_position(rng) for _ in range(N_POSITIONS)]
    # Ensure none are terminal
    positions = [p for p in positions if not p.is_game_over()[0] and p.get_legal_moves()]

    w_1999 = rng.normal(0, 0.2, 1743).astype(np.float32)
    w_1999[0] = 0.5; w_1999[1] = 2.0
    w_ana = rng.normal(0, 0.2, ANACONDA_TOTAL).astype(np.float32)
    w_ana[-2] = 0.5; w_ana[-1] = 2.0

    print(f"{'net':12} {'depth':>6} {'first(ms)':>10} {'mean(ms)':>10} {'median(ms)':>12}")
    print("-" * 56)
    for depth in (4, 6, 8):
        first, mean, med = bench_agent(
            lambda d: FastAgentJit(w_1999, d), positions, depth
        )
        print(f"{'1999':12} {depth:>6} {first:>10.2f} {mean:>10.2f} {med:>12.2f}")

    for depth in (4, 6, 8):
        first, mean, med = bench_agent(
            lambda d: FastAgentJitAnaconda(w_ana, d), positions, depth
        )
        print(f"{'anaconda':12} {depth:>6} {first:>10.2f} {mean:>10.2f} {med:>12.2f}")


if __name__ == "__main__":
    main()
