"""Profile the serial-CPU round-robin tournament to see where time goes.

Small pop (pop=10 -> 90 games) at a configurable depth. Runs cProfile
on a single tournament and reports cumulative time by caller.

Usage: python scripts/profile_tournament.py [depth]  (default: 4)
"""

import cProfile
import pstats
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from config import Config
from evolution.population import Population
from evolution.tournament import round_robin_tournament
from search.fast_minimax_jit import warmup as warmup_1999
from search.fast_minimax_jit_anaconda import warmup_anaconda


def build_pop(n, seed=42, arch="anaconda-2001"):
    cfg = Config()
    cfg.evolution.population_size = n
    cfg.evolution.initial_sigma = 0.15
    cfg.network.architecture = arch
    np.random.seed(seed)
    torch.manual_seed(seed)
    return cfg, Population(cfg.evolution, cfg.network)


def main():
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    pop_n = 10
    arch = "anaconda-2001"

    print("Warming JIT...")
    warmup_1999()
    warmup_anaconda()
    print("Ready.\n")

    cfg, pop = build_pop(pop_n, arch=arch)
    cfg.search.depth = depth

    n_games = pop_n * (pop_n - 1)
    print(f"{arch} pop={pop_n} depth={depth} games={n_games}")

    profiler = cProfile.Profile()
    t0 = time.time()
    profiler.enable()
    round_robin_tournament(
        pop, cfg.search, cfg.evolution,
        device="cpu", verbose=False, pool=None,
    )
    profiler.disable()
    elapsed = time.time() - t0
    print(f"tournament: {elapsed:.2f}s ({elapsed / n_games * 1000:.1f}ms/game)\n")

    # Top-30 by cumulative time, focused on our code.
    stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")
    print("=== top 30 by cumulative time ===")
    stats.print_stats(30)

    # Also: top-20 by tottime (self time) — shows the actual hot spots,
    # not just frames that recurse into the search.
    print("\n=== top 20 by self time (tottime) ===")
    pstats.Stats(profiler).strip_dirs().sort_stats("tottime").print_stats(20)


if __name__ == "__main__":
    main()
