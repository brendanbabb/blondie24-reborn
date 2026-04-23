"""CPU-only bench for measuring CPU JIT optimizations.

Usage: python bench_cpu_only.py <pop> <depth> [arch]
       arch ∈ {anaconda-2001, checkersnet-1999} (default: anaconda-2001)

Runs 3 generations to capture warmup vs steady-state. Skips GPU entirely.
"""
import sys
import time
import os
import multiprocessing as mp
import numpy as np
import torch

from config import Config
from evolution.population import Population
from evolution.tournament import round_robin_tournament, _mp_worker_init


def build_pop(pop_n, seed=42, arch="anaconda-2001"):
    cfg = Config()
    cfg.evolution.population_size = pop_n
    cfg.evolution.initial_sigma = 0.15
    cfg.network.architecture = arch
    np.random.seed(seed)
    torch.manual_seed(seed)
    return Population(cfg.evolution, cfg.network)


def main():
    pop_n = int(sys.argv[1])
    depth = int(sys.argv[2])
    arch = sys.argv[3] if len(sys.argv) > 3 else "anaconda-2001"
    n_gens = 3

    cfg = Config()
    cfg.evolution.population_size = pop_n
    cfg.evolution.initial_sigma = 0.15
    cfg.network.architecture = arch
    cfg.search.depth = depth
    pop = build_pop(pop_n, arch=arch)
    games = pop_n * (pop_n - 1)
    print(f"{arch} pop={pop_n} depth={depth} games/gen={games:,}")

    env_workers = os.environ.get("BENCH_WORKERS")
    if env_workers:
        n_workers = int(env_workers)
    else:
        n_workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"workers={n_workers}")
    pool = mp.Pool(processes=n_workers, initializer=_mp_worker_init)
    try:
        for g in range(n_gens):
            t0 = time.time()
            round_robin_tournament(
                pop, cfg.search, cfg.evolution,
                device="cpu", verbose=False, pool=pool,
            )
            elapsed = time.time() - t0
            print(f"  gen {g+1}: {elapsed:7.2f}s")
            if g < n_gens - 1:
                pop.select_and_reproduce()
    finally:
        pool.close()
        pool.join()


if __name__ == "__main__":
    mp.freeze_support()
    main()
