"""Ad-hoc single-config Anaconda CPU-vs-GPU round-robin bench.

Usage: python bench_one.py <pop> <depth>
"""
import sys
import time
import os
import multiprocessing as mp
import numpy as np
import torch

from config import Config
from evolution.population import Population
from evolution.tournament import (
    round_robin_tournament,
    parallel_round_robin_tournament,
    _mp_worker_init,
)


def build_pop(pop_n: int, seed: int = 42) -> Population:
    cfg = Config()
    cfg.evolution.population_size = pop_n
    cfg.evolution.initial_sigma = 0.15
    cfg.network.architecture = "anaconda-2001"
    np.random.seed(seed)
    torch.manual_seed(seed)
    return Population(cfg.evolution, cfg.network)


def run(device: str, pop_n: int, depth: int, n_gens: int = 2):
    cfg = Config()
    cfg.evolution.population_size = pop_n
    cfg.evolution.initial_sigma = 0.15
    cfg.network.architecture = "anaconda-2001"
    cfg.search.depth = depth
    pop = build_pop(pop_n)

    pool = None
    if device == "cpu":
        n_workers = max(1, (os.cpu_count() or 2) - 1)
        pool = mp.Pool(processes=n_workers, initializer=_mp_worker_init)
        fn = round_robin_tournament
    else:
        fn = parallel_round_robin_tournament

    times = []
    for g in range(n_gens):
        t0 = time.time()
        kw = {"device": device, "verbose": False}
        if pool is not None:
            kw["pool"] = pool
        fn(pop, cfg.search, cfg.evolution, **kw)
        times.append(time.time() - t0)
        if g < n_gens - 1:
            pop.select_and_reproduce()
    if pool is not None:
        pool.close(); pool.join()
    return times


def main():
    pop_n = int(sys.argv[1])
    depth = int(sys.argv[2])
    games = pop_n * (pop_n - 1)
    print(f"Anaconda pop={pop_n} depth={depth} games/gen={games:,}")

    print("[CPU]")
    cpu_times = run("cpu", pop_n, depth)
    for i, t in enumerate(cpu_times, 1):
        print(f"  gen {i}: {t:7.2f}s")

    print("[GPU]")
    gpu_times = run("cuda", pop_n, depth)
    for i, t in enumerate(gpu_times, 1):
        print(f"  gen {i}: {t:7.2f}s")

    cpu_steady = cpu_times[-1]
    gpu_steady = gpu_times[-1]
    if gpu_steady > 0:
        print(f"\nSteady-state: CPU {cpu_steady:.2f}s  vs  GPU {gpu_steady:.2f}s  "
              f"-> CPU is {gpu_steady/cpu_steady:.2f}x faster"
              if cpu_steady < gpu_steady else
              f"\nSteady-state: CPU {cpu_steady:.2f}s  vs  GPU {gpu_steady:.2f}s  "
              f"-> GPU is {cpu_steady/gpu_steady:.2f}x faster")


if __name__ == "__main__":
    mp.freeze_support()
    main()
