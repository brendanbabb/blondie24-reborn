"""Round-robin CPU-vs-GPU timing for either architecture.

CPU uses round_robin_tournament with a multiprocessing pool of Numba
JIT agents. GPU uses parallel_round_robin_tournament, which batches
leaf evaluations across all in-flight games.

Usage:
    python bench_tournament.py                          # 1999, default sizes
    python bench_tournament.py --architecture anaconda-2001
    python bench_tournament.py --pop 50 --depth 4 --gens 3
"""
import argparse
import multiprocessing as mp
import os
import time
import numpy as np
import torch
from config import Config
from evolution.population import Population
from evolution.tournament import (
    round_robin_tournament,
    parallel_round_robin_tournament,
    _mp_worker_init,
)


def _log(msg: str = ""):
    print(msg, flush=True)


def build_pop(seed: int, pop_size: int, architecture: str) -> Population:
    cfg = Config()
    cfg.evolution.population_size = pop_size
    cfg.evolution.initial_sigma = 0.15
    cfg.network.architecture = architecture
    np.random.seed(seed)
    torch.manual_seed(seed)
    return Population(cfg.evolution, cfg.network)


def run(device: str, pop_size: int, depth: int, n_gens: int,
        architecture: str, seed: int = 42) -> list[float]:
    cfg = Config()
    cfg.search.depth = depth
    cfg.evolution.population_size = pop_size
    cfg.evolution.initial_sigma = 0.15
    cfg.network.architecture = architecture

    pop = build_pop(seed, pop_size, architecture)
    times: list[float] = []

    pool = None
    tournament_fn = round_robin_tournament
    if device == "cpu":
        n_workers = max(1, (os.cpu_count() or 2) - 1)
        _log(f"  CPU workers: {n_workers}")
        pool = mp.Pool(processes=n_workers, initializer=_mp_worker_init)
    else:
        tournament_fn = parallel_round_robin_tournament

    kwargs = {"device": device, "verbose": False}
    if pool is not None:
        kwargs["pool"] = pool

    for g in range(1, n_gens + 1):
        t0 = time.time()
        tournament_fn(pop, cfg.search, cfg.evolution, **kwargs)
        dt = time.time() - t0
        times.append(dt)
        _log(f"  gen {g}: {dt:7.2f}s")
        pop.select_and_reproduce()

    if pool is not None:
        pool.close()
        pool.join()
    return times


def summarize(label: str, times: list[float]):
    first = times[0]
    rest = times[1:]
    rest_mean = sum(rest) / len(rest) if rest else float("nan")
    total = sum(times)
    _log(f"  {label}: gen1 (warmup) {first:.2f}s | "
         f"gen2-{len(times)} mean {rest_mean:.2f}s | total {total:.2f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--architecture", type=str, default="checkersnet-1999",
                        choices=["checkersnet-1999", "anaconda-2001"],
                        help="Which network to benchmark (default 1999).")
    parser.add_argument("--pop", type=int, default=100,
                        help="Population size (default 100).")
    parser.add_argument("--depth", type=int, default=6,
                        help="Search ply depth (default 6).")
    parser.add_argument("--gens", type=int, default=5,
                        help="Generations per device (default 5).")
    parser.add_argument("--skip-cpu", action="store_true",
                        help="Skip the CPU run.")
    parser.add_argument("--skip-gpu", action="store_true",
                        help="Skip the GPU run even if CUDA is available.")
    args = parser.parse_args()

    arch_label = {
        "checkersnet-1999": "1999 CheckersNet (1,743 weights)",
        "anaconda-2001":    "2001 Anaconda (5,048 weights)",
    }[args.architecture]

    games_per_gen = args.pop * (args.pop - 1)
    _log("=" * 60)
    _log(f"  Round-robin benchmark: {arch_label}")
    _log(f"  pop={args.pop}, depth={args.depth}, {args.gens} gens/device")
    _log(f"  Games per generation: {games_per_gen:,}")
    _log("=" * 60)

    cpu_times: list[float] | None = None
    if not args.skip_cpu:
        _log("\n[CPU run] round_robin_tournament (multiprocess, Numba JIT)")
        cpu_times = run("cpu", args.pop, args.depth, args.gens, args.architecture)
    else:
        _log("\n[CPU run] skipped (--skip-cpu)")

    gpu_times: list[float] | None = None
    if args.skip_gpu:
        _log("\n[GPU run] skipped (--skip-gpu)")
    elif torch.cuda.is_available():
        _log("\n[GPU run] parallel_round_robin_tournament (batched scheduler)")
        gpu_times = run("cuda", args.pop, args.depth, args.gens, args.architecture)
    else:
        _log("\n[GPU run] skipped — CUDA not available")

    _log("\n" + "=" * 60)
    _log("  Summary")
    _log("=" * 60)
    if cpu_times is not None:
        summarize("CPU", cpu_times)
    if gpu_times is not None:
        summarize("GPU", gpu_times)
    if cpu_times is not None and gpu_times is not None:
        cpu_steady = sum(cpu_times[1:]) / max(1, len(cpu_times) - 1)
        gpu_steady = sum(gpu_times[1:]) / max(1, len(gpu_times) - 1)
        if gpu_steady > 0:
            _log(f"\n  Steady-state speedup (CPU / GPU): "
                 f"{cpu_steady / gpu_steady:.2f}x")


if __name__ == "__main__":
    mp.freeze_support()
    main()
