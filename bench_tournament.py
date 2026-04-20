"""Round-robin CPU-vs-GPU timing: pop=100 d=6, 5 gens per device.

CPU uses round_robin_tournament with a multiprocessing pool.
GPU uses parallel_round_robin_tournament (batches leaf evals across games).
"""
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


N_GENS = 5
POP = 100
DEPTH = 6


def _log(msg: str = ""):
    print(msg, flush=True)


def build_pop(seed: int) -> Population:
    cfg = Config()
    cfg.evolution.population_size = POP
    cfg.evolution.initial_sigma = 0.15
    np.random.seed(seed)
    torch.manual_seed(seed)
    return Population(cfg.evolution, cfg.network)


def run(device: str, seed: int = 42) -> list[float]:
    cfg = Config()
    cfg.search.depth = DEPTH
    cfg.evolution.population_size = POP
    cfg.evolution.initial_sigma = 0.15

    pop = build_pop(seed)
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

    for g in range(1, N_GENS + 1):
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
    _log(f"  {label}: gen1 (warmup) {first:.2f}s | gen2-{len(times)} mean {rest_mean:.2f}s | total {total:.2f}s")


def main():
    games_per_gen = POP * (POP - 1)  # both colors, every pair
    _log("=" * 60)
    _log(f"  Round-robin benchmark: pop={POP}, depth={DEPTH}, {N_GENS} gens/device")
    _log(f"  Games per generation: {games_per_gen:,}")
    _log("=" * 60)

    _log("\n[CPU run] round_robin_tournament (multiprocess)")
    cpu_times = run("cpu")

    gpu_times: list[float] | None = None
    if torch.cuda.is_available():
        _log("\n[GPU run] parallel_round_robin_tournament (batched scheduler)")
        gpu_times = run("cuda")
    else:
        _log("\n[GPU run] skipped — CUDA not available")

    _log("\n" + "=" * 60)
    _log("  Summary")
    _log("=" * 60)
    summarize("CPU", cpu_times)
    if gpu_times is not None:
        summarize("GPU", gpu_times)
        cpu_steady = sum(cpu_times[1:]) / max(1, len(cpu_times) - 1)
        gpu_steady = sum(gpu_times[1:]) / max(1, len(gpu_times) - 1)
        if gpu_steady > 0:
            _log(f"\n  Steady-state speedup (CPU / GPU): {cpu_steady / gpu_steady:.2f}x")


if __name__ == "__main__":
    mp.freeze_support()
    main()
