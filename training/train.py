"""
Main training loop — the evolutionary driver.

This is where Blondie24 comes to life. Run this to evolve a population
of checkers-playing neural networks from random initialization to
(hopefully) expert-level play.

Usage:
    # Auto-detect GPU (will use RTX 5060 on Badger-1)
    python -m training.train --generations 250 --population 15 --depth 4

    # Force GPU with deeper search
    python -m training.train --generations 250 --population 15 --depth 6 --device cuda

    # CPU fallback
    python -m training.train --generations 100 --population 10 --depth 4 --device cpu

    # Quick test run
    python -m training.train --generations 5 --population 6 --depth 2 --device auto
"""

import argparse
import os
import time
import json
import multiprocessing as mp
import numpy as np
import torch
from datetime import datetime

from config import Config, TrainingConfig, EvolutionConfig, SearchConfig, NetworkConfig
from evolution.population import Population
from evolution.tournament import (
    random_pairing_tournament,
    round_robin_tournament,
    parallel_round_robin_tournament,
    _mp_worker_init,
)
from utils import get_device, optimize_for_inference, gpu_memory_report, clear_gpu_cache


def _parse_depth_schedule(s: str) -> list[tuple[int, int]]:
    """
    Parse a curriculum string like "0:2,20:4,50:6,100:8" into a sorted list
    of (start_gen, depth) pairs. Returns empty list if s is None/empty.
    """
    if not s:
        return []
    pairs = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        gen_str, depth_str = chunk.split(":")
        pairs.append((int(gen_str), int(depth_str)))
    pairs.sort()
    if pairs and pairs[0][0] != 0:
        raise ValueError("Depth schedule must include gen 0 (e.g. '0:2,20:4')")
    return pairs


def _depth_for_gen(schedule: list[tuple[int, int]], gen: int, default: int) -> int:
    """Return the active depth for a given generation under a schedule."""
    if not schedule:
        return default
    active = schedule[0][1]
    for start_gen, depth in schedule:
        if gen >= start_gen:
            active = depth
        else:
            break
    return active


def train(config: Config):
    """Run the full evolutionary training loop."""

    # === Device setup ===
    device = get_device(config.training.device)
    optimize_for_inference()

    if config.training.seed is not None:
        np.random.seed(config.training.seed)
        torch.manual_seed(config.training.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(config.training.seed)

    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    os.makedirs(config.training.log_dir, exist_ok=True)

    # === Initialize population ===
    population = Population(config.evolution, config.network)

    print("=" * 64)
    print("  BLONDIE24 REBORN — Evolutionary Checkers")
    print("=" * 64)
    print(f"  Population size:  {config.evolution.population_size}")
    print(f"  Network weights:  {population.n_weights}")
    sched = getattr(config.training, "depth_schedule", None) or []
    if sched:
        sched_str = ", ".join(f"gen{g}:d{d}" for g, d in sched)
        print(f"  Search depth:     curriculum [{sched_str}]")
    else:
        print(f"  Search depth:     {config.search.depth}")
    print(f"  Generations:      {config.training.generations}")
    print(f"  Games/individual: {config.evolution.games_per_individual}")
    print(f"  Scoring (W/D/L):  "
          f"{config.evolution.win_score:+.1f} / "
          f"{config.evolution.draw_score:+.1f} / "
          f"{config.evolution.loss_score:+.1f}")
    print(f"  Initial sigma:    {config.evolution.initial_sigma:.3f}")
    print(f"  Device:           {device}")
    if device.type == "cuda":
        mem = gpu_memory_report()
        print(f"  GPU memory:       {mem.get('allocated_mb', 0):.0f} MB allocated")
    print("=" * 64)

    # Determine tournament style. On GPU + round-robin, use the parallel
    # scheduler which batches leaf evals across all in-flight games.
    # On CPU, use multiprocessing Pool for parallel game play.
    use_round_robin = config.evolution.population_size <= 20
    pool = None
    if use_round_robin and device.type == "cuda":
        tournament_fn = parallel_round_robin_tournament
        tournament_name = "round-robin (parallel GPU)"
    elif use_round_robin and device.type == "cpu":
        tournament_fn = round_robin_tournament
        n_workers = config.training.num_workers or max(1, (os.cpu_count() or 2) - 1)
        pool = mp.Pool(processes=n_workers, initializer=_mp_worker_init)
        tournament_name = f"round-robin (multiprocess, {n_workers} workers)"
    elif use_round_robin:
        tournament_fn = round_robin_tournament
        tournament_name = "round-robin"
    else:
        tournament_fn = random_pairing_tournament
        tournament_name = "random-pairing"
    print(f"  Tournament:       {tournament_name}")
    print()

    # Training log
    log_path = os.path.join(
        config.training.log_dir,
        f"training_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    )

    best_ever_fitness = -float("inf")
    schedule = getattr(config.training, "depth_schedule", None) or []
    current_depth = config.search.depth

    for gen in range(1, config.training.generations + 1):
        gen_start = time.time()

        # === Curriculum: update search depth if schedule says so ===
        scheduled_depth = _depth_for_gen(schedule, gen - 1, config.search.depth)
        if scheduled_depth != current_depth:
            print(f"  [curriculum] gen {gen}: depth {current_depth} -> {scheduled_depth}")
            current_depth = scheduled_depth
            config.search.depth = scheduled_depth

        # === Tournament: evaluate fitness ===
        tournament_kwargs = dict(
            device=str(device), verbose=False,
        )
        if pool is not None:
            tournament_kwargs["pool"] = pool
        tournament_fn(
            population, config.search, config.evolution,
            **tournament_kwargs,
        )

        # Get stats before selection
        stats = population.stats()
        gen_time = time.time() - gen_start
        stats["time_seconds"] = round(gen_time, 2)

        # Track best-ever
        if stats["max_fitness"] > best_ever_fitness:
            best_ever_fitness = stats["max_fitness"]

        # GPU memory stats
        if device.type == "cuda":
            mem = gpu_memory_report()
            stats["gpu_allocated_mb"] = round(mem.get("allocated_mb", 0), 1)
            stats["gpu_max_allocated_mb"] = round(mem.get("max_allocated_mb", 0), 1)

        # === Logging ===
        if gen % config.training.log_every == 0 or gen == 1:
            gpu_str = ""
            if device.type == "cuda":
                gpu_str = f" | GPU: {stats.get('gpu_allocated_mb', 0):.0f}MB"
            print(
                f"Gen {gen:4d} | "
                f"Best: {stats['max_fitness']:+.1f} | "
                f"Mean: {stats['mean_fitness']:+.2f} | "
                f"sigma: {stats['mean_sigma']:.4f} | "
                f"W/L/D: {stats['best_wins']}/{stats['best_losses']}/{stats['best_draws']} | "
                f"{gen_time:.1f}s{gpu_str}"
            )

        with open(log_path, "a") as f:
            f.write(json.dumps(stats) + "\n")

        # === Checkpoint ===
        if gen % config.training.checkpoint_every == 0:
            _save_checkpoint(population, config, gen)

        # === Selection + reproduction ===
        population.select_and_reproduce()

        # Periodic GPU cache cleanup
        if device.type == "cuda" and gen % 10 == 0:
            clear_gpu_cache()

    # === Final checkpoint (skip if the loop just saved the same generation) ===
    if config.training.generations % config.training.checkpoint_every != 0:
        _save_checkpoint(population, config, config.training.generations)

    # Tear down worker pool if one was created
    if pool is not None:
        pool.close()
        pool.join()

    print("\n" + "=" * 64)
    print("  Training complete!")
    print(f"  Best fitness this run:  {best_ever_fitness:+.1f}")
    print(f"  Final best individual:  {population.best_individual().fitness:+.1f}")
    print(f"  Log saved to: {log_path}")
    if device.type == "cuda":
        mem = gpu_memory_report()
        print(f"  Peak GPU memory: {mem.get('max_allocated_mb', 0):.1f} MB")
    print("=" * 64)


def _save_checkpoint(population: Population, config: Config, generation: int):
    """Save population state to disk."""
    path = os.path.join(
        config.training.checkpoint_dir,
        f"population_gen{generation:04d}.pt"
    )

    checkpoint = {
        "generation": generation,
        "config": {
            "evolution": vars(config.evolution),
            "network": vars(config.network),
            "search": vars(config.search),
        },
        "individuals": [
            {
                "weights": ind.weights,
                "sigmas": ind.sigmas,
                "fitness": ind.fitness,
            }
            for ind in population.individuals
        ],
    }

    torch.save(checkpoint, path)
    print(f"  -> Checkpoint saved: {path}")

    # Also save just the best network for easy loading
    best = population.best_individual()
    best_path = os.path.join(
        config.training.checkpoint_dir,
        f"best_gen{generation:04d}.pt"
    )
    torch.save({"weights": best.weights, "sigmas": best.sigmas}, best_path)


def main():
    parser = argparse.ArgumentParser(description="Blondie24 Reborn — Evolutionary Checkers Training")
    parser.add_argument("--generations", type=int, default=250)
    parser.add_argument("--population", type=int, default=15)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'cuda', 'mps', or 'cpu'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--workers", type=int, default=None,
                        help="Multiprocessing workers for CPU tournament (default: cpu_count - 1)")
    parser.add_argument("--loss-score", type=float, default=None,
                        help="Score for a loss (default: Fogel -2.0; try -1.0 to "
                             "escape the draw plateau)")
    parser.add_argument("--initial-sigma", type=float, default=None,
                        help="Starting mutation step size (default: 0.05; try 0.10 "
                             "for more behavioral diversity)")
    parser.add_argument("--depth-schedule", type=str, default=None,
                        help="Curriculum: gen:depth pairs, e.g. '0:2,20:4,50:6,100:8'. "
                             "Overrides --depth for each gen based on the schedule.")
    args = parser.parse_args()

    # Auto-detect device
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda" if args.depth >= 3 else "cpu"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    # Warn on pathological GPU configs: batched leaf eval only kicks in at
    # depth >= 3, and below that GPU is slower than CPU due to kernel launch
    # overhead on a 1,743-parameter network.
    if args.device == "cuda" and args.depth < 3:
        print(
            f"  WARNING: --device cuda with --depth {args.depth} will be slower than CPU.\n"
            f"  Batched leaf evaluation requires depth >= 3; below that every leaf\n"
            f"  incurs full GPU launch+sync overhead. Use --device cpu for depth < 3."
        )

    config = Config()
    config.training.generations = args.generations
    config.evolution.population_size = args.population
    config.search.depth = args.depth
    config.evolution.games_per_individual = args.games
    config.training.device = args.device
    config.training.num_workers = args.workers
    config.training.seed = args.seed
    config.training.checkpoint_every = args.checkpoint_every
    config.training.log_every = args.log_every
    if args.loss_score is not None:
        config.evolution.loss_score = args.loss_score
    if args.initial_sigma is not None:
        config.evolution.initial_sigma = args.initial_sigma
    config.training.depth_schedule = _parse_depth_schedule(args.depth_schedule)
    if config.training.depth_schedule:
        # Seed config.search.depth with the schedule's first entry so the
        # initial population + header reflect the starting depth.
        config.search.depth = config.training.depth_schedule[0][1]

    train(config)


if __name__ == "__main__":
    main()
