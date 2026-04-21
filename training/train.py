"""
Main training loop — the evolutionary driver.

This is where Blondie24 comes to life. Run this to evolve a population
of checkers-playing neural networks from random initialization to
(hopefully) expert-level play.

Usage:
    # Paper-faithful config (Chellapilla & Fogel 1999):
    # pop=15, games=5, depth=4, random pairing, +1/0/-2 scoring, no sigma ceiling.
    python -m training.train --preset paper-1999 --generations 250

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

    resume_path = getattr(config.training, "resume_from", None)
    if resume_path:
        population.load_checkpoint(resume_path)
        population.reset_fitness()

    print("=" * 64)
    print("  BLONDIE24 REBORN — Evolutionary Checkers")
    print("=" * 64)
    print(f"  Population size:  {config.evolution.population_size}")
    if resume_path:
        print(f"  Resumed from:     {resume_path} (gen {population.generation})")
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
    print(f"  Max sigma:        {config.evolution.max_sigma:.3f}")
    print(f"  Device:           {device}")
    if device.type == "cuda":
        mem = gpu_memory_report()
        print(f"  GPU memory:       {mem.get('allocated_mb', 0):.0f} MB allocated")
    print("=" * 64)

    # Determine tournament style. Default is "random" (paper-faithful).
    # On GPU + round-robin, use the parallel scheduler which batches leaf
    # evals across all in-flight games. On CPU, use multiprocessing Pool
    # for parallel game play (applies to either tournament style).
    use_round_robin = config.training.tournament == "round-robin"
    pool = None
    n_workers = 0

    if device.type == "cpu":
        n_workers = config.training.num_workers or max(1, (os.cpu_count() or 2) - 1)
        # Warm the Numba JIT in the parent before spawning workers. With a cold
        # disk cache, N workers racing to populate __pycache__ can produce
        # cross-function link failures ("unresolved symbol $.numba.unresolved$...").
        # Parent-side warmup guarantees the cache is fully written before any
        # worker reads from it — subsequent workers hit warm cache and just load.
        print(f"  Warming JIT in parent before spawning {n_workers} workers...")
        _mp_worker_init()
        pool = mp.Pool(processes=n_workers, initializer=_mp_worker_init)

    if use_round_robin and device.type == "cuda":
        tournament_fn = parallel_round_robin_tournament
        tournament_name = "round-robin (parallel GPU)"
    elif use_round_robin:
        tournament_fn = round_robin_tournament
        tournament_name = (f"round-robin (multiprocess, {n_workers} workers)"
                           if pool else "round-robin")
    else:
        tournament_fn = random_pairing_tournament
        if pool:
            tournament_name = f"random-pairing (multiprocess, {n_workers} workers)"
        elif device.type == "cuda":
            tournament_name = "random-pairing (GPU)"
        else:
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

    start_gen = population.generation
    end_gen = start_gen + config.training.generations
    for gen in range(start_gen + 1, end_gen + 1):
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
    if end_gen % config.training.checkpoint_every != 0:
        _save_checkpoint(population, config, end_gen)

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
        "architecture": config.network.architecture,
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
    torch.save({
        "architecture": config.network.architecture,
        "weights": best.weights,
        "sigmas": best.sigmas,
    }, best_path)


def main():
    parser = argparse.ArgumentParser(description="Blondie24 Reborn — Evolutionary Checkers Training")
    parser.add_argument("--generations", type=int, default=250)
    parser.add_argument("--population", type=int, default=15)
    parser.add_argument("--depth", type=int, default=None,
                        help="Search ply depth (default: 4, or checkpoint value when resuming)")
    parser.add_argument("--games", type=int, default=None,
                        help="Games per individual (default: 5, or checkpoint value when resuming)")
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
    parser.add_argument("--win-score", type=float, default=None,
                        help="Score for a win (default: Fogel +1.0; try +2.0 to "
                             "reward aggressive play)")
    parser.add_argument("--initial-sigma", type=float, default=None,
                        help="Starting mutation step size (default: 0.05; try 0.10 "
                             "for more behavioral diversity)")
    parser.add_argument("--max-sigma", type=float, default=None,
                        help="Ceiling on per-weight sigma to prevent runaway during "
                             "flat-gradient phases (default: 0.5)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a population checkpoint (.pt) to resume from. "
                             "--generations is interpreted as additional gens to run.")
    parser.add_argument("--depth-schedule", type=str, default=None,
                        help="Curriculum: gen:depth pairs, e.g. '0:2,20:4,50:6,100:8'. "
                             "Overrides --depth for each gen based on the schedule.")
    parser.add_argument("--tournament", type=str, default="random",
                        choices=["random", "round-robin"],
                        help="Tournament style: 'random' (paper-faithful: each "
                             "individual plays --games games vs randomly chosen "
                             "opponents) or 'round-robin' (every pair, both colors).")
    parser.add_argument("--architecture", type=str, default=None,
                        choices=["checkersnet-1999", "anaconda-2001"],
                        help="Neural network architecture. 'checkersnet-1999' is the "
                             "original 1,743-weight net (default). 'anaconda-2001' is "
                             "the 5,048-weight sub-board-preprocessor net from the "
                             "2001 paper. Both run on CPU (Numba JIT) and CUDA (torch).")
    parser.add_argument("--preset", type=str, default=None,
                        choices=["paper-1999", "paper-2001"],
                        help="Named config preset. 'paper-1999' matches Chellapilla "
                             "& Fogel 1999: pop=15, games=5, depth=4, random pairing, "
                             "+1/0/-2 scoring, initial sigma 0.05, no sigma ceiling. "
                             "'paper-2001' switches to the Anaconda architecture with "
                             "the same EP hyperparameters. Explicit CLI flags still "
                             "win over the preset.")
    args = parser.parse_args()

    # Presets fill args the user didn't set explicitly. Applied before resume
    # logic so an explicit preset beats a checkpoint's saved config.
    if args.preset == "paper-1999":
        paper_defaults = {
            "architecture": "checkersnet-1999",
            "depth": 4,
            "games": 5,
            "loss_score": -2.0,
            "win_score": 1.0,
            "initial_sigma": 0.05,
            "max_sigma": float("inf"),
        }
        for attr, val in paper_defaults.items():
            if getattr(args, attr) is None:
                setattr(args, attr, val)
        print("  [preset] paper-1999 (Chellapilla & Fogel 1999) — "
              "pop=15, games=5, depth=4, random pairing, +1/0/-2, sigma=0.05, no sigma ceiling")
    elif args.preset == "paper-2001":
        paper_defaults = {
            "architecture": "anaconda-2001",
            "depth": 4,
            "games": 5,
            "loss_score": -2.0,
            "win_score": 1.0,
            "initial_sigma": 0.05,
            "max_sigma": float("inf"),
        }
        for attr, val in paper_defaults.items():
            if getattr(args, attr) is None:
                setattr(args, attr, val)
        print("  [preset] paper-2001 (Chellapilla & Fogel 2001 / Anaconda) — "
              "arch=anaconda-2001, pop=15, games=5, depth=4, random pairing, "
              "+1/0/-2, sigma=0.05, no sigma ceiling. "
              "NOTE: paper ran 840 gens for expert strength; default 250 here.")

    # === Resume: adopt hyperparameters saved in the checkpoint unless the ===
    # user explicitly overrode them on the CLI. Keeps resumed runs faithful
    # to the overnight config (depth, loss_score, initial_sigma, ...).
    resumed_cfg = None
    if args.resume:
        ckpt_probe = torch.load(args.resume, weights_only=False)
        resumed_cfg = ckpt_probe.get("config", {}) or {}

    def _fill_from_resume(attr: str, section: str, key: str, fresh_default):
        if getattr(args, attr) is not None:
            return
        if resumed_cfg:
            val = resumed_cfg.get(section, {}).get(key)
            if val is not None:
                setattr(args, attr, val)
                return
        setattr(args, attr, fresh_default)

    _fill_from_resume("depth", "search", "depth", 4)
    _fill_from_resume("games", "evolution", "games_per_individual", 5)
    _fill_from_resume("loss_score", "evolution", "loss_score", None)
    _fill_from_resume("win_score", "evolution", "win_score", None)
    _fill_from_resume("initial_sigma", "evolution", "initial_sigma", None)
    _fill_from_resume("max_sigma", "evolution", "max_sigma", None)

    # Architecture: prefer an explicit --architecture; otherwise read from
    # checkpoint (top-level key first, then nested config.network); finally
    # fall back to inferring from the weight-vector length (legacy checkpoints).
    if args.architecture is None:
        if args.resume:
            ckpt_for_arch = torch.load(args.resume, weights_only=False)
            arch = ckpt_for_arch.get("architecture")
            if arch is None:
                arch = ckpt_for_arch.get("config", {}).get("network", {}).get("architecture")
            if arch is None:
                from neural.network import architecture_from_weight_count
                ind0 = ckpt_for_arch["individuals"][0]
                arch = architecture_from_weight_count(len(np.asarray(ind0["weights"])))
                print(f"  [resume] inferred architecture={arch} from weight count")
            args.architecture = arch
        else:
            args.architecture = "checkersnet-1999"

    if resumed_cfg:
        print(
            f"  [resume] adopted from checkpoint: "
            f"depth={args.depth}, games={args.games}, "
            f"loss_score={args.loss_score}, initial_sigma={args.initial_sigma}"
        )

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
    config.network.architecture = args.architecture
    config.training.device = args.device
    config.training.num_workers = args.workers
    config.training.seed = args.seed
    config.training.checkpoint_every = args.checkpoint_every
    config.training.log_every = args.log_every
    if args.loss_score is not None:
        config.evolution.loss_score = args.loss_score
    if args.win_score is not None:
        config.evolution.win_score = args.win_score
    if args.initial_sigma is not None:
        config.evolution.initial_sigma = args.initial_sigma
    if args.max_sigma is not None:
        config.evolution.max_sigma = args.max_sigma
    config.training.resume_from = args.resume
    config.training.tournament = args.tournament
    config.training.depth_schedule = _parse_depth_schedule(args.depth_schedule)
    if config.training.depth_schedule:
        # Seed config.search.depth with the schedule's first entry so the
        # initial population + header reflect the starting depth.
        config.search.depth = config.training.depth_schedule[0][1]

    train(config)


if __name__ == "__main__":
    main()
