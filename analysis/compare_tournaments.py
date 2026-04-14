"""
Compare per-individual fitnesses from sequential vs parallel round-robin
tournaments on CPU. Lets us isolate parallel-scheduling bugs from
GPU/float-precision artifacts.
"""

import numpy as np
import torch

from config import EvolutionConfig, SearchConfig
from evolution.population import Population
from evolution.tournament import round_robin_tournament, parallel_round_robin_tournament
from utils import optimize_for_inference


def run():
    optimize_for_inference()
    np.random.seed(42)
    torch.manual_seed(42)

    evo = EvolutionConfig(population_size=6, games_per_individual=2)
    search = SearchConfig(depth=4)

    # Build one population and clone weights into four separate populations
    pop_seq_cpu = Population(evo)
    pop_par_cpu = Population(evo)
    pop_seq_gpu = Population(evo)
    pop_par_gpu = Population(evo)
    for i in range(len(pop_seq_cpu.individuals)):
        src = pop_seq_cpu.individuals[i]
        for target in (pop_par_cpu, pop_seq_gpu, pop_par_gpu):
            target.individuals[i].weights = src.weights.copy()
            target.individuals[i].sigmas = src.sigmas.copy()

    round_robin_tournament(pop_seq_cpu, search, evo, device="cpu")
    parallel_round_robin_tournament(pop_par_cpu, search, evo, device="cpu")
    if torch.cuda.is_available():
        round_robin_tournament(pop_seq_gpu, search, evo, device="cuda")
        parallel_round_robin_tournament(pop_par_gpu, search, evo, device="cuda")

    print("=== Fitness per individual across 4 paths ===")
    print(f"{'idx':>3}  {'seq-CPU':>10}  {'par-CPU':>10}  {'seq-GPU':>10}  {'par-GPU':>10}")
    pop_a = pop_seq_cpu
    pop_b = pop_par_cpu
    for i in range(len(pop_seq_cpu.individuals)):
        a = pop_seq_cpu.individuals[i].fitness
        b = pop_par_cpu.individuals[i].fitness
        c = pop_seq_gpu.individuals[i].fitness if torch.cuda.is_available() else float("nan")
        d = pop_par_gpu.individuals[i].fitness if torch.cuda.is_available() else float("nan")
        print(f"{i:>3}  {a:>+10.1f}  {b:>+10.1f}  {c:>+10.1f}  {d:>+10.1f}")

    par_cpu_match = all(pop_seq_cpu.individuals[i].fitness == pop_par_cpu.individuals[i].fitness
                        for i in range(len(pop_seq_cpu.individuals)))
    print(f"\npar-CPU vs seq-CPU: {'IDENTICAL' if par_cpu_match else 'DIVERGES'}")
    if torch.cuda.is_available():
        seq_gpu_match = all(pop_seq_cpu.individuals[i].fitness == pop_seq_gpu.individuals[i].fitness
                            for i in range(len(pop_seq_cpu.individuals)))
        par_gpu_match = all(pop_seq_cpu.individuals[i].fitness == pop_par_gpu.individuals[i].fitness
                            for i in range(len(pop_seq_cpu.individuals)))
        print(f"seq-GPU vs seq-CPU: {'IDENTICAL' if seq_gpu_match else 'DIVERGES'}")
        print(f"par-GPU vs seq-CPU: {'IDENTICAL' if par_gpu_match else 'DIVERGES'}")


if __name__ == "__main__":
    run()
