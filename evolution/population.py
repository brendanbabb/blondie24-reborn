"""
Population management: initialization, selection, and reproduction.

Implements (μ, λ) style selection where μ parents produce λ offspring,
and the best μ individuals from parents + offspring survive.
"""

import numpy as np
from evolution.strategy import Individual, initialize_individual, mutate
from neural.network import make_network
from config import EvolutionConfig, NetworkConfig


class Population:
    """
    Manages the evolving population of checkers-playing neural networks.
    """

    def __init__(self, config: EvolutionConfig, net_config: NetworkConfig = NetworkConfig()):
        self.config = config
        self.net_config = net_config

        # Determine number of weights from a reference network
        ref_net = make_network(net_config)
        self.n_weights = ref_net.num_weights()

        # Initialize population
        self.individuals: list[Individual] = [
            initialize_individual(self.n_weights, config)
            for _ in range(config.population_size)
        ]

        self.generation = 0

    def get_network(self, individual: Individual, device: str = "cpu"):
        """Build a network with this individual's weight vector, on `device`."""
        net = make_network(self.net_config)
        net.set_weight_vector(individual.weights)
        net = net.to(device)
        return net

    def load_checkpoint(self, path: str):
        """Replace individuals + generation counter from a saved checkpoint."""
        import torch
        ckpt = torch.load(path, weights_only=False)
        loaded = ckpt["individuals"]
        if len(loaded) != self.config.population_size:
            print(
                f"  [resume] population_size {self.config.population_size} -> "
                f"{len(loaded)} (adopted from checkpoint)"
            )
            self.config.population_size = len(loaded)
        first_w = np.asarray(loaded[0]["weights"])
        if first_w.shape[0] != self.n_weights:
            raise ValueError(
                f"Checkpoint weight vector has {first_w.shape[0]} params but "
                f"current network has {self.n_weights} — network architecture "
                f"does not match checkpoint."
            )
        self.individuals = [
            Individual(
                weights=np.asarray(ind["weights"]),
                sigmas=np.asarray(ind["sigmas"]),
            )
            for ind in loaded
        ]
        self.generation = int(ckpt["generation"])

    def reset_fitness(self):
        """Reset all fitness scores for a new generation."""
        for ind in self.individuals:
            ind.fitness = 0.0
            ind.games_played = 0
            ind.wins = 0
            ind.losses = 0
            ind.draws = 0
    
    def select_and_reproduce(self) -> list[Individual]:
        """
        Selection + reproduction step.
        
        1. Rank individuals by fitness
        2. Keep top fraction as parents
        3. Each parent spawns one mutated offspring
        4. New population = parents + offspring
        """
        # Sort by fitness (descending)
        ranked = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)
        
        # Keep top fraction
        n_keep = max(1, int(len(ranked) * self.config.keep_fraction))
        parents = ranked[:n_keep]
        
        # Reproduce: each parent spawns one offspring
        offspring = [mutate(parent, self.config) for parent in parents]
        
        # If we need more to fill the population, keep mutating top parents
        while len(parents) + len(offspring) < self.config.population_size:
            extra_parent = parents[len(offspring) % len(parents)]
            offspring.append(mutate(extra_parent, self.config))
        
        # New population: parents keep their weights (but fitness resets next gen)
        # Parents get fresh Individual wrappers to reset fitness
        new_pop = []
        for p in parents:
            new_pop.append(Individual(weights=p.weights.copy(), sigmas=p.sigmas.copy()))
        new_pop.extend(offspring[:self.config.population_size - len(new_pop)])
        
        self.individuals = new_pop
        self.generation += 1
        
        return self.individuals
    
    def best_individual(self) -> Individual:
        """Return the individual with the highest fitness."""
        return max(self.individuals, key=lambda ind: ind.fitness)
    
    def stats(self) -> dict:
        """Return population statistics for logging."""
        fitnesses = [ind.fitness for ind in self.individuals]
        sigmas_mean = np.mean([np.mean(ind.sigmas) for ind in self.individuals])
        
        return {
            "generation": self.generation,
            "max_fitness": max(fitnesses),
            "min_fitness": min(fitnesses),
            "mean_fitness": np.mean(fitnesses),
            "std_fitness": np.std(fitnesses),
            "mean_sigma": sigmas_mean,
            "best_wins": self.best_individual().wins,
            "best_losses": self.best_individual().losses,
            "best_draws": self.best_individual().draws,
        }
