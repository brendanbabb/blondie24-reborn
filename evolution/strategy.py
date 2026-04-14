"""
Evolutionary strategy for evolving neural network weights.

Implements Fogel's evolutionary programming (EP) approach:
- No crossover — mutation only
- Self-adaptive step sizes (each weight has its own σ)
- Gaussian perturbation of weights
"""

import numpy as np
from dataclasses import dataclass
from config import EvolutionConfig


@dataclass
class Individual:
    """
    An individual in the population.
    
    Attributes:
        weights: 1D numpy array of network weights
        sigmas: 1D numpy array of per-weight mutation step sizes
        fitness: accumulated score from tournament play
    """
    weights: np.ndarray
    sigmas: np.ndarray
    fitness: float = 0.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0


def compute_tau(n_weights: int) -> tuple[float, float]:
    """
    Compute self-adaptation learning rates from Fogel's EP.
    
    τ  (tau)       = 1 / sqrt(2 * sqrt(n))  — per-weight component
    τ' (tau_prime) = 1 / sqrt(2 * n)         — global component
    """
    tau = 1.0 / np.sqrt(2.0 * np.sqrt(n_weights))
    tau_prime = 1.0 / np.sqrt(2.0 * n_weights)
    return tau, tau_prime


def mutate(parent: Individual, config: EvolutionConfig) -> Individual:
    """
    Create a mutated offspring from a parent.
    
    The mutation follows Schwefel's self-adaptive EP:
    1. Mutate step sizes:  σ_i' = σ_i * exp(τ' * N(0,1) + τ * N_i(0,1))
    2. Mutate weights:     w_i' = w_i + σ_i' * N_i(0,1)
    
    Where N(0,1) is a fresh standard normal draw, and N_i(0,1) is a per-weight draw.
    """
    n = len(parent.weights)
    
    # Compute tau values
    tau = config.tau if config.tau is not None else compute_tau(n)[0]
    tau_prime = config.tau_prime if config.tau_prime is not None else compute_tau(n)[1]
    
    # Global random factor (same for all weights in this offspring)
    global_noise = np.random.randn()
    
    # Per-weight random factors
    per_weight_noise = np.random.randn(n)
    
    # Step 1: mutate sigmas
    new_sigmas = parent.sigmas * np.exp(tau_prime * global_noise + tau * per_weight_noise)
    new_sigmas = np.maximum(new_sigmas, config.min_sigma)  # floor
    
    # Step 2: mutate weights using the new sigmas
    weight_noise = np.random.randn(n)
    new_weights = parent.weights + new_sigmas * weight_noise
    
    return Individual(weights=new_weights, sigmas=new_sigmas)


def initialize_individual(n_weights: int, config: EvolutionConfig) -> Individual:
    """Create a random individual with small initial weights ([-0.2, 0.2] per Fogel 1999)."""
    weights = np.random.uniform(-0.2, 0.2, size=n_weights).astype(np.float64)
    sigmas = np.full(n_weights, config.initial_sigma)
    return Individual(weights=weights, sigmas=sigmas)
