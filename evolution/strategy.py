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


def mutate(parent: Individual, config: EvolutionConfig,
           king_idx: int | None = None) -> Individual:
    """
    Create a mutated offspring from a parent.

    Self-adaptive EP mutation. The sigma update rule is selected by
    config.sigma_update:
      "two_factor":  σ_i' = σ_i * exp(τ' * N(0,1) + τ * N_i(0,1))  — Schwefel
      "single_tau":  σ_i' = σ_i * exp(τ * N_i(0,1))  — Chellapilla & Fogel 1999/2001
    Then weights:    w_i' = w_i + σ_i' * N_i(0,1)

    Where N(0,1) is a fresh standard normal draw, and N_i(0,1) is a per-weight draw.

    If king_idx is given, the king-weight slot is clamped to
    [config.king_weight_min, config.king_weight_max] after mutation
    (paper: K constrained to [1, 3]).
    """
    n = len(parent.weights)

    # Compute tau values
    tau = config.tau if config.tau is not None else compute_tau(n)[0]
    tau_prime = config.tau_prime if config.tau_prime is not None else compute_tau(n)[1]

    # Per-weight random factors
    per_weight_noise = np.random.randn(n)

    # Step 1: mutate sigmas
    if getattr(config, "sigma_update", "two_factor") == "single_tau":
        # Paper rule: per-weight noise only, no correlated global factor.
        new_sigmas = parent.sigmas * np.exp(tau * per_weight_noise)
    else:
        # Global random factor (same for all weights in this offspring)
        global_noise = np.random.randn()
        new_sigmas = parent.sigmas * np.exp(tau_prime * global_noise + tau * per_weight_noise)
    new_sigmas = np.maximum(new_sigmas, config.min_sigma)  # floor
    new_sigmas = np.minimum(new_sigmas, config.max_sigma)  # ceiling — prevents runaway

    # Step 2: mutate weights using the new sigmas
    weight_noise = np.random.randn(n)
    new_weights = parent.weights + new_sigmas * weight_noise

    # King weight stays inside the paper's [1, 3] band.
    if king_idx is not None:
        new_weights[king_idx] = np.clip(
            new_weights[king_idx], config.king_weight_min, config.king_weight_max
        )

    return Individual(weights=new_weights, sigmas=new_sigmas)


def initialize_individual(n_weights: int, config: EvolutionConfig,
                          king_idx: int | None = None,
                          king_init: float | None = None) -> Individual:
    """Create a random individual with small initial weights ([-0.2, 0.2] per Fogel 1999).

    If king_idx/king_init are given, the king-weight slot is set to king_init
    (paper: K starts at exactly 2.0) instead of a random draw near zero.
    """
    weights = np.random.uniform(-0.2, 0.2, size=n_weights).astype(np.float64)
    if king_idx is not None and king_init is not None:
        weights[king_idx] = king_init
    sigmas = np.full(n_weights, config.initial_sigma)
    return Individual(weights=weights, sigmas=sigmas)
