"""
Blondie24 Reborn — Central Configuration

All hyperparameters from the original Chellapilla & Fogel (1999, 2001) papers,
with sensible defaults and notes on what to tune.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CheckersConfig:
    """Board and rules configuration."""
    board_size: int = 8
    num_playable_squares: int = 32  # dark squares only
    max_moves_per_game: int = 200   # draw if exceeded (prevents infinite games)


@dataclass
class NetworkConfig:
    """Neural network architecture — matches Blondie24 original."""
    input_size: int = 32        # one per playable square
    hidden1_size: int = 40      # first hidden layer
    hidden2_size: int = 10      # second hidden layer
    output_size: int = 1        # single board evaluation score

    # King weight: how much more a king is worth vs a regular piece in the encoding.
    # This is ALSO an evolvable parameter in the original paper.
    # Fogel 1999: initialized at K=2.0, constrained to range [1, 3].
    initial_king_weight: float = 2.0

    # Activation functions (Fogel 1999 used tanh throughout)
    hidden_activation: str = "tanh"
    output_activation: str = "tanh"     # output in [-1, 1]

    # Which network to build. "checkersnet-1999" is the 1999 paper net
    # (1,743 evolvable weights). "anaconda-2001" adds the spatial sub-board
    # preprocessor from the 2001 paper (5,048 evolvable weights).
    architecture: str = "checkersnet-1999"


@dataclass
class SearchConfig:
    """Minimax search parameters."""
    depth: int = 4              # ply depth (original used 4; try 6-8 with GPU)
    use_alpha_beta: bool = True # always use alpha-beta pruning

    # Quiescence: if a capture is available at leaf, extend search
    quiescence_enabled: bool = False  # optional enhancement
    quiescence_max_depth: int = 2     # extra ply for captures only

    # Opt-in: turn off quiescence in the CPU JIT engines to match the paper's
    # plain alpha-beta at depth 4 (Chellapilla & Fogel 1999/2001). Default False
    # leaves the engines in their current behavior (quiescence on).
    disable_quiescence: bool = False


@dataclass
class EvolutionConfig:
    """Evolutionary strategy parameters — from Fogel's EP approach."""
    population_size: int = 15       # original used 15 (small!)
    games_per_individual: int = 5   # games each network plays per generation

    # Selection scheme. "half_keep_mutate" is the existing behavior (top
    # keep_fraction survives, spawns mutated offspring to refill to μ).
    # "mu_plus_mu" is the paper-faithful (μ+μ) EP: every parent spawns one
    # offspring, combined 2μ pool is evaluated in the tournament, top μ
    # survive. Opt-in via --selection-scheme or --preset paper-2001-strict.
    selection_scheme: str = "half_keep_mutate"
    keep_fraction: float = 0.5      # top 50% survive (half_keep_mutate only)
    
    # Mutation — self-adaptive step sizes
    # Each weight has its own σ (step size), also evolved
    initial_sigma: float = 0.05     # starting mutation step size
    tau: Optional[float] = None     # if None, computed as 1/sqrt(2*sqrt(n_weights))
    tau_prime: Optional[float] = None  # if None, computed as 1/sqrt(2*n_weights)
    min_sigma: float = 1e-5         # floor to prevent σ collapse
    max_sigma: float = 0.5          # ceiling to prevent σ runaway on flat-gradient phases
    
    # Fitness scoring (Fogel 1999: +1 win, 0 draw, -2 loss)
    win_score: float = 1.0
    draw_score: float = 0.0
    loss_score: float = -2.0


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    generations: int = 250          # 250 gets A-class; 840 for expert (original)
    checkpoint_every: int = 25      # save population state
    log_every: int = 5              # print/log fitness stats
    
    # Device
    device: str = "cuda"            # "cuda" for Badger-1's RTX 5060, "cpu" fallback

    # Multiprocessing (CPU only): number of worker processes for parallel
    # tournament play. None means "auto" (all cores - 1).
    num_workers: Optional[int] = None

    # Tournament style. "random" is paper-faithful (Fogel 1999/2001: each
    # individual plays games_per_individual games vs randomly selected
    # opponents). "round-robin" evaluates every pair in both colors — more
    # accurate fitness, but not what the paper did.
    tournament: str = "random"

    # Reproducibility
    seed: Optional[int] = 42
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


@dataclass
class Config:
    """Top-level config aggregating all sub-configs."""
    checkers: CheckersConfig = field(default_factory=CheckersConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# Convenience: default config instance
DEFAULT_CONFIG = Config()
