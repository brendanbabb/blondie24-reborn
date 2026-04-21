"""
Anaconda / Blondie24 evaluator (Chellapilla & Fogel 2001).

Extends the 1999 architecture with a spatial preprocessor: 91 sub-board
filter nodes (plus a constant-1 bias channel) feeding a 92 -> 40 -> 10 -> 1
MLP with tanh activations. The piece-difference bypass and king weight
carry over from the 1999 network.

Parameter count:
  input -> spatial preprocessor: 854 connections + 91 biases = 945
  preprocessor -> fc1 (92x40, no explicit bias):             3,680
  fc1 -> fc2 (40x10 + 10 biases):                              410
  fc2 -> fc3 (10x1 + 1 bias):                                   11
  ----------------------------------------------------------------
  Paper count:                                              5,046
  + piece_diff_weight (bypass):                                  1
  + king_weight:                                                 1
  ----------------------------------------------------------------
  Flat evolvable vector:                                     5,048

The paper reports 5,046 because it excludes bypass and treats king as a
separate evolvable hyperparameter. We fold both into the flat weight
vector so the evolution loop sees a single array of length 5,048.

The 92-vector fed to fc1 is [filters(91), 1.0]: the appended constant
acts as the 40 implicit biases for fc1 (algebraically equivalent to a
bias term; the paper counts it this way to get 5,046).
"""

import numpy as np
import torch
import torch.nn as nn

from config import NetworkConfig
from neural.anaconda_windows import (
    N_FILTERS, N_PP_WEIGHTS, scatter_indices,
)


class AnacondaNet(nn.Module):
    """
    2001 Anaconda evaluator. Public interface matches CheckersNet so the
    evolution loop and checkpoint machinery treat the two interchangeably.
    """

    def __init__(self, config: NetworkConfig = NetworkConfig()):
        super().__init__()
        self.config = config

        # Spatial preprocessor: only the 854 "live" connections are stored.
        # scatter_idx maps flat position -> cell in the dense (91, 32) weight
        # matrix. Dark squares outside each window stay permanently zero.
        self.pp_weights = nn.Parameter(torch.zeros(N_PP_WEIGHTS), requires_grad=False)
        self.pp_bias    = nn.Parameter(torch.zeros(N_FILTERS),  requires_grad=False)
        self.register_buffer("pp_scatter_idx", scatter_indices())

        # 92 -> 40 -> 10 -> 1. fc1 has no explicit bias; the appended
        # constant-1 channel of the preprocessor output serves that role
        # (and is what the paper counts to arrive at 5,046).
        self.fc1 = nn.Linear(N_FILTERS + 1, 40, bias=False)
        self.fc2 = nn.Linear(40, 10)
        self.fc3 = nn.Linear(10, 1)

        self.piece_diff_weight = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.king_weight = nn.Parameter(
            torch.tensor(config.initial_king_weight), requires_grad=False,
        )

        for p in self.parameters():
            p.requires_grad = False

    def _dense_pp_weight(self) -> torch.Tensor:
        """Scatter the flat 854-vector into a dense (91, 32) weight matrix."""
        dense = torch.zeros(
            N_FILTERS * 32, device=self.pp_weights.device, dtype=self.pp_weights.dtype,
        )
        dense.scatter_(0, self.pp_scatter_idx, self.pp_weights)
        return dense.view(N_FILTERS, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (32,) or (B, 32) board encoding (from current-player perspective).
        Returns a score in [-1, 1], shape matching the input's leading dim.
        """
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)

        W_pp = self._dense_pp_weight()                       # (91, 32)
        filters = torch.tanh(x @ W_pp.T + self.pp_bias)      # (B, 91)

        ones = torch.ones(filters.shape[0], 1, device=x.device, dtype=x.dtype)
        pp_out = torch.cat([filters, ones], dim=-1)          # (B, 92)

        h = torch.tanh(self.fc1(pp_out))                     # (B, 40)
        h = torch.tanh(self.fc2(h))                          # (B, 10)
        h = self.fc3(h)                                      # (B, 1)

        # Piece-difference bypass from the RAW 32-vector input, matching
        # the 1999 network. Not from the preprocessor output.
        bypass = x.sum(dim=-1, keepdim=True) * self.piece_diff_weight
        out = torch.tanh(h + bypass)                         # (B, 1)
        return out.squeeze(0) if single else out

    def evaluate_board(self, board_tensor: torch.Tensor) -> float:
        with torch.no_grad():
            score = self.forward(board_tensor)
        return score.item() if score.numel() == 1 else score.squeeze().item()

    # --- Flat weight-vector interface (same contract as CheckersNet) ---

    def get_weight_vector(self) -> np.ndarray:
        parts = [
            self.pp_weights.data.cpu().numpy().flatten(),        # 854
            self.pp_bias.data.cpu().numpy().flatten(),           # 91
            self.fc1.weight.data.cpu().numpy().flatten(),        # 3680
            self.fc2.weight.data.cpu().numpy().flatten(),        # 400
            self.fc2.bias.data.cpu().numpy().flatten(),          # 10
            self.fc3.weight.data.cpu().numpy().flatten(),        # 10
            self.fc3.bias.data.cpu().numpy().flatten(),          # 1
            self.piece_diff_weight.data.cpu().numpy().flatten(), # 1
            self.king_weight.data.cpu().numpy().flatten(),       # 1
        ]
        return np.concatenate(parts)

    def set_weight_vector(self, weights: np.ndarray):
        sections = [
            ("pp_weights", self.pp_weights, (N_PP_WEIGHTS,)),
            ("pp_bias", self.pp_bias, (N_FILTERS,)),
            ("fc1.weight", self.fc1.weight, self.fc1.weight.shape),
            ("fc2.weight", self.fc2.weight, self.fc2.weight.shape),
            ("fc2.bias", self.fc2.bias, self.fc2.bias.shape),
            ("fc3.weight", self.fc3.weight, self.fc3.weight.shape),
            ("fc3.bias", self.fc3.bias, self.fc3.bias.shape),
            ("piece_diff_weight", self.piece_diff_weight, ()),
            ("king_weight", self.king_weight, ()),
        ]
        offset = 0
        for _name, param, shape in sections:
            n = 1
            for d in shape:
                n *= d
            chunk = weights[offset:offset + n]
            if shape == ():
                param.data = torch.tensor(
                    float(chunk[0]), dtype=param.dtype, device=param.device,
                )
            else:
                param.data = torch.tensor(
                    np.asarray(chunk).reshape(shape),
                    dtype=param.dtype, device=param.device,
                )
            offset += n
        if offset != len(weights):
            raise ValueError(
                f"set_weight_vector: consumed {offset} of {len(weights)} weights"
            )

    def num_weights(self) -> int:
        return (
            N_PP_WEIGHTS + N_FILTERS                               # 945
            + self.fc1.weight.numel()                              # 3680
            + self.fc2.weight.numel() + self.fc2.bias.numel()      # 410
            + self.fc3.weight.numel() + self.fc3.bias.numel()      # 11
            + 1 + 1                                                # bypass + king
        )

    def copy(self) -> "AnacondaNet":
        new = AnacondaNet(self.config)
        new.set_weight_vector(self.get_weight_vector())
        return new
