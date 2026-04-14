"""
Feedforward neural network for board evaluation.

Matches the Blondie24 architecture from Chellapilla & Fogel (1999):

  32 inputs → 40 hidden (tanh) → 10 hidden (tanh) → 1 output (tanh)
  + piece-difference bypass: sum(inputs) → output (weighted, bypasses hidden layers)
  + evolvable king weight K

The piece-difference bypass gives the network an implicit material count
signal without having to learn it through the hidden layers. The sum of
the 32 input values approximates (own pieces - opponent pieces), scaled
by K for kings.

Total evolvable parameters: 1,742
  fc1: 32×40 + 40 = 1,320
  fc2: 40×10 + 10 = 410
  fc3: 10×1 + 1   = 11
  piece_diff_weight: 1   (bypass connection)
  king_weight: 1
  ─────────────────
  Total: 1,743

(Fogel's paper reports ~1,742; the ±1 depends on whether the king weight
is counted as a network parameter or a separate evolvable.)
"""

import torch
import torch.nn as nn
import numpy as np
from config import NetworkConfig


class CheckersNet(nn.Module):
    """
    Blondie24-style feedforward evaluation network (1999 paper architecture).
    
    Architecture faithfully matches Chellapilla & Fogel (1999):
    - 32 input nodes (one per playable square)
    - 40-node hidden layer with tanh activation
    - 10-node hidden layer with tanh activation
    - 1 output node with tanh activation
    - Piece-difference bypass: sum of 32 inputs connects directly to output
    - Evolvable king weight K (value assigned to kings in the encoding)
    
    All weights evolved by evolutionary programming — none trained by backprop.
    """
    
    def __init__(self, config: NetworkConfig = NetworkConfig()):
        super().__init__()
        self.config = config
        
        self.fc1 = nn.Linear(config.input_size, config.hidden1_size)
        self.fc2 = nn.Linear(config.hidden1_size, config.hidden2_size)
        self.fc3 = nn.Linear(config.hidden2_size, config.output_size)
        
        # Piece-difference bypass: a single learned weight connecting
        # sum(inputs) directly to the output, bypassing hidden layers.
        # This lets the network implicitly compute material advantage.
        self.piece_diff_weight = nn.Parameter(
            torch.tensor(0.0), requires_grad=False
        )
        
        # King weight: how much a king is worth relative to a regular piece
        # in the board encoding. Evolvable, initialized at K=2.0 per the paper.
        self.king_weight = nn.Parameter(
            torch.tensor(config.initial_king_weight), requires_grad=False
        )
        
        # Disable gradient tracking — we don't use backprop
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: tensor of shape (batch, 32) or (32,) — board encoding
        
        Returns:
            tensor of shape (batch, 1) or (1,) — evaluation score in [-1, 1]
        """
        # Piece-difference bypass: sum of inputs × learned weight
        piece_diff = x.sum(dim=-1, keepdim=True) * self.piece_diff_weight
        
        # Standard feedforward path through hidden layers
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        h = self.fc3(h)  # linear output before final activation
        
        # Combine hidden path + bypass, then apply tanh
        out = torch.tanh(h + piece_diff)
        return out
    
    def evaluate_board(self, board_tensor: torch.Tensor) -> float:
        """Convenience: evaluate a single board, return Python float."""
        with torch.no_grad():
            score = self.forward(board_tensor)
        return score.item()
    
    def get_weight_vector(self) -> np.ndarray:
        """Flatten all parameters (weights + biases + king_weight) into a 1D numpy array."""
        params = []
        for param in self.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)
    
    def set_weight_vector(self, weights: np.ndarray):
        """Load a 1D numpy array back into the network parameters."""
        offset = 0
        for param in self.parameters():
            n = param.numel()
            param.data = torch.tensor(
                weights[offset:offset + n].reshape(param.shape),
                dtype=param.dtype,
                device=param.device,
            )
            offset += n
    
    def num_weights(self) -> int:
        """Total number of evolvable parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def copy(self) -> "CheckersNet":
        """Create a deep copy of this network."""
        new_net = CheckersNet(self.config)
        new_net.set_weight_vector(self.get_weight_vector())
        return new_net
