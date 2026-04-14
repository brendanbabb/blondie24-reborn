"""
Board state → tensor encoding for the neural network.

Encoding scheme (from Chellapilla & Fogel):
  For each of the 32 squares, from the perspective of the current player:
    +1.0         = own regular piece
    +king_weight = own king
    -1.0         = opponent regular piece
    -king_weight = opponent king
    0.0          = empty

This means the network always "sees" the board from the perspective of the
player to move, which is important — the same network can play both colors.
"""

import torch
import numpy as np
from checkers.board import Board, BLACK, WHITE, BLACK_PIECE, BLACK_KING, WHITE_PIECE, WHITE_KING, EMPTY


def encode_board(board: Board, king_weight: float = 1.3) -> torch.Tensor:
    """
    Encode a board state as a tensor from the current player's perspective.
    
    Args:
        board: the current board state
        king_weight: value assigned to kings (evolvable parameter)
    
    Returns:
        tensor of shape (32,) with values in [-king_weight, +king_weight]
    """
    player = board.current_player
    encoding = np.zeros(32, dtype=np.float32)
    
    for i in range(32):
        piece = board.squares[i]
        if piece == EMPTY:
            encoding[i] = 0.0
        elif piece == BLACK_PIECE:
            encoding[i] = 1.0 if player == BLACK else -1.0
        elif piece == BLACK_KING:
            encoding[i] = king_weight if player == BLACK else -king_weight
        elif piece == WHITE_PIECE:
            encoding[i] = 1.0 if player == WHITE else -1.0
        elif piece == WHITE_KING:
            encoding[i] = king_weight if player == WHITE else -king_weight
    
    return torch.tensor(encoding, dtype=torch.float32)


def encode_board_batch(boards: list[Board], king_weight: float = 1.3) -> torch.Tensor:
    """
    Encode multiple boards as a batched tensor.
    
    Args:
        boards: list of Board objects
        king_weight: value assigned to kings
    
    Returns:
        tensor of shape (N, 32)
    """
    encodings = [encode_board(b, king_weight).numpy() for b in boards]
    return torch.tensor(np.stack(encodings), dtype=torch.float32)
