"""
Pure-numpy evaluation for the Blondie24 network.

Mirrors CheckersNet.forward bit-for-bit on CPU, but without any PyTorch
dispatch overhead. A 1,743-parameter MLP is small enough that framework
overhead per call dominates actual math — dropping PyTorch for training
inference is a ~5x win on the CPU path.

Weight layout (matches CheckersNet.parameters() order — PyTorch yields a
module's own parameters BEFORE its child modules', so the scalars come
first, then the nn.Linear submodules in order):
    piece_diff_weight   :    1   [0]
    king_weight         :    1   [1]
    fc1.weight (40, 32) : 1280   [2..1281]
    fc1.bias   (40,)    :   40   [1282..1321]
    fc2.weight (10, 40) :  400   [1322..1721]
    fc2.bias   (10,)    :   10   [1722..1731]
    fc3.weight (1,  10) :   10   [1732..1741]
    fc3.bias   (1,)     :    1   [1742]
                         -----
                          1743
"""

import numpy as np


PD_IDX   = 0
KW_IDX   = 1
FC1_W_S  = 2
FC1_W_E  = FC1_W_S + 1280         # 1282
FC1_B_E  = FC1_W_E + 40           # 1322
FC2_W_E  = FC1_B_E + 400          # 1722
FC2_B_E  = FC2_W_E + 10           # 1732
FC3_W_E  = FC2_B_E + 10           # 1742
FC3_B_E  = FC3_W_E + 1            # 1743


def unpack_weights(w: np.ndarray):
    """Split flat 1743-length weight vector into typed matrices/scalars."""
    w = np.asarray(w, dtype=np.float32)
    piece_diff = float(w[PD_IDX])
    king_weight = float(w[KW_IDX])
    W1 = w[FC1_W_S:FC1_W_E].reshape(40, 32)
    b1 = w[FC1_W_E:FC1_B_E]
    W2 = w[FC1_B_E:FC2_W_E].reshape(10, 40)
    b2 = w[FC2_W_E:FC2_B_E]
    W3 = w[FC2_B_E:FC3_W_E].reshape(1, 10)
    b3 = w[FC3_W_E:FC3_B_E]
    return W1, b1, W2, b2, W3, b3, piece_diff, king_weight


def forward_batch(
    x: np.ndarray,
    W1: np.ndarray, b1: np.ndarray,
    W2: np.ndarray, b2: np.ndarray,
    W3: np.ndarray, b3: np.ndarray,
    piece_diff: float,
) -> np.ndarray:
    """
    Batched forward pass.

    Args:
        x: (N, 32) float32 board encoding
        W1..b3: decoded network weights
        piece_diff: scalar bypass weight

    Returns:
        (N,) float32 scores in [-1, 1]
    """
    # x @ W.T is equivalent to PyTorch's Linear (y = x W^T + b)
    h1 = np.tanh(x @ W1.T + b1)          # (N, 40)
    h2 = np.tanh(h1 @ W2.T + b2)         # (N, 10)
    h3 = h2 @ W3.T + b3                  # (N, 1)
    pd = x.sum(axis=-1, keepdims=True) * piece_diff  # (N, 1)
    out = np.tanh(h3 + pd)
    return out[:, 0]


def encode_boards(
    squares_batch: np.ndarray,
    player_batch: np.ndarray,
    king_weight: float,
) -> np.ndarray:
    """
    Encode a batch of boards into network input vectors.

    Args:
        squares_batch: (N, 32) int8 raw square states
        player_batch: (N,) int8 — 1 (BLACK) or -1 (WHITE), viewpoint
        king_weight: evolvable king value

    Returns:
        (N, 32) float32 encodings, each from the current player's viewpoint
    """
    N = squares_batch.shape[0]
    enc = np.zeros((N, 32), dtype=np.float32)

    # Signed piece values: +1/+K for own pieces, -1/-K for opponent.
    # Multiply by player viewpoint to flip signs for white-to-move.
    raw = squares_batch.astype(np.float32)
    # Raw encoding: BLACK_PIECE=1, BLACK_KING=2, WHITE_PIECE=-1, WHITE_KING=-2.
    # We want: pieces → ±1, kings → ±king_weight.
    is_piece = (np.abs(raw) == 1.0)
    is_king  = (np.abs(raw) == 2.0)
    sign = np.sign(raw)  # +1 black, -1 white, 0 empty
    enc = sign * (is_piece.astype(np.float32) + king_weight * is_king.astype(np.float32))
    # Flip perspective: multiply by player viewpoint
    enc *= player_batch[:, None].astype(np.float32)
    return enc
