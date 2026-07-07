"""
Five fixed board positions used to probe a network's discriminative health.

Shared by two consumers:
  - scripts/export_weights_to_js.py writes (board, score) pairs so the JS
    port can assert bit-level agreement with the Python forward pass.
  - training/train.py logs the max-min spread of the five scores each
    generation. A healthy network separates these positions; a saturated
    one (weights so large that tanh pins to ±1) scores them all the same.
    Past runs showed spread peaking mid-run and collapsing to ~0 as weights
    blow up — so the spread identifies which checkpoint to ship.

Board convention: 32-entry list over the playable squares, +1/-1 men,
+2/-2 kings, from black's perspective; current_player is +1 (black) or
-1 (white).
"""

import numpy as np
import torch


def _starting_position() -> list[int]:
    squares = [0] * 32
    for i in range(12):
        squares[i] = 1       # black men
    for i in range(20, 32):
        squares[i] = -1      # white men
    return squares


def _material_advantage() -> list[int]:
    squares = _starting_position()
    squares[20] = 0
    squares[21] = 0          # black up two pieces
    return squares


def _two_kings_sparse() -> list[int]:
    squares = [0] * 32
    squares[0] = 2           # black king
    squares[31] = -2         # white king
    squares[15] = 1          # black man midboard
    squares[16] = -1         # white man midboard
    return squares


def _endgame() -> list[int]:
    squares = [0] * 32
    squares[12] = 2          # black king
    squares[28] = -1         # white man
    return squares


# (label, squares_32, current_player)
FIXTURE_POSITIONS: list[tuple[str, list[int], int]] = [
    ("starting-position-black-to-move", _starting_position(), +1),
    ("starting-position-white-to-move", _starting_position(), -1),
    ("black-up-two-pieces-black-to-move", _material_advantage(), +1),
    ("two-kings-sparse-black-to-move", _two_kings_sparse(), +1),
    ("endgame-black-king-vs-white-man", _endgame(), +1),
]


def encode_fixture(squares_32: list[int], current_player: int,
                   king_weight: float) -> np.ndarray:
    """Encode a fixture board the same way the JS side does: ±1 men, ±K kings,
    from the current side-to-move's perspective."""
    x = np.zeros(32, dtype=np.float32)
    for i, p in enumerate(squares_32):
        if p == 0:
            continue
        mag = king_weight if abs(p) == 2 else 1.0
        x[i] = mag if (p * current_player) > 0 else -mag
    return x


def fixture_scores(net) -> list[float]:
    """Forward the five fixture positions through `net` (a CheckersNet or
    AnacondaNet whose weights are already loaded). Returns the five scores."""
    king_weight = float(net.king_weight.data.item())
    scores = []
    for _label, squares, current_player in FIXTURE_POSITIONS:
        x = encode_fixture(squares, current_player, king_weight)
        with torch.no_grad():
            scores.append(float(net.forward(torch.from_numpy(x)).item()))
    return scores


def fixture_spread(net) -> float:
    """max - min of the five fixture scores. ~0 means a saturated network."""
    scores = fixture_scores(net)
    return max(scores) - min(scores)
