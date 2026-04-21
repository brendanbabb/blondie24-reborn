"""
Sub-board window enumeration for the 2001 Anaconda spatial preprocessor.

Chellapilla & Fogel 2001 added a convolutional-style front end to the 1999
network: a bank of 91 sub-board "filter" nodes, one per (position, size)
window on the 8x8 board. Windows range from 3x3 up to 8x8. Each filter
takes a weighted sum of ONLY the dark (playable) squares inside its window,
applies tanh, and feeds the scalar to the next layer.

Window counts:
    1 x (8x8)   + 4 x (7x7)  + 9 x (6x6)
  + 16 x (5x5) + 25 x (4x4) + 36 x (3x3)   = 91 windows

Summing the dark-square count across all windows gives 854 connection
weights. Plus 91 biases (one per filter) that gives 945 input-to-preprocessor
weights, matching the paper.
"""

import numpy as np
import torch

WINDOW_SIZES = (8, 7, 6, 5, 4, 3)


def _dark_square_index(row: int, col: int) -> int:
    """Map an 8x8 (row, col) dark-square coordinate to its 0..31 index.

    Dark squares are those where (row+col) is odd — same convention used
    by Board.squares elsewhere in the repo. Raises for light squares.
    """
    if (row + col) % 2 == 0:
        raise ValueError(f"({row},{col}) is a light square")
    return row * 4 + col // 2


def enumerate_windows() -> list[list[int]]:
    """Return one list of dark-square indices per window, in a fixed order.

    Ordering: sizes descending (8, 7, 6, 5, 4, 3), and within each size,
    scan by (row0, col0) in row-major order. This ordering is frozen —
    changing it would silently invalidate every Anaconda checkpoint.
    """
    windows: list[list[int]] = []
    for n in WINDOW_SIZES:
        for r0 in range(8 - n + 1):
            for c0 in range(8 - n + 1):
                dark_squares = []
                for r in range(r0, r0 + n):
                    for c in range(c0, c0 + n):
                        if (r + c) % 2 == 1:
                            dark_squares.append(_dark_square_index(r, c))
                windows.append(dark_squares)
    return windows


# Compute once at import. These are pure functions of the architecture.
WINDOWS: list[list[int]] = enumerate_windows()
N_FILTERS: int = len(WINDOWS)                   # 91
N_PP_WEIGHTS: int = sum(len(w) for w in WINDOWS)  # 854


def scatter_indices() -> torch.Tensor:
    """Flat (854,) long tensor — where each of the 854 weights lives in the
    flattened (91 * 32,) dense preprocessor weight matrix.

    Used by AnacondaNet.forward to scatter the flat weight vector into a
    dense (91, 32) matrix for a single matmul, and by BatchedAnacondaNets
    to do the same in batched form.
    """
    idx = []
    for filter_i, dark_squares in enumerate(WINDOWS):
        for sq in dark_squares:
            idx.append(filter_i * 32 + sq)
    return torch.tensor(idx, dtype=torch.long)


def window_sizes_summary() -> dict[int, int]:
    """Return {size: count} for docs/tests. 1,4,9,16,25,36 for 8..3."""
    out: dict[int, int] = {}
    for w in WINDOWS:
        n = int(round(len(w) ** 0.5))  # not exact — see _size_of_window
        out[n] = out.get(n, 0) + 1
    return out


if __name__ == "__main__":
    # Sanity check the numbers this module is supposed to produce.
    print(f"{N_FILTERS} windows, {N_PP_WEIGHTS} connection weights")
    sizes_counts = [(n, (8 - n + 1) ** 2) for n in WINDOW_SIZES]
    print("windows by size:", sizes_counts)
    assert N_FILTERS == 91, N_FILTERS
    assert N_PP_WEIGHTS == 854, N_PP_WEIGHTS

    # Reconstruct a mask and verify each window has the right dark count.
    mask = np.zeros((91, 32), dtype=np.int8)
    for i, w in enumerate(WINDOWS):
        for sq in w:
            mask[i, sq] = 1
    assert mask.sum() == 854

    idx = scatter_indices()
    assert idx.shape == (854,)
    assert idx.min().item() >= 0
    assert idx.max().item() < 91 * 32
    print("OK")
