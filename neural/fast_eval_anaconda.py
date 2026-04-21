"""
Pure-numpy evaluation for the 2001 Anaconda network.

Mirrors AnacondaNet.forward bit-for-bit on CPU without any PyTorch
dispatch overhead. The 5,048-parameter net is still small enough that
framework overhead per call dominates actual math, so the numpy path
keeps tournament play fast.

Weight layout (matches AnacondaNet.get_weight_vector — the 854 flat
preprocessor connections come first, then the 91 biases, then fc1/2/3
in PyTorch parameter order, and finally the two scalars):

    pp_weights (854,)        :  854   [0..853]
    pp_bias    (91,)         :   91   [854..944]
    fc1.weight (40, 92)      : 3680   [945..4624]       (no explicit bias)
    fc2.weight (10, 40)      :  400   [4625..5024]
    fc2.bias   (10,)         :   10   [5025..5034]
    fc3.weight (1,  10)      :   10   [5035..5044]
    fc3.bias   (1,)          :    1   [5045]
    piece_diff_weight        :    1   [5046]
    king_weight              :    1   [5047]
                              -----
                               5048
"""

import numpy as np

from neural.anaconda_windows import WINDOWS, N_FILTERS, N_PP_WEIGHTS


PP_W_S   = 0
PP_W_E   = PP_W_S + N_PP_WEIGHTS       # 854
PP_B_E   = PP_W_E  + N_FILTERS         # 945
FC1_W_E  = PP_B_E  + 40 * (N_FILTERS + 1)   # 4625
FC2_W_E  = FC1_W_E + 10 * 40           # 5025
FC2_B_E  = FC2_W_E + 10                # 5035
FC3_W_E  = FC2_B_E + 10                # 5045
FC3_B_E  = FC3_W_E + 1                 # 5046
PD_IDX   = FC3_B_E                     # 5046
KW_IDX   = PD_IDX + 1                  # 5047
TOTAL    = KW_IDX + 1                  # 5048


def _build_scatter_numpy() -> np.ndarray:
    """Pure-numpy analog of anaconda_windows.scatter_indices()."""
    idx = np.empty(N_PP_WEIGHTS, dtype=np.int64)
    k = 0
    for filter_i, dark_squares in enumerate(WINDOWS):
        for sq in dark_squares:
            idx[k] = filter_i * 32 + sq
            k += 1
    return idx


SCATTER_IDX_NP: np.ndarray = _build_scatter_numpy()


def dense_pp_weights(flat_pp: np.ndarray) -> np.ndarray:
    """Scatter a flat (854,) pp weight vector into a dense (91, 32) matrix."""
    dense = np.zeros(N_FILTERS * 32, dtype=np.float32)
    dense[SCATTER_IDX_NP] = np.asarray(flat_pp, dtype=np.float32)
    return dense.reshape(N_FILTERS, 32)


def unpack_weights_anaconda(w: np.ndarray):
    """
    Split the flat 5048-length Anaconda weight vector into typed tensors.

    Returns
    -------
    W_pp       : (91, 32) float32, dense preprocessor weight matrix
    b_pp       : (91,)    float32, preprocessor biases
    W1         : (40, 92) float32, fc1 weights (no bias — constant-1 channel handles it)
    W2, b2     : (10, 40), (10,)   fc2
    W3, b3     : (1, 10),  (1,)    fc3
    piece_diff : float              bypass weight
    king_weight: float              evolvable king value
    """
    w = np.asarray(w, dtype=np.float32)
    if w.shape[0] != TOTAL:
        raise ValueError(
            f"Expected {TOTAL}-element Anaconda weight vector, got {w.shape[0]}"
        )
    W_pp = dense_pp_weights(w[PP_W_S:PP_W_E])
    b_pp = w[PP_W_E:PP_B_E].copy()
    W1   = w[PP_B_E:FC1_W_E].reshape(40, N_FILTERS + 1).copy()
    W2   = w[FC1_W_E:FC2_W_E].reshape(10, 40).copy()
    b2   = w[FC2_W_E:FC2_B_E].copy()
    W3   = w[FC2_B_E:FC3_W_E].reshape(1, 10).copy()
    b3   = w[FC3_W_E:FC3_B_E].copy()
    piece_diff  = float(w[PD_IDX])
    king_weight = float(w[KW_IDX])
    return W_pp, b_pp, W1, W2, b2, W3, b3, piece_diff, king_weight


def forward_batch_anaconda(
    x: np.ndarray,
    W_pp: np.ndarray, b_pp: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray, b2: np.ndarray,
    W3: np.ndarray, b3: np.ndarray,
    piece_diff: float,
) -> np.ndarray:
    """
    Batched forward pass through the Anaconda network.

    Args:
        x: (N, 32) float32 encoding
        W_pp: (91, 32) dense scattered preprocessor weights
        b_pp: (91,) preprocessor biases
        W1: (40, 92) fc1 weights (no explicit bias)
        W2, b2: fc2 (10, 40) + (10,)
        W3, b3: fc3 (1, 10) + (1,)
        piece_diff: scalar bypass weight

    Returns:
        (N,) float32 scores in [-1, 1]
    """
    filters = np.tanh(x @ W_pp.T + b_pp)                      # (N, 91)
    ones = np.ones((filters.shape[0], 1), dtype=np.float32)
    pp_out = np.concatenate([filters, ones], axis=1)          # (N, 92)
    h1 = np.tanh(pp_out @ W1.T)                               # (N, 40) — no bias
    h2 = np.tanh(h1 @ W2.T + b2)                              # (N, 10)
    h3 = h2 @ W3.T + b3                                       # (N, 1)
    pd = x.sum(axis=-1, keepdims=True) * piece_diff           # (N, 1)
    out = np.tanh(h3 + pd)
    return out[:, 0]
