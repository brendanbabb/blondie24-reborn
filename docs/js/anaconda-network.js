/*
 * Anaconda (2001) evaluator — JS port of neural/anaconda_network.py.
 *
 * Architecture (matches Chellapilla & Fogel 2001):
 *   32-square board (from side-to-move's perspective)
 *     → 91 sub-board filter nodes (3x3..8x8 windows, tanh),
 *     → 92-vector [filters..., 1.0]  (the trailing 1 is the fc1 bias channel)
 *     → 40-node tanh layer (no explicit bias: the 92nd channel serves that role)
 *     → 10-node tanh layer (with bias)
 *     → 1 scalar (with bias)
 *     → + piece-difference bypass × sum(board)
 *     → tanh  → final eval in [-1, +1]
 *
 * Flat weight layout (5,048 floats total):
 *   [0     .. 853]    pp_weights       (854, scatter-indexed into 91×32 dense)
 *   [854   .. 944]    pp_bias          (91)
 *   [945   .. 4624]   fc1.weight       (40×92 row-major — no bias)
 *   [4625  .. 5024]   fc2.weight       (10×40 row-major)
 *   [5025  .. 5034]   fc2.bias         (10)
 *   [5035  .. 5044]   fc3.weight       (10)
 *   [5045]            fc3.bias         (1)
 *   [5046]            piece_diff_weight
 *   [5047]            king_weight
 *
 * This is the exact layout emitted by scripts/export_weights_to_js.py
 * and consumed from docs/weights/anaconda.bin (little-endian float32).
 *
 * The 91 windows are enumerated deterministically at module load using the
 * same order as neural/anaconda_windows.py:
 *   sizes descending (8,7,6,5,4,3), within size scan (row0, col0) row-major.
 * Changing that order silently invalidates checkpoints, so don't.
 */

(function (global) {
  "use strict";

  const N_WEIGHTS = 5048;
  const N_FILTERS = 91;
  const N_PP_WEIGHTS = 854;

  const OFF_PP_W   = 0;
  const OFF_PP_B   = OFF_PP_W + N_PP_WEIGHTS;            // 854
  const OFF_FC1    = OFF_PP_B + N_FILTERS;               // 945
  const OFF_FC2W   = OFF_FC1 + 40 * 92;                  // 4625
  const OFF_FC2B   = OFF_FC2W + 10 * 40;                 // 5025
  const OFF_FC3W   = OFF_FC2B + 10;                      // 5035
  const OFF_FC3B   = OFF_FC3W + 10;                      // 5045
  const OFF_PD     = OFF_FC3B + 1;                       // 5046
  const OFF_KING   = OFF_PD + 1;                         // 5047

  // ---- Windows table (built once at module load) ------------------------
  //
  // Flat layout to avoid Array-of-Arrays allocation churn in the hot path.
  //   WINDOWS_FLAT[k]  — the dark-square index (0..31) of the k-th connection
  //   WINDOW_STARTS[i] — offset into WINDOWS_FLAT where filter i's squares
  //                      begin; WINDOW_STARTS[N_FILTERS] = N_PP_WEIGHTS.
  //
  // Iterating pp_weights in lockstep with WINDOWS_FLAT reproduces the
  // scatter_indices mapping used by the Python forward pass.

  const WINDOW_SIZES = [8, 7, 6, 5, 4, 3];

  function darkSquareIndex(r, c) {
    // Same as neural/anaconda_windows.py::_dark_square_index. Undefined for
    // light squares — we only call with dark coordinates.
    return r * 4 + (c >> 1);
  }

  const WINDOWS_FLAT = new Int8Array(N_PP_WEIGHTS);
  const WINDOW_STARTS = new Int32Array(N_FILTERS + 1);
  (function buildWindows() {
    let filterI = 0;
    let flatK = 0;
    for (const n of WINDOW_SIZES) {
      for (let r0 = 0; r0 <= 8 - n; r0++) {
        for (let c0 = 0; c0 <= 8 - n; c0++) {
          WINDOW_STARTS[filterI] = flatK;
          for (let r = r0; r < r0 + n; r++) {
            for (let c = c0; c < c0 + n; c++) {
              if (((r + c) & 1) === 1) {
                WINDOWS_FLAT[flatK] = darkSquareIndex(r, c);
                flatK++;
              }
            }
          }
          filterI++;
        }
      }
    }
    WINDOW_STARTS[N_FILTERS] = flatK;
    if (filterI !== N_FILTERS || flatK !== N_PP_WEIGHTS) {
      throw new Error(
        `AnacondaNetwork: windows table built incorrectly — got ` +
        `filters=${filterI} (want ${N_FILTERS}), weights=${flatK} ` +
        `(want ${N_PP_WEIGHTS})`
      );
    }
  })();

  // Use exact Math.tanh here (not the Padé approximation used by the 1999
  // live-evolve demo). Rationale: the play-strong page runs one minimax
  // search per human move, not a high-throughput training loop, so paying
  // a few ns per tanh for bit-exact agreement with the Python network is
  // a trivial cost for a cleaner correctness story. Cross-checked against
  // scripts/export_weights_to_js.py fixtures to ~1e-8.
  const ftanh = Math.tanh;

  // Board encoder — identical convention to the 1999 network: ±1 man,
  // ±K king, 0 empty, from the side-to-move's perspective. Writes into
  // `out` (length 32). Kept here so this file doesn't depend on
  // network.js load order.
  const _inBuf = new Float32Array(32);
  function encodeInPlace(board, kingWeight, out) {
    const sq = board.squares;
    const side = board.currentPlayer;
    for (let i = 0; i < 32; i++) {
      const p = sq[i];
      if (p === 0) { out[i] = 0; continue; }
      const isKing = (p === 2 || p === -2);
      const val = isKing ? kingWeight : 1.0;
      out[i] = (p * side > 0) ? val : -val;
    }
    return out;
  }

  // ---- Network wrapper ---------------------------------------------------

  function makeNetwork(weights) {
    if (weights.length !== N_WEIGHTS) {
      throw new Error(
        `AnacondaNetwork: expected ${N_WEIGHTS} weights, got ${weights.length}`
      );
    }
    const w = weights;

    // Scratch buffers reused across forward calls (single-threaded).
    const filters = new Float32Array(N_FILTERS);
    const h1 = new Float32Array(40);
    const h2 = new Float32Array(10);

    function forward(board) {
      const kw = w[OFF_KING];
      const x = _inBuf;
      encodeInPlace(board, kw, x);

      // 1. Sub-board filters: filters[i] = tanh(sum_{sq in window_i} pp_w[k]*x[sq] + pp_bias[i]).
      //    We iterate pp_weights linearly (index k) alongside WINDOWS_FLAT.
      let k = 0;
      for (let i = 0; i < N_FILTERS; i++) {
        let acc = w[OFF_PP_B + i];
        const end = WINDOW_STARTS[i + 1];
        for (let m = WINDOW_STARTS[i]; m < end; m++) {
          acc += w[OFF_PP_W + k] * x[WINDOWS_FLAT[m]];
          k++;
        }
        filters[i] = ftanh(acc);
      }

      // 2. fc1 (40 × 92, no bias): pp_out = [filters..., 1.0]. The 91st
      //    column (index 91 in each row) multiplies the constant-1 channel
      //    and acts as the bias.
      for (let i = 0; i < 40; i++) {
        const rowOff = OFF_FC1 + i * 92;
        let acc = 0;
        for (let j = 0; j < N_FILTERS; j++) acc += w[rowOff + j] * filters[j];
        acc += w[rowOff + N_FILTERS];  // × 1.0
        h1[i] = ftanh(acc);
      }

      // 3. fc2 (10 × 40, with bias)
      for (let i = 0; i < 10; i++) {
        const rowOff = OFF_FC2W + i * 40;
        let acc = w[OFF_FC2B + i];
        for (let j = 0; j < 40; j++) acc += w[rowOff + j] * h1[j];
        h2[i] = ftanh(acc);
      }

      // 4. fc3 (1 × 10, with bias) + piece-difference bypass + final tanh
      let out = w[OFF_FC3B];
      for (let j = 0; j < 10; j++) out += w[OFF_FC3W + j] * h2[j];
      let pdSum = 0;
      for (let j = 0; j < 32; j++) pdSum += x[j];
      out += w[OFF_PD] * pdSum;
      return ftanh(out);
    }

    function getKingWeight() { return w[OFF_KING]; }
    function getWeights() { return w; }

    return { forward, getKingWeight, getWeights };
  }

  // Load a weights binary (little-endian float32) from a URL.
  // Returns a Promise<Float32Array> of length N_WEIGHTS.
  function loadWeightsFromUrl(url) {
    return fetch(url).then((r) => {
      if (!r.ok) throw new Error(`fetch ${url}: HTTP ${r.status}`);
      return r.arrayBuffer();
    }).then((buf) => {
      const expected = N_WEIGHTS * 4;
      if (buf.byteLength !== expected) {
        throw new Error(
          `${url}: expected ${expected} bytes (${N_WEIGHTS} float32s), got ${buf.byteLength}`
        );
      }
      return new Float32Array(buf);
    });
  }

  global.AnacondaNetwork = {
    N_WEIGHTS,
    N_FILTERS,
    N_PP_WEIGHTS,
    makeNetwork,
    loadWeightsFromUrl,
    encodeInPlace,  // exposed for tests
    // Exposed for test cross-checks; don't mutate externally.
    _WINDOWS_FLAT: WINDOWS_FLAT,
    _WINDOW_STARTS: WINDOW_STARTS,
  };
})(typeof self !== "undefined" ? self : this);
