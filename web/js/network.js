/*
 * 1999 CheckersNet — 1,743-weight MLP with piece-difference bypass and
 * evolvable king weight. Matches neural/network.py in the Python repo.
 *
 * Layout of the flat weights vector (Float32Array(1743)):
 *   [0     .. 1279]  W1 (32x40 row-major, fc1 kernel)
 *   [1280  .. 1319]  b1 (40)
 *   [1320  .. 1719]  W2 (40x10 row-major, fc2 kernel)
 *   [1720  .. 1729]  b2 (10)
 *   [1730  .. 1739]  W3 (10x1, fc3 kernel)
 *   [1740]           b3 (1)
 *   [1741]           piece_diff_weight
 *   [1742]           king_weight
 *
 * Totals: 1280 + 40 + 400 + 10 + 10 + 1 + 1 + 1 = 1743  ✓
 *
 * Encoding convention: +1 own man, +K own king, -1 opp man, -K opp king,
 * 0 empty. "Own" = current side-to-move, so the network is symmetric.
 */

(function (global) {
  "use strict";

  const N_WEIGHTS = 1743;
  const C = global.Checkers;

  // Offsets into the flat weights vector
  const OFF_W1 = 0;
  const OFF_B1 = OFF_W1 + 32 * 40;
  const OFF_W2 = OFF_B1 + 40;
  const OFF_B2 = OFF_W2 + 40 * 10;
  const OFF_W3 = OFF_B2 + 10;
  const OFF_B3 = OFF_W3 + 10;
  const OFF_PIECE_DIFF = OFF_B3 + 1;
  const OFF_KING = OFF_PIECE_DIFF + 1;

  function newRandomWeights(sigma) {
    // Paper-faithful initialization:
    //   - All evolvable weights, including the piece-difference bypass, are
    //     drawn from N(0, sigma). The paper did NOT seed the bypass with a
    //     useful value — it emerges through selection like everything else.
    //   - King weight is the one exception: paper explicitly initialized K
    //     at 2.0 (constrained to [1, 3] — we don't clamp, but it rarely
    //     drifts at sigma=0.05).
    // This means gen-0 networks play erratically (may give back material
    // freely) — a feature, not a bug, for the "watch it learn" demo.
    sigma = sigma == null ? 0.05 : sigma;
    const w = new Float32Array(N_WEIGHTS);
    for (let i = 0; i < N_WEIGHTS; i++) {
      w[i] = gauss() * sigma;
    }
    w[OFF_KING] = 2.0;
    return w;
  }

  function newSigmas(init) {
    init = init == null ? 0.05 : init;
    const s = new Float32Array(N_WEIGHTS);
    s.fill(init);
    return s;
  }

  // Padé [3/2] rational approximation of tanh, accurate to ~1e-4 on [-4, 4]
  // (and saturates cleanly beyond that). Avoids Math.tanh's exponential path,
  // which is the single biggest JS-side cost per forward pass when called
  // 51 times per eval × thousands of evals per minimax search.
  function ftanh(x) {
    if (x >  4.0) return  1.0;
    if (x < -4.0) return -1.0;
    const x2 = x * x;
    return x * (27.0 + x2) / (27.0 + 9.0 * x2);
  }

  // Box-Muller. Random N(0,1).
  function gauss() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  // Encode the board into a 32-element input vector from the perspective of
  // the side-to-move.
  const _inBuf = new Float32Array(32);
  function encodeInPlace(board, kingWeight, out) {
    const sq = board.squares;
    const side = board.currentPlayer; // +1 black, -1 white
    for (let i = 0; i < 32; i++) {
      const p = sq[i];
      if (p === 0) { out[i] = 0; continue; }
      const isKing = (p === 2 || p === -2);
      const val = isKing ? kingWeight : 1.0;
      out[i] = (p * side > 0) ? val : -val;
    }
    return out;
  }

  // Allocate and return a Network wrapper that holds views into the flat
  // weights and exposes forward() for minimax leaves.
  function makeNetwork(weights) {
    const w = weights;
    const h1 = new Float32Array(40);
    const h2 = new Float32Array(10);

    function forward(board) {
      const kw = w[OFF_KING];
      const x = _inBuf;
      encodeInPlace(board, kw, x);

      // fc1: h1 = tanh(W1 x + b1)
      let pdSum = 0;
      for (let i = 0; i < 40; i++) {
        let acc = w[OFF_B1 + i];
        const rowOff = OFF_W1 + i * 32;
        for (let j = 0; j < 32; j++) acc += w[rowOff + j] * x[j];
        h1[i] = ftanh(acc);
      }
      for (let j = 0; j < 32; j++) pdSum += x[j];

      // fc2: h2 = tanh(W2 h1 + b2)
      for (let i = 0; i < 10; i++) {
        let acc = w[OFF_B2 + i];
        const rowOff = OFF_W2 + i * 40;
        for (let j = 0; j < 40; j++) acc += w[rowOff + j] * h1[j];
        h2[i] = ftanh(acc);
      }

      // fc3: out = tanh(W3 h2 + b3 + piece_diff_weight * sum(x))
      let out = w[OFF_B3] + w[OFF_PIECE_DIFF] * pdSum;
      for (let j = 0; j < 10; j++) out += w[OFF_W3 + j] * h2[j];
      return ftanh(out);
    }

    function getKingWeight() { return w[OFF_KING]; }
    function getWeights() { return w; }

    return { forward, getKingWeight, getWeights };
  }

  // Schwefel self-adaptive EP mutation (matches evolution/strategy.py).
  //   sigma_i' = sigma_i * exp(tau_prime * N(0,1) + tau * N_i(0,1))
  //   w_i'     = w_i + sigma_i' * N_i(0,1)
  // No king-weight clamping (matches the repo's current behavior).
  function mutate(parentWeights, parentSigmas) {
    const n = parentWeights.length;
    const tau = 1.0 / Math.sqrt(2.0 * Math.sqrt(n));
    const tauPrime = 1.0 / Math.sqrt(2.0 * n);
    const globalNoise = gauss();
    const w = new Float32Array(n);
    const s = new Float32Array(n);
    const minSigma = 1e-5;
    for (let i = 0; i < n; i++) {
      const localNoise = gauss();
      let ns = parentSigmas[i] * Math.exp(tauPrime * globalNoise + tau * localNoise);
      if (ns < minSigma) ns = minSigma;
      s[i] = ns;
      w[i] = parentWeights[i] + ns * gauss();
    }
    return { weights: w, sigmas: s };
  }

  global.Network = {
    N_WEIGHTS,
    newRandomWeights, newSigmas,
    makeNetwork,
    mutate,
    encodeInPlace,   // exposed for tests
  };
})(typeof self !== "undefined" ? self : this);
