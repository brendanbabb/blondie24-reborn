/*
 * Negamax alpha-beta with iterative deepening, a per-search Zobrist
 * transposition table, and a neural-network leaf evaluator.
 *
 * Score convention: always from the side-to-move's perspective. A terminal
 * "no moves" position returns a large negative score (the side to move lost).
 * Internal nodes flip sign across recursion.
 *
 * Speed-side details (all correctness-preserving — same alpha-beta the paper
 * describes; the returned score and root move match plain depth-N alpha-beta
 * modulo tie-breaks):
 *   - make/unmake on a single board (no Int8Array(32) clone per node)
 *   - per-search 16k-slot Zobrist TT with EXACT/LOWER/UPPER flags, valid for
 *     the lifetime of one pickMove() call (cleared via generation counter)
 *   - iterative deepening 1..N at the root, with TT-best-move tried first
 *
 * Depth is in plies. Default 4 matches the paper.
 */

(function (global) {
  "use strict";

  const C = global.Checkers;

  const WIN = 1e6;

  // ---- Zobrist hash tables ----
  // Random 32-bit int per (square, piece-code) pair, plus one for "white to
  // move". Piece codes are mapped to 0..3 by pieceIdx(). Module-load init.
  const ZOB = new Int32Array(32 * 4);
  let ZOB_SIDE = 0;
  (function initZobrist() {
    for (let i = 0; i < ZOB.length; i++) {
      ZOB[i] = (Math.random() * 0x100000000) | 0;
    }
    ZOB_SIDE = (Math.random() * 0x100000000) | 0;
  })();

  function pieceIdx(p) {
    if (p === 1) return 0;   // black man
    if (p === 2) return 1;   // black king
    if (p === -1) return 2;  // white man
    return 3;                // -2, white king
  }

  function hashBoard(board) {
    let h = 0;
    const sq = board.squares;
    for (let i = 0; i < 32; i++) {
      const p = sq[i];
      if (p !== 0) h ^= ZOB[i * 4 + pieceIdx(p)];
    }
    if (board.currentPlayer === C.WHITE) h ^= ZOB_SIDE;
    return h | 0;
  }

  // ---- Transposition table (per-search) ----
  // 16k entries × ~12 bytes = ~192 KB. Always-replace, indexed by low bits
  // of the Zobrist hash with a full-hash collision check on probe. The
  // generation counter avoids needing to clear the arrays between searches:
  // an entry is valid only if ttGen[idx] === currentTtGen.
  const TT_SIZE = 1 << 14;
  const TT_MASK = TT_SIZE - 1;

  const FLAG_EXACT = 1;
  const FLAG_LOWER = 2;  // value is a lower bound (fail-high / beta cutoff)
  const FLAG_UPPER = 3;  // value is an upper bound (fail-low / no improvement)

  const ttHash     = new Int32Array(TT_SIZE);
  const ttDepth    = new Int8Array(TT_SIZE);
  const ttFlag     = new Int8Array(TT_SIZE);
  const ttScore    = new Float32Array(TT_SIZE);
  const ttBestFrom = new Int8Array(TT_SIZE);
  const ttBestTo   = new Int8Array(TT_SIZE);
  const ttGen      = new Uint16Array(TT_SIZE);
  let currentTtGen = 0;

  function ttBumpGen() {
    currentTtGen = (currentTtGen + 1) & 0xFFFF;
    if (currentTtGen === 0) {
      ttGen.fill(0);
      currentTtGen = 1;
    }
  }

  // Returns a probe descriptor or null if no valid entry. Structure is
  // re-used internally — callers must read fields immediately, not stash.
  const _probe = { flag: 0, depth: 0, score: 0, bestFrom: -1, bestTo: -1 };
  function ttProbe(hash) {
    const idx = hash & TT_MASK;
    if (ttGen[idx] !== currentTtGen) return null;
    if (ttHash[idx] !== hash) return null;
    _probe.flag = ttFlag[idx];
    _probe.depth = ttDepth[idx];
    _probe.score = ttScore[idx];
    _probe.bestFrom = ttBestFrom[idx];
    _probe.bestTo = ttBestTo[idx];
    return _probe;
  }

  function ttStore(hash, depth, flag, score, bestFrom, bestTo) {
    const idx = hash & TT_MASK;
    ttHash[idx] = hash;
    ttDepth[idx] = depth;
    ttFlag[idx] = flag;
    ttScore[idx] = score;
    ttBestFrom[idx] = bestFrom;
    ttBestTo[idx] = bestTo;
    ttGen[idx] = currentTtGen;
  }

  // Order moves: bigger captures first (forced-jump rules: when any jump
  // exists every move IS a jump, so this ranks longer chains earlier). If
  // the TT has a best-move hint for this position, bump it to index 0.
  function orderMoves(moves, hintFrom, hintTo) {
    if (moves.length <= 1) return moves;
    const sorted = moves.slice().sort((a, b) => b.length - a.length);
    if (hintFrom < 0) return sorted;
    for (let i = 0; i < sorted.length; i++) {
      const m = sorted[i];
      if (m[0] === hintFrom && m[m.length - 1] === hintTo) {
        if (i !== 0) {
          const tmp = sorted[i];
          for (let j = i; j > 0; j--) sorted[j] = sorted[j - 1];
          sorted[0] = tmp;
        }
        return sorted;
      }
    }
    return sorted;
  }

  // Per-search counters; reset at the start of every pickMove. Exposed so
  // callers (the play-strong UI) can show "search effort" stats.
  const searchStats = { evaluated: 0, pruned: 0 };

  function pickMove(board, depth, network) {
    const raw = C.getLegalMoves(board);
    if (raw.length === 0) {
      return { move: null, score: -WIN, pv: [], nodesEvaluated: 0, nodesPruned: 0 };
    }

    ttBumpGen();
    searchStats.evaluated = 0;
    searchStats.pruned = 0;

    let bestMove = raw[0];
    let bestScore = -Infinity;

    // Iterative deepening: each iteration's TT entries seed move ordering
    // for the next. Total cost is dominated by the final iteration (earlier
    // depths cost ~1/B, 1/B^2, ... of the deepest), so the overhead is small
    // and the better ordering at depth N typically more than pays it back.
    for (let d = 1; d <= depth; d++) {
      const r = rootSearch(board, d, network, raw);
      bestMove = r.move;
      bestScore = r.score;
    }

    // Walk the TT from root to extract the AI's predicted line (principal
    // variation). The TT is still keyed by the current generation, so root
    // and one-ply-down entries from the deepest ID iteration are still
    // probe-able. Stops short if a probe misses or no move matches the
    // (from, to) hint (rare; can happen if a multi-jump's TT key was
    // overwritten by a sibling search).
    const pv = extractPv(board, depth);

    return {
      move: bestMove,
      score: bestScore,
      pv: pv,
      nodesEvaluated: searchStats.evaluated,
      nodesPruned: searchStats.pruned,
    };
  }

  function extractPv(rootBoard, maxLen) {
    const pv = [];
    let board = C.cloneBoard(rootBoard);
    for (let i = 0; i < maxLen; i++) {
      const probe = ttProbe(hashBoard(board));
      if (!probe) break;
      if (probe.bestFrom < 0) break;
      const moves = C.getLegalMoves(board);
      if (moves.length === 0) break;
      const m = moves.find(mv =>
        mv[0] === probe.bestFrom && mv[mv.length - 1] === probe.bestTo
      );
      if (!m) break;
      pv.push(m);
      board = C.applyMove(board, m);
    }
    return pv;
  }

  function rootSearch(board, depth, network, rawMoves) {
    const hash = hashBoard(board);
    const probe = ttProbe(hash);
    const hintFrom = probe ? probe.bestFrom : -1;
    const hintTo   = probe ? probe.bestTo   : -1;
    const moves = orderMoves(rawMoves, hintFrom, hintTo);

    let bestMove = moves[0];
    let bestScore = -Infinity;
    let alpha = -Infinity;
    const beta = Infinity;

    for (let i = 0; i < moves.length; i++) {
      const m = moves[i];
      const undo = C.applyMoveInPlace(board, m);
      const s = -negamax(board, depth - 1, -beta, -alpha, network);
      C.undoMove(board, undo);
      if (s > bestScore) { bestScore = s; bestMove = m; }
      if (bestScore > alpha) alpha = bestScore;
    }

    ttStore(hash, depth, FLAG_EXACT, bestScore,
            bestMove[0], bestMove[bestMove.length - 1]);

    return { move: bestMove, score: bestScore };
  }

  function negamax(board, depth, alpha, beta, network) {
    const alpha0 = alpha;
    const hash = hashBoard(board);

    const probe = ttProbe(hash);
    let hintFrom = -1;
    let hintTo = -1;
    if (probe !== null) {
      hintFrom = probe.bestFrom;
      hintTo = probe.bestTo;
      if (probe.depth >= depth) {
        const f = probe.flag;
        const s = probe.score;
        if (f === FLAG_EXACT) return s;
        if (f === FLAG_LOWER && s >= beta) return s;
        if (f === FLAG_UPPER && s <= alpha) return s;
      }
    }

    const rawMoves = C.getLegalMoves(board);
    if (rawMoves.length === 0) {
      // Side to move has no moves → they lost.
      // Subtract depth so "mate in 1" is preferred over "mate in 3".
      return -WIN + (100 - depth);
    }
    if (depth === 0) {
      searchStats.evaluated++;
      return network.forward(board);
    }

    const moves = orderMoves(rawMoves, hintFrom, hintTo);
    let best = -Infinity;
    let bestFrom = moves[0][0];
    let bestTo = moves[0][moves[0].length - 1];

    for (let i = 0; i < moves.length; i++) {
      const m = moves[i];
      const undo = C.applyMoveInPlace(board, m);
      const s = -negamax(board, depth - 1, -beta, -alpha, network);
      C.undoMove(board, undo);
      if (s > best) {
        best = s;
        bestFrom = m[0];
        bestTo = m[m.length - 1];
      }
      if (best > alpha) alpha = best;
      if (alpha >= beta) {
        // Beta cutoff — every remaining sibling move is now guaranteed
        // worse than what we've already found. Counted once per cutoff
        // (not per pruned move) so the stat reads as "subtrees skipped".
        searchStats.pruned++;
        break;
      }
    }

    let flag;
    if (best <= alpha0)    flag = FLAG_UPPER;
    else if (best >= beta) flag = FLAG_LOWER;
    else                   flag = FLAG_EXACT;
    ttStore(hash, depth, flag, best, bestFrom, bestTo);

    return best;
  }

  // Evaluate the current position's score from side-to-move's perspective,
  // without searching. Used for UI display only.
  function leafEval(board, network) {
    return network.forward(board);
  }

  global.Minimax = { pickMove, leafEval, WIN };
})(typeof self !== "undefined" ? self : this);
