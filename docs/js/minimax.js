/*
 * Negamax alpha-beta with a neural-network leaf evaluator.
 *
 * Score convention: always from the side-to-move's perspective. A terminal
 * "no moves" position returns a large negative score (the side to move lost).
 * Internal nodes flip sign across recursion.
 *
 * Depth is in plies. Default 4 matches the paper.
 */

(function (global) {
  "use strict";

  const C = global.Checkers;

  const WIN = 1e6;

  function pickMove(board, depth, network) {
    const moves = C.getLegalMoves(board);
    if (moves.length === 0) return { move: null, score: -WIN };

    let bestMove = moves[0];
    let bestScore = -Infinity;
    let alpha = -Infinity;
    const beta = Infinity;

    // Optional: order moves by quick leaf eval to improve pruning. For pop=6
    // and depth=4 with only ~8 moves at the root, unordered is already fast.

    for (let i = 0; i < moves.length; i++) {
      const next = C.applyMove(board, moves[i]);
      const s = -negamax(next, depth - 1, -beta, -alpha, network);
      if (s > bestScore) { bestScore = s; bestMove = moves[i]; }
      if (bestScore > alpha) alpha = bestScore;
    }
    return { move: bestMove, score: bestScore };
  }

  function negamax(board, depth, alpha, beta, network) {
    const [over, _winner] = C.isGameOver(board);
    if (over) {
      // Side to move has no moves → they lost.
      // Subtract depth so "mate in 1" is preferred over "mate in 3".
      return -WIN + (100 - depth);
    }
    if (depth === 0) return network.forward(board);

    const moves = C.getLegalMoves(board);
    let best = -Infinity;
    for (let i = 0; i < moves.length; i++) {
      const next = C.applyMove(board, moves[i]);
      const s = -negamax(next, depth - 1, -beta, -alpha, network);
      if (s > best) best = s;
      if (best > alpha) alpha = best;
      if (alpha >= beta) break;
    }
    return best;
  }

  // Evaluate the current position's score from side-to-move's perspective,
  // without searching. Used for UI display only.
  function leafEval(board, network) {
    return network.forward(board);
  }

  global.Minimax = { pickMove, leafEval, WIN };
})(typeof self !== "undefined" ? self : this);
