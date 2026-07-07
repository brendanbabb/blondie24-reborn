/*
 * Search worker — runs minimax off the main thread for the play-strong and
 * match pages, so deep (budgeted) searches never freeze the UI.
 *
 * Stateless except for a cache of Network wrappers keyed by slotId (wrapper
 * reuse matters: fresh forward-closures per search deopt V8's inline cache
 * at the minimax call site — same lesson as game-worker.js).
 *
 *   from client:
 *     { type: "weights", slotId, weights }         cache a net under slotId
 *     { type: "search", reqId, slotId,
 *       squares, currentPlayer, moveCount,         board snapshot
 *       depth, budgetMs, maxDepth }                pickMove args
 *
 *   from worker:
 *     { type: "ready" }
 *     { type: "result", reqId, move, score, pv, depthReached,
 *       nodesEvaluated, nodesPruned }
 *     { type: "error", reqId, message, stack }     per-request failure
 */

importScripts("checkers.js?v=8", "anaconda-network.js?v=35", "minimax.js?v=9");

const A = self.AnacondaNetwork;
const M = self.Minimax;

const nets = Object.create(null);  // slotId → Network wrapper

self.onmessage = function (ev) {
  const msg = ev.data || {};
  try {
    if (msg.type === "weights") {
      nets[msg.slotId] = A.makeNetwork(msg.weights);
      return;
    }
    if (msg.type === "search") {
      const net = nets[msg.slotId];
      if (!net) {
        postMessage({
          type: "error", reqId: msg.reqId,
          message: `no weights loaded for slot "${msg.slotId}"`, stack: "",
        });
        return;
      }
      const board = {
        squares: msg.squares,
        currentPlayer: msg.currentPlayer,
        moveCount: msg.moveCount,
      };
      const opts = msg.budgetMs
        ? { budgetMs: msg.budgetMs, maxDepth: msg.maxDepth }
        : undefined;
      const r = M.pickMove(board, msg.depth, net, opts);
      postMessage({
        type: "result", reqId: msg.reqId,
        move: r.move, score: r.score, pv: r.pv,
        depthReached: r.depthReached,
        nodesEvaluated: r.nodesEvaluated, nodesPruned: r.nodesPruned,
      });
      return;
    }
  } catch (err) {
    postMessage({
      type: "error", reqId: msg.reqId,
      message: (err && err.message) || String(err),
      stack: (err && err.stack) || "",
    });
  }
};

postMessage({ type: "ready" });
