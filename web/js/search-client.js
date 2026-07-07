/*
 * SearchClient — main-thread wrapper around search-worker.js with a
 * synchronous fallback.
 *
 * Usage:
 *   const sc = SearchClient.create({});
 *   sc.setWeights("paper-strict", float32Weights);
 *   const r = await sc.search(board, "paper-strict", 6, {budgetMs, maxDepth});
 *   // r has the same shape as Minimax.pickMove's return value
 *
 * If Workers are unavailable (file:// pages, construction failure) or the
 * worker crashes, the client falls back to running Minimax.pickMove
 * synchronously on the main thread — same results, just with the old
 * UI-freezing behavior — so the pages degrade instead of breaking. Weights
 * are retained locally for that path.
 */

(function (global) {
  "use strict";

  function create(opts) {
    opts = opts || {};
    const WorkerCtor = opts.WorkerCtor || global.Worker;
    const workerUrl = opts.workerUrl || "js/search-worker.js?v=1";

    let worker = null;
    let failed = false;
    let nextReqId = 1;
    const pending = new Map();               // reqId → { resolve, reject, args }
    const localWeights = Object.create(null); // slotId → Float32Array
    const localNets = Object.create(null);    // slotId → Network (built lazily on fallback)

    function runLocal(args) {
      const A = global.AnacondaNetwork;
      const M = global.Minimax;
      let net = localNets[args.slotId];
      if (!net) {
        const w = localWeights[args.slotId];
        if (!w) throw new Error(`no weights loaded for slot "${args.slotId}"`);
        net = localNets[args.slotId] = A.makeNetwork(w);
      }
      return M.pickMove(args.board, args.depth, net, args.searchOpts);
    }

    // Switch to main-thread search and settle everything in flight with it.
    function activateFallback(reason) {
      if (failed) return;
      failed = true;
      console.warn("SearchClient: worker unavailable, searching on the main thread —", reason);
      if (worker) { try { worker.terminate(); } catch (e) {} worker = null; }
      const inFlight = Array.from(pending.values());
      pending.clear();
      for (const p of inFlight) {
        try { p.resolve(runLocal(p.args)); }
        catch (err) { p.reject(err); }
      }
    }

    try {
      worker = new WorkerCtor(workerUrl);
      worker.onmessage = (ev) => {
        const msg = ev.data || {};
        if (msg.type === "ready") return;
        const p = pending.get(msg.reqId);
        if (!p) return;  // stale/unknown (e.g. settled by a fallback switch)
        pending.delete(msg.reqId);
        if (msg.type === "result") p.resolve(msg);
        else p.reject(new Error(msg.message || "search worker error"));
      };
      // Fires on script-load failure or an uncaught crash — everything the
      // per-request error messages don't cover.
      worker.onerror = (ev) => {
        activateFallback((ev && ev.message) || "worker error");
      };
    } catch (err) {
      activateFallback(err.message || String(err));
    }

    function setWeights(slotId, weights) {
      localWeights[slotId] = weights;
      delete localNets[slotId];
      if (!failed && worker) {
        worker.postMessage({ type: "weights", slotId, weights });
      }
    }

    // board is snapshotted (squares copied) at call time, so callers may
    // advance their own board immediately.
    function search(board, slotId, depth, searchOpts) {
      const args = {
        board: {
          squares: new Int8Array(board.squares),
          currentPlayer: board.currentPlayer,
          moveCount: board.moveCount,
        },
        slotId, depth, searchOpts,
      };
      if (failed) {
        try { return Promise.resolve(runLocal(args)); }
        catch (err) { return Promise.reject(err); }
      }
      const reqId = nextReqId++;
      return new Promise((resolve, reject) => {
        pending.set(reqId, { resolve, reject, args });
        worker.postMessage({
          type: "search", reqId, slotId,
          squares: args.board.squares,
          currentPlayer: args.board.currentPlayer,
          moveCount: args.board.moveCount,
          depth: depth,
          budgetMs: (searchOpts && searchOpts.budgetMs) || 0,
          maxDepth: (searchOpts && searchOpts.maxDepth) || depth,
        });
      });
    }

    return {
      setWeights,
      search,
      isFallback: () => failed,
    };
  }

  global.SearchClient = { create };
})(typeof self !== "undefined" ? self : this);
