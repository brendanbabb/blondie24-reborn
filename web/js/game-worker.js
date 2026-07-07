/*
 * Stateless self-play game worker — one of a small pool driven by the
 * main-thread coordinator in evolution.js.
 *
 * Holds NO evolution state. Each generation the coordinator broadcasts the
 * population's weight vectors; each "play" message then references players
 * by population index. One game at a time (pull dispatch: the coordinator
 * hands this worker its next game when the previous result arrives).
 *
 *   from coordinator:
 *     { type: "weights", epoch, gen, weights: Float32Array[] }
 *         cache Network wrappers for this generation's population
 *     { type: "play", epoch, gen, gameId, blackIdx, whiteIdx }
 *         play one game, reply with its result
 *
 *   from worker:
 *     { type: "ready" }
 *     { type: "result", epoch, gen, gameId, blackIdx, whiteIdx,
 *       winner, frames }             frames' buffers sent as transferables
 *     { type: "error", message, stack }
 */

importScripts("checkers.js?v=8", "network.js?v=8", "minimax.js?v=8");

const C = self.Checkers;
const N = self.Network;
const M = self.Minimax;

// Paper-faithful: flat depth 4 in self-play, matching the paper's training
// and game-play depth (see the old worker.js rationale — no endgame-depth
// heuristics, that would be a knowledge injection Fogel didn't make).
const TRAIN_SEARCH_DEPTH = 4;
const MAX_GAME_MOVES = 80;  // shorter than the user-facing cap to keep training decisive

// Wrap weights in Network wrappers ONCE per generation and reuse them across
// every game this worker plays that gen. Creating fresh wrappers per game
// generates new `forward` closures each time, which blows V8's inline cache
// at the minimax call site from polymorphic to megamorphic and deoptimizes
// the forward path (measured 2-3× slower in the old single-worker design).
let nets = null;
let netsEpoch = -1;
let netsGen = -1;

// Play one self-play game. Returns { winner, frames } — winner is +1
// (black), -1 (white), or 0 (draw); frames is an array of Int8Array(32)
// board snapshots for the replay panel.
function playGame(blackNet, whiteNet) {
  let board = C.makeBoard();
  const stateCounts = Object.create(null);
  const frames = [new Int8Array(board.squares)];

  while (board.moveCount < MAX_GAME_MOVES) {
    const key = C.stateKey(board);
    stateCounts[key] = (stateCounts[key] || 0) + 1;
    if (stateCounts[key] >= 3) return { winner: 0, frames };

    const net = board.currentPlayer === C.BLACK ? blackNet : whiteNet;
    // pickMove returns a null move iff the side to move has no legal moves
    // (= they lost) — no separate isGameOver pre-check needed.
    const { move } = M.pickMove(board, TRAIN_SEARCH_DEPTH, net);
    if (!move) return { winner: -board.currentPlayer, frames };
    board = C.applyMove(board, move);
    frames.push(new Int8Array(board.squares));
  }
  return { winner: 0, frames };
}

self.onmessage = function (ev) {
  const msg = ev.data || {};
  try {
    if (msg.type === "weights") {
      nets = msg.weights.map((w) => N.makeNetwork(w));
      netsEpoch = msg.epoch;
      netsGen = msg.gen;
      return;
    }
    if (msg.type === "play") {
      if (msg.epoch !== netsEpoch || msg.gen !== netsGen || !nets) {
        // Shouldn't happen (messages are FIFO per worker and the coordinator
        // only dispatches against the weights it just broadcast), but reply
        // anyway so the coordinator's slot accounting never wedges.
        postMessage({
          type: "result", epoch: msg.epoch, gen: msg.gen, gameId: msg.gameId,
          blackIdx: msg.blackIdx, whiteIdx: msg.whiteIdx,
          winner: 0, frames: null,
        });
        return;
      }
      const t0 = performance.now();
      const r = playGame(nets[msg.blackIdx], nets[msg.whiteIdx]);
      const gameMs = performance.now() - t0;
      const transfers = [];
      for (const f of r.frames) transfers.push(f.buffer);
      postMessage({
        type: "result", epoch: msg.epoch, gen: msg.gen, gameId: msg.gameId,
        blackIdx: msg.blackIdx, whiteIdx: msg.whiteIdx,
        winner: r.winner, frames: r.frames, gameMs: gameMs,
      }, transfers);
      return;
    }
  } catch (err) {
    postMessage({
      type: "error",
      message: (err && err.message) || String(err),
      stack: (err && err.stack) || "",
    });
  }
};

postMessage({ type: "ready" });
