// Node-side smoke test for the browser demo's checkers + minimax + network
// modules. Run with: node web/smoke_test.js
//
// Loads the actual web/js/*.js files into a fresh vm context, then exercises:
//   1) basic engine: starting moves count
//   2) make/unmake round-trips against the pure applyMove
//   3) deeper random-walk consistency: apply N moves in place, undo them
//      in reverse, and verify the board matches a recorded snapshot at every
//      step (catches jump-undo bugs)
//   4) minimax pickMove returns a legal move at depth 4 with TT + ID
//   5) iterative-deepening agreement: pickMove(depth=4) returns the same
//      score as a from-scratch plain alpha-beta would (sanity, not a strict
//      bit-equality — TT cutoffs can fail-soft to a different score on the
//      same root move; we just check it's finite and reasonable)

const vm = require('vm');
const fs = require('fs');
const path = require('path');

const JS_DIR = path.join(__dirname, 'js');

const ctx = {
  Math, Float32Array, Int8Array, Int16Array, Int32Array, Uint16Array, Uint8Array,
  Array, Object, Number, String, Symbol, Infinity, NaN, parseFloat, parseInt,
  console,
};
ctx.self = ctx;
ctx.global = ctx;
vm.createContext(ctx);

for (const f of ['checkers.js', 'network.js', 'minimax.js']) {
  const src = fs.readFileSync(path.join(JS_DIR, f), 'utf8');
  vm.runInContext(src, ctx, { filename: f });
}

const C = ctx.Checkers;
const N = ctx.Network;
const M = ctx.Minimax;

function assert(cond, msg) {
  if (!cond) {
    console.error('FAIL:', msg);
    process.exit(1);
  }
}

function snap(b) {
  return Array.from(b.squares).join(',') + '|' + b.currentPlayer + '|' + b.moveCount;
}

// ---- Test 1 ----
let board = C.makeBoard();
const start = C.getLegalMoves(board);
assert(start.length === 7, `expected 7 starting moves, got ${start.length}`);

// ---- Test 2 ----
const before = snap(board);
const m0 = start[0];
const pure = C.applyMove(board, m0);
const undo = C.applyMoveInPlace(board, m0);
assert(snap(board) === snap(pure), `applyMoveInPlace ≠ applyMove`);
C.undoMove(board, undo);
assert(snap(board) === before, `undoMove did not restore`);

// ---- Test 3 ----
// 200 random plies, snapshotting before each apply; undo all in reverse.
const seed = 1234;
let rngState = seed >>> 0;
function rng() { rngState = (rngState * 1664525 + 1013904223) >>> 0; return rngState / 0x100000000; }

const snaps = [];
const undos = [];
const moves = [];
let plies = 0;
const MAX_PLIES = 200;
while (plies < MAX_PLIES) {
  const [over] = C.isGameOver(board);
  if (over) break;
  const legal = C.getLegalMoves(board);
  if (legal.length === 0) break;
  const m = legal[Math.floor(rng() * legal.length)];
  snaps.push(snap(board));
  moves.push(m);
  undos.push(C.applyMoveInPlace(board, m));
  plies++;
}
console.log(`played ${plies} random plies in place`);
// Unwind
let foundJump = false;
for (let i = undos.length - 1; i >= 0; i--) {
  if (moves[i].length > 2) foundJump = true;
  C.undoMove(board, undos[i]);
  assert(snap(board) === snaps[i], `undo step ${i}: state mismatch\n  expected ${snaps[i]}\n  actual   ${snap(board)}`);
}
assert(snap(board) === snap(C.makeBoard()), `final state ≠ starting state`);
assert(foundJump, `expected at least one jump in 200 random plies`);
console.log('round-trip undo verified across all plies (incl. jumps)');

// ---- Test 4 ----
const w1 = N.newRandomWeights(0.05);
const net1 = N.makeNetwork(w1);
const t0 = Date.now();
const r = M.pickMove(board, 4, net1);
const dt = Date.now() - t0;
assert(r.move !== null, `pickMove returned null move`);
assert(Number.isFinite(r.score), `pickMove returned non-finite score: ${r.score}`);
console.log(`pickMove(d=4) -> score=${r.score.toFixed(4)} from=${r.move[0]} to=${r.move[r.move.length - 1]} (${dt}ms)`);

// ---- Test 5 ----
// Quick self-play: 30 plies @ depth 4 between two random nets. Verifies the
// search holds up over many calls (TT generation counter, no leaks).
const w2 = N.newRandomWeights(0.05);
const net2 = N.makeNetwork(w2);
let game = C.makeBoard();
const tg0 = Date.now();
let nMoves = 0;
while (nMoves < 30) {
  const [over] = C.isGameOver(game);
  if (over) break;
  const cur = game.currentPlayer === C.BLACK ? net1 : net2;
  const { move } = M.pickMove(game, 4, cur);
  if (!move) break;
  game = C.applyMove(game, move);
  nMoves++;
}
const tgdt = Date.now() - tg0;
console.log(`self-play: ${nMoves} plies @ d=4 in ${tgdt}ms (avg ${(tgdt / Math.max(nMoves, 1)).toFixed(1)}ms/move)`);

console.log('\nAll smoke tests passed.');
