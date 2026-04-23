// Verifies the JS Anaconda port matches Python's AnacondaNet on the same
// weights and the same fixed boards. Requires:
//   web/weights/anaconda.bin         (5048 little-endian float32)
//   web/weights/anaconda-fixtures.json
// Run with: node web/test_anaconda.js
const vm = require('vm');
const fs = require('fs');
const path = require('path');

const JS_DIR = path.join(__dirname, 'js');
const W_DIR  = path.join(__dirname, 'weights');

const ctx = {
  Math, Float32Array, Int8Array, Int16Array, Int32Array, Uint16Array, Uint8Array,
  Array, Object, Number, String, Symbol, Infinity, NaN, parseFloat, parseInt,
  console, Error, Promise,
};
ctx.self = ctx; ctx.global = ctx;
vm.createContext(ctx);

for (const f of ['checkers.js', 'anaconda-network.js', 'minimax.js']) {
  const src = fs.readFileSync(path.join(JS_DIR, f), 'utf8');
  vm.runInContext(src, ctx, { filename: f });
}

const C = ctx.Checkers;
const A = ctx.AnacondaNetwork;
const M = ctx.Minimax;

// Load weights.bin
const binPath = path.join(W_DIR, 'anaconda.bin');
const buf = fs.readFileSync(binPath);
const expected = A.N_WEIGHTS * 4;
if (buf.byteLength !== expected) {
  console.error(`FAIL: ${binPath} is ${buf.byteLength} bytes, expected ${expected}`);
  process.exit(1);
}
// Node's Buffer isn't guaranteed aligned to 4 bytes, so slice into a
// fresh ArrayBuffer before reinterpreting.
const ab = new ArrayBuffer(buf.byteLength);
new Uint8Array(ab).set(buf);
const weights = new Float32Array(ab);
console.log(`loaded ${weights.length} weights from ${path.relative(process.cwd(), binPath)}`);

const net = A.makeNetwork(weights);

// Load fixtures.json
const fixPath = path.join(W_DIR, 'anaconda-fixtures.json');
const fixData = JSON.parse(fs.readFileSync(fixPath, 'utf8'));
if (fixData.arch !== '2001') {
  console.error(`FAIL: fixtures arch=${fixData.arch}, expected 2001`);
  process.exit(1);
}
console.log(`running ${fixData.fixtures.length} fixtures...`);

let worst = 0;
let failed = 0;
// Anaconda JS port uses exact Math.tanh (not the Padé from the 1999 demo),
// so mismatch vs. Python should be float32-precision only. Budget: 1e-6
// (comfortably looser than the ~1e-8 observed, tight enough to catch any
// real layout or encoder regression).
const TOL = 1e-6;

for (const fx of fixData.fixtures) {
  // Reconstruct a board matching the fixture. JS Anaconda's encoder
  // uses (board.squares, board.currentPlayer), so we build that directly.
  const board = {
    squares: new Int8Array(fx.squares),
    currentPlayer: fx.currentPlayer,
    moveCount: 0,
  };
  const jsScore = net.forward(board);
  const diff = Math.abs(jsScore - fx.score);
  const ok = diff < TOL;
  worst = Math.max(worst, diff);
  if (!ok) failed++;
  const mark = ok ? 'ok ' : 'FAIL';
  console.log(
    `  [${mark}] ${fx.label.padEnd(40)}  ` +
    `py=${fx.score.toFixed(8)}  js=${jsScore.toFixed(8)}  |Δ|=${diff.toExponential(2)}`
  );
}

console.log(`worst |Δ| = ${worst.toExponential(3)}  (tolerance ${TOL})`);
if (failed > 0) {
  console.error(`${failed} fixture(s) failed.`);
  process.exit(1);
}
console.log('All fixtures passed.');

// End-to-end: make sure the Anaconda network is compatible with the JS
// minimax's network interface, and that a depth-4 pickMove from the
// starting position returns a legal move. This catches interface bugs
// (forward signature mismatches, weight-binding errors) that the
// fixture-only test can miss.
console.log('\nEnd-to-end minimax test...');
const startBoard = C.makeBoard();
const t0 = Date.now();
const result = M.pickMove(startBoard, 4, net);
const dt = Date.now() - t0;
if (!result.move) {
  console.error('FAIL: pickMove returned no move from the starting position');
  process.exit(1);
}
if (!Number.isFinite(result.score)) {
  console.error('FAIL: pickMove returned non-finite score');
  process.exit(1);
}
const legal = C.getLegalMoves(startBoard);
const isLegal = legal.some(m => {
  if (m.length !== result.move.length) return false;
  for (let i = 0; i < m.length; i++) if (m[i] !== result.move[i]) return false;
  return true;
});
if (!isLegal) {
  console.error('FAIL: pickMove returned an illegal move:', result.move);
  process.exit(1);
}
console.log(
  `  pickMove(d=4) from start -> move ${result.move[0]+1} → ${result.move[result.move.length-1]+1}, ` +
  `score=${result.score.toFixed(5)} (${dt}ms)`
);
console.log('\nAll tests passed.');
