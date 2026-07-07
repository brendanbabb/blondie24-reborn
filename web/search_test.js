// Node-side test for search-worker.js + search-client.js.
// Run with: node web/search_test.js
//
// Uses the same fake-Worker shim as pool_test.js to run the real
// search-worker.js in an isolated vm context. Exercises:
//   1) worker protocol: weights → search → legal move + depthReached
//   2) budgeted search through the client (floor honored, maxDepth capped)
//   3) request matching across interleaved searches on two slots
//   4) error path: searching an unloaded slot rejects
//   5) fallback: WorkerCtor that throws → main-thread search, same shape

const vm = require('vm');
const fs = require('fs');
const path = require('path');

const JS_DIR = path.join(__dirname, 'js');

function baseGlobals() {
  return {
    Math, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array,
    Uint16Array, Uint8Array, Array, Object, Number, String, Symbol, Promise,
    Map, Set, Infinity, NaN, parseFloat, parseInt, console, Error,
    setTimeout, clearTimeout,
    performance: { now: () => Number(process.hrtime.bigint()) / 1e6 },
  };
}

function loadInto(ctx, files) {
  for (const f of files) {
    const src = fs.readFileSync(path.join(JS_DIR, f), 'utf8');
    vm.runInContext(src, ctx, { filename: f });
  }
}

class FakeWorker {
  constructor(_url) {
    this.onmessage = null;
    this.onerror = null;
    const wctx = baseGlobals();
    wctx.self = wctx;
    wctx.global = wctx;
    wctx.importScripts = (...names) => loadInto(wctx, names.map((n) => n.split('?')[0]));
    wctx.postMessage = (msg) => {
      setImmediate(() => { if (this.onmessage) this.onmessage({ data: msg }); });
    };
    vm.createContext(wctx);
    this._ctx = wctx;
    loadInto(wctx, ['search-worker.js']);
  }
  postMessage(msg) {
    setImmediate(() => { if (this._ctx.onmessage) this._ctx.onmessage({ data: msg }); });
  }
  terminate() {}
}

class ThrowingWorker {
  constructor() { throw new Error('no workers here'); }
}

const ctx = baseGlobals();
ctx.self = ctx;
ctx.global = ctx;
vm.createContext(ctx);
loadInto(ctx, ['checkers.js', 'anaconda-network.js', 'minimax.js', 'search-client.js']);

const C = ctx.Checkers;

const buf = fs.readFileSync(path.join(__dirname, 'weights', 'anaconda-paper-strict.bin'));
const weights = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
const buf2 = fs.readFileSync(path.join(__dirname, 'weights', 'anaconda-enhanced.bin'));
const weights2 = new Float32Array(buf2.buffer, buf2.byteOffset, buf2.byteLength / 4);

function assert(cond, msg) {
  if (!cond) {
    console.error('FAIL:', msg);
    process.exit(1);
  }
}

function assertLegal(board, move, label) {
  assert(move, `${label}: null move`);
  const legal = C.getLegalMoves(board);
  assert(legal.some((m) => m.length === move.length && m.every((v, i) => v === move[i])),
    `${label}: illegal move ${JSON.stringify(move)}`);
}

(async () => {
  // ---- 1+2) worker-backed client, plain + budgeted search ----
  const sc = ctx.SearchClient.create({ WorkerCtor: FakeWorker });
  sc.setWeights('strict', weights);
  sc.setWeights('enhanced', weights2);
  assert(!sc.isFallback(), 'client should not be in fallback mode');

  const board = C.makeBoard();
  const r1 = await sc.search(board, 'strict', 4, null);
  assertLegal(board, r1.move, 'plain d4');
  assert(r1.depthReached === 4, `plain search depthReached ${r1.depthReached}, want 4`);
  assert(Number.isFinite(r1.score), 'plain search score not finite');

  const r2 = await sc.search(board, 'strict', 4, { budgetMs: 150, maxDepth: 10 });
  assertLegal(board, r2.move, 'budgeted d4+');
  assert(r2.depthReached >= 4 && r2.depthReached <= 10,
    `budgeted depthReached ${r2.depthReached} out of [4,10]`);
  console.log(`worker search OK: plain d4, budgeted reached d${r2.depthReached}`);

  // ---- 3) interleaved requests on two slots resolve to their own results ----
  const [a, b] = await Promise.all([
    sc.search(board, 'strict', 3, null),
    sc.search(board, 'enhanced', 3, null),
  ]);
  assertLegal(board, a.move, 'interleaved strict');
  assertLegal(board, b.move, 'interleaved enhanced');
  console.log('interleaved two-slot searches OK');

  // ---- 4) unloaded slot rejects ----
  let rejected = false;
  try { await sc.search(board, 'nope', 3, null); }
  catch (e) { rejected = /no weights/.test(e.message); }
  assert(rejected, 'search on unloaded slot should reject with "no weights"');
  console.log('unloaded-slot error path OK');

  // ---- 5) fallback client (Worker construction throws) ----
  const fb = ctx.SearchClient.create({ WorkerCtor: ThrowingWorker });
  assert(fb.isFallback(), 'client should be in fallback mode');
  fb.setWeights('strict', weights);
  const r3 = await fb.search(board, 'strict', 4, { budgetMs: 100, maxDepth: 8 });
  assertLegal(board, r3.move, 'fallback d4+');
  assert(r3.depthReached >= 4, `fallback depthReached ${r3.depthReached}`);
  console.log(`fallback (main-thread) search OK: reached d${r3.depthReached}`);

  console.log('\nAll search tests passed.');
  process.exit(0);
})().catch((err) => {
  console.error('FAIL:', err.message);
  process.exit(1);
});
