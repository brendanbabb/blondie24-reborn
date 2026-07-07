// Node-side test for the main-thread evolution coordinator + game-worker
// pool. Run with: node web/pool_test.js
//
// Loads the actual web/js files. Game workers are simulated with a fake
// Worker class that runs game-worker.js in its own vm context and delivers
// messages asynchronously (setImmediate), preserving the real FIFO
// per-worker ordering. Exercises:
//   1) reset → async gen-0 event with a zeroed leaderboard
//   2) resume → generations complete; each gen's leaderboard accounts for
//      exactly 18 games (36 win/loss/draw participations), sample games
//      carry frames, and gens are strictly increasing
//   3) pause → at most one more gen completes (the in-flight one), then none
//   4) snapshot → resolves at the gen boundary with 1743-weight champion
//   5) reset mid-generation → stale results dropped, evolution restarts
//      cleanly from gen 0 with no wedged workers or double-counted fitness

const vm = require('vm');
const fs = require('fs');
const path = require('path');

const JS_DIR = path.join(__dirname, 'js');

function baseGlobals() {
  return {
    Math, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array,
    Uint16Array, Uint8Array, Array, Object, Number, String, Symbol, Promise,
    Infinity, NaN, parseFloat, parseInt, console,
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

// ---- Fake Worker running game-worker.js in an isolated vm context ---------

class FakeWorker {
  constructor(_url) {
    this.onmessage = null;
    this.onerror = null;
    const wctx = baseGlobals();
    wctx.self = wctx;
    wctx.global = wctx;
    // importScripts: strip the ?v= cache-buster and load from JS_DIR.
    wctx.importScripts = (...names) => {
      loadInto(wctx, names.map((n) => n.split('?')[0]));
    };
    // worker → main (async, like the real thing)
    wctx.postMessage = (msg, _transfers) => {
      setImmediate(() => { if (this.onmessage) this.onmessage({ data: msg }); });
    };
    vm.createContext(wctx);
    this._ctx = wctx;
    loadInto(wctx, ['game-worker.js']);
  }
  // main → worker (async)
  postMessage(msg, _transfers) {
    setImmediate(() => {
      if (this._ctx.onmessage) this._ctx.onmessage({ data: msg });
    });
  }
}

// ---- Main-thread context ---------------------------------------------------

const ctx = baseGlobals();
ctx.self = ctx;
ctx.global = ctx;
vm.createContext(ctx);
loadInto(ctx, ['network.js', 'evolution.js']);

function assert(cond, msg) {
  if (!cond) {
    console.error('FAIL:', msg);
    process.exit(1);
  }
}

const genEvents = [];
let errors = [];
let reentrantHook = null;  // set by test 6 to re-enter evo from inside onGen
const evo = ctx.Evolution.create({
  WorkerCtor: FakeWorker,
  poolSize: 4,
  onGen: (msg) => {
    genEvents.push(msg);
    if (reentrantHook) reentrantHook(msg);
  },
  onError: (m) => errors.push(m),
});
assert(evo.poolSize === 4, `poolSize override: expected 4, got ${evo.poolSize}`);

function waitFor(pred, timeoutMs, label) {
  return new Promise((resolve, reject) => {
    const t0 = Date.now();
    (function poll() {
      if (pred()) return resolve();
      if (Date.now() - t0 > timeoutMs) return reject(new Error('timeout: ' + label));
      setTimeout(poll, 5);
    })();
  });
}

function checkGenEvent(e) {
  assert(e.gen > 0, 'gen event has positive gen');
  assert(Array.isArray(e.leaderboard) && e.leaderboard.length === 6,
    `leaderboard has 6 entries, got ${e.leaderboard && e.leaderboard.length}`);
  // 18 games × 2 participants = 36 win/loss/draw slots, wins === losses.
  let w = 0, l = 0, d = 0;
  for (const row of e.leaderboard) { w += row.wins; l += row.losses; d += row.draws; }
  assert(w + l + d === 36, `gen ${e.gen}: W+L+D = ${w + l + d}, expected 36`);
  assert(w === l, `gen ${e.gen}: wins ${w} != losses ${l}`);
  assert(e.sampleGameA && e.sampleGameA.frames && e.sampleGameA.frames.length > 1,
    `gen ${e.gen}: sampleGameA missing frames`);
  assert(typeof e.genMs === 'number' && e.genMs >= 0, `gen ${e.gen}: bad genMs`);
}

(async () => {
  // ---- 1) reset → gen-0 event ----
  evo.reset();
  await waitFor(() => genEvents.length >= 1, 2000, 'gen-0 event after reset');
  assert(genEvents[0].gen === 0, `first event is gen 0, got ${genEvents[0].gen}`);
  assert(genEvents[0].leaderboard.every((r) => r.fitness === 0 && r.wins === 0),
    'gen-0 leaderboard is zeroed');
  genEvents.length = 0;

  // ---- 2) resume → gens complete with sane accounting ----
  evo.resume();
  await waitFor(() => genEvents.length >= 3, 60000, '3 generations');
  for (const e of genEvents) checkGenEvent(e);
  for (let i = 1; i < genEvents.length; i++) {
    assert(genEvents[i].gen === genEvents[i - 1].gen + 1,
      `gens strictly increasing: ${genEvents[i - 1].gen} → ${genEvents[i].gen}`);
  }
  console.log(`ran ${genEvents.length} gens; per-gen W/L/D accounting OK ` +
    `(genMs ≈ ${genEvents.map((e) => e.genMs.toFixed(0)).join(', ')} ms)`);

  // ---- 3+4) pause, snapshot at the boundary, then verify quiescence ----
  evo.pause();
  const snap = await evo.snapshot();
  assert(snap.weights instanceof ctx.Float32Array && snap.weights.length === 1743,
    `snapshot weights: expected Float32Array(1743)`);
  assert(snap.sigmas.length === 1743, 'snapshot sigmas length 1743');
  assert(snap.gen === genEvents[genEvents.length - 1].gen,
    `snapshot gen ${snap.gen} matches last completed gen ${genEvents[genEvents.length - 1].gen}`);
  const gensAtPause = genEvents.length;
  await new Promise((r) => setTimeout(r, 600));
  assert(genEvents.length === gensAtPause,
    `no gens after pause+boundary (had ${gensAtPause}, now ${genEvents.length})`);
  console.log(`pause honored at gen ${snap.gen}; snapshot returned champion weights`);

  // ---- 5) reset mid-generation → clean restart, stale results dropped ----
  evo.resume();
  await new Promise((r) => setTimeout(r, 30));  // let games get in flight
  genEvents.length = 0;
  evo.reset();                                   // mid-gen: epoch bump
  await waitFor(() => genEvents.length >= 1, 2000, 'gen-0 event after mid-gen reset');
  assert(genEvents[0].gen === 0, 'post-reset event is gen 0');
  genEvents.length = 0;
  evo.resume();
  await waitFor(() => genEvents.length >= 2, 60000, '2 gens after mid-gen reset');
  assert(genEvents[0].gen === 1, `first gen after reset is 1, got ${genEvents[0].gen}`);
  for (const e of genEvents) checkGenEvent(e);   // stale results would break the W+L+D=36 invariant
  evo.pause();
  await evo.snapshot();
  console.log('mid-gen reset: stale results dropped, restarted cleanly from gen 0');

  // ---- 6) re-entrant pause+resume inside onGen (the warmup flow) ----
  // main.js's warmup handler calls evo.pause() then evo.resume()
  // synchronously from INSIDE the onGen callback, which finishGen invokes
  // mid-function. This double-started the generation (pending reset with
  // games in flight → wedged pool) before the genInFlight guard existed.
  genEvents.length = 0;
  let reentered = false;
  reentrantHook = (msg) => {
    if (!reentered && msg.gen >= 1) {
      reentered = true;
      evo.pause();
      evo.resume();  // synchronous re-entry, like maybeStartAiTurn()
    }
  };
  evo.reset();
  await waitFor(() => genEvents.length >= 1, 2000, 'gen-0 after reset (reentrancy phase)');
  genEvents.length = 0;
  evo.resume();
  await waitFor(() => genEvents.length >= 3, 60000, '3 gens despite re-entrant pause+resume');
  for (const e of genEvents) checkGenEvent(e);  // double-start would double-count W/L/D
  reentrantHook = null;
  evo.pause();
  await evo.snapshot();
  console.log('re-entrant pause+resume inside onGen: no double-start, accounting intact');

  assert(errors.length === 0, `worker errors: ${errors.join('; ')}`);
  console.log('\nAll pool tests passed.');
  process.exit(0);
})().catch((err) => {
  console.error('FAIL:', err.message);
  process.exit(1);
});
