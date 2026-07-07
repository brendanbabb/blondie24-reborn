/*
 * Main-thread evolution coordinator + pull-dispatch game-worker pool.
 *
 * Replaces the old single evolution worker (worker.js). The EP loop itself
 * (pairings, fitness accounting, selection, mutation) runs here on the main
 * thread — it's microseconds per generation — while the expensive part, the
 * 18 self-play games per generation, fans out to a small pool of stateless
 * game workers (game-worker.js).
 *
 * Pull dispatch: each worker holds exactly one game at a time and is handed
 * the next from the queue the moment it reports a result. Game lengths vary
 * ~5× (15-move blunder-fests vs 80-move shuffle-draws), so a static
 * games-per-worker split would idle the fast workers behind the slowest —
 * with pull dispatch the end-of-gen straggler wait is at most one game.
 *
 * Barrier-cost experiments (2026-07, 24-thread hybrid-core box, 8 workers) —
 * measured via the per-gen genStats diagnostics below:
 *   - CAVEAT discovered later: these experiments ran with the browser window
 *     UNFOCUSED, where Windows 11 EcoQoS demotes worker threads to E-cores.
 *     Focused (how people actually play), the same build runs ~51 ms/gen
 *     (~18.7 gens/sec) — near-ideal parallel efficiency. Benchmark workers
 *     with the window foreground.
 *   - In-worker game time sums to ~1,050 ms/gen but genMs is ~170 ms
 *     (unfocused); the gap to the ideal makespan max(sum/8, longest game)
 *     is only ~35 ms (~20%). The rest of the unfocused "loss" was the
 *     EcoQoS effect above, not scheduling.
 *   - The ~35 ms residual is TAIL PACKING: in the last round each worker
 *     holds 2-3 variable-length games and early finishers idle.
 *   - Double-buffering (keep 2 games queued per worker while the queue is
 *     deep) measured neutral on an idle main thread but is kept: it removes
 *     the dispatch round-trip, which matters when the main thread is busy
 *     rendering during real play.
 *   - Hedged tail dispatch (race a DUPLICATE of a still-running game on an
 *     idle worker, first result wins, losers dropped by gameId/gen tag)
 *     also measured neutral: a duplicate must replay from move 0 (~55 ms)
 *     to beat a straggler that's typically ~30 ms from done, so it only
 *     pays against extreme slow-core outliers. Off by default; enable with
 *     create({hedge: true}).
 *
 * Generations are a barrier: gen N+1's population depends on gen N's full
 * ranking, so a new gen starts only after all 18 results are in. reset()
 * during a gen bumps an epoch counter; results tagged with a stale epoch or
 * gen are dropped on arrival (workers are stateless, nothing to unwind).
 *
 * API (plain main-thread calls — no postMessage protocol for callers):
 *   const evo = Evolution.create({ onGen, onError });
 *   evo.reset()     back to gen 0; emits a zeroed gen-0 event (async)
 *   evo.resume()    run generations back-to-back until pause()
 *   evo.pause()     stop AFTER the in-flight generation completes
 *   evo.snapshot()  Promise of { gen, weights, sigmas, fitness } for the
 *                   current champion; resolves at the next gen boundary so
 *                   it reflects every game played this burst
 *
 * onGen fires once per completed generation with the payload shape the old
 * worker sent: { gen, leaderboard, meanFitness, maxFitness, sampleGameA,
 * sampleGameB, genMs }. genMs is wall-clock per gen (parallel time, not CPU
 * time). Samples ride along every gen — they're same-thread references now,
 * the old every-3-gens transfer throttle is obsolete.
 */

(function (global) {
  "use strict";

  const N = global.Network;

  // Demo-tuned EP hyperparameters — smaller than paper for browser
  // responsiveness. (Search depth and the game-move cap live in
  // game-worker.js, next to the game loop they govern.)
  const POP_SIZE = 6;
  const GAMES_PER_INDIVIDUAL = 3;   // paper uses 5; 3 gives reasonable ranking at pop=6

  const WIN_SCORE = 1.0;
  const DRAW_SCORE = 0.0;
  const LOSS_SCORE = -2.0;

  // 4–8 workers: no benefit past ~9 (only 18 games/gen to hand out), and we
  // leave headroom for the UI thread and OS. hardwareConcurrency overcounts
  // useful cores (hyperthreads, E-cores), hence the -2.
  const POOL_MIN = 4;
  const POOL_MAX = 8;

  // Max concurrent copies of one game under hedged dispatch (original + 1
  // duplicate). More copies burn workers on the same race for shrinking
  // returns.
  const HEDGE_MAX_COPIES = 2;

  function now() {
    return (typeof performance !== "undefined") ? performance.now() : Date.now();
  }

  function create(opts) {
    opts = opts || {};
    const onGen = opts.onGen || function () {};
    const onError = opts.onError || function () {};
    const WorkerCtor = opts.WorkerCtor || global.Worker;
    const workerUrl = opts.workerUrl || "js/game-worker.js?v=10";
    const hc = (global.navigator && global.navigator.hardwareConcurrency) || POOL_MAX;
    const poolSize = opts.poolSize ||
      Math.max(POOL_MIN, Math.min(POOL_MAX, hc - 2));
    const hedge = opts.hedge === true;  // hedged tail dispatch, off by default (measured neutral)

    // ---- Pool ---------------------------------------------------------

    const workers = [];  // { w, outstanding: games sent, results not yet back }
    for (let i = 0; i < poolSize; i++) {
      const w = new WorkerCtor(workerUrl);
      const slot = { w, outstanding: 0 };
      w.onmessage = (ev) => onWorkerMessage(slot, ev.data || {});
      w.onerror = (ev) => {
        onError("game worker crashed: " + ((ev && ev.message) || "unknown"));
      };
      workers.push(slot);
    }

    // ---- Population state ----------------------------------------------

    let population = null;  // { weights, sigmas, fitness, wins, losses, draws }
    let generation = 0;
    let running = false;
    let epoch = 0;          // bumped on reset; stale results are dropped by tag

    // Per-generation flight state.
    let genInFlight = false;
    let specs = [];         // all game specs this gen, by gameId
    let queue = [];         // specs awaiting first dispatch
    let done = [];          // by gameId — first result in?
    let doneCount = 0;
    let inFlight = [];      // by gameId — concurrent copies out (for hedging)
    let recordings = [];    // by gameId — all 18 present at gen end
    let genStartMs = 0;
    // Diagnostics for the gen barrier: total in-worker game CPU, the longest
    // single game, and how many hedge duplicates were dispatched.
    let genGameMsSum = 0;
    let genGameMsMax = 0;
    let genHedges = 0;

    let lastTournamentSnapshot = null;
    let snapshotWaiters = [];  // { resolve, reject } — settled at gen boundary

    function makeIndividual(weights, sigmas) {
      // No Network wrapper here — wrappers live in the game workers, built
      // once per generation from the broadcast weights.
      return { weights, sigmas, fitness: 0, wins: 0, losses: 0, draws: 0 };
    }

    function initPopulation() {
      population = [];
      for (let i = 0; i < POP_SIZE; i++) {
        // Paper-faithful σ=0.05 for both weights and sigma-state.
        population.push(makeIndividual(
          N.newRandomWeights(0.05),
          N.newSigmas(0.05),
        ));
      }
      generation = 0;
    }

    function resetFitness() {
      for (const ind of population) {
        ind.fitness = 0;
        ind.wins = 0;
        ind.losses = 0;
        ind.draws = 0;
      }
    }

    function sampleWithoutReplacement(n, k, excluded) {
      const pool = [];
      for (let i = 0; i < n; i++) if (i !== excluded) pool.push(i);
      const picked = [];
      for (let i = 0; i < k && pool.length > 0; i++) {
        const idx = Math.floor(Math.random() * pool.length);
        picked.push(pool[idx]);
        pool[idx] = pool[pool.length - 1];
        pool.pop();
      }
      return picked;
    }

    // Random pairings: each individual plays GAMES_PER_INDIVIDUAL games
    // against distinct randomly chosen opponents, random color each game.
    function buildPairings() {
      const specs = [];
      for (let i = 0; i < POP_SIZE; i++) {
        const opps = sampleWithoutReplacement(POP_SIZE, GAMES_PER_INDIVIDUAL, i);
        for (const opp of opps) {
          const aIsBlack = Math.random() < 0.5;
          specs.push({
            gameId: specs.length,
            blackIdx: aIsBlack ? i : opp,
            whiteIdx: aIsBlack ? opp : i,
          });
        }
      }
      return specs;
    }

    // ---- Generation lifecycle -------------------------------------------

    function startGen() {
      // Re-entrancy guard: onGen handlers may call resume() synchronously
      // from inside finishGen (the warmup flow does exactly that), which
      // starts the next gen before finishGen's own trailing startGen runs.
      // Starting the same gen twice would reset the flight state with games
      // still in flight and wedge the pool.
      if (genInFlight) return;
      genInFlight = true;
      genStartMs = now();
      resetFitness();
      recordings = [];
      specs = buildPairings();
      queue = specs.slice();
      done = new Array(specs.length).fill(false);
      doneCount = 0;
      inFlight = new Array(specs.length).fill(0);
      genGameMsSum = 0;
      genGameMsMax = 0;
      genHedges = 0;

      // Broadcast this generation's weights, then hand every idle worker its
      // first game. postMessage is FIFO per worker, so plays can never
      // overtake the weights they reference.
      const weights = population.map((ind) => ind.weights);
      for (const slot of workers) {
        slot.w.postMessage({ type: "weights", epoch, gen: generation, weights });
      }
      for (const slot of workers) dispatchTo(slot);
    }

    // Tail hedging: pick a still-running game with the fewest copies out.
    function pickHedge() {
      let best = null;
      for (const spec of specs) {
        const id = spec.gameId;
        if (done[id] || inFlight[id] >= HEDGE_MAX_COPIES) continue;
        if (best === null || inFlight[id] < inFlight[best.gameId]) best = spec;
      }
      return best;
    }

    // Double-buffering: keep up to 2 games queued per worker while the game
    // queue is deep — the worker starts its next game straight from its own
    // message queue instead of idling a main-thread dispatch round-trip
    // (~2 ms × 18 games/gen, measured). Near the tail (queue shorter than
    // the pool) drop to 1 outstanding so the last games stay stealable by
    // whichever worker frees up first instead of committed to a busy one.
    function maxOutstanding() {
      return queue.length > workers.length ? 2 : 1;
    }

    function dispatchTo(slot) {
      while (slot.outstanding < maxOutstanding()) {
        let spec = queue.shift() || null;
        if (spec === null && hedge && genInFlight && slot.outstanding === 0) {
          spec = pickHedge();
          if (spec !== null) genHedges++;
        }
        if (spec === null) return;
        slot.outstanding++;
        inFlight[spec.gameId]++;
        slot.w.postMessage({
          type: "play", epoch, gen: generation,
          gameId: spec.gameId, blackIdx: spec.blackIdx, whiteIdx: spec.whiteIdx,
        });
      }
    }

    function onWorkerMessage(slot, msg) {
      if (msg.type === "ready") return;
      if (msg.type === "error") {
        onError(msg.message);
        return;
      }
      if (msg.type !== "result") return;

      if (slot.outstanding > 0) slot.outstanding--;

      // Stale result from before a reset (or a hedge loser that ran past its
      // gen's end): drop it, but the slot is free now — pull from the CURRENT
      // gen so it isn't stranded.
      if (msg.epoch !== epoch || msg.gen !== generation || !genInFlight) {
        dispatchTo(slot);
        return;
      }

      inFlight[msg.gameId]--;

      // Hedge race already decided by another copy — drop the duplicate.
      if (done[msg.gameId]) {
        dispatchTo(slot);
        return;
      }

      done[msg.gameId] = true;
      doneCount++;
      if (typeof msg.gameMs === "number") {
        genGameMsSum += msg.gameMs;
        if (msg.gameMs > genGameMsMax) genGameMsMax = msg.gameMs;
      }
      applyResult(msg);

      if (doneCount === specs.length) {
        finishGen();
      } else {
        dispatchTo(slot);
      }
    }

    function applyResult(msg) {
      const winner = msg.winner;
      recordings[msg.gameId] = {
        frames: msg.frames,
        blackIdx: msg.blackIdx,
        whiteIdx: msg.whiteIdx,
        winner: winner,
      };

      const black = population[msg.blackIdx];
      const white = population[msg.whiteIdx];
      if (winner === 0) {
        black.fitness += DRAW_SCORE; black.draws++;
        white.fitness += DRAW_SCORE; white.draws++;
      } else if (winner === 1) {  // black won
        black.fitness += WIN_SCORE;  black.wins++;
        white.fitness += LOSS_SCORE; white.losses++;
      } else {
        white.fitness += WIN_SCORE;  white.wins++;
        black.fitness += LOSS_SCORE; black.losses++;
      }
    }

    // Rank → pick replay samples → capture standings → select + mutate.
    // Straight port of the old worker's runOneGen back half.
    function finishGen() {
      genInFlight = false;

      // Each network's rank in this tournament (1 = best).
      const rankOf = new Array(POP_SIZE);
      const scored = population.map((ind, i) => ({ idx: i, fit: ind.fitness }));
      scored.sort((a, b) => b.fit - a.fit);
      for (let r = 0; r < scored.length; r++) rankOf[scored[r].idx] = r + 1;
      for (const rec of recordings) {
        rec.blackRank = rankOf[rec.blackIdx];
        rec.whiteRank = rankOf[rec.whiteIdx];
      }

      // Pick 2 games for the replay panel: prefer decisive games, then wide
      // rank gaps ("strong vs weak" is more instructive), lightly shuffled.
      const decisive = recordings.filter((r) => r.winner !== 0);
      const draws = recordings.filter((r) => r.winner === 0);
      const pool = (decisive.length >= 2) ? decisive : decisive.concat(draws);
      pool.sort((a, b) => {
        const gapA = Math.abs(a.blackRank - a.whiteRank);
        const gapB = Math.abs(b.blackRank - b.whiteRank);
        return gapB - gapA;
      });
      const topN = pool.slice(0, 4);
      for (let i = topN.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        const tmp = topN[i]; topN[i] = topN[j]; topN[j] = tmp;
      }
      const sampleGameA = topN[0] || null;
      const sampleGameB = topN[1] || null;

      // Capture standings BEFORE selection zeroes fitness. Mean/max fitness
      // are computed here too — the old worker computed them from the
      // post-selection population, whose fitness had just been wiped to 0,
      // so the history chart always plotted flat zeros.
      population.sort((x, y) => y.fitness - x.fitness);
      lastTournamentSnapshot = population.map((ind, i) => ({
        rank: i + 1,
        fitness: ind.fitness,
        wins: ind.wins,
        losses: ind.losses,
        draws: ind.draws,
      }));
      let fitSum = 0;
      let fitMax = -Infinity;
      for (const ind of population) {
        fitSum += ind.fitness;
        if (ind.fitness > fitMax) fitMax = ind.fitness;
      }

      // Selection: half-keep-mutate. Survivors keep their weight arrays
      // (nothing mutates weights in place).
      const survivors = population.slice(0, POP_SIZE / 2);
      for (const s of survivors) {
        s.fitness = 0; s.wins = 0; s.losses = 0; s.draws = 0;
      }
      const offspring = [];
      for (let i = 0; i < POP_SIZE / 2; i++) {
        const parent = survivors[i % survivors.length];
        const m = N.mutate(parent.weights, parent.sigmas);
        offspring.push(makeIndividual(m.weights, m.sigmas));
      }
      population = survivors.concat(offspring);

      generation++;

      onGen({
        gen: generation,
        leaderboard: lastTournamentSnapshot,
        meanFitness: fitSum / POP_SIZE,
        maxFitness: fitMax,
        sampleGameA: sampleGameA,
        sampleGameB: sampleGameB,
        genMs: now() - genStartMs,
        // Barrier diagnostics: with perfect scheduling genMs would approach
        // max(gameMsSum / poolSize, gameMsMax).
        genStats: {
          gameMsSum: genGameMsSum,
          gameMsMax: genGameMsMax,
          hedges: genHedges,
        },
      });

      // Snapshot requests resolve at the gen boundary so the champion
      // reflects every game played this burst.
      if (snapshotWaiters.length > 0) {
        const snap = topSnapshot();
        const waiters = snapshotWaiters;
        snapshotWaiters = [];
        for (const wtr of waiters) wtr.resolve(snap);
      }

      if (running && !genInFlight) startGen();
    }

    function leaderboardSnapshot() {
      if (lastTournamentSnapshot) return lastTournamentSnapshot;
      return population.map((ind, i) => ({
        rank: i + 1, fitness: 0, wins: 0, losses: 0, draws: 0,
      }));
    }

    function topSnapshot() {
      // population[0] is the current champion after the last gen's sort.
      if (!population) initPopulation();
      const top = population[0];
      return {
        gen: generation,
        weights: new Float32Array(top.weights),
        sigmas: new Float32Array(top.sigmas),
        fitness: top.fitness,
      };
    }

    // ---- Public API ------------------------------------------------------

    function reset() {
      epoch++;  // in-flight results now arrive stale and get dropped
      running = false;
      genInFlight = false;
      specs = [];
      queue = [];
      done = [];
      doneCount = 0;
      inFlight = [];
      recordings = [];
      lastTournamentSnapshot = null;
      const waiters = snapshotWaiters;
      snapshotWaiters = [];
      for (const wtr of waiters) wtr.reject(new Error("reset"));
      initPopulation();
      // Async like the old worker's reset reply, so callers finish their own
      // reset bookkeeping before the gen-0 event touches the DOM.
      setTimeout(() => {
        onGen({
          gen: 0,
          leaderboard: leaderboardSnapshot(),
          meanFitness: 0,
          maxFitness: 0,
          sampleGameA: null,
          sampleGameB: null,
        });
      }, 0);
    }

    function resume() {
      if (!population) initPopulation();
      if (running) return;
      running = true;
      if (!genInFlight) startGen();
    }

    function pause() {
      // The in-flight generation always completes (the population must sit
      // at a gen boundary); we just don't start another.
      running = false;
    }

    function snapshot() {
      return new Promise((resolve, reject) => {
        if (!genInFlight) {
          resolve(topSnapshot());
        } else {
          snapshotWaiters.push({ resolve, reject });
        }
      });
    }

    if (!population) initPopulation();

    return { reset, resume, pause, snapshot, poolSize };
  }

  global.Evolution = { create };
})(typeof self !== "undefined" ? self : this);
