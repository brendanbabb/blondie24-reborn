/*
 * Evolution worker.
 *
 * Runs the EP loop in the background — independent of the UI thread. Exposes
 * a small message protocol to the main thread:
 *
 *   from main:
 *     { type: "reset" }                 reinitialize population to gen 0
 *     { type: "resume" }                start (or resume) evolving
 *     { type: "pause" }                 stop evolving after the current gen
 *     { type: "snapshot" }              reply with top-ranked weights
 *
 *   from worker:
 *     { type: "ready" }                 sent once, on load
 *     { type: "gen",
 *         gen, leaderboard, meanFitness, maxFitness }   per-gen stats
 *     { type: "snapshot",
 *         gen, weights, sigmas, fitness }
 *
 * The worker is single-threaded; each gen is a synchronous burst of work. We
 * setTimeout(runOneGen, 0) between gens so the event loop can drain messages
 * (pause/snapshot requests) between gens.
 */

importScripts("checkers.js", "network.js", "minimax.js");

const C = self.Checkers;
const N = self.Network;
const M = self.Minimax;

// Demo-tuned EP hyperparameters — smaller than paper for browser responsiveness.
const POP_SIZE = 6;
const GAMES_PER_INDIVIDUAL = 3;   // paper uses 5; 3 gives reasonable ranking at pop=6
// Adaptive training depth. Openings have wide branching and shallow search is
// enough to rank networks; endgames have narrow branching but require more
// plies to actually convert material into wins — at a flat depth 3 most
// winning endgames degenerate into shuffle draws via threefold repetition.
const TRAIN_DEPTH_OPENING  = 3;
const TRAIN_DEPTH_ENDGAME  = 5;
const ENDGAME_PIECE_THRESHOLD = 6;   // <= this many total pieces → use endgame depth
const MAX_GAME_MOVES = 100;       // move cap for self-play games

const WIN_SCORE = 1.0;
const DRAW_SCORE = 0.0;
const LOSS_SCORE = -2.0;

let population = null;   // array of { weights, sigmas, fitness, wins, losses, draws }
let generation = 0;
let running = false;
let nextTask = null;

function initPopulation() {
  population = [];
  for (let i = 0; i < POP_SIZE; i++) {
    population.push({
      weights: N.newRandomWeights(0.1),
      sigmas: N.newSigmas(0.05),
      fitness: 0, wins: 0, losses: 0, draws: 0,
    });
  }
  generation = 0;
}

function sampleWithoutReplacement(n, k, excluded) {
  const pool = [];
  for (let i = 0; i < n; i++) if (i !== excluded) pool.push(i);
  // Fisher-Yates partial shuffle
  const picked = [];
  for (let i = 0; i < k && pool.length > 0; i++) {
    const idx = Math.floor(Math.random() * pool.length);
    picked.push(pool[idx]);
    pool[idx] = pool[pool.length - 1];
    pool.pop();
  }
  return picked;
}

function resetFitness() {
  for (const ind of population) {
    ind.fitness = 0;
    ind.wins = 0;
    ind.losses = 0;
    ind.draws = 0;
  }
}

// Play a self-play game between two individuals. Returns { winner, frames }
// where winner is +1 (black), -1 (white), or 0 (draw), and `frames` is an
// array of Int8Array(32) board snapshots (optional, when `record` is true —
// always captured here for the replay panel).
function playGame(black, white, record) {
  let board = C.makeBoard();
  const blackNet = N.makeNetwork(black.weights);
  const whiteNet = N.makeNetwork(white.weights);
  const stateCounts = Object.create(null);
  const frames = record ? [new Int8Array(board.squares)] : null;

  while (board.moveCount < MAX_GAME_MOVES) {
    const [over, winner] = C.isGameOver(board);
    if (over) return { winner: winner === null ? 0 : winner, frames };

    const key = C.stateKey(board);
    stateCounts[key] = (stateCounts[key] || 0) + 1;
    if (stateCounts[key] >= 3) return { winner: 0, frames };

    const net = board.currentPlayer === C.BLACK ? blackNet : whiteNet;
    const [bc, wc] = C.pieceCount(board);
    const depth = (bc + wc) <= ENDGAME_PIECE_THRESHOLD
      ? TRAIN_DEPTH_ENDGAME
      : TRAIN_DEPTH_OPENING;
    const { move } = M.pickMove(board, depth, net);
    if (!move) return { winner: -board.currentPlayer, frames };
    board = C.applyMove(board, move);
    if (frames) frames.push(new Int8Array(board.squares));
  }
  return { winner: 0, frames };
}

// Result of the most recently completed tournament. Captured BEFORE selection
// wipes fitness, so the leaderboard shows meaningful numbers.
let lastTournamentSnapshot = null;

// Two recorded self-play games per gen, for the replay panel.
let _lastSampleGameA = null;
let _lastSampleGameB = null;

// Run one full generation: tournament → rank → mutate offspring → advance.
function runOneGen() {
  resetFitness();

  // Build random pairings: each individual plays GAMES_PER_INDIVIDUAL games
  // against distinct randomly chosen opponents, random color each game.
  const pairings = [];
  for (let i = 0; i < POP_SIZE; i++) {
    const opps = sampleWithoutReplacement(POP_SIZE, GAMES_PER_INDIVIDUAL, i);
    for (const opp of opps) {
      pairings.push({ a: i, b: opp, aIsBlack: Math.random() < 0.5 });
    }
  }

  // Pick two distinct pairings to record for the replay panel (Game A and
  // Game B). If there are somehow fewer than 2 pairings, sampleIdxB stays
  // -1 and nothing is recorded for B.
  const sampleIdxA = Math.floor(Math.random() * pairings.length);
  let sampleIdxB = -1;
  if (pairings.length > 1) {
    do {
      sampleIdxB = Math.floor(Math.random() * pairings.length);
    } while (sampleIdxB === sampleIdxA);
  }
  let sampleGameA = null;
  let sampleGameB = null;

  for (let pi = 0; pi < pairings.length; pi++) {
    const p = pairings[pi];
    const shouldRecord = pi === sampleIdxA || pi === sampleIdxB;
    const black = p.aIsBlack ? population[p.a] : population[p.b];
    const white = p.aIsBlack ? population[p.b] : population[p.a];
    const result = playGame(black, white, shouldRecord);
    const winner = result.winner;
    if (shouldRecord) {
      const recorded = {
        frames: result.frames,
        blackIdx: p.aIsBlack ? p.a : p.b,
        whiteIdx: p.aIsBlack ? p.b : p.a,
        winner: winner,
      };
      if (pi === sampleIdxA) sampleGameA = recorded;
      else sampleGameB = recorded;
    }

    const playerA = population[p.a];
    const playerB = population[p.b];
    if (winner === 0) {
      playerA.fitness += DRAW_SCORE; playerA.draws++;
      playerB.fitness += DRAW_SCORE; playerB.draws++;
    } else {
      const blackIdx = p.aIsBlack ? p.a : p.b;
      const whiteIdx = p.aIsBlack ? p.b : p.a;
      if (winner === C.BLACK) {
        population[blackIdx].fitness += WIN_SCORE;  population[blackIdx].wins++;
        population[whiteIdx].fitness += LOSS_SCORE; population[whiteIdx].losses++;
      } else {
        population[whiteIdx].fitness += WIN_SCORE;  population[whiteIdx].wins++;
        population[blackIdx].fitness += LOSS_SCORE; population[blackIdx].losses++;
      }
    }
  }

  // Stash for the next gen event.
  _lastSampleGameA = sampleGameA;
  _lastSampleGameB = sampleGameB;

  // Capture the tournament standings BEFORE selection zeroes fitness.
  population.sort((x, y) => y.fitness - x.fitness);
  lastTournamentSnapshot = population.map((ind, i) => ({
    rank: i + 1,
    fitness: ind.fitness,
    wins: ind.wins,
    losses: ind.losses,
    draws: ind.draws,
  }));

  // Selection: half-keep-mutate (matches the repo default). Keep top half,
  // spawn that many offspring by mutating the survivors.
  const survivors = population.slice(0, POP_SIZE / 2);
  const offspring = [];
  for (let i = 0; i < POP_SIZE / 2; i++) {
    const parent = survivors[i % survivors.length];
    const m = N.mutate(parent.weights, parent.sigmas);
    offspring.push({
      weights: m.weights, sigmas: m.sigmas,
      fitness: 0, wins: 0, losses: 0, draws: 0,
    });
  }
  population = survivors.map(s => ({
    weights: new Float32Array(s.weights),
    sigmas:  new Float32Array(s.sigmas),
    fitness: 0, wins: 0, losses: 0, draws: 0,
  })).concat(offspring);

  generation++;
}

function leaderboardSnapshot() {
  if (lastTournamentSnapshot) return lastTournamentSnapshot;
  // Fallback for the initial gen 0 report before any tournament has run.
  return population.map((ind, i) => ({
    rank: i + 1, fitness: 0, wins: 0, losses: 0, draws: 0,
  }));
}

function topSnapshot() {
  // Called during idle time (between gens). The current "champion" is
  // population[0] after the last sort. We report its weights by value.
  if (!population) initPopulation();
  const top = population[0];
  return {
    gen: generation,
    weights: new Float32Array(top.weights),
    sigmas:  new Float32Array(top.sigmas),
    fitness: top.fitness,
  };
}

function scheduleNext() {
  if (!running) { nextTask = null; return; }
  nextTask = setTimeout(() => {
    const prevGen = generation;
    runOneGen();
    // After each gen, report stats + leaderboard.
    postMessage({
      type: "gen",
      gen: generation,
      leaderboard: leaderboardSnapshot(),
      meanFitness: population.reduce((s, x) => s + x.fitness, 0) / POP_SIZE,
      maxFitness: Math.max.apply(null, population.map(x => x.fitness)),
      sampleGameA: _lastSampleGameA,
      sampleGameB: _lastSampleGameB,
    });
    scheduleNext();
  }, 0);
}

self.onmessage = function (ev) {
  const msg = ev.data || {};
  switch (msg.type) {
    case "reset":
      running = false;
      if (nextTask) { clearTimeout(nextTask); nextTask = null; }
      lastTournamentSnapshot = null;
      _lastSampleGameA = null;
      _lastSampleGameB = null;
      initPopulation();
      postMessage({
        type: "gen",
        gen: 0,
        leaderboard: leaderboardSnapshot(),
        meanFitness: 0,
        maxFitness: 0,
        sampleGameA: null,
        sampleGameB: null,
      });
      break;
    case "resume":
      if (!population) initPopulation();
      if (!running) {
        running = true;
        scheduleNext();
      }
      break;
    case "pause":
      running = false;
      if (nextTask) { clearTimeout(nextTask); nextTask = null; }
      break;
    case "snapshot":
      postMessage(Object.assign({ type: "snapshot" }, topSnapshot()));
      break;
  }
};

initPopulation();
postMessage({ type: "ready", gen: 0 });
