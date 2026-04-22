/*
 * Main game loop and UI glue.
 *
 * Milestone 3: evolution runs in a Web Worker. On each AI turn:
 *   1. Resume evolution for ~1s (gens accumulate in the worker).
 *   2. Pause, request a snapshot of the top-ranked weights.
 *   3. Run depth-4 minimax locally with that snapshot.
 *   4. Commit the move. The gen counter reflects the gen that moved.
 *
 * During the human's turn, evolution is paused. Gens in the worker only
 * accumulate during AI-turn bursts. A full 30-move game yields ~30 seconds of
 * evolution total.
 */

(function () {
  "use strict";

  const C = Checkers;
  const R = Render;
  const N = Network;
  const M = Minimax;

  const canvas = document.getElementById("board");
  const ctx = canvas.getContext("2d");
  const miniCanvas = document.getElementById("mini-board");
  const miniCtx = miniCanvas.getContext("2d");
  const miniSize = miniCanvas.width;
  const moveLog = document.getElementById("move-log");
  const moveHistoryEl = document.getElementById("move-history");
  const historyCanvas = document.getElementById("history-chart");
  const historyCtx = historyCanvas.getContext("2d");
  const newGameBtn = document.getElementById("new-game");
  const offerDrawBtn = document.getElementById("offer-draw");
  const resignBtn = document.getElementById("resign");
  const humanColorSel = document.getElementById("human-color");

  const AI_DEPTH = 4;
  const TRAIN_BURST_MS = 2000;   // evolution runs this long per AI turn
  const MIN_SEARCH_PAD_MS = 200; // small UX pad so moves don't snap instantly

  const state = {
    board: C.makeBoard(),
    selectedFrom: -1,
    pendingJumpPath: null,
    lastFrom: -1,
    lastTo: -1,
    lastCaptured: [],
    stateCounts: Object.create(null),
    finished: false,
    humanColor: "white",
    aiNet: null,
    aiThinking: false,
    latestGen: 0,
    turnStartGen: 0,
    turnGens: [],  // accumulates per-gen stats during the current AI burst (for Milestone 4 replay)
    genBurstStart: 0,  // timestamp, for gens/sec
    aiEvolveMs: 0,     // cumulative training-burst time across all AI turns
    aiSearchMs: 0,     // cumulative minimax-search time across all AI turns
    aiMoveCount: 0,    // AI moves committed this game
    latestSampleGameA: null,
    latestSampleGameB: null,
    miniSlot: "A",         // which sample game is currently playing back: "A" or "B"
    miniPlayback: null,    // { game, step, finished, timer }
    historyGens: [],      // cumulative per-gen fitness stats across the whole game
    turnBoundaries: [],   // generation numbers at which each AI turn started
  };

  // ---- Worker setup ---------------------------------------------------------

  const worker = new Worker("js/worker.js");
  let pendingSnapshot = null;  // Promise resolver while a snapshot is in flight

  worker.onerror = (ev) => {
    console.error("Worker error:", ev.message || ev);
    showBanner("error", "Training worker crashed: " + (ev.message || "unknown"));
  };

  worker.onmessage = (ev) => {
    const msg = ev.data;
    if (msg.type === "ready") {
      // Worker initialized at gen 0; nothing to do here.
      return;
    }
    if (msg.type === "gen") {
      state.latestGen = msg.gen;
      if (state.aiThinking) {
        state.turnGens.push({
          gen: msg.gen,
          meanFitness: msg.meanFitness,
          maxFitness: msg.maxFitness,
          leaderboard: msg.leaderboard,
        });
        // Also append to the cumulative game-long history. Skip the gen 0
        // reset ping (meanFitness and maxFitness are both 0 before any
        // tournament has run).
        if (msg.gen > 0) {
          state.historyGens.push({
            gen: msg.gen,
            meanFitness: msg.meanFitness,
            maxFitness: msg.maxFitness,
          });
          renderHistoryChart();
        }
        const delta = msg.gen - state.turnStartGen;
        document.getElementById("gen-delta").textContent = "+" + delta + " this turn";
        const elapsed = (performance.now() - state.genBurstStart) / 1000;
        if (elapsed > 0) {
          document.getElementById("gens-per-sec").textContent = (delta / elapsed).toFixed(1);
        }
      }
      document.getElementById("gen-counter").textContent = "gen " + msg.gen;
      renderLeaderboard(msg.leaderboard);
      if (msg.sampleGameA) state.latestSampleGameA = msg.sampleGameA;
      if (msg.sampleGameB) state.latestSampleGameB = msg.sampleGameB;
      return;
    }
    if (msg.type === "snapshot") {
      if (pendingSnapshot) {
        pendingSnapshot(msg);
        pendingSnapshot = null;
      }
      return;
    }
  };

  function snapshot() {
    return new Promise((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        if (pendingSnapshot === resolve) pendingSnapshot = null;
        reject(new Error("snapshot timeout"));
      }, 3000);
      pendingSnapshot = (msg) => { clearTimeout(timeoutId); resolve(msg); };
      worker.postMessage({ type: "snapshot" });
    });
  }

  function renderLeaderboard(entries) {
    const ol = document.getElementById("leaderboard");
    if (!entries || entries.length === 0) {
      ol.innerHTML = "<li class=\"placeholder\">Waiting for first generation…</li>";
      return;
    }
    const maxAbs = Math.max(1, ...entries.map(e => Math.abs(e.fitness)));
    const allZero = entries.every(e => e.fitness === 0 && e.wins === 0 && e.losses === 0 && e.draws === 0);
    if (allZero) {
      ol.innerHTML = "<li class=\"placeholder\">Population initialized. Run a generation to rank them…</li>";
      return;
    }
    const html = entries.map((e, rank) => {
      const pct = (Math.abs(e.fitness) / maxAbs) * 100;
      const side = e.fitness >= 0 ? "pos" : "neg";
      const wld = `${e.wins}/${e.losses}/${e.draws}`;
      return `<li>
        <span class="rank">#${rank + 1}</span>
        <span class="bar"><div class="${side}" style="transform: scaleX(${(pct / 100).toFixed(3)})"></div></span>
        <span class="score">${e.fitness >= 0 ? "+" : ""}${e.fitness.toFixed(1)}</span>
        <span class="wld" title="wins/losses/draws">${wld}</span>
      </li>`;
    }).join("");
    ol.innerHTML = html;
  }

  function renderHistoryChart() {
    const W = historyCanvas.width;
    const H = historyCanvas.height;
    const pad = 8;
    const g = historyCtx;
    g.clearRect(0, 0, W, H);

    const rows = state.historyGens;
    const capEl = document.getElementById("history-caption");
    if (rows.length === 0) {
      capEl.textContent = "no generations yet";
      return;
    }

    // Range
    const maxGen = rows[rows.length - 1].gen;
    let lo = Infinity, hi = -Infinity;
    for (const r of rows) {
      if (r.maxFitness > hi) hi = r.maxFitness;
      if (r.meanFitness < lo) lo = r.meanFitness;
      if (r.maxFitness < lo) lo = r.maxFitness;
      if (r.meanFitness > hi) hi = r.meanFitness;
    }
    // Guarantee some spread so flat lines don't look weird.
    if (hi - lo < 1) { const mid = (hi + lo) / 2; lo = mid - 1; hi = mid + 1; }

    const xOf = (gen) => pad + (gen / Math.max(1, maxGen)) * (W - 2 * pad);
    const yOf = (v) => {
      const t = (v - lo) / (hi - lo);
      return H - pad - t * (H - 2 * pad);
    };

    // Zero line
    g.strokeStyle = "rgba(255,255,255,0.15)";
    g.lineWidth = 1;
    if (lo <= 0 && hi >= 0) {
      g.beginPath();
      const y0 = yOf(0);
      g.moveTo(pad, y0); g.lineTo(W - pad, y0);
      g.stroke();
    }

    // Turn-boundary ticks
    g.strokeStyle = "rgba(255,255,255,0.08)";
    for (const tb of state.turnBoundaries) {
      if (tb <= 0 || tb > maxGen) continue;
      const x = xOf(tb);
      g.beginPath();
      g.moveTo(x, pad); g.lineTo(x, H - pad);
      g.stroke();
    }

    // Mean line
    g.strokeStyle = "rgba(170,180,200,0.65)";
    g.lineWidth = 1.2;
    g.beginPath();
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i];
      const x = xOf(r.gen);
      const y = yOf(r.meanFitness);
      if (i === 0) g.moveTo(x, y); else g.lineTo(x, y);
    }
    g.stroke();

    // Max line
    g.strokeStyle = "rgba(243,193,74,0.95)";
    g.lineWidth = 1.5;
    g.beginPath();
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i];
      const x = xOf(r.gen);
      const y = yOf(r.maxFitness);
      if (i === 0) g.moveTo(x, y); else g.lineTo(x, y);
    }
    g.stroke();

    capEl.textContent = `gen 1 — ${maxGen} · best ${hi >= 0 ? "+" : ""}${hi.toFixed(1)} · ${state.turnBoundaries.length} AI turn${state.turnBoundaries.length === 1 ? "" : "s"}`;
  }

  // Layered architecture view. Draws the network as 4 columns of nodes
  // (32 → 40 → 10 → 1) with the top-K strongest connections per layer drawn
  // as lines. Same diverging palette: red negative, blue positive.
  function renderLayeredNetwork(gen, weights) {
    const canvas = document.getElementById("layers");
    const g = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;

    g.clearRect(0, 0, W, H);
    g.fillStyle = "#1f242f";
    g.fillRect(0, 0, W, H);

    // Layer offsets into the flat weights buffer (see network.js for the layout).
    const OFF_W1 = 0;
    const OFF_W2 = 0 + 32 * 40 + 40;     // past W1 + b1
    const OFF_W3 = OFF_W2 + 40 * 10 + 10; // past W2 + b2
    const LAYERS = [
      { count: 32, x: 28,  yTop: 14,  yBot: H - 14 },
      { count: 40, x: 125, yTop: 6,   yBot: H - 6  },
      { count: 10, x: 225, yTop: 30,  yBot: H - 30 },
      { count: 1,  x: 315, yTop: H/2, yBot: H/2    },
    ];

    function nodeY(layer, i) {
      if (layer.count === 1) return H / 2;
      return layer.yTop + i * (layer.yBot - layer.yTop) / (layer.count - 1);
    }

    // Helper: draw a slab of connections. `matrix` is a function giving the
    // weight for (dst, src). We pick the top-K by absolute magnitude and draw
    // them with color and alpha scaled to magnitude.
    function drawLayer(fromIdx, toIdx, matrix, topK) {
      const layerFrom = LAYERS[fromIdx];
      const layerTo = LAYERS[toIdx];
      const entries = [];
      for (let d = 0; d < layerTo.count; d++) {
        for (let s = 0; s < layerFrom.count; s++) {
          entries.push({ s, d, w: matrix(d, s) });
        }
      }
      entries.sort((a, b) => Math.abs(b.w) - Math.abs(a.w));
      const keep = entries.slice(0, topK);
      const normAbs = Math.max(1e-6, Math.abs(keep[0] ? keep[0].w : 1));

      // Draw in magnitude order so strongest lines render on top.
      keep.reverse();
      for (const e of keep) {
        const t = Math.max(-1, Math.min(1, e.w / normAbs));
        const a = Math.abs(t);
        const x1 = layerFrom.x + 4;
        const y1 = nodeY(layerFrom, e.s);
        const x2 = layerTo.x - 4;
        const y2 = nodeY(layerTo, e.d);
        g.beginPath();
        g.moveTo(x1, y1);
        g.lineTo(x2, y2);
        const rgb = t < 0 ? `255,60,60` : `80,170,255`;
        g.strokeStyle = `rgba(${rgb},${(0.15 + 0.7 * a).toFixed(3)})`;
        g.lineWidth = 0.5 + 1.8 * a;
        g.stroke();
      }
    }

    // Layer 1 → Layer 2: W1 is [40 × 32] row-major.
    drawLayer(0, 1, (d, s) => weights[OFF_W1 + d * 32 + s], 40);
    // Layer 2 → Layer 3: W2 is [10 × 40] row-major.
    drawLayer(1, 2, (d, s) => weights[OFF_W2 + d * 40 + s], 40);
    // Layer 3 → Layer 4: W3 is [1 × 10].
    drawLayer(2, 3, (d, s) => weights[OFF_W3 + s], 10);

    // Piece-diff bypass — draw a single soft curve from the input column's
    // centroid to the output, labeled, so the paper's "material signal" is
    // visible at a glance.
    const OFF_PIECE_DIFF = OFF_W3 + 10 + 1; // past W3 + b3
    const pd = weights[OFF_PIECE_DIFF];
    const pdA = Math.min(1, Math.abs(pd) / 0.5);
    g.beginPath();
    g.moveTo(LAYERS[0].x + 4, H / 2);
    g.bezierCurveTo(W * 0.4, H * 0.2, W * 0.7, H * 0.8, LAYERS[3].x - 4, H / 2);
    const pdRgb = pd < 0 ? `255,150,80` : `120,220,200`;
    g.strokeStyle = `rgba(${pdRgb},${(0.25 + 0.6 * pdA).toFixed(3)})`;
    g.lineWidth = 1 + 2 * pdA;
    g.stroke();

    // Nodes, drawn on top.
    for (let li = 0; li < LAYERS.length; li++) {
      const layer = LAYERS[li];
      const r = li === 3 ? 6 : 2.6;
      for (let i = 0; i < layer.count; i++) {
        g.beginPath();
        g.arc(layer.x, nodeY(layer, i), r, 0, Math.PI * 2);
        g.fillStyle = "#e6ecf3";
        g.fill();
        g.strokeStyle = "rgba(0,0,0,0.5)";
        g.lineWidth = 1;
        g.stroke();
      }
    }

    // Column labels.
    g.fillStyle = "rgba(200,210,225,0.75)";
    g.font = "10px -apple-system, Segoe UI, sans-serif";
    g.textAlign = "center";
    g.fillText("32 in",  LAYERS[0].x, H - 2);
    g.fillText("40",     LAYERS[1].x, H - 2);
    g.fillText("10",     LAYERS[2].x, H - 2);
    g.fillText("out",    LAYERS[3].x, H - 2);
    g.textAlign = "start";

    document.getElementById("layers-caption").textContent =
      `gen ${gen} · 3 dense layers + piece-diff bypass · top-40 edges drawn per slab`;
  }

  function appendMoveHistory(actor, move, captured, extra) {
    const path = describeMove(move);
    const capMsg = captured && captured.length ? ` × ${captured.join(",")}` : "";
    const meta = extra ? `<span class="meta"> ${extra}</span>` : "";
    const klass = actor.startsWith("AI") ? "actor-ai" : "actor-you";
    const li = document.createElement("li");
    li.className = klass;
    li.innerHTML = `<span>${actor}: ${path}${capMsg}${meta}</span>`;
    moveHistoryEl.appendChild(li);
    moveHistoryEl.scrollTop = moveHistoryEl.scrollHeight;
  }

  function showBanner(kind, text) {
    const b = document.getElementById("status-banner");
    if (!text) { b.hidden = true; return; }
    b.hidden = false;
    b.className = kind === "info" ? "info" : "";
    b.textContent = text;
  }

  // ---- Game state + rendering ----------------------------------------------

  function resetGame() {
    state.board = C.makeBoard();
    state.selectedFrom = -1;
    state.pendingJumpPath = null;
    state.lastFrom = -1;
    state.lastTo = -1;
    state.lastCaptured = [];
    state.stateCounts = Object.create(null);
    state.finished = false;
    state.humanColor = humanColorSel.value;
    state.aiThinking = false;
    state.latestGen = 0;
    state.turnStartGen = 0;
    state.turnGens = [];
    state.aiEvolveMs = 0;
    state.aiSearchMs = 0;
    state.aiMoveCount = 0;
    state.latestSampleGameA = null;
    state.latestSampleGameB = null;
    state.historyGens = [];
    state.turnBoundaries = [];
    stopAllMiniPlaybacks();
    Render.drawMini(miniCtx, C.makeBoard().squares, { size: miniSize });
    document.getElementById("mini-label").textContent = "—";
    document.getElementById("mini-caption").textContent = "waiting for first self-play game…";
    moveHistoryEl.innerHTML = "";
    // Clear the layered-architecture view.
    const layersCanvas = document.getElementById("layers");
    if (layersCanvas) {
      const lg = layersCanvas.getContext("2d");
      lg.fillStyle = "#1f242f";
      lg.fillRect(0, 0, layersCanvas.width, layersCanvas.height);
      document.getElementById("layers-caption").textContent = "waiting for the AI's first move…";
    }
    renderHistoryChart();

    // Send the worker back to gen 0 for a fresh run.
    worker.postMessage({ type: "reset" });

    document.getElementById("gen-counter").textContent = "gen 0";
    document.getElementById("gen-delta").textContent = "+0 this turn";
    document.getElementById("gens-per-sec").textContent = "—";
    updatePosEval();
    updateAiTimeDisplay();

    logClear();
    render();
    updatePieceCounts();
    updateButtons();
    maybeStartAiTurn();
  }

  function log(msg) { moveLog.textContent = msg; }
  function logClear() { moveLog.textContent = ""; }

  function humanSide() {
    return state.humanColor === "black" ? C.BLACK : C.WHITE;
  }

  function render() {
    const hints = computeHints();
    R.draw(ctx, state.board, {
      lastFrom: state.lastFrom,
      lastTo: state.lastTo,
      captured: state.lastCaptured,
      legalTargets: hints.legalTargets,
      captureTargets: hints.captureTargets,
      selectedFrom: state.pendingJumpPath
        ? state.pendingJumpPath[state.pendingJumpPath.length - 1]
        : state.selectedFrom,
      humanColor: state.humanColor,
    });
    updateForcedCaptureBanner();
  }

  function updateForcedCaptureBanner() {
    if (state.finished || state.aiThinking) { showBanner(null, null); return; }
    if (state.board.currentPlayer !== humanSide()) { showBanner(null, null); return; }

    const moves = C.getLegalMoves(state.board);
    if (moves.length === 0) { showBanner(null, null); return; }

    // getLegalMoves returns ONLY jumps if any exist — so if the first move is
    // a jump (length >= 3), captures are mandatory for the human this turn.
    const forced = moves[0].length >= 3;
    if (forced) {
      const pieces = new Set();
      for (const m of moves) pieces.add(m[0]);
      if (state.pendingJumpPath) {
        showBanner("", "You must continue the jump — click the next green dot.");
      } else {
        const count = pieces.size;
        const piecesList = Array.from(pieces).sort((a, b) => a - b).join(", ");
        showBanner("", `Capture is mandatory. ${count} of your pieces can jump (square${count > 1 ? "s" : ""}: ${piecesList}).`);
      }
    } else {
      showBanner(null, null);
    }
  }

  function computeHints() {
    const out = { legalTargets: [], captureTargets: [] };
    if (state.finished) return out;
    if (state.board.currentPlayer !== humanSide()) return out;

    const moves = C.getLegalMoves(state.board);
    if (state.pendingJumpPath) {
      for (const m of moves) {
        if (m.length < 3) continue;
        if (!pathStartsWith(m, state.pendingJumpPath)) continue;
        const nextLand = m[state.pendingJumpPath.length + 1];
        if (nextLand != null && out.legalTargets.indexOf(nextLand) === -1) {
          out.legalTargets.push(nextLand);
          out.captureTargets.push(nextLand);
        }
      }
      return out;
    }
    if (state.selectedFrom !== -1) {
      for (const m of moves) {
        if (m[0] !== state.selectedFrom) continue;
        if (m.length === 2) {
          out.legalTargets.push(m[1]);
        } else {
          out.legalTargets.push(m[2]);
          out.captureTargets.push(m[2]);
        }
      }
    }
    return out;
  }

  function pathStartsWith(move, prefix) {
    if (move.length < prefix.length) return false;
    for (let i = 0; i < prefix.length; i++) if (move[i] !== prefix[i]) return false;
    return true;
  }

  function arrayEq(a, b) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  }

  function onCanvasClick(ev) {
    if (state.finished || state.aiThinking) return;
    if (state.board.currentPlayer !== humanSide()) return;

    const sq = R.clickToSquare(canvas, ev, state.humanColor);
    if (sq === -1) return;

    const moves = C.getLegalMoves(state.board);
    if (moves.length === 0) return;

    if (state.pendingJumpPath) { tryExtendJump(sq, moves); return; }

    if (state.selectedFrom === -1) {
      if (moves.some(m => m[0] === sq)) {
        state.selectedFrom = sq;
        render();
      }
      return;
    }

    if (sq === state.selectedFrom) {
      state.selectedFrom = -1;
      render();
      return;
    }
    const candidates = moves.filter(m => m[0] === state.selectedFrom);
    const slides = candidates.filter(m => m.length === 2 && m[1] === sq);
    const jumps  = candidates.filter(m => m.length >= 3 && m[2] === sq);

    if (slides.length > 0) { commitMove(slides[0], "You"); return; }
    if (jumps.length > 0) { startHumanJump(jumps[0], candidates); return; }

    if (C.owner(state.board.squares[sq]) === humanSide()
        && moves.some(m => m[0] === sq)) {
      state.selectedFrom = sq;
      render();
    }
  }

  function startHumanJump(firstCandidate, allCandidates) {
    const firstHopPath = firstCandidate.slice(0, 3);
    const matching = allCandidates.filter(m => pathStartsWith(m, firstHopPath));
    if (matching.length === 1) { commitMove(matching[0], "You"); return; }
    if (matching.every(m => arrayEq(m, matching[0]))) {
      commitMove(matching[0], "You");
      return;
    }
    state.pendingJumpPath = firstHopPath;
    state.selectedFrom = -1;
    render();
  }

  function tryExtendJump(sq, moves) {
    const path = state.pendingJumpPath;
    const continuations = moves.filter(m =>
      pathStartsWith(m, path) && m.length > path.length
    );
    const matches = continuations.filter(m => m[path.length + 1] === sq);
    if (matches.length === 0) return;

    const nextCap = matches[0][path.length];
    const nextLand = matches[0][path.length + 1];
    const newPath = path.concat([nextCap, nextLand]);

    const furtherChoices = matches.filter(m => m.length > newPath.length);
    if (furtherChoices.length === 0) {
      const terminalMatch = matches.find(m => m.length === newPath.length);
      if (terminalMatch) { commitMove(terminalMatch, "You"); return; }
    }
    state.pendingJumpPath = newPath;
    render();
  }

  function commitMove(move, actor) {
    const captured = [];
    for (let i = 1; i < move.length; i += 2) captured.push(move[i]);

    state.board = C.applyMove(state.board, move);
    state.lastFrom = move[0];
    state.lastTo = move[move.length - 1];
    state.lastCaptured = captured;
    state.selectedFrom = -1;
    state.pendingJumpPath = null;

    const key = C.stateKey(state.board);
    state.stateCounts[key] = (state.stateCounts[key] || 0) + 1;

    const path = describeMove(move);
    const capMsg = captured.length ? ` (captured ${captured.join(", ")})` : "";
    log(`${actor}: ${path}${capMsg}`);
    const metaGen = actor.startsWith("AI ") ? actor.replace(/^AI\s*/, "(") + ")" : "";
    appendMoveHistory(actor.startsWith("AI") ? "AI" : "You", move, captured, metaGen);

    render();
    updatePosEval();
    updatePieceCounts();
    updateButtons();
    if (checkEnd()) { updateButtons(); return; }

    maybeStartAiTurn();
  }

  function describeMove(move) {
    if (move.length === 2) return `${move[0]} → ${move[1]}`;
    const stops = [move[0]];
    for (let i = 2; i < move.length; i += 2) stops.push(move[i]);
    return stops.join(" → ");
  }

  function checkEnd() {
    const key = C.stateKey(state.board);
    if (state.stateCounts[key] >= 3) {
      state.finished = true;
      log("Draw by threefold repetition.");
      return true;
    }
    if (state.board.moveCount >= 200) {
      state.finished = true;
      log("Draw by move cap.");
      return true;
    }
    const [over, winner] = C.isGameOver(state.board);
    if (over) {
      state.finished = true;
      if (winner === humanSide()) log("You win!");
      else if (winner === -humanSide()) log("AI wins.");
      else log("Draw.");
      return true;
    }
    return false;
  }

  // ---- AI turn orchestration ------------------------------------------------

  function maybeStartAiTurn() {
    if (state.finished) return;
    if (state.board.currentPlayer === humanSide()) return;

    state.aiThinking = true;
    state.turnStartGen = state.latestGen;
    state.turnGens = [];
    state.turnBoundaries.push(state.latestGen);
    state.genBurstStart = performance.now();
    state._evolveStart = performance.now();
    // Pause the mini replay while the AI trains — we'll start a fresh one
    // (from this turn's games) as soon as the AI has finished moving.
    stopAllMiniPlaybacks();
    document.getElementById("mini-label").textContent = "—";
    document.getElementById("mini-caption").textContent = "AI is training — new games in progress…";
    log("AI is training…");
    updateButtons();

    // Resume evolution in the worker. After TRAIN_BURST_MS, pause, snapshot,
    // search, and commit.
    worker.postMessage({ type: "resume" });
    setTimeout(onTrainBurstEnd, TRAIN_BURST_MS);
  }

  async function onTrainBurstEnd() {
    if (!state.aiThinking) return;  // resetGame may have fired

    // Stop the evolve timer; snapshot + search contribute to the search timer.
    state.aiEvolveMs += performance.now() - state._evolveStart;
    updateAiTimeDisplay();

    worker.postMessage({ type: "pause" });

    let snap;
    const snapStart = performance.now();
    try {
      snap = await snapshot();
    } catch (err) {
      console.error("snapshot failed:", err);
      state.aiThinking = false;
      showBanner("", "Training worker timed out. Click New Game to reset.");
      return;
    }

    try {
      state.aiNet = N.makeNetwork(snap.weights);
      log(`AI is moving (gen ${snap.gen})…`);
      renderLayeredNetwork(snap.gen, snap.weights);
      updatePosEval();
      const searchStart = performance.now();
      const result = M.pickMove(state.board, AI_DEPTH, state.aiNet);
      const searchMs = performance.now() - searchStart;
      state.aiSearchMs += (searchStart - snapStart) + searchMs;
      state.aiMoveCount += 1;
      updateAiTimeDisplay();
      const pad = Math.max(0, MIN_SEARCH_PAD_MS - searchMs);

      setTimeout(() => {
        state.aiThinking = false;
        updatePosEval();
        if (!result.move) {
          state.finished = true;
          log("AI has no legal moves; you win!");
          updateButtons();
          return;
        }
        commitMove(result.move, `AI gen ${snap.gen}`);
        if (!state.finished && (state.latestSampleGameA || state.latestSampleGameB)) {
          startAllMiniPlaybacks();
        }
        updateButtons();
      }, pad);
    } catch (err) {
      console.error("AI move failed:", err);
      state.aiThinking = false;
      showBanner("", "AI move failed: " + err.message + ". Click New Game to reset.");
    }
  }

  // ---- Mini self-play replay (alternates Game A and Game B) ----------------

  const MINI_STEP_MS = 220;   // ms between frames
  const MINI_END_HOLD_MS = 1800;  // how long the winner banner stays up before swapping

  function latestSampleFor(slotId) {
    return slotId === "A" ? state.latestSampleGameA : state.latestSampleGameB;
  }

  function startAllMiniPlaybacks() {
    // Picks the currently-selected slot. Falls back to the other if empty.
    startMiniPlayback(state.miniSlot || "A");
  }

  function stopAllMiniPlaybacks() {
    stopMiniPlayback();
  }

  function setMiniLabel(slotId, caption) {
    document.getElementById("mini-label").textContent = "Game " + slotId;
    document.getElementById("mini-caption").textContent = caption;
  }

  function startMiniPlayback(slotId) {
    stopMiniPlayback();
    let game = latestSampleFor(slotId);
    let usedSlot = slotId;
    if (!game || !game.frames || game.frames.length === 0) {
      // No game in the requested slot; try the other before giving up.
      const other = slotId === "A" ? "B" : "A";
      const otherGame = latestSampleFor(other);
      if (otherGame && otherGame.frames && otherGame.frames.length > 0) {
        game = otherGame;
        usedSlot = other;
      } else {
        setMiniLabel(slotId, "waiting for self-play game…");
        return;
      }
    }
    state.miniSlot = usedSlot;
    state.miniPlayback = { game, slotId: usedSlot, step: 0, finished: false, timer: null };
    tickMiniPlayback();
  }

  function stopMiniPlayback() {
    if (state.miniPlayback && state.miniPlayback.timer != null) {
      clearTimeout(state.miniPlayback.timer);
    }
    state.miniPlayback = null;
  }

  function tickMiniPlayback() {
    const pb = state.miniPlayback;
    if (!pb) return;
    const frames = pb.game.frames;
    const squares = frames[pb.step];
    Render.drawMini(miniCtx, squares, { size: miniSize });
    if (pb.step >= frames.length - 1) {
      drawWinnerOverlay(miniCtx, miniSize, pb.game.winner);
    }
    setMiniLabel(pb.slotId, miniCaptionFor(pb.game, pb.step));

    pb.step += 1;
    if (pb.step >= frames.length) {
      pb.finished = true;
      pb.timer = setTimeout(() => {
        // Alternate: swap to the other slot if it has a game, else replay this one.
        const otherSlot = pb.slotId === "A" ? "B" : "A";
        const otherGame = latestSampleFor(otherSlot);
        if (otherGame && otherGame.frames && otherGame.frames.length > 0) {
          startMiniPlayback(otherSlot);
        } else {
          // No other-slot game; restart the current one (possibly with a
          // refreshed frame set if a newer version arrived).
          startMiniPlayback(pb.slotId);
        }
      }, MINI_END_HOLD_MS);
      return;
    }
    pb.timer = setTimeout(tickMiniPlayback, MINI_STEP_MS);
  }

  function miniCaptionFor(game, step) {
    const frames = game.frames;
    const moveNum = step;
    if (step >= frames.length - 1) {
      const w = game.winner;
      const outcome = w === 1 ? "Black wins" : w === -1 ? "White wins" : "Draw";
      return `#${game.blackIdx} vs #${game.whiteIdx} → ${outcome} (${frames.length - 1} plies)`;
    }
    return `#${game.blackIdx} vs #${game.whiteIdx} · move ${moveNum}/${frames.length - 1}`;
  }

  function drawWinnerOverlay(ctx, size, winner) {
    const text = winner === 1 ? "BLACK WINS" : winner === -1 ? "WHITE WINS" : "DRAW";
    const bandHeight = Math.floor(size * 0.26);
    const y = Math.floor((size - bandHeight) / 2);
    ctx.save();
    // Dimming veil so the board behind is still visible but subdued.
    ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
    ctx.fillRect(0, 0, size, size);
    // Band
    ctx.fillStyle = winner === 1
      ? "rgba(20, 20, 20, 0.92)"
      : winner === -1
        ? "rgba(230, 230, 230, 0.92)"
        : "rgba(110, 120, 150, 0.92)";
    ctx.fillRect(0, y, size, bandHeight);
    // Text
    ctx.fillStyle = winner === -1 ? "#1a1410" : "#ffffff";
    ctx.font = "bold 18px -apple-system, Segoe UI, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, size / 2, size / 2);
    ctx.restore();
  }

  function updateAiTimeDisplay() {
    const evolveS = state.aiEvolveMs / 1000;
    const searchS = state.aiSearchMs / 1000;
    const totalS = evolveS + searchS;
    document.getElementById("ai-total-time").textContent = totalS.toFixed(1) + " s";
    document.getElementById("ai-evolve-time").textContent = evolveS.toFixed(1) + " s";
    document.getElementById("ai-search-time").textContent = searchS.toFixed(2) + " s";
    document.getElementById("ai-move-count").textContent = String(state.aiMoveCount);
  }

  function updatePosEval() {
    const numEl = document.getElementById("eval-number");
    const verdictEl = document.getElementById("eval-verdict");
    const fillEl = document.getElementById("eval-bar-fill");
    if (!state.aiNet) {
      numEl.textContent = "—";
      verdictEl.textContent = "waiting for AI to move…";
      if (fillEl) { fillEl.style.width = "0%"; fillEl.style.left = "50%"; }
      return;
    }
    // Network outputs from the side-to-move's perspective; flip to always
    // show things from the HUMAN's perspective so "+" always means you're ahead.
    const rawV = state.aiNet.forward(state.board);
    const humanV = (state.board.currentPlayer === humanSide()) ? rawV : -rawV;
    const clamped = Math.max(-1, Math.min(1, humanV));

    numEl.textContent = (humanV >= 0 ? "+" : "") + humanV.toFixed(3);
    verdictEl.textContent = verdictFor(humanV);

    // Fill grows from center outward, to the right when positive, left when negative.
    const halfPct = Math.abs(clamped) * 50; // 0..50
    if (fillEl) {
      fillEl.style.width = halfPct.toFixed(2) + "%";
      if (humanV >= 0) {
        fillEl.style.left = "50%";
        fillEl.style.background = "rgba(80, 200, 120, 0.85)";
      } else {
        fillEl.style.left = (50 - halfPct).toFixed(2) + "%";
        fillEl.style.background = "rgba(230, 90, 90, 0.85)";
      }
    }
  }

  function verdictFor(v) {
    const a = Math.abs(v);
    if (a < 0.10) return "about even";
    const who = v > 0 ? "you" : "AI";
    if (a < 0.30) return "slight edge to " + who;
    if (a < 0.60) return who + " ahead";
    if (a < 0.90) return who + " clearly winning";
    return who + " has a near-decided advantage";
  }

  function updatePieceCounts() {
    let bMen = 0, bKings = 0, wMen = 0, wKings = 0;
    for (let i = 0; i < 32; i++) {
      const p = state.board.squares[i];
      if (p === C.BLACK_PIECE)  bMen++;
      else if (p === C.BLACK_KING) bKings++;
      else if (p === C.WHITE_PIECE) wMen++;
      else if (p === C.WHITE_KING) wKings++;
    }
    const you = humanSide() === C.BLACK
      ? { men: bMen, kings: bKings }
      : { men: wMen, kings: wKings };
    const ai = humanSide() === C.BLACK
      ? { men: wMen, kings: wKings }
      : { men: bMen, kings: bKings };

    document.getElementById("count-you").textContent = you.men + you.kings;
    document.getElementById("count-ai").textContent  = ai.men + ai.kings;
    document.getElementById("kings-you").textContent = you.kings;
    document.getElementById("kings-ai").textContent  = ai.kings;

    document.getElementById("piece-row-you").classList.toggle("low", (you.men + you.kings) <= 3);
    document.getElementById("piece-row-ai").classList.toggle("low", (ai.men + ai.kings) <= 3);

    document.body.classList.toggle("human-black", state.humanColor === "black");
  }

  function onOfferDraw() {
    if (state.finished || state.aiThinking) return;
    if (state.board.currentPlayer !== humanSide()) return;
    if (!state.aiNet) {
      log("You can offer a draw after the AI has moved at least once.");
      return;
    }
    // AI evaluates from ITS OWN side-to-move perspective. The eval from the
    // human's perspective is the current eval (since it's the human's turn).
    // Flip to get the AI's view of the position, then let it decide.
    const humanPerspective = state.aiNet.forward(state.board);
    const aiPerspective = -humanPerspective;
    // Paper's heuristic: AI accepts unless it considers itself clearly ahead.
    const ACCEPT_THRESHOLD = 0.30;
    if (aiPerspective <= ACCEPT_THRESHOLD) {
      state.finished = true;
      log(`Draw agreed. AI eval from its side: ${aiPerspective >= 0 ? "+" : ""}${aiPerspective.toFixed(2)}.`);
      updateButtons();
    } else {
      log(`AI declines the draw — it evaluates its position at +${aiPerspective.toFixed(2)}.`);
    }
  }

  function onResign() {
    if (state.finished || state.aiThinking) return;
    if (state.board.currentPlayer !== humanSide()) return;
    if (!confirm("Resign this game?")) return;
    state.finished = true;
    log("You resigned. AI wins.");
    updateButtons();
  }

  function updateButtons() {
    const yourTurn = !state.finished && !state.aiThinking
                     && state.board.currentPlayer === humanSide();
    offerDrawBtn.disabled = !yourTurn || !state.aiNet;
    resignBtn.disabled = !yourTurn;
  }

  canvas.addEventListener("click", onCanvasClick);
  newGameBtn.addEventListener("click", resetGame);
  offerDrawBtn.addEventListener("click", onOfferDraw);
  resignBtn.addEventListener("click", onResign);
  humanColorSel.addEventListener("change", resetGame);

  // Draw the starting board on the mini canvas immediately so the panel is
  // populated before the first self-play game arrives.
  Render.drawMini(miniCtx, C.makeBoard().squares, { size: miniSize });
  document.getElementById("mini-label").textContent = "—";
  document.getElementById("mini-caption").textContent = "starting position — self-play replays here during your turn";

  resetGame();
})();
