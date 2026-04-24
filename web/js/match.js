/*
 * match.js
 *
 * Auto-play viewer: loads both trained Anaconda checkpoints
 * (paper-strict + enhanced) and lets them play each other move by move,
 * with a watchable inter-move delay. Reuses the shared engine modules
 * (checkers, render, anaconda-network, minimax). No human input during
 * play — controls are: Start, Pause, Step, Reset, Swap colors, Speed,
 * Depth.
 *
 * Each side's eval of the current position is shown so the user can see
 * where the two models disagree. Eval is always rendered from the side's
 * own perspective (+ = good for that side).
 */

(function () {
  "use strict";

  const C = self.Checkers;
  const R = self.Render;
  const A = self.AnacondaNetwork;
  const M = self.Minimax;

  const SLOTS = [
    {
      id: "paper-strict",
      label: "Paper-strict",
      weightsUrl: "weights/anaconda-paper-strict.bin?v=18",
      metaUrl:    "weights/anaconda-paper-strict.meta.json?v=18",
    },
    {
      id: "enhanced",
      label: "Enhanced",
      weightsUrl: "weights/anaconda-enhanced.bin?v=18",
      metaUrl:    "weights/anaconda-enhanced.meta.json?v=18",
    },
  ];
  const MAX_MOVES = 200;
  const EVAL_COLOR_POS = "#20c060";
  const EVAL_COLOR_NEG = "#e03030";

  // ---- DOM refs ----
  const canvas         = document.getElementById("board");
  const ctx            = canvas.getContext("2d");
  const startBtn       = document.getElementById("start");
  const pauseBtn       = document.getElementById("pause");
  const stepBtn        = document.getElementById("step");
  const resetBtn       = document.getElementById("reset");
  const swapBtn        = document.getElementById("swap");
  const speedInput     = document.getElementById("speed");
  const speedLabel     = document.getElementById("speed-label");
  const depthSelect    = document.getElementById("depth");
  const moveCountEl    = document.getElementById("move-count");
  const moveLogEl      = document.getElementById("move-log");
  const moveHistoryEl  = document.getElementById("move-history");
  const bannerEl       = document.getElementById("status-banner");
  const blackNameEl    = document.getElementById("black-name");
  const whiteNameEl    = document.getElementById("white-name");

  // ---- State ----
  // Slot-0 is black by default, slot-1 is white. Swap toggles them.
  const state = {
    board: C.makeBoard(),
    moveCount: 0,
    lastFrom: -1,
    lastTo: -1,
    lastCaptured: [],
    stateCounts: Object.create(null),
    finished: false,
    playing: false,
    timer: null,
    blackSlotIdx: 0,
    whiteSlotIdx: 1,
    nets: { 0: null, 1: null },   // keyed by slot index, populated on load
    speedMs: parseInt(speedInput.value, 10),
    depth: parseInt(depthSelect.value, 10),
  };

  // ---- Helpers ----
  function blackSlot() { return SLOTS[state.blackSlotIdx]; }
  function whiteSlot() { return SLOTS[state.whiteSlotIdx]; }
  function blackNet()  { return state.nets[state.blackSlotIdx]; }
  function whiteNet()  { return state.nets[state.whiteSlotIdx]; }

  function log(msg) { moveLogEl.textContent = msg; }
  function showBanner(text, kind) {
    if (!text) { bannerEl.hidden = true; bannerEl.className = ""; bannerEl.textContent = ""; return; }
    bannerEl.hidden = false;
    bannerEl.className = kind || "info";
    bannerEl.textContent = text;
  }

  function render() {
    R.draw(ctx, state.board, {
      lastFrom: state.lastFrom,
      lastTo: state.lastTo,
      captured: state.lastCaptured,
    });
  }

  function updateLabels() {
    blackNameEl.textContent = blackSlot().label;
    whiteNameEl.textContent = whiteSlot().label;
  }

  // ---- Eval bars ----
  // For each side, run their own network on the current position and render
  // the score from that side's own perspective. So "Black: +0.42" means
  // Black's network thinks Black is ahead by 0.42; "White: +0.42" means
  // White's network thinks White is ahead by 0.42. They can disagree.
  function updateEvals() {
    paintEval("black", blackNet(), C.BLACK);
    paintEval("white", whiteNet(), C.WHITE);
  }
  function paintEval(side, net, sideColor) {
    if (!net) return;
    const fillEl    = document.getElementById(side + "-eval-fill");
    const numberEl  = document.getElementById(side + "-eval-number");
    const verdictEl = document.getElementById(side + "-eval-verdict");
    // network.forward returns score from current-side-to-move's perspective.
    // Flip to show from this side's perspective.
    const raw = net.forward(state.board);
    const fromSide = (state.board.currentPlayer === sideColor) ? raw : -raw;
    const pct = Math.max(-1, Math.min(1, fromSide));
    const half = Math.abs(pct) * 50;
    if (pct >= 0) {
      fillEl.style.left  = "50%";
      fillEl.style.width = half + "%";
      fillEl.style.background = EVAL_COLOR_POS;
      fillEl.style.color = EVAL_COLOR_POS;
    } else {
      fillEl.style.left  = (50 - half) + "%";
      fillEl.style.width = half + "%";
      fillEl.style.background = EVAL_COLOR_NEG;
      fillEl.style.color = EVAL_COLOR_NEG;
    }
    numberEl.textContent = (pct >= 0 ? "+" : "") + pct.toFixed(3);
    let v;
    if (Math.abs(pct) < 0.08) v = "roughly even";
    else if (pct >=  0.30)    v = "I'm ahead";
    else if (pct >=  0.08)    v = "slight edge";
    else if (pct <= -0.30)    v = "I'm losing";
    else                       v = "slight disadvantage";
    verdictEl.textContent = v;
  }

  // ---- Move history ----
  function describeMove(move) {
    if (move.length === 2) return `${move[0] + 1} → ${move[1] + 1}`;
    const stops = [move[0] + 1];
    for (let i = 2; i < move.length; i += 2) stops.push(move[i] + 1);
    return stops.join(" → ");
  }
  function appendHistory(actor, move) {
    const li = document.createElement("li");
    li.textContent = `${actor}: ${describeMove(move)}`;
    li.className = (actor === blackSlot().label) ? "actor-you" : "actor-ai";
    moveHistoryEl.appendChild(li);
    moveHistoryEl.scrollTop = moveHistoryEl.scrollHeight;
  }

  // ---- Game-end checks ----
  function checkEnd() {
    const [over, winner] = C.isGameOver(state.board);
    if (over) {
      state.finished = true;
      const w = winner === C.BLACK ? blackSlot().label
              : winner === C.WHITE ? whiteSlot().label
              : null;
      showBanner(w ? `${w} wins (no legal moves for opponent)` : "Draw", w ? null : "info");
      return true;
    }
    if (state.moveCount >= MAX_MOVES) {
      state.finished = true;
      showBanner(`Draw — move cap (${MAX_MOVES})`, "info");
      return true;
    }
    const key = C.stateKey(state.board);
    if (state.stateCounts[key] >= 3) {
      state.finished = true;
      showBanner("Draw — threefold repetition", "info");
      return true;
    }
    return false;
  }

  // ---- The actual move loop ----
  function playOneMove() {
    if (state.finished) return false;
    if (checkEnd()) { afterEnd(); return false; }

    const isBlackTurn = state.board.currentPlayer === C.BLACK;
    const net   = isBlackTurn ? blackNet() : whiteNet();
    const slot  = isBlackTurn ? blackSlot() : whiteSlot();
    const result = M.pickMove(state.board, state.depth, net);
    if (!result || !result.move) {
      state.finished = true;
      const winnerLabel = isBlackTurn ? whiteSlot().label : blackSlot().label;
      showBanner(`${winnerLabel} wins (opponent has no moves)`);
      afterEnd();
      return false;
    }

    state.lastFrom = result.move[0];
    state.lastTo = result.move[result.move.length - 1];
    state.lastCaptured = [];
    if (result.move.length > 2) {
      for (let i = 1; i < result.move.length; i += 2) {
        state.lastCaptured.push(result.move[i]);
      }
    }
    state.board = C.applyMove(state.board, result.move);
    state.moveCount += 1;
    moveCountEl.textContent = String(state.moveCount);

    const key = C.stateKey(state.board);
    state.stateCounts[key] = (state.stateCounts[key] || 0) + 1;

    appendHistory(slot.label, result.move);
    log(`${slot.label} (${isBlackTurn ? "B" : "W"}) plays ${describeMove(result.move)} · score ${(result.score >= 0 ? "+" : "") + result.score.toFixed(3)}`);
    render();
    updateEvals();

    if (checkEnd()) { afterEnd(); return false; }
    return true;
  }

  function loop() {
    state.timer = null;
    if (!state.playing) return;
    const ok = playOneMove();
    if (!ok) return;
    state.timer = setTimeout(loop, state.speedMs);
  }

  function setPlaying(playing) {
    state.playing = playing;
    startBtn.disabled = playing || state.finished;
    pauseBtn.disabled = !playing;
    stepBtn.disabled = playing || state.finished;
    swapBtn.disabled = playing || state.moveCount > 0;  // can't swap mid-game
  }
  function afterEnd() {
    setPlaying(false);
    startBtn.disabled = true;
  }

  // ---- Controls ----
  startBtn.addEventListener("click", () => {
    if (state.finished) return;
    showBanner(null);
    setPlaying(true);
    loop();
  });
  pauseBtn.addEventListener("click", () => {
    if (state.timer) { clearTimeout(state.timer); state.timer = null; }
    setPlaying(false);
  });
  stepBtn.addEventListener("click", () => {
    if (state.playing || state.finished) return;
    showBanner(null);
    playOneMove();
  });
  resetBtn.addEventListener("click", () => {
    if (state.timer) { clearTimeout(state.timer); state.timer = null; }
    state.board = C.makeBoard();
    state.moveCount = 0;
    state.lastFrom = -1;
    state.lastTo = -1;
    state.lastCaptured = [];
    state.stateCounts = Object.create(null);
    state.finished = false;
    moveCountEl.textContent = "0";
    moveHistoryEl.innerHTML = "";
    showBanner(null);
    log("Reset. Click Start.");
    render();
    updateEvals();
    setPlaying(false);
  });
  swapBtn.addEventListener("click", () => {
    if (state.playing || state.moveCount > 0) return;
    [state.blackSlotIdx, state.whiteSlotIdx] = [state.whiteSlotIdx, state.blackSlotIdx];
    updateLabels();
    updateEvals();
  });
  speedInput.addEventListener("input", () => {
    state.speedMs = parseInt(speedInput.value, 10);
    speedLabel.textContent = state.speedMs + " ms";
  });
  depthSelect.addEventListener("change", () => {
    state.depth = parseInt(depthSelect.value, 10);
  });

  // ---- Boot ----
  async function loadAll() {
    log("Loading networks…");
    try {
      const weights = await Promise.all(SLOTS.map(s => A.loadWeightsFromUrl(s.weightsUrl)));
      state.nets[0] = A.makeNetwork(weights[0]);
      state.nets[1] = A.makeNetwork(weights[1]);
      updateLabels();
      render();
      updateEvals();
      log("Ready. Click Start to watch them play.");
      setPlaying(false);
    } catch (err) {
      console.error(err);
      log("Failed to load networks: " + (err.message || err));
      startBtn.disabled = true;
      stepBtn.disabled = true;
    }
  }

  speedLabel.textContent = state.speedMs + " ms";
  loadAll();
})();
