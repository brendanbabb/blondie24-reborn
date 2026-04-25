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
      weightsUrl: "weights/anaconda-paper-strict.bin?v=31",
      metaUrl:    "weights/anaconda-paper-strict.meta.json?v=31",
    },
    {
      id: "enhanced",
      label: "Enhanced",
      weightsUrl: "weights/anaconda-enhanced.bin?v=31",
      metaUrl:    "weights/anaconda-enhanced.meta.json?v=31",
    },
    {
      id: "risky",
      label: "Risky",
      weightsUrl: "weights/anaconda-risky.bin?v=31",
      metaUrl:    "weights/anaconda-risky.meta.json?v=31",
    },
  ];
  const MAX_MOVES = 200;
  const EVAL_COLOR_POS = "#20c060";
  const EVAL_COLOR_NEG = "#e03030";
  const NEXT_GAME_DELAY_MS = 1500;     // pause between games in a series

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
  const gamesInput     = document.getElementById("games");
  const randomOpenInput = document.getElementById("random-opening");
  const seriesPanel    = document.getElementById("series-panel");
  const seriesProgress = document.getElementById("series-progress");
  const seriesTallyA   = document.getElementById("series-tally-a");
  const seriesTallyB   = document.getElementById("series-tally-b");
  const seriesLog      = document.getElementById("series-log");
  const tallyLabelA    = document.getElementById("tally-label-a");
  const tallyLabelB    = document.getElementById("tally-label-b");
  const seriesInline   = document.getElementById("series-inline");
  const inlineProgress = document.getElementById("inline-progress");
  const inlineTallyA   = document.getElementById("inline-tally-a");
  const inlineTallyB   = document.getElementById("inline-tally-b");
  const moveCountEl    = document.getElementById("move-count");
  const moveLogEl      = document.getElementById("move-log");
  const moveHistoryEl  = document.getElementById("move-history");
  const bannerEl       = document.getElementById("status-banner");
  const blackNameEl    = document.getElementById("black-name");
  const whiteNameEl    = document.getElementById("white-name");
  const blackOppSel    = document.getElementById("black-opp");
  const whiteOppSel    = document.getElementById("white-opp");

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
    lastAdjudication: null,
    playing: false,
    timer: null,
    blackSlotIdx: 0,
    whiteSlotIdx: 1,
    nets: { 0: null, 1: null },   // keyed by slot index, populated on load
    speedMs: parseInt(speedInput.value, 10),
    depth: parseInt(depthSelect.value, 10),
    randomOpening: parseInt(randomOpenInput.value, 10),
    // Series state. seriesTotal=1 → single game, no series UI shown.
    // For N>1, we track per-slot (not per-color) W/L/D so the tally is
    // attributed to the correct network even as colors swap each game.
    // seriesGameIdx is 0-indexed; on game k, slot 0 plays BLACK if k is even.
    seriesTotal: 1,
    seriesGameIdx: 0,
    seriesActive: false,
    // Pair that kicked off the current series — so mid-series color swaps
    // alternate WITHIN the selected pair instead of resetting to slot-0/slot-1.
    seriesBlackSlotIdx: 0,
    seriesWhiteSlotIdx: 1,
    // Tally is keyed by SLOTS index — 0,1,2. Expanded to cover all three
    // possible opponents; only the two currently playing accumulate scores.
    tally: { 0: { w: 0, l: 0, d: 0 }, 1: { w: 0, l: 0, d: 0 }, 2: { w: 0, l: 0, d: 0 } },
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
    updatePieceCounts();
  }

  function updatePieceCounts() {
    const sq = state.board.squares;
    let bMen = 0, bKings = 0, wMen = 0, wKings = 0;
    for (let i = 0; i < 32; i++) {
      const p = sq[i];
      if (p === C.BLACK_PIECE) bMen++;
      else if (p === C.BLACK_KING) bKings++;
      else if (p === C.WHITE_PIECE) wMen++;
      else if (p === C.WHITE_KING) wKings++;
    }
    document.getElementById("count-black").textContent = String(bMen + bKings);
    document.getElementById("kings-black").textContent = String(bKings);
    document.getElementById("count-white").textContent = String(wMen + wKings);
    document.getElementById("kings-white").textContent = String(wKings);
    document.getElementById("piece-row-black").classList.toggle("low", (bMen + bKings) <= 3);
    document.getElementById("piece-row-white").classList.toggle("low", (wMen + wKings) <= 3);
  }

  function updateLabels() {
    blackNameEl.textContent = blackSlot().label;
    whiteNameEl.textContent = whiteSlot().label;
    paintKingChip("black", blackNet());
    paintKingChip("white", whiteNet());
  }

  // Render each side's evolved K as a gold pill next to the model name.
  // Called from updateLabels so it stays correct across color swaps and
  // on initial load. Pulled from the network wrapper (AnacondaNetwork
  // exposes getKingWeight); `hidden` while the net isn't loaded yet so
  // the header doesn't flash "K=—".
  function paintKingChip(side, net) {
    const chip = document.getElementById(side + "-king-chip");
    if (!chip) return;
    if (!net) { chip.hidden = true; return; }
    chip.hidden = false;
    chip.textContent = "K=" + net.getKingWeight().toFixed(2);
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
  // Returns one of "BLACK", "WHITE", "DRAW", or null (game still going).
  // Sets state.finished and shows the banner as a side effect when ended.
  //
  // For draws, we also compute a "material adjudication" — *does not* change
  // the tournament score (a draw is still a draw per the rules), but annotates
  // obvious should-have-won positions so the user sees that the network
  // failed to convert. These networks have a well-documented draw plateau:
  // tanh-saturated evals treat "winning in 3" and "winning in 30" identically,
  // so overwhelming material advantages can shuffle into threefold repetition.
  function checkEnd() {
    const [over, winner] = C.isGameOver(state.board);
    if (over) {
      state.finished = true;
      if (winner === C.BLACK) {
        showBanner(`${blackSlot().label} wins (no legal moves for opponent)`);
        return "BLACK";
      }
      if (winner === C.WHITE) {
        showBanner(`${whiteSlot().label} wins (no legal moves for opponent)`);
        return "WHITE";
      }
      // No-moves-for-anyone draw (rare — usually only if both sides have no
      // pieces, which isn't reachable). Still adjudicate just in case.
      const adj = adjudicateDraw();
      state.lastAdjudication = adj;
      showBanner("Draw" + (adj ? ` — ${adj.desc}` : ""), "info");
      return "DRAW";
    }
    if (state.moveCount >= MAX_MOVES) {
      state.finished = true;
      const adj = adjudicateDraw();
      state.lastAdjudication = adj;
      showBanner(
        `Draw — move cap (${MAX_MOVES})` + (adj ? ` · ${adj.desc}` : ""),
        "info"
      );
      return "DRAW";
    }
    const key = C.stateKey(state.board);
    if (state.stateCounts[key] >= 3) {
      state.finished = true;
      const adj = adjudicateDraw();
      state.lastAdjudication = adj;
      showBanner(
        "Draw — threefold repetition" + (adj ? ` · ${adj.desc}` : ""),
        "info"
      );
      return "DRAW";
    }
    return null;
  }

  // Count material on the current board. King = 2 (matches the paper's K
  // convention); man = 1. Weighted score is what the "should have won"
  // verdict looks at; raw piece counts are what we display ("4m+2K").
  function materialCounts(board) {
    let bMen = 0, bK = 0, wMen = 0, wK = 0;
    for (let i = 0; i < 32; i++) {
      const p = board.squares[i];
      if (p === C.BLACK_PIECE) bMen++;
      else if (p === C.BLACK_KING) bK++;
      else if (p === C.WHITE_PIECE) wMen++;
      else if (p === C.WHITE_KING) wK++;
    }
    return { bMen, bK, wMen, wK, bTotal: bMen + bK, wTotal: wMen + wK };
  }

  // Return a { winner, label, desc, short } annotation when one side has a
  // clearly-winning material advantage at draw time, else null. Triggers
  // on a weighted material gap of ≥ 3 (king=2, man=1 — so 3K vs 1K hits
  // at gap=4, 6m vs 3m hits at gap=3, 7m+2K vs 1K hits at gap=10).
  //
  // Deliberately below that threshold:
  //   - 2K vs 1K is a theoretical DRAW for the weaker side (lone king can
  //     always reach the long-diagonal haven) — gap=2, not adjudicated.
  //   - 5m vs 4m is a small material edge that doesn't reliably convert
  //     — gap=1, not adjudicated.
  //   - 1K+2m vs 2m is ambiguous endgame material — gap=2, not adjudicated.
  function adjudicateDraw() {
    const m = materialCounts(state.board);
    const bw = m.bMen + 2 * m.bK;
    const ww = m.wMen + 2 * m.wK;
    const gap = Math.abs(bw - ww);
    if (gap < 3) return null;

    function fmt(men, k) {
      if (men === 0 && k === 0) return "0";
      if (men === 0) return `${k}K`;
      if (k === 0) return `${men}m`;
      return `${men}m+${k}K`;
    }
    const bDesc = fmt(m.bMen, m.bK);
    const wDesc = fmt(m.wMen, m.wK);

    if (bw > ww) {
      return {
        winner: "BLACK",
        label: blackSlot().label,
        desc: `${blackSlot().label} had ${bDesc} vs ${wDesc} but couldn't convert`,
        short: `${blackSlot().label} adv ${bDesc}/${wDesc}`,
      };
    }
    return {
      winner: "WHITE",
      label: whiteSlot().label,
      desc: `${whiteSlot().label} had ${wDesc} vs ${bDesc} but couldn't convert`,
      short: `${whiteSlot().label} adv ${wDesc}/${bDesc}`,
    };
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

    const ended = checkEnd();
    if (ended) { afterEnd(ended); return false; }
    return true;
  }

  // ---- Series ----
  function startGame() {
    state.board = C.makeBoard();
    state.moveCount = 0;
    state.lastFrom = -1;
    state.lastTo = -1;
    state.lastCaptured = [];
    state.stateCounts = Object.create(null);
    state.finished = false;
    state.lastAdjudication = null;
    moveCountEl.textContent = "0";
    moveHistoryEl.innerHTML = "";
    showBanner(null);
    // Optional random opening: play N random legal plies before the AIs
    // take over. Without this, deterministic agents from the start
    // position always produce the exact same game on repeat.
    if (state.randomOpening > 0) {
      applyRandomOpening(state.randomOpening);
    }
    render();
    updateEvals();
  }

  function applyRandomOpening(plies) {
    let lastFrom = -1, lastTo = -1, lastCaptured = [];
    for (let i = 0; i < plies; i++) {
      const moves = C.getLegalMoves(state.board);
      const [over] = C.isGameOver(state.board);
      if (!moves.length || over) {
        // Random walk dead-ended; bail and let the AI play from here.
        break;
      }
      const move = moves[Math.floor(Math.random() * moves.length)];
      lastFrom = move[0];
      lastTo = move[move.length - 1];
      lastCaptured = [];
      if (move.length > 2) {
        for (let j = 1; j < move.length; j += 2) lastCaptured.push(move[j]);
      }
      state.board = C.applyMove(state.board, move);
      const key = C.stateKey(state.board);
      state.stateCounts[key] = (state.stateCounts[key] || 0) + 1;
      // Don't increment moveCountEl yet — these aren't AI moves and we
      // don't want them to count against the 200-move cap (well, they
      // do, but we don't show them in the move counter so the user sees
      // "AI moves played" cleanly).
    }
    state.lastFrom = lastFrom;
    state.lastTo = lastTo;
    state.lastCaptured = lastCaptured;
  }
  function applySwapForGameIdx() {
    // Preserve the pair chosen when the series started; just alternate which
    // of the two plays Black each game so colors balance over the series.
    const evenIdx = (state.seriesGameIdx % 2 === 0);
    const a = state.seriesBlackSlotIdx;
    const b = state.seriesWhiteSlotIdx;
    state.blackSlotIdx = evenIdx ? a : b;
    state.whiteSlotIdx = evenIdx ? b : a;
    updateLabels();
  }
  function recordGameResult(outcome) {
    // outcome is "BLACK", "WHITE", or "DRAW". Translate to per-slot tally.
    // Draw tally/score is unchanged whether or not material adjudication
    // triggered — a draw is still a draw by the rules. The adjudication
    // note is informational only.
    const blackIdx = state.blackSlotIdx;
    const whiteIdx = state.whiteSlotIdx;
    let summary;
    if (outcome === "DRAW") {
      state.tally[blackIdx].d += 1;
      state.tally[whiteIdx].d += 1;
      const adj = state.lastAdjudication;
      summary = adj ? `draw (${adj.short})` : "draw";
    } else if (outcome === "BLACK") {
      state.tally[blackIdx].w += 1;
      state.tally[whiteIdx].l += 1;
      summary = `${blackSlot().label} wins (B)`;
    } else {
      state.tally[whiteIdx].w += 1;
      state.tally[blackIdx].l += 1;
      summary = `${whiteSlot().label} wins (W)`;
    }
    if (state.seriesActive) {
      const li = document.createElement("li");
      li.textContent = `Game ${state.seriesGameIdx + 1}: ${summary} (${state.moveCount} moves)`;
      seriesLog.appendChild(li);
      seriesLog.scrollTop = seriesLog.scrollHeight;
    }
    refreshTally();
  }
  function refreshTally() {
    function fmt(t) { return `${t.w}W ${t.l}L ${t.d}D`; }
    function fmtShort(t) { return `${t.w}-${t.l}-${t.d}`; }
    // Show the tally for the two slots actually in this series/game (pair
    // anchored by seriesBlack/WhiteSlotIdx when active, otherwise the
    // currently-selected pair).
    const aIdx = state.seriesActive ? state.seriesBlackSlotIdx : state.blackSlotIdx;
    const bIdx = state.seriesActive ? state.seriesWhiteSlotIdx : state.whiteSlotIdx;
    seriesTallyA.textContent = fmt(state.tally[aIdx]);
    seriesTallyB.textContent = fmt(state.tally[bIdx]);
    tallyLabelA.textContent = SLOTS[aIdx].label;
    tallyLabelB.textContent = SLOTS[bIdx].label;
    const progressStr =
      `${Math.min(state.seriesGameIdx + 1, state.seriesTotal)} / ${state.seriesTotal}`;
    seriesProgress.textContent = progressStr;
    // Compact inline live status, always visible during a series.
    inlineProgress.textContent = `Game ${progressStr}`;
    inlineTallyA.textContent = `${SLOTS[aIdx].label} ${fmtShort(state.tally[aIdx])}`;
    inlineTallyB.textContent = `${SLOTS[bIdx].label} ${fmtShort(state.tally[bIdx])}`;
  }
  function clearSeries() {
    state.seriesActive = false;
    state.seriesGameIdx = 0;
    state.seriesTotal = 1;
    state.tally = { 0: { w: 0, l: 0, d: 0 }, 1: { w: 0, l: 0, d: 0 }, 2: { w: 0, l: 0, d: 0 } };
    seriesLog.innerHTML = "";
    seriesPanel.hidden = true;
    seriesInline.hidden = true;
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
    startBtn.disabled = playing || (state.finished && !state.seriesActive);
    pauseBtn.disabled = !playing;
    stepBtn.disabled = playing || state.finished;
    // Can't swap mid-game or mid-series. Outside series, only between games.
    swapBtn.disabled = playing || state.moveCount > 0 || state.seriesActive;
    gamesInput.disabled = playing || state.seriesActive;
  }
  function afterEnd(outcome) {
    setPlaying(false);
    if (outcome) recordGameResult(outcome);
    if (state.seriesActive) {
      const next = state.seriesGameIdx + 1;
      if (next < state.seriesTotal) {
        state.seriesGameIdx = next;
        log(`Game ${next + 1} of ${state.seriesTotal} starting…`);
        // Quick pause so the user can read the result, then restart.
        state.timer = setTimeout(() => {
          applySwapForGameIdx();
          startGame();
          setPlaying(true);
          loop();
        }, NEXT_GAME_DELAY_MS);
      } else {
        // Series complete — reveal the big breakdown panel now.
        state.seriesActive = false;
        const a = state.tally[0], b = state.tally[1];
        log(`Series complete. ${SLOTS[0].label}: ${a.w}-${a.l}-${a.d}, ${SLOTS[1].label}: ${b.w}-${b.l}-${b.d}`);
        seriesPanel.hidden = false;
        gamesInput.disabled = false;
        startBtn.disabled = true;
        swapBtn.disabled = false;
      }
    } else {
      startBtn.disabled = true;
    }
  }

  // ---- Controls ----
  startBtn.addEventListener("click", () => {
    if (state.finished && !state.seriesActive) return;
    const total = Math.max(1, Math.min(50, parseInt(gamesInput.value, 10) || 1));
    state.seriesTotal = total;
    state.seriesActive = total > 1;
    if (state.seriesActive) {
      // Fresh series: reset tally + log + game index. Show the compact
      // inline live tally now; the big breakdown panel only appears when
      // the series finishes (per user preference — keep the side column
      // uncluttered during play).
      state.seriesGameIdx = 0;
      state.tally = { 0: { w: 0, l: 0, d: 0 }, 1: { w: 0, l: 0, d: 0 }, 2: { w: 0, l: 0, d: 0 } };
      // Freeze the pair that kicks off this series — applySwapForGameIdx
      // alternates colors between these two across the series without
      // ever swapping in a third slot.
      state.seriesBlackSlotIdx = state.blackSlotIdx;
      state.seriesWhiteSlotIdx = state.whiteSlotIdx;
      seriesLog.innerHTML = "";
      seriesPanel.hidden = true;
      seriesInline.hidden = false;
      refreshTally();
      applySwapForGameIdx();
      startGame();
      log(`Game 1 of ${total} starting…`);
    } else {
      seriesPanel.hidden = true;
      seriesInline.hidden = true;
      // Single game: don't reset board if it's already mid-game and was paused.
      if (state.finished) startGame();
    }
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
    clearSeries();
    startGame();
    log("Reset. Click Start.");
    setPlaying(false);
  });
  swapBtn.addEventListener("click", () => {
    if (!acceptOppChange()) return;
    [state.blackSlotIdx, state.whiteSlotIdx] = [state.whiteSlotIdx, state.blackSlotIdx];
    if (blackOppSel) blackOppSel.value = String(state.blackSlotIdx);
    if (whiteOppSel) whiteOppSel.value = String(state.whiteSlotIdx);
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
  randomOpenInput.addEventListener("input", () => {
    state.randomOpening = Math.max(0, Math.min(20,
      parseInt(randomOpenInput.value, 10) || 0));
  });
  // Update the Start button label so it's obvious that >1 means a series
  // (otherwise the Games input is too easy to miss).
  function refreshStartLabel() {
    if (state.playing) return;
    const n = Math.max(1, Math.min(50, parseInt(gamesInput.value, 10) || 1));
    startBtn.textContent = n > 1 ? `Start ${n}-game series` : "Start";
  }
  gamesInput.addEventListener("input", refreshStartLabel);
  refreshStartLabel();

  // Populate the opponent dropdowns from SLOTS. Disable mid-game to prevent
  // swapping out a network while it's thinking. Changing a dropdown while
  // idle mirrors the Swap button — it just reassigns the color's slot and
  // repaints labels/evals; no reload needed (all SLOTS networks preload on
  // boot).
  function populateOppDropdowns() {
    for (const sel of [blackOppSel, whiteOppSel]) {
      sel.innerHTML = "";
      SLOTS.forEach((s, i) => {
        const opt = document.createElement("option");
        opt.value = String(i);
        opt.textContent = s.label;
        sel.appendChild(opt);
      });
    }
    blackOppSel.value = String(state.blackSlotIdx);
    whiteOppSel.value = String(state.whiteSlotIdx);
  }
  // True if the change is acceptable (game not actively playing or in a
  // running series). When a previous game has finished or moves were
  // stepped, we silently reset the board/banner/series state so the new
  // matchup starts cleanly — without this, the eval panel labels could
  // disagree with the network actually playing each color.
  function acceptOppChange() {
    if (state.playing || state.seriesActive) return false;
    if (state.moveCount > 0 || state.finished) {
      clearSeries();
      startGame();
      showBanner(null);
      setPlaying(false);
      log("Opponent changed — board reset.");
    }
    return true;
  }
  blackOppSel.addEventListener("change", () => {
    if (!acceptOppChange()) {
      blackOppSel.value = String(state.blackSlotIdx);
      return;
    }
    state.blackSlotIdx = parseInt(blackOppSel.value, 10);
    updateLabels();
    updateEvals();
  });
  whiteOppSel.addEventListener("change", () => {
    if (!acceptOppChange()) {
      whiteOppSel.value = String(state.whiteSlotIdx);
      return;
    }
    state.whiteSlotIdx = parseInt(whiteOppSel.value, 10);
    updateLabels();
    updateEvals();
  });

  // ---- Boot ----
  async function loadAll() {
    log("Loading networks…");
    try {
      const weights = await Promise.all(SLOTS.map(s => A.loadWeightsFromUrl(s.weightsUrl)));
      SLOTS.forEach((_, i) => { state.nets[i] = A.makeNetwork(weights[i]); });
      populateOppDropdowns();
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
