/*
 * play-strong.js
 *
 * The "play the fully evolved champion" page. Loads a frozen Anaconda
 * network from web/weights/anaconda-<slot>.bin on startup (default slot
 * is "paper-strict"; user can switch via the opponent dropdown) and plays
 * the human via minimax — no training, no worker, no evolution panels.
 *
 * Deliberately shares as little as possible with the live-evolve demo's
 * main.js so the two pages can evolve independently. Reuses only the
 * shared modules: checkers.js, render.js, anaconda-network.js, minimax.js.
 */

(function () {
  "use strict";

  const C = self.Checkers;
  const R = self.Render;
  const A = self.AnacondaNetwork;
  const M = self.Minimax;

  const AI_DEPTH = 6;
  const MIN_SEARCH_PAD_MS = 200;       // UX pad so fast endgames don't snap-move
  const THINKING_YIELD_MS = 20;        // paint the "AI is thinking…" banner before blocking
  const PLAN_PLIES = 6;                // # of mini-boards in the AI's-plan strip
  const PLAN_BOARD_PX = 72;            // each mini-board canvas size
  const PLAN_REVEAL_MS = 90;           // delay between progressive reveals
  const EVAL_COLOR_POS = "#20c060";    // bright green: AI sees you ahead
  const EVAL_COLOR_NEG = "#e03030";    // bright red: AI sees itself ahead

  // Opponent slots. Each entry is one weights bin + sidecar meta. Adding a
  // third slot is just appending another object here and shipping the files.
  const OPPONENTS = [
    {
      id: "paper-strict",
      label: "Paper-strict",
      weightsUrl: "weights/anaconda-paper-strict.bin?v=7",
      metaUrl:    "weights/anaconda-paper-strict.meta.json?v=7",
      available:  true,
    },
    {
      id: "enhanced",
      label: "Enhanced (not trained yet)",
      weightsUrl: "weights/anaconda-enhanced.bin?v=7",
      metaUrl:    "weights/anaconda-enhanced.meta.json?v=7",
      available:  false,  // flip to true once the .bin is exported
    },
  ];
  const DEFAULT_OPPONENT_ID = "paper-strict";
  function opponentById(id) {
    return OPPONENTS.find(o => o.id === id) || OPPONENTS[0];
  }

  // ---- DOM refs ----
  const canvas        = document.getElementById("board");
  const ctx           = canvas.getContext("2d");
  const moveLogEl     = document.getElementById("move-log");
  const moveHistoryEl = document.getElementById("move-history");
  const bannerEl      = document.getElementById("status-banner");
  const newGameBtn    = document.getElementById("new-game");
  const offerDrawBtn  = document.getElementById("offer-draw");
  const askResignBtn  = document.getElementById("ask-resign");
  const resignBtn     = document.getElementById("resign");
  const humanColorSel = document.getElementById("human-color");

  // ---- State ----
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
    aiNet: null,                 // filled once the selected slot's bin loads
    netReady: false,
    currentOpponentId: DEFAULT_OPPONENT_ID,
    aiThinking: false,
    aiMoveCount: 0,
    aiThinkMs: 0,                // cumulative search time
    gameActive: false,
  };

  // ---- Logging helpers ----
  function log(msg) {
    moveLogEl.textContent = msg;
  }
  function showBanner(kind, text) {
    if (!text) { bannerEl.hidden = true; bannerEl.className = ""; bannerEl.textContent = ""; return; }
    bannerEl.hidden = false;
    bannerEl.className = kind || "";
    bannerEl.textContent = text;
  }
  function humanSide() {
    return state.humanColor === "black" ? C.BLACK : C.WHITE;
  }

  // ---- Load weights + metadata ----
  async function loadOpponent(opponentId) {
    const opp = opponentById(opponentId || state.currentOpponentId);
    state.currentOpponentId = opp.id;
    state.netReady = false;
    state.aiNet = null;
    updateButtons();

    if (!opp.available) {
      document.getElementById("weights-label").textContent = "—";
      document.getElementById("training-label").textContent = "not trained yet";
      showBanner("",
        `"${opp.label.replace(/ \(.*\)$/, '')}" weights aren't available yet. ` +
        `Pick a different opponent from the dropdown.`
      );
      return;
    }

    document.getElementById("weights-label").textContent = "loading…";
    document.getElementById("training-label").textContent = "loading…";
    try {
      const [weights, metaResp] = await Promise.all([
        A.loadWeightsFromUrl(opp.weightsUrl),
        fetch(opp.metaUrl).then(r => r.ok ? r.json() : null).catch(() => null),
      ]);
      state.aiNet = A.makeNetwork(weights);
      state.netReady = true;
      document.getElementById("weights-label").textContent = String(weights.length);
      if (metaResp) {
        const gens = metaResp.generations;
        const src = metaResp.source || "unknown";
        document.getElementById("training-label").textContent =
          (gens != null ? `${gens} gens — ` : "") + src;
        if ((gens == null || gens === 0) && /random/i.test(src)) {
          showBanner("",
            "Playing against UNTRAINED (random-init) weights — this is a " +
            "pipeline stub, not the real Anaconda opponent. See README for " +
            "the training command."
          );
        } else {
          showBanner(null, null);
        }
      } else {
        document.getElementById("training-label").textContent = "(no metadata)";
      }
      updateButtons();
      render();
    } catch (err) {
      console.error(err);
      showBanner("", "Failed to load opponent weights: " + (err.message || err));
      document.getElementById("weights-label").textContent = "error";
    }
  }

  // ---- Board hints + forced-capture banner ----
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

  function computeHints() {
    const out = { legalTargets: [], captureTargets: [] };
    if (state.finished || !state.gameActive) return out;
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

  function updateForcedCaptureBanner() {
    if (state.finished || state.aiThinking || !state.gameActive) { showBanner(null, null); return; }
    if (state.board.currentPlayer !== humanSide()) { showBanner(null, null); return; }
    const moves = C.getLegalMoves(state.board);
    if (moves.length === 0) { showBanner(null, null); return; }
    const forced = moves[0].length >= 3;
    if (!forced) { showBanner(null, null); return; }
    if (state.pendingJumpPath) {
      showBanner("", "You must continue the jump — click the next green dot.");
      return;
    }
    const pieces = new Set();
    for (const m of moves) pieces.add(m[0]);
    const count = pieces.size;
    const list = Array.from(pieces).sort((a, b) => a - b).map(sq => sq + 1).join(", ");
    showBanner("", `Capture is mandatory. ${count} of your pieces can jump (square${count > 1 ? "s" : ""}: ${list}).`);
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

  // ---- Click-to-move ----
  function onCanvasClick(ev) {
    if (!state.gameActive || state.finished || state.aiThinking) return;
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
      const terminator = matches.find(m => arrayEq(m, newPath)) || matches[0];
      commitMove(terminator, "You");
      return;
    }
    state.pendingJumpPath = newPath;
    render();
  }

  // ---- Committing a move (human or AI) ----
  function describeMove(move) {
    if (move.length === 2) return `${move[0] + 1} → ${move[1] + 1}`;
    const stops = [move[0] + 1];
    for (let i = 2; i < move.length; i += 2) stops.push(move[i] + 1);
    return stops.join(" → ");
  }

  function commitMove(move, actor) {
    state.lastFrom = move[0];
    state.lastTo = move[move.length - 1];
    state.lastCaptured = [];
    if (move.length > 2) {
      for (let i = 1; i < move.length; i += 2) state.lastCaptured.push(move[i]);
    }
    state.board = C.applyMove(state.board, move);
    state.selectedFrom = -1;
    state.pendingJumpPath = null;

    const key = C.stateKey(state.board);
    state.stateCounts[key] = (state.stateCounts[key] || 0) + 1;

    appendHistory(actor, move);
    render();
    updatePosEval();
    updatePieceCounts();
    updateButtons();
    if (checkEnd()) { updateButtons(); return; }

    maybeStartAiTurn();
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

  // ---- AI turn ----
  function maybeStartAiTurn() {
    if (state.finished) return;
    if (!state.gameActive) return;
    if (state.board.currentPlayer === humanSide()) return;
    if (!state.netReady) {
      log("(waiting for opponent weights to finish loading…)");
      return;
    }

    state.aiThinking = true;
    log(`AI is thinking at depth ${AI_DEPTH}…`);
    updateButtons();

    // Yield one paint so the banner updates before we block the main thread
    // inside pickMove. setTimeout + a short pad make the move feel deliberate
    // for fast endgames without artificially dragging out the user-facing
    // latency of deep searches.
    setTimeout(() => {
      const searchStart = performance.now();
      const result = M.pickMove(state.board, AI_DEPTH, state.aiNet);
      const searchMs = performance.now() - searchStart;
      state.aiThinkMs += searchMs;
      state.aiMoveCount += 1;
      updateThinkDisplay();
      updateSearchEffort(result);
      // Render the AI's predicted line into the mini-board strip. This is
      // the planning view — done before commit so the user sees what the
      // AI thought *would* happen vs. what actually plays out.
      renderAiPlan(state.board, result.pv || []);
      const pad = Math.max(0, MIN_SEARCH_PAD_MS - searchMs);
      setTimeout(() => {
        state.aiThinking = false;
        if (!result.move) {
          state.finished = true;
          log("AI has no legal moves; you win!");
          updateButtons();
          return;
        }
        commitMove(result.move, "AI");
      }, pad);
    }, THINKING_YIELD_MS);
  }

  // ---- Search-effort stat ----
  function updateSearchEffort(result) {
    const el = document.getElementById("search-effort");
    if (!el) return;
    if (!result || result.nodesEvaluated == null) {
      el.textContent = "—";
      return;
    }
    const ev = result.nodesEvaluated;
    const pr = result.nodesPruned;
    el.textContent =
      ev.toLocaleString() + " evals · " + pr.toLocaleString() + " α-β cuts";
  }

  // ---- AI's plan (principal variation) mini-board strip ----
  function renderAiPlan(rootBoard, pv) {
    const row = document.getElementById("plan-row");
    if (!row) return;
    row.innerHTML = "";

    if (!pv || pv.length === 0) {
      const empty = document.createElement("div");
      empty.className = "plan-empty";
      empty.textContent = "(no predicted line — search returned no PV)";
      row.appendChild(empty);
      return;
    }

    // Walk the line, rendering one mini-board per ply. Score is the network's
    // raw eval at each successor position, shown from that side's perspective.
    let board = C.cloneBoard(rootBoard);
    const cells = [];
    const limit = Math.min(PLAN_PLIES, pv.length);

    for (let i = 0; i < limit; i++) {
      const move = pv[i];
      // Side that's ABOUT to move at this snapshot (the one playing this ply).
      const mover = board.currentPlayer;
      const aiSide = -humanSide();
      const isAi = (mover === aiSide);
      // Grammar: "You play" (2nd person) vs "AI plays" (3rd person).
      const actorLabel = isAi ? "AI plays" : "You play";
      const movedTo = move[move.length - 1];

      board = C.applyMove(board, move);
      const evalRaw = state.aiNet ? state.aiNet.forward(board) : 0;
      // After the move, currentPlayer flipped. The eval is from the *new*
      // side-to-move's perspective. Display from the mover's perspective so
      // the sign tells a consistent story ("the side that just moved sees X").
      const fromMover = -evalRaw;

      const cell = document.createElement("div");
      cell.className = "plan-cell";

      const plyNum = document.createElement("div");
      plyNum.className = "ply-num";
      plyNum.textContent = "ply " + (i + 1);
      cell.appendChild(plyNum);

      const cv = document.createElement("canvas");
      cv.width = PLAN_BOARD_PX;
      cv.height = PLAN_BOARD_PX;
      const cvCtx = cv.getContext("2d");
      // highlightSq draws a blue ring around the piece that just moved
      // — makes the move visible at the small mini-board scale. pieceScale
      // is shrunk from the default 0.38 so pieces don't crowd the cell at
      // 72px boards. Edge width also dialed down so the outline reads as
      // a refined detail rather than dominating the small piece.
      R.drawMini(cvCtx, board.squares, {
        size: PLAN_BOARD_PX,
        highlightSq: movedTo,
        pieceScale: 0.30,
        pieceEdgeWidth: 1.1,
      });
      cell.appendChild(cv);

      const actor = document.createElement("div");
      actor.className = "ply-actor";
      actor.textContent = actorLabel;
      cell.appendChild(actor);

      const score = document.createElement("div");
      score.className = "ply-score";
      score.textContent =
        (fromMover >= 0 ? "+" : "") + fromMover.toFixed(2);
      cell.appendChild(score);

      row.appendChild(cell);
      cells.push(cell);
    }

    // Progressive reveal: each cell pops in PLAN_REVEAL_MS later than the
    // previous one. Total animation time is PLAN_REVEAL_MS × N ≈ 540 ms for
    // a depth-6 line, which feels like the AI is "thinking forward" without
    // dragging out the actual move.
    cells.forEach((cell, i) => {
      setTimeout(() => cell.classList.add("show"), i * PLAN_REVEAL_MS);
    });
  }

  function clearAiPlan() {
    const row = document.getElementById("plan-row");
    if (!row) return;
    row.innerHTML = "";
    const empty = document.createElement("div");
    empty.className = "plan-empty";
    empty.textContent = "(plan appears after the AI's first move)";
    row.appendChild(empty);
  }

  // ---- Side-panel updates ----
  function updatePieceCounts() {
    const squares = state.board.squares;
    let bMen = 0, bKings = 0, wMen = 0, wKings = 0;
    for (let i = 0; i < 32; i++) {
      const p = squares[i];
      if (p === C.BLACK_PIECE) bMen++;
      else if (p === C.BLACK_KING) bKings++;
      else if (p === C.WHITE_PIECE) wMen++;
      else if (p === C.WHITE_KING) wKings++;
    }
    const you = humanSide() === C.BLACK ? [bMen + bKings, bKings] : [wMen + wKings, wKings];
    const ai  = humanSide() === C.BLACK ? [wMen + wKings, wKings] : [bMen + bKings, bKings];
    document.getElementById("count-you").textContent = String(you[0]);
    document.getElementById("kings-you").textContent = String(you[1]);
    document.getElementById("count-ai").textContent  = String(ai[0]);
    document.getElementById("kings-ai").textContent  = String(ai[1]);
    const row = document.getElementById("piece-row-you");
    const rowAi = document.getElementById("piece-row-ai");
    row.classList.toggle("low", you[0] <= 3);
    rowAi.classList.toggle("low", ai[0] <= 3);
  }

  function updatePosEval() {
    if (!state.netReady) return;
    // network.forward gives the score from current-side-to-move's perspective.
    // Flip to show from the human's perspective so +ve always means "you ahead".
    const rawScore = state.aiNet.forward(state.board);
    const fromCurrent = rawScore;
    const fromHuman = (state.board.currentPlayer === humanSide())
      ? fromCurrent
      : -fromCurrent;
    const pct = Math.max(-1, Math.min(1, fromHuman));

    // Fill grows from the center (50% mark) outward to the eval position,
    // colored vividly by sign so the direction is unmistakable.
    const fill = document.getElementById("eval-bar-fill");
    const half = Math.abs(pct) * 50;        // % width on the appropriate side
    if (pct >= 0) {
      fill.style.left  = "50%";
      fill.style.width = half + "%";
      fill.style.background = EVAL_COLOR_POS;
      fill.style.color = EVAL_COLOR_POS;    // drives the box-shadow glow
    } else {
      fill.style.left  = (50 - half) + "%";
      fill.style.width = half + "%";
      fill.style.background = EVAL_COLOR_NEG;
      fill.style.color = EVAL_COLOR_NEG;
    }
    document.getElementById("eval-number").textContent =
      (pct >= 0 ? "+" : "") + pct.toFixed(3);
    let verdict;
    if (Math.abs(pct) < 0.08) verdict = "roughly even";
    else if (pct >=  0.30)    verdict = "you ahead";
    else if (pct >=  0.08)    verdict = "slight edge to you";
    else if (pct <= -0.30)    verdict = "AI ahead";
    else                       verdict = "slight edge to AI";
    document.getElementById("eval-verdict").textContent = verdict;
  }

  function updateThinkDisplay() {
    document.getElementById("ai-move-count").textContent = String(state.aiMoveCount);
    document.getElementById("think-time").textContent =
      (state.aiThinkMs / 1000).toFixed(1) + " s total";
  }

  // ---- Move history ----
  function appendHistory(actor, move) {
    const li = document.createElement("li");
    li.textContent = `${actor}: ${describeMove(move)}`;
    li.className = (actor === "You") ? "mh-you" : "mh-ai";
    moveHistoryEl.appendChild(li);
    moveHistoryEl.scrollTop = moveHistoryEl.scrollHeight;
  }

  // ---- Buttons ----
  function onOfferDraw() {
    if (state.finished || state.aiThinking || !state.netReady) return;
    if (state.board.currentPlayer !== humanSide()) return;
    const human = state.aiNet.forward(state.board);
    const ai = -human;
    const ACCEPT_THRESHOLD = 0.30;
    if (ai <= ACCEPT_THRESHOLD) {
      state.finished = true;
      log(`Draw agreed. AI eval from its side: ${ai >= 0 ? "+" : ""}${ai.toFixed(2)}.`);
      updateButtons();
    } else {
      log(`AI declines the draw — it evaluates its position at +${ai.toFixed(2)}.`);
    }
  }

  function onAskResign() {
    if (state.finished || state.aiThinking || !state.netReady) return;
    if (state.board.currentPlayer !== humanSide()) return;
    const ai = -state.aiNet.forward(state.board);
    const RESIGN_THRESHOLD = -0.65;
    if (ai <= RESIGN_THRESHOLD) {
      state.finished = true;
      log(`AI resigns — it evaluates its position at ${ai.toFixed(2)}. You win.`);
      updateButtons();
    } else {
      log(`AI plays on — it evaluates its position at ${ai >= 0 ? "+" : ""}${ai.toFixed(2)}.`);
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
    const yourTurn = state.gameActive && !state.finished && !state.aiThinking
                     && state.board.currentPlayer === humanSide();
    offerDrawBtn.disabled = !yourTurn || !state.netReady;
    askResignBtn.disabled = !yourTurn || !state.netReady;
    resignBtn.disabled    = !yourTurn;
  }

  // ---- New game ----
  function resetGame(autoStart) {
    state.board = C.makeBoard();
    state.selectedFrom = -1;
    state.pendingJumpPath = null;
    state.lastFrom = -1;
    state.lastTo = -1;
    state.lastCaptured = [];
    state.stateCounts = Object.create(null);
    state.finished = false;
    state.humanColor = humanColorSel.value;
    state.gameActive = !!autoStart;
    state.aiThinking = false;
    state.aiMoveCount = 0;
    state.aiThinkMs = 0;
    updateThinkDisplay();
    moveHistoryEl.innerHTML = "";
    log(state.gameActive
      ? "New game — your move."
      : "Pick your color, then click New game to start.");

    render();
    updatePieceCounts();
    updatePosEval();
    updateButtons();
    clearAiPlan();
    updateSearchEffort(null);

    if (state.gameActive) maybeStartAiTurn();
  }

  canvas.addEventListener("click", onCanvasClick);
  newGameBtn.addEventListener("click", () => resetGame(true));
  offerDrawBtn.addEventListener("click", onOfferDraw);
  askResignBtn.addEventListener("click", onAskResign);
  resignBtn.addEventListener("click", onResign);
  humanColorSel.addEventListener("change", () => resetGame(false));

  // Populate opponent dropdown.
  const opponentSel = document.getElementById("opponent-select");
  for (const opp of OPPONENTS) {
    const opt = document.createElement("option");
    opt.value = opp.id;
    opt.textContent = opp.label;
    if (!opp.available) opt.disabled = true;
    opponentSel.appendChild(opt);
  }
  opponentSel.value = DEFAULT_OPPONENT_ID;
  opponentSel.addEventListener("change", () => {
    loadOpponent(opponentSel.value);
    resetGame(false);
  });

  // Boot.
  resetGame(false);
  loadOpponent(DEFAULT_OPPONENT_ID);
})();
