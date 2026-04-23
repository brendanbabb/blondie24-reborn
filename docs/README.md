# Browser demo — developer notes

A self-contained, build-free static site that re-implements Chellapilla & Fogel's 1999
evolutionary-checkers architecture in vanilla JavaScript. The AI trains for ~2 seconds of
self-play on every AI turn, then picks its move with the freshly-evolved champion.

Serve locally:

```bash
cd docs && python -m http.server --bind 127.0.0.1 8765
# open http://localhost:8765/
```

For GitHub Pages: repo Settings → Pages → Source = `main` / `/docs`.

## File layout

```
docs/
├── index.html              single-page layout + inline "What am I looking at?" explainer
├── css/style.css           dark theme, 3-col grid, mobile fallback at <=900px and <=520px
├── js/checkers.js          32-square engine (legal moves, multi-jumps, king promotion,
│                             threefold-repetition hash)
├── js/network.js           1,743-weight MLP (32 → 40 → 10 → 1 tanh) with piece-diff bypass,
│                             evolvable king weight, fast Padé-tanh approximation,
│                             Schwefel self-adaptive EP mutation
├── js/minimax.js           negamax + alpha-beta, capture-first move ordering
├── js/render.js            main board canvas renderer + mini-board (for the self-play replay)
├── js/worker.js            Web Worker: EP loop, pop=6, games-per-ind=3, adaptive training
│                             depth (3 opening / 5 endgame), records all games each gen and
│                             picks decisive strong-vs-weak pairings for the replay
└── js/main.js              UI glue: click-to-move, forced-jump enforcement, training panel,
                              live leaderboard (currently hidden), eval bar, self-play
                              replay, network-architecture viz, move history,
                              Offer draw / Resign, 1-indexed square notation
```

## Worker message protocol

Main thread sends to worker:

| Message | Effect |
|---|---|
| `{type: "reset"}` | Re-initialize population at gen 0, posts a gen-0 gen event. |
| `{type: "resume"}` | Begin/resume evolving (schedules runOneGen via setTimeout 0). |
| `{type: "pause"}` | Stop evolving after the current gen. |
| `{type: "snapshot"}` | Reply with the current champion's weights + sigmas + fitness. |

Worker sends to main:

| Message | Effect |
|---|---|
| `{type: "ready", gen: 0}` | Sent once on load after `initPopulation`. |
| `{type: "gen", gen, leaderboard, meanFitness, maxFitness, sampleGameA, sampleGameB}` | Per-gen stats. `sampleGameA/B` are recorded self-play games (frames + B/W idx + ranks + winner) chosen from the tournament, preferring decisive + wide-rank-gap games. |
| `{type: "snapshot", gen, weights, sigmas, fitness}` | Response to a snapshot request. |
| `{type: "error", message, stack}` | Any exception inside the gen loop. Main surfaces this in the status banner so silent hangs become visible. |

`pause` and `snapshot` only take effect **between** gens (each gen is a synchronous chunk
of worker CPU), so deep-depth endgame gens can stall response by up to ~2 seconds. The
main thread's snapshot promise has an 8-second timeout to absorb that.

## Key constants you might want to tweak

In `docs/js/worker.js`:

```js
POP_SIZE               = 6     // networks per generation
GAMES_PER_INDIVIDUAL   = 3     // self-play games per network per gen
TRAIN_DEPTH_OPENING    = 3     // minimax depth while pieces > ENDGAME_PIECE_THRESHOLD
TRAIN_DEPTH_ENDGAME    = 5     // minimax depth when total pieces <= threshold
ENDGAME_PIECE_THRESHOLD= 6     // pieces-on-board cutoff for adaptive depth
MAX_GAME_MOVES         = 80    // self-play draw cap
WIN_SCORE  =  1.0              // paper fitness
DRAW_SCORE =  0.0
LOSS_SCORE = -2.0              // paper's asymmetric loss penalty
```

In `docs/js/main.js`:

```js
AI_DEPTH            = 4     // depth for the move the AI plays against you
TRAIN_BURST_MS      = 2000  // how long the worker evolves between AI moves
MIN_SEARCH_PAD_MS   = 200   // UX pad so the AI doesn't snap-move instantly
PRETRAIN_GENS       = 1     // warmup gens run when you click New game
MINI_STEP_MS        = 220   // ms per frame in the self-play replay animation
MINI_END_HOLD_MS    = 1800  // pause on the winner banner before alternating
```

## What's intentionally not there

- **No build step / bundler.** Three `<script>` tags, one Worker file. Edit and refresh.
- **No framework / dependencies.** Pure DOM + Canvas + Web Worker.
- **No network calls.** Everything runs client-side. You can serve over `file://` if your
  browser permits Web Workers from there (most don't, hence the `python -m http.server`
  line above).
- **No transposition table or iterative deepening** in the JS minimax. The Python repo has
  both; we didn't port them because at the demo's depth (3–5) the overhead of the
  bookkeeping doesn't pay back. If you crank training depth past 6, add TT first.
- **No make/unmake** — every `applyMove` allocates a new `Int8Array(32)`. That's the single
  biggest remaining speed-up if you care about gens/sec: reusing a board buffer + undo
  stack would probably 2–3× it.

## Relationship to the Python repo

This is deliberately a **1999-paper** rebuild. The Python repo has both the 1999 net and
the 2001 Anaconda (5,048-weight, sub-board preprocessor) architecture, plus three presets
(`paper-1999`, `paper-2001`, `paper-2001-strict`) and adaptive-depth curriculum training.
If you want the deeper / stronger version, train there and play via
`python -m play.human_vs_ai`. This demo is about **visible evolution in the browser**, not
peak strength.
