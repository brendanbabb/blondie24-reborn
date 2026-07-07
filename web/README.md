# Browser demo — developer notes

A self-contained, build-free static site that re-implements Chellapilla & Fogel's 1999
evolutionary-checkers architecture in vanilla JavaScript. The AI trains for ~3 seconds of
self-play on every AI turn, then picks its move with the freshly-evolved champion.

## Relationship to the Python codebase (containment rules)

This `web/` tree is treated as **independent code** from the Python training stack
(`checkers/ evolution/ neural/ search/ training/`). The evolve-as-it-plays page
(`index.html` + `js/main.js` + `js/evolution.js` + `js/game-worker.js` +
`js/network.js` + `js/minimax.js` + `js/checkers.js`) does its own in-browser EP loop — it is **not** expected to stay
bit-compatible with the Python engine, and changes on either side do not need to be
mirrored.

The **one** deliberate bridge is the frozen-opponent page (`play-strong.html`):
`js/anaconda-network.js` is a bit-exact port of `neural/anaconda_network.py` (inference
only) so it can faithfully play weights trained offline in Python. That parity is
guarded by `node web/test_anaconda.js` (|Δ| < 1e-6 on five fixed positions). Touch
either side of that port and run the test.

## Serve locally

```bash
cd web && python -m http.server --bind 127.0.0.1 8765
# open http://localhost:8765/
```

For GitHub Pages: repo Settings → Pages → Source = **GitHub Actions**. The workflow at
`.github/workflows/pages.yml` (one level up) uploads this folder on every push to `main`.

## File layout

```
web/
├── index.html              single-page layout + inline "What am I looking at?" explainer
├── css/style.css           dark theme, 3-col grid, mobile fallback at <=900px and <=520px
├── js/checkers.js          32-square engine (legal moves, multi-jumps, king promotion,
│                             threefold-repetition hash)
├── js/network.js           1,743-weight MLP (32 → 40 → 10 → 1 tanh) with piece-diff bypass,
│                             evolvable king weight (init 2.0, clamped [1,3]), fast
│                             Padé-tanh approximation, single-τ self-adaptive EP mutation
├── js/minimax.js           negamax + alpha-beta with make/unmake, a per-search Zobrist
│                             transposition table, and iterative deepening + TT-move-first
│                             ordering; capture-length move ordering as fallback
├── js/render.js            main board canvas renderer + mini-board (for the self-play replay)
├── js/evolution.js         main-thread EP coordinator: pop=6, games-per-ind=3 pairings,
│                             fitness, half-keep-mutate selection; pull-dispatches the 18
│                             games/gen to a pool of 4-8 game workers; records all games
│                             and picks decisive strong-vs-weak pairings for the replay
├── js/game-worker.js       stateless Web Worker: plays ONE flat depth-4 self-play game
│                             per message (paper-faithful), caches Network wrappers per gen
└── js/main.js              UI glue: click-to-move, forced-jump enforcement, training panel,
                              live leaderboard (currently hidden), eval bar, self-play
                              replay, network-architecture viz, move history,
                              Offer draw / Ask AI to resign / Resign, 1-indexed
                              square notation
```

## Evolution architecture

The EP loop (pairings, fitness, selection, mutation — microseconds per gen) runs on the
**main thread** in `js/evolution.js`. The expensive part — 18 independent self-play games
per generation — fans out to a pool of 4-8 stateless **game workers** (`js/game-worker.js`,
pool size = `clamp(hardwareConcurrency - 2, 4, 8)`).

Dispatch is **pull-based**: each worker holds one game at a time and is handed the next
from the queue when its result arrives, so one 80-move shuffle-draw only ever strands one
worker. Generations are a barrier (gen N+1's population needs gen N's full ranking); a
`reset()` mid-gen bumps an epoch counter and in-flight results are dropped by tag.

Main-thread API (`Evolution.create({onGen, onError})`):

| Call | Effect |
|---|---|
| `evo.reset()` | Re-initialize population at gen 0, emits a gen-0 gen event (async). |
| `evo.resume()` | Run generations back-to-back until `pause()`. |
| `evo.pause()` | Stop after the in-flight generation completes. |
| `evo.snapshot()` | Promise of the champion's `{gen, weights, sigmas, fitness}`, resolved at the next gen boundary. |

`onGen` fires per completed generation with
`{gen, leaderboard, meanFitness, maxFitness, sampleGameA, sampleGameB, genMs}`.
`sampleGameA/B` are recorded self-play games (frames + B/W idx + ranks + winner) chosen
from the tournament, preferring decisive + wide-rank-gap games. Worker↔coordinator
messages (`weights` broadcast per gen, `play`/`result` per game) are internal to
evolution.js/game-worker.js; frames come back as transferables.

`pause` and `snapshot` take effect at the next **generation boundary** (the in-flight
games must finish so the population stays consistent). The main thread's snapshot promise
keeps an 8-second safety timeout in case a game worker dies mid-gen.

The coordinator/pool protocol is guarded by `node web/pool_test.js` (fake-Worker shim
around the real game-worker.js): per-gen W/L/D accounting, pause/snapshot at the
boundary, mid-gen reset staleness, and re-entrant pause+resume from inside onGen (the
warmup flow) — the last one wedged the pool before the genInFlight guard existed.

## Key constants you might want to tweak

In `web/js/evolution.js`:

```js
POP_SIZE             = 6     // networks per generation
GAMES_PER_INDIVIDUAL = 3     // self-play games per network per gen
WIN_SCORE  =  1.0            // paper fitness
DRAW_SCORE =  0.0
LOSS_SCORE = -2.0            // paper's asymmetric loss penalty
POOL_MIN / POOL_MAX  = 4 / 8 // game-worker pool bounds
```

In `web/js/game-worker.js`:

```js
TRAIN_SEARCH_DEPTH   = 4     // paper-faithful flat depth for self-play
MAX_GAME_MOVES       = 80    // self-play draw cap
```

In `web/js/main.js`:

```js
AI_DEPTH            = 4     // depth for the move the AI plays against you
TRAIN_BURST_MS      = 2500  // how long evolution runs per AI turn
MIN_SEARCH_PAD_MS   = 200   // UX pad so the AI doesn't snap-move instantly
PRETRAIN_GENS       = 5     // warmup gens run when you click New game
MINI_STEP_MS        = 220   // ms per frame in the self-play replay animation
MINI_END_HOLD_MS    = 1800  // pause on the winner banner before alternating
```

### Paper-faithfulness notes

The demo deliberately stays close to Chellapilla & Fogel 1999:

- **Flat self-play depth** (4 ply) matching the paper — no "search deeper in the endgame"
  injection, even though that heuristic would reduce shuffle-draws.
- **Piece-difference bypass weight** is initialized from N(0, σ), same as every other weight
  — not seeded to a useful positive value. Gen-0 networks play chaotically; selection finds
  the useful bypass value over 10–50 generations.
- **Asymmetric scoring** (+1 / 0 / −2), **random pairing**, **no crossover**, **single-τ
  self-adaptive σ mutation** (σ′ = σ·exp(τ·Nᵢ), the paper's rule — no correlated global
  noise factor), and **king weight K initialized at 2.0 and clamped to [1, 3]** all match
  the paper.
- **Differences from the paper** for browser-practicality: population of 6 (paper: 15),
  3 games per individual (paper: 5), and a 3-second per-turn evolution budget instead of
  the paper's overnight-per-generation runs. None of these change the algorithm, only its
  scale.

## What's intentionally not there

- **No build step / bundler.** A handful of `<script>` tags, one Worker file. Edit and refresh.
- **No framework / dependencies.** Pure DOM + Canvas + Web Workers. No SharedArrayBuffer
  (weights are ~7 KB, structured clone is trivial), so no COOP/COEP headers needed —
  deploys on GitHub Pages as-is.
- **No network calls.** Everything runs client-side. You can serve over `file://` if your
  browser permits Web Workers from there (most don't, hence the `python -m http.server`
  line above).
- **No hall-of-fame / coevolution.** The population is the only source of opponents. The
  main Python repo's writeup flags this as the likely cause of the draw-plateau failure mode
  at deeper self-play; the demo is shallow enough not to hit it, but adding a frozen
  strongest-ever anchor opponent would be the natural next lever.
- **No evolution during the human's turn.** Deliberate: gens only accumulate in the
  per-AI-turn 3-second bursts, so the gen counter reflects training the AI "earned"
  during play, not how long the human stared at the board.

## Relationship to the Python repo

This is deliberately a **1999-paper** rebuild. The Python repo has both the 1999 net and
the 2001 Anaconda (5,048-weight, sub-board preprocessor) architecture, plus three presets
(`paper-1999`, `paper-2001`, `paper-2001-strict`) and adaptive-depth curriculum training.
If you want the deeper / stronger version, train there and play via
`python -m play.human_vs_ai`. This demo is about **visible evolution in the browser**, not
peak strength.
