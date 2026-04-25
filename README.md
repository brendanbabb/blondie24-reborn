# Blondie24 Reborn

**A modern Python reimplementation of Chellapilla & Fogel's evolutionary checkers work**

A feedforward neural network with no hand-crafted evaluation, no opening book, no endgame database,
and no backprop — trained entirely by self-play under an evolutionary strategy.

## Try it in your browser

A GitHub-Pages-deployable demo lives under [`web/`](./web). It's a pure-static, vanilla-JS
re-implementation of the 1999 architecture with a twist: the AI **evolves while you play it**.
Every time the AI's turn arrives it runs ~2 seconds of self-play evolution in a Web Worker, then
picks a move with the freshly-evolved champion network. Over a 30-move game the population runs
through ~100–200 generations — a real slice of the paper's training curve, compressed into a
single play session.

No install. Just serve the folder:

```bash
cd web && python -m http.server --bind 127.0.0.1 8765
```

Or publish via GitHub Pages and share the URL. This repo ships a workflow at
[`.github/workflows/pages.yml`](./.github/workflows/pages.yml) that uploads `web/` as the
Pages artifact on every push to `main`; enable it by setting Settings → Pages → Source =
**GitHub Actions**. The demo code deliberately has no build step, no dependencies, and no
network calls. See [`web/index.html`](./web/index.html) for the explanation that renders
in-page.

### Companion page: play the fully-evolved Anaconda

[`web/play-strong.html`](./web/play-strong.html) is a second browser page that plays against
a **frozen, pre-trained** 2001 Anaconda network (5,048 weights, 91 sub-board filters +
92→40→10→1 MLP) loaded from `web/weights/anaconda-paper-strict.bin` — or the Enhanced
slot, picked via the dropdown in the page header. Unlike the live-evolve page, no training
happens while you play — it's the finished product of an offline training run. The AI
searches at depth 6 on every move. Pure static — same GitHub Pages deployment, no extra setup.

The repo ships real trained weights in both slots, so the page is playable out of the box.
To retrain either opponent yourself, see the
[training section below](#training-the-anaconda-opponent). Each bin has a sidecar
`.meta.json` with the exact checkpoint, generation count, and training recipe it came from.

The JS Anaconda inference (`web/js/anaconda-network.js`) is a pure-JavaScript port of
[`neural/anaconda_network.py`](./neural/anaconda_network.py). It's cross-checked against the
Python forward pass on five fixed positions to ~1e-8 agreement (see
[`web/test_anaconda.js`](./web/test_anaconda.js)) so the browser plays the **same** network
the Python side trained.

What you'll see:

- **Playable board** with circular checker pieces, gold king markers, last-move highlights,
  green legal-move dots and red capture-landing dots. Squares are labeled 1–32 (standard
  checkers notation). Forced captures surface a red banner above the board listing which of
  your pieces can jump.
- **Controls row** (New game · Offer draw · Ask AI to resign · Resign · color picker) sits
  right above the board. The AI doesn't start training until you click **New game**, so you
  can pick your color first. Offered draws are evaluated by the current champion network: if
  its own eval isn't > +0.30 it accepts. **Ask AI to resign** is the mirror: the AI concedes
  only if its own-side eval is ≤ −0.65 (clearly losing); otherwise it plays on.
- **Pieces panel** — large tabular-figures count of your remaining pieces vs. the AI's, kings
  tracked separately. Turns orange when either side drops to ≤3 pieces.
- **Training stats** — generation counter, gens-this-turn delta, gens/sec, cumulative AI
  think-time (broken down into "evolving" vs. "searching"), AI-moves-this-game count.
- **Board evaluation bar** — the AI's score of the current position, flipped to be shown from
  **your** perspective (so + = you're ahead). Red → Blue gradient with a verdict line
  ("slight edge to you", "AI ahead", etc.).
- **Self-play replay** — a mini-board animates one of the AI's training games from the last
  generation, alternating between two different match-ups ("Game A" ↔ "Game B"). When
  possible we pick decisive games (not draws) that pit a top-ranked network against a
  lower-ranked one — more instructive than mid-pack matchups. Caption reads
  `B:#3 r1 vs W:#0 r5 → Black wins`.
- **Network architecture** — a canvas that redraws on every AI move. The 1,743-weight
  champion net laid out as 4 columns of nodes (32 → 40 → 10 → 1), with the 40 strongest
  connections per layer rendered as colored lines (red negative, blue positive, thickness
  ∝ magnitude). The piece-difference bypass is drawn as a soft curve from the input centroid
  to the output. Watch it shift as evolution reshapes the champion from move to move.
- **Move history** — a scrollable numbered log of every move this game, colored by actor
  (your moves one hue, AI's another).

### How the AI trains (the 2-second cycle)

Every AI turn, for ~2 seconds:

1. Population of **6 networks** runs a self-play tournament — each plays **3 games** against
   random opponents from the group.
2. Training search: **flat depth 4** — same as game-play, same as the paper. No hand-crafted
   "search deeper in the endgame" boost; Fogel didn't use one either.
3. Scoring is paper-faithful: **+1 win / 0 draw / −2 loss**.
4. Top 3 networks survive; bottom 3 are replaced by **Gaussian self-adaptive EP mutations**
   of the survivors (Schwefel rule for σ, no crossover — pure evolutionary programming).

Network weights — including the piece-difference bypass — are initialized from **N(0, σ)**
with σ=0.05, exactly as in the paper. No hand-seeded "material bias" to bootstrap early play:
gen-0 networks play chaotically and selection finds the useful weights on its own. After
clicking New game the worker runs **3 warmup generations** before the first move so the AI
at least isn't literally random on move 1.

At 2 seconds per AI turn at depth 4, the browser typically fits **~4–8 generations** per turn
(the JS search uses make/unmake, a per-search Zobrist transposition table, and iterative
deepening with TT-move-first ordering — all correctness-preserving speed-ups; see
[`web/README.md`](./web/README.md) for details). Across ~30 AI moves in a game, the
population runs through **~100–240 generations** total. The paper reached Class-A play at
~250 generations, so one casual play session covers a meaningful slice of the A-class learning
curve.

When the AI's 2 seconds are up, the current top-ranked network runs **depth-4 minimax** from
the board you see and plays its move. Evolution pauses while you think, so the opponent you
face at any point is frozen — it only changes between AI turns.

> **A note on the name.** "Blondie24" is the screen name used on Zone.com by the 2001 "Anaconda"
> system (Chellapilla & Fogel 2001), which added a spatial-preprocessing layer on top of the 1999
> network and reached expert-level (~2045 USCF) play. Both architectures are implemented here —
> the **1999 precursor** (32→40→10→1 MLP with piece-diff bypass, 1,743 evolvable weights) and the
> **2001 Anaconda** (91 sub-board filters → 92→40→10→1 with piece-diff bypass, 5,048 evolvable
> weights) — trained by pure EP with a fitness signal of tournament wins/draws/losses.

## Run the paper

Reproduce the 1999 config:

```bash
python -m training.train --preset paper-1999 --generations 250
```

Reproduce the 2001 Anaconda config:

```bash
python -m training.train --preset paper-2001 --generations 250
```

That's pop=15, 5 games per individual per generation vs. randomly chosen opponents, fixed 4-ply
minimax, +1/0/−2 scoring, initial σ = 0.05, random pairing, no σ ceiling. Explicit flags still win,
so `--preset paper-1999 --depth 6` keeps depth 6 while applying the rest.

### Strict paper reproduction (for time comparisons)

The `paper-2001` preset above matches the paper's **hyperparameters** but uses two engine
accelerations not in the original: (1) half-keep-mutate selection (top 50% spawn offspring to
refill) instead of the paper's (μ+μ) scheme; and (2) quiescence-extended alpha-beta instead
of plain depth-4 alpha-beta. To turn **both** off and run a true paper copy:

```bash
python -m training.train --preset paper-2001-strict --generations 850 --workers 20 --device cpu
```

Individual flags if you want to isolate one deviation at a time:

- `--selection-scheme mu_plus_mu` — every parent spawns one offspring, the 2μ pool is
  evaluated, top μ survive. Paper-faithful (Chellapilla & Fogel 1999/2001). Roughly doubles
  tournament compute per generation.
- `--no-quiescence` — disable the quiescence extension in the CPU JIT engines. Shorter
  per-position search, smaller JIT warmup.

Head-to-head, 850 generations, 20 CPU workers on a 24-core box:

| | `paper-2001` | `paper-2001-strict` |
|---|---|---|
| Wall time (850 gens) | 39.9 min | 47.1 min (+18%) |
| Gens 2+ avg | 2.70 s | 3.25 s |
| JIT warmup (gen 1) | 100.5 s | 62.9 s |
| Late mean pop fitness | −1.24 | **−0.72** |
| Late best losses/gen | 0.11 | **0.04** |

The (μ+μ) selection keeps strong parents in the evaluation pool alongside offspring, which
raises the population's mean fitness and tightens elite consistency. The quiescence-off
per-search speedup compensates for most of the 2× tournament work, so true-paper fidelity
costs only about 7 extra minutes per 850 generations.

### Training the Anaconda opponent

The [companion browser page](#companion-page-play-the-fully-evolved-anaconda) has two
opponent slots, picked from a dropdown:

- **Paper-strict** — `web/weights/anaconda-paper-strict.bin`. The strict paper-faithful
  training: `paper-2001-strict` preset, (μ+μ), no quiescence, symmetric σ evolution.
- **Enhanced** — `web/weights/anaconda-enhanced.bin`. A second run with non-paper tweaks
  (asymmetric win-favoring scoring, quiescence on, optional depth-schedule curriculum)
  intended to break out of the paper-strict draw plateau.

All three slots ship with real trained weights — see each slot's `.meta.json` sidecar for
the exact checkpoint and training recipe.

- **Paper-strict** (`anaconda-paper-strict.bin`) — gen 500 from a 2000-gen run (gen 850+
  saturated under unbounded σ and was unusable).
- **Enhanced** (`anaconda-enhanced.bin`) — **gen 270**, the two-phase paper-2001 base
  (through gen 240) plus a 30-gen d7/d8 finishing curriculum. Beats the previous shipped
  Enhanced (gen 240) by +9 tournament points over a 150-game head-to-head at depth 6
  (24W/15L/111D; 62% decisive-game win rate).
- **Risky** (`anaconda-risky.bin`) — **gen 285**, resumed from Enhanced gen 270 with a
  synthetic pop-50 seed and trained 15 gens at depth 8, **quiescence off**, with
  aggressive +3/0/-1 asymmetric scoring (wins worth 3× vs losses worth -1). The scoring
  reshapes selection toward decisive play instead of draw-shuffling. Head-to-head vs
  Enhanced (50 games d6): 6W/11L/33D — loses overall (+5 for Enhanced) but produces a
  **34% decisive-game rate** vs the usual 20–26%. Pair it with Paper-strict on the match
  page for the most exciting games.

To retrain paper-strict from scratch and re-ship it:

```bash
# Step 1: train. ~47 min on a 24-core box for 850 generations at strict paper fidelity.
# Always pass --device cpu; auto-detect picks CUDA at depth>=4 which disables the worker
# pool and runs serially (200x slower per gen).
python -m training.train --preset paper-2001-strict --generations 850 \
    --workers 20 --device cpu

# Step 2: export the champion's flat weight vector to the browser demo's bin file.
python scripts/export_weights_to_js.py checkpoints/best_gen0850.pt \
    web/weights/anaconda-paper-strict.bin \
    --fixtures web/weights/anaconda-paper-strict-fixtures.json
```

To train and slot the enhanced opponent:

```bash
python -m training.train --preset paper-2001 --generations 2000 \
    --win-score 2.0 --loss-score -1.0 \
    --depth-schedule 0:3,20:4,80:5,200:6 \
    --max-sigma 0.5 \
    --workers 20 --device cpu

python scripts/export_weights_to_js.py checkpoints/best_gen2000.pt \
    web/weights/anaconda-enhanced.bin \
    --fixtures web/weights/anaconda-enhanced-fixtures.json

# Then flip `available: true` for the "enhanced" entry in web/js/play-strong.js.
```

Each export writes both the `.bin` (5,048 × float32 = ~20 KB) and a sidecar `.meta.json`
with provenance (checkpoint name, gen count) that the play-strong page displays. Commit
both and the GitHub-Pages-deployed opponent updates automatically.

Verify the JS port still matches the new weights with:

```bash
node web/test_anaconda.js
```

`test_anaconda.js` runs the JS Anaconda network on five representative boards and asserts
|js − python| < 1e-6 — catches any drift in layout, encoding, or forward-pass assumptions.

---

## Architecture Overview

The original Blondie24 system has four main components:

```
┌─────────────────────────────────────────────────────┐
│                  EVOLUTIONARY LOOP                   │
│                                                      │
│  1. Population of 15 neural networks                 │
│  2. Each plays 5 games against random opponents      │
│  3. Rank by win/loss/draw record                     │
│  4. Keep top 50%, replace bottom 50% with            │
│     mutated copies of survivors                      │
│  5. Repeat for ~250-840 generations                  │
│                                                      │
│  ┌─────────────┐    ┌──────────────────────────┐     │
│  │  Checkers    │───▶│  Minimax + Alpha-Beta    │     │
│  │  Engine      │    │  (depth 4-8)             │     │
│  │             │    │                          │     │
│  │  - Legal    │    │  Evaluation fn =         │     │
│  │    moves    │    │  Neural Network          │     │
│  │  - Captures │    │                          │     │
│  │  - Kings    │    │  ┌──────────────────┐    │     │
│  │  - Jumps    │    │  │ Input: 32 squares│    │     │
│  │             │    │  │ (1 value each)   │    │     │
│  └─────────────┘    │  │ = 32 inputs      │    │     │
│                      │  │                  │    │     │
│                      │  │ Hidden layers:   │    │     │
│                      │  │  40 → 10 → 1    │    │     │
│                      │  │                  │    │     │
│                      │  │ 1,742 total      │    │     │
│                      │  │ evolvable weights│    │     │
│                      │  └──────────────────┘    │     │
│                      └──────────────────────────┘     │
└─────────────────────────────────────────────────────┘
```

## Neural Network Architecture (Chellapilla & Fogel 1999)

This implementation faithfully reproduces the architecture from the original 1999 paper:

- **Input layer**: 32 nodes (one per playable square on the 8×8 board), linear activation
  - Each square encoded from the current player's perspective:
    +1 (own piece), +K (own king), -1 (opponent piece), -K (opponent king), 0 (empty)
  - K (king weight) is itself an evolvable parameter, initialized at K=2.0, constrained to [1, 3]
- **Hidden layer 1**: 40 neurons, tanh activation
- **Hidden layer 2**: 10 neurons, tanh activation  
- **Output layer**: 1 neuron, tanh activation (board evaluation in [-1, +1])
- **Piece-difference bypass**: the sum of all 32 input values connects directly to the
  output node via a single learned weight, bypassing both hidden layers. This gives the
  network an implicit material-advantage signal without needing to learn it from scratch.
- **Total evolvable parameters: 1,743 = 1,742 (neural network) + 1 (king weight)**
  - fc1: 32×40 weights + 40 biases = 1,320 &nbsp;&nbsp;&nbsp;┐
  - fc2: 40×10 weights + 10 biases = 410  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│ 1,742 → neural network proper
  - fc3: 10×1 weights + 1 bias = 11       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│ (what the 1999 paper reports)
  - piece_diff_weight: 1 (bypass connection)&nbsp;┘
  - king_weight: 1  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ← scalar K ∈ [1,3], scales king squares in the input encoding
  - *The 1999 paper's "1,742" counts only the MLP + piece-diff bypass and treats K as a
    separate evolvable hyperparameter. This repo evolves K with the same self-adaptive σ
    rule as the other 1,742 weights, so we fold it into the flat vector and report 1,743.*
- All evolved via evolutionary programming — none trained by backpropagation

> **The 2001 "Anaconda" / Blondie24 system is also implemented here.** It adds a spatial
> sub-board preprocessor (91 overlapping 3×3…8×8 filter windows feeding the 92→40→10→1 MLP),
> bringing the total to 5,048 evolvable weights. The 2001 paper reports 5,046; we fold the
> piece-diff bypass and king weight into the flat evolvable vector (+2). Train it with:
>
> ```
> python -m training.train --preset paper-2001 --generations 250
> ```
>
> Anaconda runs on both CPU (Numba JIT alpha-beta via `search/fast_minimax_jit_anaconda.py`)
> and CUDA (torch); CPU JIT is currently the faster path on this repo's hardware. The 2001
> paper reached expert-level (~2045 USCF) play after ~840 generations; budget accordingly.

---

## Improvements Over Fogel & Chellapilla (1999)

The 1999 Blondie24 paper was bottlenecked by late-90s hardware: a single desktop running
one game at a time, Python-equivalent overhead in the search tree, and flat depth-4 minimax
throughout training. A full 840-generation run took weeks. This reimplementation keeps the
exact same neural architecture (1,743 evolvable weights, 32→40→10→1 tanh MLP with piece-diff
bypass) and the same self-adaptive ES mutation, but rebuilds the hot path around modern
compute.

### What's faithful to the paper

- **Network topology**: 32 inputs → 40 → 10 → 1 tanh MLP, with the piece-difference bypass
  connection. Exactly 1,743 evolvable parameters.
- **Encoding**: ±1 for men, ±K for kings, 0 for empty squares, from the side-to-move's
  perspective. K itself evolved, bounded to [1, 3].
- **Evolutionary strategy**: pure EP (no crossover), Gaussian mutation with self-adaptive
  per-weight σ, τ = 1/√(2√n). Top 50% survive, bottom 50% replaced by mutated copies.
- **Default population size of 15** and **5 games per individual** match the original
  paper's knobs when no CLI flags override them.
- **Fitness as the only learning signal**: no backprop, no gradient descent, no curated
  positions, no opening book, no endgame database.

### Where this implementation deliberately differs

These are intentional deviations from the 1999 paper — all exposed as flags so the faithful
configuration is still reachable via defaults or explicit arguments:

| Aspect | Paper (1999) | This repo | Why |
|---|---|---|---|
| Tournament structure | Each individual plays **5 games vs. randomly chosen opponents** | Full **round-robin** by default (every pair plays twice, once per color); paper-style random pairing available via `random_pairing_tournament` | Round-robin gives a much lower-variance fitness signal at pop 15–20, which matters once games are cheap |
| Color balance | Random color per game | **Each pair plays both colors** in round-robin | Removes first-move bias from the fitness signal |
| Search depth | Fixed **4-ply** throughout training | Configurable per-generation via `--depth-schedule`, e.g. `0:2,8:4,16:6,24:8` | The speedup makes depth-8 practical, and curriculum lets shallow phases do cheap early shaping |
| Scoring | **Asymmetric** +1 / 0 / −2 (wins, draws, losses) | Default matches paper; `--loss-score -1.0` exposes a symmetric variant | Anti-plateau experiment — see "draw plateau" section below |
| Draw detection | Move cap only (≈100 ply) | Move cap **plus threefold repetition** via position hashing | Without repetition detection, draw-seekers shuffle pieces until the move cap; this gave false positive decisive results in early runs |
| Terminal handling | Loss on no-legal-moves | Same, but propagated as ±∞ inside alpha-beta so forced mates aren't drowned by eval noise | |
| Transposition table | Not reported | **1 M-slot Zobrist TT** with EXACT/LOWER/UPPER flags, open-addressing, cleared per search because values are relative to the root side | Correctness-preserving speedup; matches a modern alpha-beta setup |
| Leaf evaluation | Single forward pass per leaf | **Batched** forward pass — across siblings (CPU numpy path) and across in-flight games (`ParallelSearchScheduler`, GPU path) | Makes a 1,743-weight network worth running on CUDA; serial eval is slower than CPU |
| Implementation | Custom C/C++ with a tight inner loop | Python glue + **Numba-JIT'd alpha-beta** (`FastAgentJit`) for the inner loop, or PyTorch for the GPU path | Keeps the whole pipeline hackable from Python while still getting native-code speed in the tree |
| Parallelism | Single-threaded | `multiprocessing.Pool` across CPU cores for tournaments; GPU lockstep scheduler across games | Modern hardware has cores the paper never had |
| Monitoring | Not described in detail | Per-generation JSONL log, checkpoints every N gens, best-individual snapshots, GPU memory telemetry | Lets us debug the draw plateau empirically instead of waiting weeks for a result |

Crucially, **the eval function, selection rule, and weight mutation operator are
identical to the paper when defaults are used.** Every deviation in the table above is
a training-loop or compute-infrastructure change, not a change to the learning
algorithm itself. That separation is deliberate: it means any behavior difference
between this repo and the published 1999 trajectory is attributable to compute/search
choices, not to a changed optimizer.

#### Known algorithmic deviations from the 1999 paper

One code-level deviation we have not yet corrected, flagged here for honesty:

- **King weight mutation.** The paper uses a dedicated multiplicative log-normal update
  for K (K' = K·exp(N(0,1)/√2)) with the constraint K ∈ [1, 3]. This repo folds K into
  the flat weight vector and mutates it with the same self-adaptive σᵢ rule as the
  rest, with no explicit [1, 3] clamp. In practice the σ-rule keeps K near its
  initialization across typical runs (K₀ = 2.0, σ₀ = 0.05), but a long run or a
  high-σ setting could drift K out of [1, 3].

### Speed improvements

| Layer | Original (1999) | Reborn | Notes |
|---|---|---|---|
| Leaf evaluation | scalar C forward pass | Numba-JIT'd scalar loop **or** batched GPU forward | GPU path batches all in-flight tournament games in a single kernel |
| Alpha-beta recursion | interpreted/C hybrid | fully JIT-compiled (`FastAgentJit`) | Python interpreter eliminated from the tree |
| Transposition table | none reported | 1 M-slot Zobrist TT, open-addressing, replace-always | ~2× alone at depth 8, ~45% fewer leaf evals |
| Tournament scheduling | serial games | multiprocess pool (CPU) / lockstep scheduler (GPU) | ~23 workers on a 24-core CPU path; GPU path batches NN evals *across games* |
| Search depth | 4 ply throughout training | depth schedule, e.g. `0:2,8:4,16:6,24:8` | Shallow play is cheap; deep play gates harder selection later |

**End-to-end:** a baseline generation (population 20, depth 8, full round-robin) dropped from
**~660 s/gen** in the initial port to **~37 s/gen** with the JIT + TT + multiprocess path —
a **~12× speedup**. Correctness is bit-verified against the pure-Python reference (0
disagreements across 15 random positions at depths 4, 6, 8). The GPU lockstep scheduler in
`search/parallel_search.py` takes this further for round-robin tournaments by running one
forward pass per *network* per tick instead of one per *game*.

This is what moved depth-8 training from "overnight per generation" into the "few minutes
per generation" regime on a single consumer box.

#### A note on GPU in practice: smaller runs go *faster* on CPU

One of the more counter-intuitive findings in this project is that **GPU is a net loss for
small populations, small depths, or early in training**. The 1,743-weight network is
tiny — a single forward pass is dominated by Python→CUDA kernel launch and
host↔device sync overhead, not by actual floating-point work. Until you can batch
hundreds of positions into one kernel, the CPU numpy+Numba path is strictly faster.

Empirically on a consumer box (RTX 5060 + 24-core CPU):

- **Depth 2**: CPU ~0.3 s/gen at pop 20 full round-robin. GPU: several seconds, worse.
  Almost every leaf is a terminal or near-terminal state and the branching factor is
  small, so the batch never fills up.
- **Depth 4**: CPU ~1.1 s/gen. GPU ~comparable, slightly slower.
- **Depth 6**: CPU ~10 s/gen. GPU starts to pull ahead when the game count × branching
  factor saturates a kernel.
- **Depth 8**: CPU ~95 s/gen with multiprocess pool. GPU via `ParallelSearchScheduler` wins
  here because each tick batches across *all still-running games in the tournament* — one
  forward pass can evaluate 100+ positions at once.

The tripwire is **depth ≥ 3 *and* enough concurrent games to fill a batch**. Below that,
the training CLI actively warns and recommends `--device cpu`:

```
WARNING: --device cuda with --depth 2 will be slower than CPU.
Batched leaf evaluation requires depth >= 3; below that every leaf
incurs full GPU launch+sync overhead.
```

In practice, almost all of the non-final curriculum and anti-plateau experiments in this
repo were run on CPU because a typical interactive iteration is ≤ 50 generations at
depth ≤ 6 — exactly the regime where GPU loses. The GPU path is reserved for long
depth-8 runs and for the planned Anaconda follow-up, where the larger network tips the
math back in CUDA's favor.

### The draw plateau (where modern compute meets an old problem)

The speedup uncovered a failure mode that was mostly hidden in the 1999 paper's wall-clock
budget: **at deep search on self-play, populations collapse into almost all draws**. In our
runs at depth 8, population 20, a typical generation has the best individual going
6W/1L/31D out of 38 games — a fitness signal dominated by draws, with barely any gradient
for selection to work with.

Interventions tried in this repo (all exposed as CLI flags):

- `--loss-score -1.0` (symmetric W/D/L = +1/0/-1 instead of Fogel's asymmetric +1/0/-2)
  — marginal: peak 9W vs 8W baseline over a 50-gen run
- `--initial-sigma 0.10` (double Fogel's default to keep behavioral variance alive)
  — kept σ from collapsing but didn't break the plateau on its own
- `--depth-schedule 0:2,8:4,16:6,24:8` (curriculum: start shallow and decisive, bump depth
  as the population matures) — the shallow phases produce high-wincount individuals (21W at
  depth 2), but that skill does *not* transfer: every depth bump collapses the best-wins
  count (21→14→7→7), because deeper search filters out shallow-level tactical tricks faster
  than selection can replace them

The plateau itself is consistent with what Fogel reported when he noted that "most games
among evolved players end in draws" — but the 1999 paper's compute budget let it be treated
as a nuisance rather than a structural obstacle. At ~37 s/gen we can run long enough to see
it clearly, and it's unresolved: none of the above interventions produced a run that
breaks out.

Likely next moves (not yet in-tree):

- **Coevolutionary hall-of-fame**: cache strong historical opponents so the current
  population can't converge on a single draw-optimal policy
- **Move-level fitness / PV quality signal**: evaluate positions on tactical-puzzle suites
  instead of — or alongside — head-to-head wins, so draws still produce gradient
- **Larger populations + Swiss pairing**: 15–20 is small enough that one good draw-seeker
  dominates; a bigger pool with Swiss-style pairings gives more decisive games

### Further speed ideas (not yet implemented)

Every speedup so far has been correctness-preserving — same alpha-beta, same eval, same
fitness. The levers below keep that property:

- **Move ordering**: killer-move and history-heuristic tables cut the branching factor of
  alpha-beta by a large constant. At depth 8 this typically translates to a 1.5–3× wall-clock
  drop without changing the game tree.
- **Iterative deepening** (depth 1 → 2 → … → N) with TT carry-over: the shallower searches
  seed the TT with good move orderings, so the final deep search prunes faster. Also lets us
  apply a wall-clock budget per move instead of a fixed depth.
- **Aspiration windows**: start each iterative-deepening level with a narrow (alpha, beta)
  around the previous level's score; re-search if it falls outside. Free 10–20% on top of
  plain iterative deepening.
- **Null-move pruning**: skip the current side's move and do a reduced-depth search; if the
  opponent still can't improve their position, prune. Very effective in checkers where
  zugzwang is rare. Risk: interacts with forced-capture rules — needs careful gating.
- **Incremental Zobrist**: update the hash in `apply_move_fast` instead of rehashing from
  scratch on each TT probe. Probably small (~5%) but trivial to add.
- **NN inference batching across depth** (GPU): the current GPU scheduler batches across
  *games*; batching across *leaf siblings within a single game* would shave another chunk
  at very deep searches, at the cost of more complex alpha-beta bookkeeping.
- **Opening book cache**: the first ~6 plies of every game are wasted re-searching the same
  positions. A shared read-only opening TT across the whole tournament would cut generation
  wall time meaningfully once the population stabilizes.

### Ideas for longer / larger runs

The plateau discussion above is really about what to *spend* extra compute on. If we want
to push beyond ~100 generations and a population of 20, these are the scaling knobs that
matter:

- **Swiss tournaments** instead of full round-robin once pop > 30. Round-robin is O(n²)
  games/gen; Swiss gives a comparable ranking signal in O(n log n) with pairing that
  concentrates games on boundary cases. Already stubbed in `tournament.py` as a natural
  extension of `random_pairing_tournament`.
- **Population sharding**: run k sub-populations with occasional migration (island model).
  Each island can use a different `--loss-score` / `--initial-sigma` / `--depth-schedule`
  and the migration step pulls in diversity. This is the most natural way to use an
  8-core+ box without just growing a single pop.
- **Warm-starting from checkpoints across configs**: `--resume-from checkpoints/...pt` so
  anti-plateau experiments can branch from a shared ancestor instead of re-running the
  first 100 gens of boring shallow play every time.
- **Distributed tournament**: the multiprocess pool path is already picklable; swapping
  `mp.Pool` for a `concurrent.futures.ProcessPoolExecutor` backed by multiple machines is
  a small change. Not worth it until the plateau story is figured out.
- **GPU residency for the full population**: right now each generation loads networks onto
  the GPU once and reuses them. For very large populations we'd run out of VRAM; the fix
  is a rolling-window of resident networks plus lazy eviction.
- **Tensor-core utilization**: the 32→40→10→1 net is tiny. Batching 1024+ positions per
  forward pass (which the GPU scheduler already does when many games are in-flight) is
  what makes CUDA pay off on a network this small. For Anaconda's ~5046-weight successor
  the per-position cost goes up but batching gets more efficient.

### Infrastructure additions

- CUDA / MPS / CPU device auto-detection with per-device optimized search paths
- JSONL per-generation logging (fitness, σ, W/L/D, wall time, GPU mem)
- Checkpoint-on-cadence + best-individual snapshots
- Threefold-repetition draw detection (the 1999 paper used move caps only)
- Smoke tests + correctness tests for every speedup (TT vs no-TT, JIT vs Python)
- `play/human_vs_ai.py` for playing a loaded checkpoint interactively

### Comparison to the 2001 "Anaconda" / Blondie24 model (also in this repo)

| | 1999 precursor | Anaconda / Blondie24 (2001) |
|---|---|---|
| Input representation | 32 raw squares | 91 overlapping sub-board filters (3×3 through 8×8) |
| Total evolvable weights | **1,743** (1,742 + king) | **5,048** (5,046 + bypass + king) |
| Training generations reported | ~250 for A-class play | ~840 for expert (~2045 USCF) |
| Training search depth | 4-ply | 4-ply (same as 1999) |
| Evaluation search depth | 4-ply | 6-ply and 8-ply vs. humans on Zone.com |
| Feature learning | none (raw board only) | positional sub-features evolved jointly |

The Anaconda expansion is specifically the *input pre-processor* — the 92→40→10→1 MLP core
is a near-clone of the 1999 core (the constant-1 channel of the preprocessor output replaces
the 40 biases of the 1999 fc1, which is how the paper arrives at 5,046). Pick between them at
the CLI with `--preset paper-1999` or `--preset paper-2001`. Anaconda requires CUDA — the
Numba CPU path hard-codes the 1999 layer shapes.

---

## Evolutionary Strategy (from Fogel 1999)

- Population size: 30 (15 parents + 15 offspring); Fogel kept the top 15
- Each generation: every network plays 5 games against randomly chosen opponents
- Scoring: +1 win, 0 draw, -2 loss (note the asymmetric penalty for losing)
- Selection: rank all 15 by score, keep top ~8, discard bottom ~7
- Reproduction: each survivor spawns one offspring via Gaussian mutation
  - Each weight w_i is mutated: w_i' = w_i + N(0, σ_i)
  - σ_i (step size) is also evolved: σ_i' = σ_i * exp(τ * N(0,1))
  - τ = 1/sqrt(2 * sqrt(n)) where n = number of weights
- No crossover — this is evolutionary programming (EP), not a genetic algorithm in the strict sense
  (though colloquially often called one)

## Project Structure

```
blondie24-reborn/
├── README.md                  # This file
├── requirements.txt           # Dependencies
├── config.py                  # All hyperparameters in one place
├── test_smoke.py              # Smoke tests for all components
│
├── checkers/
│   ├── board.py               # Pure-Python board state (32-square array)
│   ├── fast_board.py          # Numba-compatible board ops for the JIT search path
│   └── game.py                # Game loop: play a complete game between two agents
│
├── neural/
│   ├── network.py             # PyTorch 1999 net (32→40→10→1), 1,743 weights
│   ├── anaconda_network.py    # PyTorch 2001 Anaconda net, 5,048 weights
│   ├── anaconda_windows.py    # Deterministic 91 sub-board windows (3×3…8×8)
│   ├── encoding.py            # Board state → tensor encoding (vectorized numpy)
│   ├── fast_eval.py           # Numba-JIT forward pass for the 1999 net
│   └── fast_eval_anaconda.py  # Numba-JIT forward pass for Anaconda
│
├── search/
│   ├── minimax.py             # Reference alpha-beta + GPU batched leaves (1999 net)
│   ├── fast_minimax.py        # Pure-numpy alpha-beta (correctness reference)
│   ├── fast_minimax_jit.py    # Numba-JIT alpha-beta + Zobrist TT (1999)
│   ├── fast_minimax_jit_anaconda.py   # Numba-JIT alpha-beta for Anaconda
│   ├── fast_minimax_gpu_anaconda.py   # CUDA batched leaf-eval for Anaconda
│   ├── parallel_search.py     # Lockstep GPU scheduler (batches across games)
│   └── _order_helpers.py      # Move-ordering utilities shared across engines
│
├── evolution/
│   ├── population.py          # Population management, selection, reproduction
│   ├── strategy.py            # Evolutionary strategy (self-adaptive σ mutation)
│   └── tournament.py          # Round-robin / random-pairing tournaments (GPU-aware)
│
├── training/
│   └── train.py               # Main training loop (evolution driver, device auto-detect)
│
├── play/
│   └── human_vs_ai.py         # Play against a loaded checkpoint in the terminal
│
├── utils/
│   └── __init__.py            # GPU detection, TF32 optimization, memory management
│
├── analysis/
│   ├── benchmark_gpu.py       # CPU vs GPU throughput benchmark
│   └── compare_tournaments.py # Compare round-robin vs random-pairing fitness signals
│
├── scripts/
│   ├── ai_vs_ai.py                    # Head-to-head matches between two checkpoints
│   ├── export_weights_to_js.py        # Checkpoint → web/weights/*.bin + fixtures
│   ├── bench_iddfs.py                 # Iterative-deepening benchmark
│   ├── profile_tournament.py          # cProfile harness for one tournament gen
│   ├── test_jit_correctness.py        # JIT alpha-beta vs. reference bit-check (1999)
│   ├── test_jit_anaconda_correctness.py
│   └── test_tt_correctness.py         # Transposition-table correctness check
│
└── web/                                 # Static browser demo (no build, no deps)
    ├── index.html                       # Live-evolve page (1999 net, in-browser EP)
    ├── play-strong.html                 # Play the frozen Anaconda champion
    ├── match.html                       # Paper-strict vs Enhanced auto-play viewer
    ├── bench.html                       # Main-thread throughput bench
    ├── js/                              # Engine + UI (vanilla JS + Web Worker)
    └── weights/                         # Shipped *.bin + *.meta.json for each slot
```

## Getting Started

### Requirements

```
Python 3.10+
PyTorch 2.x (with CUDA for GPU acceleration)
numpy
matplotlib
tensorboard (optional, for monitoring)
tqdm
```

### Install

```bash
cd blondie24-reborn
pip install -r requirements.txt
```

### Train

```bash
# CPU (slow but works)
python -m training.train --generations 250 --population 15 --depth 4

# GPU (e.g. RTX 5060)
python -m training.train --generations 250 --population 15 --depth 6 --device cuda
```

### Play Against It

```bash
python -m play.human_vs_ai --checkpoint checkpoints/best_gen250.pt
```

## Key Design Decisions

### Why PyTorch instead of raw numpy?
The neural net forward pass happens thousands of times per game (once per minimax leaf node).
With depth-6 search, that's potentially millions of evaluations per generation. PyTorch lets us
batch evaluations on GPU, which is where an RTX 5060 pays off.

### Why evolutionary programming instead of a "true" genetic algorithm?
Fogel's original used EP (mutation only, no crossover) because crossover on neural network
weight vectors tends to be destructive — swapping weight subsets between two working networks
usually breaks both. This is consistent with findings across neuroevolution research.

### What about modern improvements?
The scaffold is designed to be extensible. Natural follow-ups:
- **NEAT/HyperNEAT**: Evolve topology alongside weights
- **Larger populations**: 15 is small by modern standards; try 50-100
- **Deeper search**: Original used 4-ply; with GPU batching, 6-8 is feasible
- **Coevolutionary dynamics**: Hall-of-fame opponents to prevent cycling
- **N-tuple systems**: Al-Khateeb & Kendall showed these can outperform NNs for checkers

## References

1. Chellapilla, K. & Fogel, D.B. (1999). "Evolving neural networks to play checkers without
   relying on expert knowledge." IEEE Trans. Neural Networks, 10(6), 1382-1391.
2. Chellapilla, K. & Fogel, D.B. (2001). "Evolving an expert checkers playing program without
   using human expertise." IEEE Trans. Evol. Comput., 5(4), 422-428.
3. Fogel, D.B. (2002). *Blondie24: Playing at the Edge of AI*. Academic Press.
4. Al-Khateeb, B. & Kendall, G. (2012). "Introducing a round robin tournament into evolutionary
   individual and social learning checkers."

## License

MIT — go forth and evolve.
