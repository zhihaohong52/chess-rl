# ferrum — Rust alpha-beta + NNUE engine to genuine 3000+ Elo (design)

**Date:** 2026-07-14
**Status:** approved by user (brainstorming session), pending spec review
**Working name:** `ferrum` (renameable; nothing depends on the name)

## 1. Context and decision history

chess-rl's transformer+MCTS engine is at ~2000–2050 honest Elo and at its
distillation ceiling (~51% teacher agreement; 4× more training bought +35 Elo).
The user's goal is a **genuine 3000+ Elo chess bot** on a hobbyist budget
(~$20–30 of ThunderCompute credit, current balance $0.44, plus evenings).

Three paths were considered:

- **A. Own alpha-beta engine + NNUE eval** — the community-proven hobbyist
  path (Koivisto, Berserk, Seer, Viridithas et al., mostly solo devs): ~70%
  search engineering, ~30% small fast net. High confidence of genuine 3000+.
- **B. Scale the transformer+MCTS stack** — needs ~$500–1,500 of training plus
  C++ inference engineering, and realistically tops out ~2800 (no raw neural
  policy anywhere, at any budget, has reached 3000 without massive search).
- **C. Maximize the current system within $25** — proven +100–200 Elo, lands
  ~2400–2600, no 3000.

**Decision: all-in on A.** Rust, from scratch. Gen-0 NNUE trained directly on
ChessBench Stockfish-16 labels. The chess-rl repo is not abandoned — it becomes
the supply chain: data tooling, Syzygy/book know-how, and the 77M transformer
as a position labeler from gen-1 onward.

## 2. Goal and success metric

**Deliverable:** a UCI chess engine in Rust, living in a `ferrum/`
subdirectory of the chess-rl repository (user decision 2026-07-14: no separate
repo; the crate is self-contained and touches no existing chess-rl code).

**Success metric:** in blitz gauntlets against a **fixed anchor pool of open
engines with published CCRL blitz ratings** spanning ~2600–3200 (pinned
versions of e.g. Stash, Halogen, Weiss, Ethereal — all build on Apple
Silicon), run with `fastchess` on the user's Mac using standard testing
opening books, ≥1,000 games, rated with Ordo anchored to the pool:
**ferrum's 95% confidence interval lies entirely above 3000.**

Explicit non-metrics: the chess-rl depth-4 Stockfish ladder (saturates,
flattering), puzzle top-1 (anti-correlates with strength per chess-rl's own
ablations), Lichess ratings (inflated scale; a possible future deployment, not
the success bar).

## 3. Non-goals

- No changes to chess-rl's Python MCTS engine (it stays as-is; separate repo).
- No multi-threaded search before the single-threaded engine has proven Elo
  (lazy SMP is future work).
- No Lichess bot deployment in this phase (listed as future work).
- No originality constraints beyond standard practice: techniques are the
  well-documented community canon; the implementation is ours.

## 4. Engine architecture (Rust)

Single binary crate, modules with hard boundaries. Each unit is testable
alone; the seams below are the module interfaces.

- **`board`** — bitboard position: magic-bitboard sliding attacks, incremental
  Zobrist hashing, make/unmake. Public surface: `Board` (from FEN/startpos,
  `make`, `unmake`, `legal_moves`, attack/check queries), `Move`. Correctness
  is locked by a **perft suite** (standard + castling/EP/promotion trap
  positions) in CI before any search work begins.
- **`eval`** — trait `Eval { fn eval(&self, board: &Board) -> i32 }` with two
  implementations: `Hce` (material + piece-square tables, M0 only) and `Nnue`
  (gen-0 onward). The NNUE accumulator updates incrementally inside
  make/unmake via a thin hook, but search only ever sees the trait.
- **`search`** — iterative deepening, PVS negamax, lockless fixed-size
  transposition table (atomic 16-byte entries), quiescence with SEE, staged
  move ordering (TT move → good captures MVV-LVA/SEE → killers → history).
  Pruning/extension features are added **one SPRT-tested patch at a time** in
  the community-canonical order: null-move pruning, LMR, late-move pruning,
  futility, aspiration windows, check extensions; then (M3) singular
  extensions, correction history, SEE pruning refinements.
- **`uci`** — complete UCI protocol (plays in any GUI and under fastchess),
  plus the community-standard **`bench`** subcommand: fixed position set whose
  total node count is a checksum that a patch changed exactly what it claimed.
- **`time`** — soft/hard time bounds from wtime/btime/winc/binc, node-count
  fallback. Worth real Elo in blitz on its own.
- **`datagen`** (M3) — subcommand: fixed-node self-play games emitting
  (position, search score, game result) records for gen-1 training.

Performance guardrail: NEON (Apple Silicon) and AVX2 (cloud x86) SIMD paths
for the NNUE inner loops, **microbenchmarked in M0** so eval speed never
silently underperforms (target: ≥1M evals/sec/thread on the M-series Mac).

## 5. NNUE stack

### Gen-0 (trained on ChessBench, no engine needed)

- **Architecture:** standard perspective network — 768 input features
  (piece × square × color from each side's view) → 512 hidden per perspective,
  SCReLU activation, single output; quantized int16 accumulator / int8 weights
  with incremental updates on make/unmake. Output buckets and HalfKA-class
  feature sets are deliberately deferred to gen-2 (proven upgrades, gen-0
  risk reduction).
- **Trainer:** `bullet` (community-standard Rust/CUDA NNUE trainer) on a
  ThunderCompute A6000 (~$0.35/h). One training run over 100M+ positions ≈
  1–4 GPU-hours ≈ **$1–2 per net**, so 5–10 experiments fit the budget.
- **Data pipeline:** raw ChessBench dense shards
  (`prdev/chessbench-full-policy-value`: fen + all legal moves + SF-16 win%)
  → bullet format (~32 B/position; 100M ≈ 3 GB). Score = win% inverted through
  the sigmoid to centipawns; WDL blend λ=1.0 (score-only) in gen-0 since
  ChessBench has no game results. Filters: drop in-check positions and
  positions whose best move is a capture or gives check (quiet-position
  training — search owns tactics), clamp extreme scores, dedupe by Zobrist.
  Conversion runs on the cloud box during the training session (fast network,
  spares the Mac's disk); one raw shard already sits locally for pipeline
  development (`data/raw_hf/train-00000.msgpack.zst`).

### Gen-1+ (M3)

Engine self-generated data: fixed-node (~5k nodes) self-play from varied
openings, recording (position, search score, result); blended with labels from
the **77M chess-rl transformer** on those same positions (its value: cheap
labels for positions ChessBench never covers — the distribution the engine's
own search actually visits). Retrain with WDL blend λ≈0.7 score / 0.3 result.

## 6. Testing methodology

- **SPRT gate on every functional patch:** fastchess self-play, STC 8s+0.08s,
  bounds [0, +5] Elo early, tightening to [0, +3] in M3; 2–4 concurrent games
  on the 8GB Mac, overnight. No proven Elo, no merge.
- **Absolute calibration at milestones:** anchor-pool gauntlets (section 2),
  Ordo ratings with error bars. This is the only number reported as "Elo".
- **CI:** perft correctness, `bench` node checksum, clippy + fmt.
- **Openings:** standard engine-testing books (UHO-class) for both SPRT and
  gauntlets.

## 7. Milestones

| Milestone | Content | Exit criterion |
|---|---|---|
| **M0** | Movegen (perft-perfect), UCI, alpha-beta + quiescence + TT, HCE eval; eval microbench | Perft suite green; plays legal complete games; ~1800–2000 vs anchors |
| **M1** | Search features batch 1 (ID, PVS, NMP, LMR, ordering, time mgmt), each SPRT-gated | ~2300–2500 vs anchors |
| **M2** | Gen-0 NNUE trained + integrated, SIMD paths on NEON/AVX2 | Beats M1 by SPRT; ~2700–2900 vs anchors |
| **M3** | SPRT grind (singular ext., correction history, SEE pruning, tuning) + gen-1 NNUE from self-gen data | **95% CI above 3000** vs anchor pool |

**Timeline honesty:** M0–M2 is a few weeks of elapsed sessions, mostly agent
work plus training runs. M3 is the long tail — hobby engines grind 2900→3000+
through dozens of SPRT patches; elapsed 2–4 months, most of it unattended
overnight test compute, not user attention. Confidence that the path ends
3000+ is high (the most-replicated result in hobby engine dev); the
uncertainty is M3's duration, not its destination.

## 8. Budget and ops

Planned spend ~$20–25 of the approved $20–30 (ThunderCompute; current balance
$0.44 — user tops up):

| Item | Est. |
|---|---|
| ChessBench raw download + bullet-format conversion (cloud session) | $1–2 |
| Gen-0 NNUE training, 3–4 iterations (A6000) | $4–6 |
| 77M labeling session for gen-1 | $1–2 |
| Gen-1/gen-2 retrains | $4–6 |
| Reserve (failed runs, extra generation, cloud-CPU SPRT burst) | $8–10 |

Ops rules (established in prior phases): pull artifacts, then **delete the
instance the moment a session ends** (TC bills while running); snapshot only
when state must survive; spend ledger at `ferrum/docs/ledger.md` updated after
every session; **hard stop + user alert when cumulative spend crosses $10 and $20**;
nothing beyond the approved pot without asking. SPRT compute is the user's
Mac: $0.

## 9. Monitoring and reporting (user request)

- Training runs execute headless on ThunderCompute; the agent watches loss and
  validation curves, pulls the net, kills the instance, and reports per run.
- SPRT tests run as background jobs on the Mac; the agent reports pass/fail
  with Elo estimates.
- A strength ledger (current anchored rating, spend to date, patch history)
  lives at `ferrum/docs/ledger.md`; plain-language status at every milestone
  and at the $10/$20 spend thresholds.

## 10. Risks and mitigations

- **M3 grind throughput on one 8GB Mac** → efficient STC testing; optional
  $2–3 cloud-CPU bursts for big tests.
- **ChessBench 50ms labels cap gen-0 quality** → expected; gen-1 self-gen data
  is the standard fix and is already in the plan.
- **NEON SIMD underperformance** → M0 microbenchmark gate before NNUE work.
- **bullet requires CUDA** → all training on TC A6000; nothing GPU-bound runs
  locally.
- **Anchor-pool validity** → pin exact engine versions + settings in the repo
  so the scale never drifts.

## 11. Relationship to chess-rl

ferrum lives in a `ferrum/` subdirectory of the chess-rl repository but is a
self-contained Rust crate. chess-rl remains the data/teacher side: ChessBench
tooling, the 77M transformer (gen-1+ labeler), Polyglot book and Syzygy
integration experience (ferrum gets TB probing in M3). No existing chess-rl
code is modified by this project; a small export/labeling script may be added
when gen-1 starts.

## 12. Future work (explicitly out of scope now)

Lazy SMP; HalfKA feature sets and output buckets (gen-2 NNUE); Syzygy probing
in search; Lichess bot deployment (a ~2500+ engine with book/TB and good time
management plausibly holds a 2800–3100 bullet rating there — a public number,
if ever wanted); OpenBench-style distributed testing if more hardware appears.
