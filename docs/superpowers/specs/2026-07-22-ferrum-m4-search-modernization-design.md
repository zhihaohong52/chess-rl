# ferrum M4 — search modernization to ~2800–2870 (design)

**Date:** 2026-07-22
**Status:** approved by user (brainstorming session)
**Parent spec:** `2026-07-14-ferrum-nnue-engine-design.md` (this extends the roadmap past M3)

## 1. Context

`ferrum-v0.3.0` (gen-0 NNUE) exits M2 at **2669 ± 25 CCRL** — a +634 Elo jump
over the M1 HCE exit (2035) from swapping eval alone. M3 then established a
rigorous **negative**: the eval-architecture lever is saturated at the ChessBench
data ceiling (gen-0 already trains on all 63M Stockfish-labeled positions, so
king-buckets / bigger nets add capacity but no new signal). That leaves exactly
two live levers to 3000+: **more search** and **genuinely new data** (a gen-2
self-play effort). M4 spends the search lever — it is $0 (all local SPRT), high
certainty, and compounds with the eval already in hand; a stronger search also
produces better self-play games for the later data lever.

The current search (`ferrum/src/search.rs`) has: iterative deepening +
aspiration windows, an always-replace TT with bound cutoffs, SEE capture
ordering + two-slot killers, reverse futility pruning, null-move pruning, late
move pruning, futility pruning, **crude full-window LMR** (`1 + (legal>6)`),
check extensions, SEE-pruned quiescence, and soft/hard time management.

The decisive gap: there is **zero move-ordering history** — the `history` field
is only the position-hash list for repetition detection; all quiet moves are
ordered equal (key `0`). History is both the single biggest ordering lever and
the *substrate* the other modern techniques ride on (history-aware LMR, history
pruning). The LMR formula is also pre-modern (a two-step integer, no log table).

## 2. Goal and exit criterion

**Goal:** add the community-canonical *search* techniques ferrum is missing —
each as an SPRT-gated bundle — to reach ~2800–2870 anchored strength. Eval stays
gen-0 NNUE unchanged; no cloud, no data work.

**Exit criterion:** after all accepted bundles, the reusable CCRL-anchored
gauntlet (§5, 1120 games, Ordo fixed-anchor MLE) places ferrum's rating with a
reported 95% CI materially above the 2669 baseline. Tag `ferrum-v0.4.0` as a
**search-modernization checkpoint** — honest framing, report the CI, do not
over-claim the band (same discipline as v0.2.0 / v0.3.0). Per-bundle SPRT Elo is
a relative development signal, not an absolute rating.

## 3. Methodology decisions (this session)

- **Per-bundle SPRT, not one mega-bundle.** Given low throughput (Apple M1,
  concurrency 2 is the clean ceiling at 8+0.08 — higher spills onto E-cores and
  contaminates games), each coherent bundle is SPRT'd against the running-best
  binary. Accept → the bundle becomes the new baseline; reject → revert (B2
  bisects) and continue from the last accepted baseline. This gives attribution
  and monotonic improvement, and isolates the two riskier bundles (singular,
  correction history) so a failure there doesn't sink the confident core.
  Rationale over the M1 mega-bundle style: M1 bundled features each worth ~+3
  Elo (below the [0,8] SPRT resolution) where attribution didn't matter; M4's
  four bundles are mechanistically distinct and two are error-prone.
- **SPRT config per bundle.** fastchess self-play A/B (running-best vs
  candidate), STC 8s+0.08s, `-sprt elo0=0 elo1=8 alpha=0.05 beta=0.05`,
  concurrency 2, the reproducible M1 opening book, `-pgnout file=…`, `-repeat`,
  a game cap (~3000) so an inconclusive bundle resolves rather than grinding
  forever. Accept = LLR ≥ +2.94; reject = LLR ≤ −2.94; cap-reached-inconclusive
  is treated as a reject (not worth the added complexity/risk) and documented.
- **Full-window negamax stays the invariant in the main move loop.** Three of
  the four bundles are fully null-window-free. The sole exception is singular
  extensions' auxiliary singularity probe (§4.3): a reduced-depth null-window
  search that answers the boolean "is the TT move singular?" — this is *not* PV
  scouting / PVS (which is what was rejected in M1). The invariant is scoped to:
  **the main move loop searches every move full-window, with full-window
  re-searches.**
- **No automated SPSA tuning.** Infeasible at our game throughput (tens of
  thousands of games). Use literature-informed default constants; hand-tune only
  a borderline param via a one-off A/B SPRT if a bundle sits on the fence.
- **Compute:** all SPRT and the exit gauntlet run on the user's Mac. **M4 spend
  = $0.** ThunderCompute is reserved for the separate gen-2 data lever. Hard
  alerts at $10 / $20 cumulative remain in force.

## 4. Components — four bundles (dependency order)

Each bundle is built subagent-driven (Sonnet implementers, TDD: unit tests +
invariants pin every mechanic before the SPRT), passes spec-compliance then
code-quality review, and is committed as one coherent change before its SPRT.
Terse conventional commit subjects, no trailer/co-author (ferrum convention).

### 4.0 Shared infrastructure (added with the bundle that first needs it)

- **Per-ply static-eval stack** `stack_eval: [i32; MAX_PLY]` — the node's static
  eval, with `i32::MIN` as an "in-check / invalid" sentinel. Feeds the
  "improving" flag (B2) and correction history (B4). Added in B2.
- **Per-ply previous-move index** — the `piece*64 + to` index of the move that
  entered each node (sentinel at root and across null moves), threaded through
  the move loop to index continuation history and counter-moves. Added in B1.
- **`excluded: Move` parameter on `negamax`** — lets a node search while skipping
  one move, for the singular probe. When set: skip the early TT-cutoff return and
  skip the TT store for that node (standard singular-search handling). Added in
  B3; defaults to `Move::NONE` so existing call sites are unaffected.
- **Incremental pawn-king zobrist** on `Board` (`pawn_hash`, updated in
  `make`/`unmake` for pawn and king from/to/capture, and on EP/promotion),
  keying correction history. Locked by a from-scratch-recompute invariant test
  mirroring the existing main-zobrist test. Added in B4.

### 4.1 Bundle 1 — Move-ordering history (the substrate; biggest lever)

- **Tables** (boxed to keep them off the stack): main quiet history
  `[color][from][to]`; 1-ply continuation / counter-move history
  `[prev_piece_to (0..768)][piece (0..12)][to (0..64)]`; capture history
  `[piece (0..12)][to (0..64)][captured_pt (0..6)]`.
- **Update:** on a beta cutoff by a quiet move, a depth-scaled bonus
  (`bonus = min(k·depth², cap)`) is applied to the cutoff move's main + 1-ply
  continuation entries with **history gravity** (`e += bonus − e·|bonus|/MAX`,
  keeping entries bounded), and an equal malus is applied to the quiet moves
  tried before it in this node (collected during the loop). A capture beta cutoff
  updates capture history the same way.
- **Ordering:** in `order_moves`, quiets are ranked by
  `main_hist + continuation_hist` (mapped so higher history sorts earlier,
  slotting **below** the killer band and **above** zero-history quiets, with no
  collision against the existing tt / capture / promo / killer bands); captures'
  winning/equal band is augmented by capture history.
- **Tests:** gravity bonus/malus + clamp unit test; ordering test (a high-history
  quiet sorts before a low-history quiet, both below killers); all existing
  tactical / mate / SEE-ordering tests still pass.
- **Framing:** this is the M1-rejected butterfly-history idea (T8) revisited in
  its stronger *continuation + capture* form, at NNUE strength where move
  ordering has far more leverage, and as the prerequisite for B2's
  history-aware reductions. If it still fails SPRT, it is dropped — same
  discipline as M1.
- **SPRT:** vs `v0.3.0`.

### 4.2 Bundle 2 — Search shaping (vs post-B1 baseline)

- **Log-based LMR table** computed once: `reduction[d][m] = base + ln(d)·ln(m)/div`
  (literature-informed `base ≈ 0.77`, `div ≈ 2.36`), stored as `[[i32; 64]; 64]`,
  clamped ≥ 1, replacing the `1 + (legal>6)` integer. Reduction is **adjusted**:
  more when not improving; less on high history / killer / counter-move. The
  reduced search and the full-depth re-search both use the full `[-beta, -alpha]`
  window (invariant preserved).
- **"Improving" heuristic:** `improving = !in_check && stack_eval[ply] >
  stack_eval[ply-2]` (guarded on ply ≥ 2 and both evals valid). Scales the RFP
  margin, futility margin, LMP move-count threshold, and LMR.
- **Internal iterative reductions (IIR):** at depth ≥ 4 with no TT move, reduce
  depth by 1 (a cheap, full-window way to avoid over-searching a node with no
  guidance).
- **Razoring:** at shallow depth (≤ 3), if `static_eval + margin < alpha`, drop
  to `qsearch`; if that also fails low (< alpha), return it. Fail-soft; guarded
  against check and mate scores.
- **History-aware LMP:** widen the current LMP depth cap and make the move-count
  threshold improving-aware; optionally prune very-negative-history late quiets
  earlier.
- **Tests:** LMR-table monotonicity (reduction non-decreasing in depth and move
  number, always ≥ 1); improving-flag unit test; IIR reduces depth only when no
  TT move; razoring drop condition + tactic-preservation; the existing
  deep-tactic LMR regression still passes.
- **Contingency:** if B1 was rejected, B2 drops its history-aware LMR/LMP terms —
  the log-LMR table, improving, IIR, and razoring are history-independent and
  still tested standalone.
- **SPRT:** vs post-B1. If B2 fails, **bisect**: SPRT LMR-table-only vs the
  +extras variant to salvage the part that gates.

### 4.3 Bundle 3 — Singular extensions (vs post-B2 baseline)

- **Trigger:** non-root node, depth ≥ 8, a usable TT move (bound LOWER or EXACT,
  tt-depth ≥ depth − 3), not already in a singular probe.
- **Probe (the scoped null-window search):** `singular_beta = tt_score − c·depth`;
  run a reduced-depth (`(depth − 1) / 2`) search of **all moves except the TT
  move** against the null window `[singular_beta − 1, singular_beta]`, via
  `negamax(…, excluded = tt_move)`. If it fails low (< `singular_beta`), the TT
  move is singular → **extend it by 1 ply**. First cut includes multi-cut (if the
  probe fails high ≥ beta, return beta); negative extensions are deferred unless
  the SPRT sits on the fence.
- **Correctness handling:** the excluded-move node skips the early TT-cutoff
  return and the TT store; the probe runs on the current board *before* the TT
  move is made; the extension is applied to the TT move's child depth; total
  extension growth is bounded by the existing `ply >= MAX_PLY` guard.
- **Tests:** an excluded-move search never tries the excluded move (unit); TT
  store/probe stays correct with `excluded` set (no corruption); a discriminating
  deep-tactic position that the extension finds and the pre-extension build
  misses (in the style of the existing check-extension regression); existing
  tests pass.
- **SPRT:** vs post-B2.

### 4.4 Bundle 4 — Correction history (vs post-B3 baseline)

- **Table:** `corr_hist: [[i32; 16384]; 2]`, keyed by `[side][pawn_hash % 16384]`
  (pawn_hash from §4.0).
- **Read:** where the node computes a static eval (non-check),
  `corrected = clamp(static_eval + corr_hist[side][key] / scale)` to the non-mate
  range; the **corrected** value is what feeds pruning/ordering (RFP, null-move
  gate, razoring, futility) and is stored in `stack_eval`; the **raw** static is
  retained for the update.
- **Update:** after the node's search resolves (non-check, TT-storable), nudge
  `corr_hist[side][key]` toward `(search_score − raw_static)`, depth-weighted and
  clamped (gravity), with bound-consistency guards (don't push toward a score
  that is only a one-sided bound in the wrong direction).
- **Tests:** read shifts the eval by the stored amount (unit); update converges
  toward the error and clamps (unit); the pawn_hash from-scratch invariant test;
  existing tactical tests pass.
- **SPRT:** vs post-B3.

## 5. Exit — anchored gauntlet → v0.4.0

Reuse the existing M1 harness and anchor pool unchanged: `ferrum/tools/sprt.sh`,
`ferrum/tools/gauntlet.sh`, `ferrum/bench/anchors/` (stash-v17 / v21 / v37 +
Weiss 2.0, pinned, Apple-Silicon build recipes, `anchors.txt` in Ordo
`"Name",Rating` format). Run the final accepted binary vs each anchor, **1120
games** (280/anchor), 8+0.08, concurrency 2, using the **0-forfeit detached
protocol** proven in M2/M3 (harness-tracked background jobs get reaped, so the
gauntlet runs `nohup`-detached with a detached finisher that auto-runs Ordo and
drops a sentinel; monitor via foreground blocking polls, not tracked bg
watchers). Ordo fixed-anchor MLE emits ferrum's new CCRL rating + 95% CI. Tag
`ferrum-v0.4.0`; archive raw PGN + Ordo output under
`ferrum/bench/anchors/results/`.

## 6. Testing strategy

- **Correctness:** per-mechanic unit tests + invariants (history gravity, LMR
  monotonicity, excluded-move skip, pawn-hash recompute) before each SPRT; all
  existing tactical / mate / SEE / aspiration tests stay green at every commit;
  `cargo run --release -- bench` node count *changes* per bundle (proves the
  change actually altered the search).
- **Strength (relative):** per-bundle SPRT, self-play, vs the running-best
  binary.
- **Strength (absolute):** the anchored gauntlet at the milestone exit only.
- **CI:** `.github/workflows/ferrum-ci.yml` (clippy `-D warnings` + test + bench)
  green at every commit — unchanged.

## 7. Files

- Modify: `ferrum/src/search.rs` (all four bundles + shared per-ply infra),
  `ferrum/src/board.rs` (B4 incremental pawn-king zobrist in make/unmake),
  possibly `ferrum/src/nnue.rs` only if the static-eval read path needs a hook
  (expected: none), `ferrum/docs/ledger.md` (per-bundle SPRT rows + M4 exit).
- Reuse unchanged: `ferrum/tools/sprt.sh`, `ferrum/tools/gauntlet.sh`,
  `ferrum/bench/anchors/**`, `ferrum/books/openings_m1.epd`.
- History / correction tables live within `search.rs` (or a small `history.rs`
  helper module if `search.rs` grows unwieldy — implementer's call at review).

## 8. Risks and mitigations

- **Mac wall-clock** (4 SPRT bundles × a few hours, plus the 1120-game gauntlet)
  → detached overnight jobs; game cap per SPRT; concurrency 2 is fixed (higher
  contaminates).
- **B1 history fails again (as T8 did)** → it is now the stronger
  continuation+capture form at NNUE strength and the substrate for B2; if it
  still fails, drop it and re-scope B2 to its history-independent parts
  (contingency in §4.2). Honest ledger either way.
- **Singular extensions are error-prone** (excluded-move TT handling, extension
  runaway) → isolated as its own bundle with explicit correctness unit tests and
  a discriminating regression; a failure doesn't block B1/B2/B4.
- **B4 touches `Board::make`/`unmake`** (the hottest path) → the pawn-hash update
  is a couple of zobrist XORs mirroring the existing main-hash update, locked by
  a recompute invariant test; measure bench nps before/after to confirm no
  regression.
- **Bundle interactions / search instability** → per-bundle re-baselining keeps
  improvement monotonic within SPRT error; aspiration + selective extensions
  already introduce benign few-cp instability (documented in existing tests).

## 9. Budget

M4 is local-only: **$0.** Cumulative ThunderCompute spend to date ~$1.75; the
gen-2 data lever (separate milestone) is where cloud spend resumes. Hard alerts
at $10 / $20 remain in force.

## 10. Non-goals (M4)

No eval-architecture or NNUE-retraining work (M3 proved that lever saturated on
ChessBench data). No self-play / new-data generation (the separate gen-2 lever).
No PVS / null-window in the main move loop (invariant). No lazy SMP /
multithreading. No Syzygy tablebases. No automated SPSA tuning. No Lichess
deployment.
