# ferrum ledger

Strength, spend, and milestone history. Absolute Elo claims are reported only
from anchored matches; per-patch self-play results are labeled SPRT Delta-Elo
estimates. Internal metrics (bench, perft) track regressions.

## Milestones

| date | version | bench (nodes) | tests | result |
|---|---|---|---|---|
| 2026-07-15 | M0 (v0.1.0) | 6,009,132 | 25 + deep perft | 100g vs Stockfish UCI_Elo=2000, 8+0.08, 24-opening book → **40.5%** (35W/54L/11D), Elo −66.8 ± 68.7 → est ~1930 |
| 2026-07-16 | M1 Task 4 (magic bitboards) — rejected/reverted | 6,009,132 | candidate 27 pass; restored 25 pass/1 ignored | Warmed paired median **+4.45%**, ratio 1.0445 (95% CI 1.0276–1.0649); failed frozen ≥10% gate; profiling put theoretical maximum below 10%; SPRT not run; ray-scan source restored |
| 2026-07-17 | M1 Task 5 (PVS) — rejected/reverted | candidate 4,500,816 / restored 6,009,132 (−25.10%) | candidate focused 4 pass/23 filtered; full 26 pass/1 ignored; restored 25 pass/1 ignored; clippy clean | definitive 2,000-game SPRT: 243W/226L/1531D, SPRT Delta-Elo +2.95 ± 6.04 (95% CI [−3.09, +8.99]), LLR +0.46; cap reached with no boundary and CI crosses zero, so no SPRT acceptance; source restored |

## M1 Task 4 — REJECTED / REVERTED

The runtime-found fancy-magic implementation preserved exact move-generation
correctness and the 6,009,132-node bench workload, but did not clear its frozen
performance gate. Fifteen warmed alternating baseline/candidate pairs measured
a median candidate uplift of **+4.45%**, with the 95% confidence interval for
the paired candidate/baseline ratio wholly positive at **1.0276–1.0649**.

The acceptance bar required both a ratio CI wholly above 1.00 and median uplift
of at least 10%. Profiling an exact benchmark attack trace measured a 1.953x
slider-kernel speedup but only about 5.45–8.73% of baseline work in ray scans,
placing the theoretical whole-engine maximum below 10% on the Apple ARM host.
The materiality gate therefore rejected the feature before SPRT. The magic
source was reverted to the pre-experiment ray-scan implementation. Candidate
verification finished with 27 passing tests (including UCI smoke), plus the
ignored deep-perft run. The restored baseline finished with 25 passing tests,
one ignored, and clean all-target release clippy. Cost: **$0**.

## M1 Task 5 — REJECTED / REVERTED

The PVS candidate preserved its pre-test correctness gates: 4 focused search
tests passed with 23 filtered, the full release suite had 26 passing tests and
one ignored test, all-target release clippy was clean, and UCI smoke completed
with `uciok`, `readyok`, and legal `bestmove b1c3`. Its bench result was
**4,500,816 nodes**, a **1,508,316-node (−25.10%)** reduction from the
**6,009,132-node** baseline; recorded candidate throughput was 2,751,612 NPS.

The definitive, unpooled 2,000-game fastchess SPRT at 8+0.08 finished
normally after 02:29:56 with **243W/226L/1531D**, relative SPRT Delta-Elo
**+2.95 ± 6.04**, and 95% CI **[−3.09, +8.99]**. Its LLR **+0.46** remained
inside the **(−2.94, +2.94)** boundaries, so neither boundary was reached. At
the cap, acceptance required a SPRT Delta-Elo confidence interval wholly above
zero; this interval crosses zero. There was therefore **no SPRT acceptance**.
The log recorded no
crash, illegal move, disconnect, or error. The earlier interrupted 103-game
run was not pooled.

Under direct authorization, only `ferrum/src/search.rs` was restored to
baseline commit `da4d4062f4d18eacf10db336322bc7c8564af96b`; the rejected PVS
behavior is absent. Fresh restored-baseline verification completed with 25
passing tests, zero failures, one ignored test, clean all-target release
clippy, and the 6,009,132-node bench. The rejected patch is preserved outside
tracked source at `.superpowers/sdd/task-5-rejected.patch`
(SHA-256 `f78b97504dfd290d1471e3e7eae969c4f9b7e7c586f6435f7553705c569089ba`);
its baseline apply-check passed. Cost: **$0**.

## M0 exit — PASSED

**Bar:** ferrum scores ≥ 25% vs Stockfish limited to UCI_Elo=2000 (i.e. within
~200 Elo of the anchor). **Result: 40.5%** — cleared with margin; estimated
strength ~1900–1950, in the M0 ~1800–2000 target band.

**Verification (all green at v0.1.0):**
- Move generation: perft exact on 6 standard positions to depth 5–6
  (startpos depth-6 = 119,060,324 nodes; Kiwipete depth-4 = 4,085,603).
- 25 unit/integration tests + `uci_smoke` pass; `cargo clippy` clean.
- Self-play smoke: 10 games via fastchess, zero illegal moves / disconnects /
  crashes.
- Live UCI: full handshake, iterative deepening to depth 8, legal play.

**Caveat on the anchor.** Stockfish's `UCI_Elo` is a self-referential dial, not
a CCRL/FIDE rating, and its calibration to CCRL is loose — treat ~1930 as a
rough internal sanity number, not a published rating. The honest,
CCRL-anchored gauntlet (Stash/Halogen/Weiss/Ethereal, Ordo) is set up in M1;
that is the number the 3000+ goal is measured against.

## Spend

| date | item | cost | cumulative |
|---|---|---|---|
| — | M0 is local-only (Mac); no ThunderCompute spend | $0.00 | $0.00 |
| 2026-07-16 | M1 Task 4 local magic-bitboard experiment (rejected) | $0.00 | $0.00 |
| 2026-07-17 | M1 Task 5 local PVS experiment (rejected/reverted) | $0.00 | $0.00 |

Budget: ~$20–25 approved. Cloud spend begins at M2 (gen-0 NNUE training on an
A6000). Hard alerts at $10 and $20 cumulative.

## Config at M0

- Search: iterative-deepening full-window negamax, quiescence (captures+promotions),
  always-replace TT, MVV-LVA + TT-move ordering, mate-distance scoring,
  50-move + repetition draws, node-checked soft time management.
- Eval: hand-crafted material + computed piece-square terms (HCE). Replaced by
  NNUE in M2.
- Movegen: bitboard, ray-scan sliders (magic bitboards deferred to M1 as a
  bench-gated perf patch).

## Next (M1)

Runtime-found magic bitboards (Task 4) and PVS (Task 5) were rejected and
reverted under their frozen gates. **Task 6, aspiration windows, is the next
planned task once Task 5 review closes.** PVS is absent from the restored
engine. Later work may include a semantics-neutral, readable-negamax
restructuring only as part of a separately gated later experiment; it must not
reintroduce the rejected PVS behavior without a new approved experiment and
acceptance criterion. The remaining search stack is
null-move, LMR, aspiration windows, and better time management, followed by the
CCRL-anchored gauntlet for the first honest absolute rating. No magic retry is
planned; any future compact redesign requires separate approval. Target:
~2300–2500.
