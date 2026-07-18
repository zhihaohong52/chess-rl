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
| 2026-07-17 | M1 Task 6 (aspiration windows) — accepted | baseline 6,009,132 / candidate 5,612,148 (−6.61%) | focused 2 pass/26 filtered; full search 5 pass/23 filtered; release 27 pass/1 ignored; deep perft; clippy clean | definitive 2,000-game SPRT: 279W/239L/1482D, relative Delta-Elo +6.95 ± 5.86 (95% CI [+1.09, +12.81]), LLR +1.86 inside boundaries; cap acceptance because CI is wholly positive; source retained |
| 2026-07-17 | M1 Task 7 (killer moves) — accepted | baseline 5,612,148 / candidate 3,166,392 (−43.58%) | focused 1+1 pass/31 filtered each; full search 9 pass/23 filtered; release 31 pass/1 ignored; deep perft; clippy clean | H1 accepted after 992 games: 160W/93L/739D, relative Delta-Elo +23.50 ± 10.09, LLR +2.98 crossed +2.94; source retained |
| 2026-07-18 | M1 Task 8 (butterfly history heuristic) — rejected/reverted | baseline 3,166,392 / candidate 3,108,001 (−1.84%) | candidate full 34 pass/1 ignored; restored 31 pass/1 ignored; clippy clean | definitive 2,000-game SPRT: 274W/246L/1480D, relative SPRT Delta-Elo +4.86 ± 6.46 (95% CI [−1.60, +11.32]), LLR +0.99; cap reached with no boundary and CI crosses zero, so no SPRT acceptance; source restored |
| 2026-07-18 | M1 Task 9 (null-move pruning) — accepted | baseline 3,166,392 / candidate 1,449,025 (−54.24%) | full release 34 pass/0 fail/1 ignored (31 baseline + 3 new null-move tests); deep perft; clippy clean | H1 accepted after 558 games: 120W/47L/391D, relative SPRT Delta-Elo +45.72 ± 14.48, LLR +2.97 crossed +2.94; source retained |

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

## M1 Task 6 — ACCEPTED

Task 6 adds aspiration windows at the iterative-deepening root. The frozen
comparison was **6,009,132 nodes / 3,813,821 NPS** for the baseline and
**5,612,148 nodes / 3,856,047 NPS** for the candidate: 396,984 fewer nodes
(**−6.61%**). A fresh finalizer re-bench reproduced 5,612,148 nodes at
3,857,760 NPS; throughput is timing-dependent and not an Elo claim.

Validation was exact: focused aspiration tests were **2 passed, 26 filtered**;
the complete search set was **5 passed, 23 filtered**; `cargo test --release`
had **27 passed, 0 failed, 1 ignored**; all-target release clippy with
`-D warnings` was clean; ignored `perft_deep` was **1 passed, 27 filtered**;
and live release `perft 6` produced **119,060,324**. The UCI smoke returned
`uciok`, `readyok`, and legal `bestmove b1c3`; `git diff --check` was clean.

The definitive unpooled 2,000-game fastchess SPRT at 8+0.08 finished normally
after 02:29:45: **279W/239L/1482D**. Its relative self-play SPRT Delta-Elo was
**+6.95 ± 5.86**, with an approximate 95% CI of **[+1.09, +12.81]**. The LLR
was **+1.86**, inside **(−2.94, +2.94)**, so no boundary was reached. The
2,000-game cap acceptance rule requires the entire reported CI to be above
zero; it is, so the task is accepted. This is a relative self-play development
estimate, not an absolute anchored Elo. The definitive log had no error,
illegal-move, disconnect, crash, killed, or failed entry; its PGN accounts for
all 2,000 games.

The accepted source is retained. PVS remains absent, and the readable
full-window negamax restructuring is retained as part of this accepted
aspiration patch; it does not introduce a first-move or null-window search.
Cost: **$0**.

## M1 Task 7 — ACCEPTED

Task 7 retains bounded killer-move storage and normal-negamax ordering. The
frozen benchmark comparison is **5,612,148 nodes / 3,172,643 NPS** for the
baseline and **3,166,392 nodes / 2,936,389 NPS** for the candidate: 2,445,756
fewer nodes (**−43.58%**). A fresh finalizer re-bench reproduced 3,166,392
nodes at 2,837,648 NPS; NPS is timing-dependent and is not an Elo claim.

Validation was exact: `killer_stored_and_deduped` and
`move_ordering_has_exact_killer_precedence` each had **1 passed, 31 filtered**;
the complete search set had **9 passed, 23 filtered**; `cargo test --release`
had **31 passed, 0 failed, 1 ignored**; all-target release clippy with
`-D warnings` was clean; ignored `perft_deep` was **1 passed, 31 filtered**;
and live release `perft 6` produced **119,060,324**. The UCI smoke returned
`uciok`, `readyok`, and legal `bestmove b1c3`; `git diff --check` was clean.

The definitive normalized fastchess SPRT at 8+0.08 ended normally after
01:14:15 when **H1 was accepted**. It contains **992 games: 160W/93L/739D**
for candidate versus baseline, relative self-play SPRT Delta-Elo **+23.50 ±
10.09**, and LLR **+2.98**, crossing the **+2.94** upper boundary. This is a
relative self-play development estimate, not an absolute anchored Elo. The
definitive log has exactly one `Finished match` and no error, illegal-move,
disconnect, crash, killed, or failed entry. Its 2,412,598-byte PGN independently
has all 992 games and 992 unique `(Round, White, Black)` identities, with the
same candidate W/L/D accounting.

The accepted source is retained. Killers reset once at `think` entry and persist
across aspiration retries and iterative depths within that think. qsearch
remains killer-free, using its existing free capture ordering; PVS, first-move,
null-window, and re-search behavior remain absent. Cost: **$0**.

## M1 Task 8 — REJECTED / REVERTED

The butterfly history heuristic candidate preserved its pre-test correctness
gates: the full release suite had **34 passing tests** (31 baseline + 3 new —
deterministic history accumulation, rescale-above-threshold, and move-ordering
precedence between killer and unscored quiet), zero failures, one ignored
test, all-target release clippy was clean, and UCI smoke completed with
`uciok`, `readyok`, and legal `bestmove b1c3`. Its bench result was
**3,108,001 nodes**, a **58,391-node (−1.84%)** reduction from the
**3,166,392-node** baseline; NPS is timing-dependent and is not an Elo claim.

An initial SPRT attempt at concurrency 4 was OS-OOM-killed with **0 games**
completed; the stub is preserved at
`.superpowers/sdd/task-8-sprt-killed-attempt-1.log` and excluded from all
accounting. The authoritative run used concurrency 2 — an environment-only
override — with games remaining independent and the SPRT valid at the same
8+0.08 time control.

The definitive, unpooled 2,000-game fastchess SPRT at 8+0.08 finished
normally after 06:40:48 with **274W/246L/1480D**, relative SPRT Delta-Elo
**+4.86 ± 6.46**, and 95% CI **[−1.60, +11.32]**. Its LLR **+0.99** remained
inside the **(−2.94, +2.94)** boundaries, so neither boundary was reached. At
the cap, acceptance required a SPRT Delta-Elo confidence interval wholly
above zero; this interval crosses zero. There was therefore **no SPRT
acceptance**. This is a relative self-play development estimate, not an
absolute anchored Elo. Timeouts were mild and favored the candidate (base 13,
cand 8), so the rejection is not an artifact of engine instability; the log
recorded no crash, illegal move, or disconnect.

Under direct authorization, only `ferrum/src/search.rs` was restored to
baseline commit `2574594cde7ef13c10474c88128b393dfe220585`; the rejected
history-heuristic behavior is absent. Fresh restored-baseline verification
completed with 31 passing tests, zero failures, one ignored test, clean
all-target release clippy, and the 3,166,392-node bench. The rejected patch
is preserved outside tracked source at
`.superpowers/sdd/task-8-rejected.patch`
(SHA-256 `98d8c7af08dbdcac06b6d8503a5c8dde109c1d638519d5f4c01ab0481d7e9efb`);
it reverse-applies to the prior candidate working tree and forward-applies to
the pristine baseline of `search.rs` at `2574594`. Cost: **$0**.

## M1 Task 9 — ACCEPTED

Task 9 adds null-move pruning (NMP) to `negamax`, backed by new board-level
null-move helpers. `Board` gained `NullUndo` (private fields mirroring
`Undo`), `has_non_pawn_material`, and `make_null`/`unmake_null`; `negamax`
prunes when `depth >= 3 && ply > 0 && beta.abs() < MATE_BOUND &&
!b.in_check(b.side) && b.has_non_pawn_material(b.side)`, using reduction
`r = 2 + depth / 4` and a fail-hard cutoff (`return beta`) on a null-window
recursive search that scores `>= beta`; `self.history` push/pop bracket the
null recursive call symmetrically, matching the existing real-move loop. The
frozen benchmark comparison is **3,166,392 nodes / 3,006,102 NPS** for the
baseline and **1,449,025 nodes / 2,887,852 NPS** for the candidate:
1,717,367 fewer nodes (**−54.24%**). A fresh finalizer re-bench reproduced
1,449,025 nodes exactly; NPS is timing-dependent and is not an Elo claim.

Validation was exact: three new board tests — `null_move_round_trips`,
`null_move_round_trips_with_ep` (post-1.e4 FEN, exercising the ep-file XOR
path), and `non_pawn_material_guard` — were added first and confirmed RED
(missing-method compile errors) before implementation, then GREEN.
`cargo test --release` had **34 passed, 0 failed, 1 ignored** (31 baseline +
3 new); all-target release clippy with `-D warnings` was clean; ignored
`perft_deep` was **1 passed, 34 filtered**, its chained asserts confirming
live startpos depth 6 = **119,060,324**. The UCI smoke returned `uciok`,
`readyok`, and legal `bestmove b1c3`; `git diff --check` was clean; the diff
was a pure addition across `board.rs` (+65) and `search.rs` (+20), no
deletions.

The definitive normalized fastchess SPRT at 8+0.08 (concurrency 2) ended when
**H1 was accepted** after **558 games: 120W/47L/391D** (120+47+391=558) for
candidate versus baseline. Relative self-play SPRT Delta-Elo was **+45.72 ±
14.48**, and LLR **+2.97** crossed the **+2.94** upper boundary. This is a
relative self-play development estimate, not an absolute anchored Elo. The
definitive log has exactly one `Finished match`, took 01:22:41, and recorded
no error, illegal-move, disconnect, crash, killed, or failed entry; an
independent recount of its `{...}` result tags (108 White-mates + 59
Black-mates + 390 threefold draws + 1 insufficient-material draw = 558)
matches the printed W/L/D exactly. **This SPRT run survived a mid-run
session restart** — the runner and fastchess subprocess continued as
orphaned processes across the restart and completed cleanly to the H1
boundary; the definitive fastchess config was recovered from the isolated
run directory afterward, and no killed/partial stub exists for this task
(unlike Task 8's OOM-killed first attempt).

The accepted source is retained. PVS and the history heuristic remain
absent; quiet-move ordering still falls back to the killer/unscored
precedence established in Task 7. The consecutive-null-move guard
(disallowing a null move immediately after another null move) was
intentionally omitted from this patch and is documented as future
hardening, not a defect. Cost: **$0**.

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
| 2026-07-17 | M1 Task 6 local aspiration-window experiment (accepted) | $0.00 | $0.00 |
| 2026-07-17 | M1 Task 7 local killer-move experiment (accepted) | $0.00 | $0.00 |
| 2026-07-18 | M1 Task 8 local history-heuristic experiment (rejected/reverted) | $0.00 | $0.00 |
| 2026-07-18 | M1 Task 9 local null-move-pruning experiment (accepted) | $0.00 | $0.00 |

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

Runtime-found magic bitboards (Task 4), PVS (Task 5), and the butterfly
history heuristic (Task 8) remain rejected and reverted under their frozen
gates. **Task 6, aspiration windows; Task 7, killer moves; and Task 9,
null-move pruning, are accepted and retained**; PVS remains absent, and the
history table is absent — quiet-move ordering falls back to the
killer/unscored precedence established in Task 7. **Task 10, reverse futility
pruning (static null-move), is next after review closure.** Task 11, late
move reductions, builds on the accepted full-window negamax loop
independently of history and does not depend on it. The remaining search
stack is reverse futility pruning, LMR, and better time management, followed
by the CCRL-anchored gauntlet for the first honest absolute rating. No magic
or history retry is planned; any future compact redesign requires separate
approval. Target: ~2300–2500.
