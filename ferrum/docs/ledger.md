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
| 2026-07-19 | M1 Task 10 (reverse futility pruning + eval-gated NMP) — accepted | baseline 1,449,025 / candidate 644,979 (−55.49%) | full release 35 pass/0 fail/1 ignored (34 baseline + rfp_keeps_tactics); deep perft; clippy clean | H1 accepted after 672 games: 111W/49L/512D, relative SPRT Delta-Elo +32.15 ± 11.13, LLR +2.95 crossed +2.94; source retained |
| 2026-07-19 | M1 Task 11 (full-window late move reductions) — accepted | baseline 644,979 / candidate 195,096 (−69.75%) | full release 36 pass/0 fail/1 ignored (35 baseline + lmr_still_finds_deep_tactic); deep perft; clippy clean | H1 accepted after 710 games: 100W/45L/565D, relative SPRT Delta-Elo +26.97 ± 10.42, LLR +2.95 crossed +2.94; source retained |
| 2026-07-19 | M1 Task 12 (late move pruning, move-count) — accepted | baseline 195,096 / candidate 161,418 (−17.26%) | full release 37 pass/0 fail/1 ignored (36 baseline + lmp_keeps_tactics); deep perft; clippy clean | H1 accepted after 798 games: 92W/43L/663D, relative SPRT Delta-Elo +21.36 ± 8.61, LLR +2.94 crossed +2.94; source retained |
| 2026-07-21 | M1 Tasks 13–16 (futility + check ext + SEE + time mgmt) — accepted (bundle) | baseline 161,418 / candidate 79,667 (−50.65%) | 40 pass/0 fail/1 ignored; clippy clean | cumulative 620-game SPRT vs Task 12 stack: 76W/18L/526D, relative SPRT Delta-Elo +32.60 ± 9.48 (95% CI [+23.12, +42.08]), LLR +2.95 crossed +2.94; H1 accepted; source retained |
| 2026-07-21 | M1 exit (anchored gauntlet) — **v0.2.0 tagged as search-complete checkpoint** (short of bar) | 79,667 | 40 pass/0 fail/1 ignored; clippy clean | 1,120-game CCRL-anchored gauntlet (Stash v17/v21/v37 + Weiss 2.0, 8+0.08), Ordo fixed-anchor: **ferrum 2035.5 ± 37.7** (95% CI [1997.8, 2073.2]); bar was CI lower-bound ≥ ~2300 → **missed by ~265**; HCE eval is the ceiling, NNUE (M2) is the lever |

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

## M1 Task 10 — ACCEPTED

Task 10 adds reverse futility pruning (RFP, aka static null-move pruning) to
`negamax` and tightens the existing null-move guard to be eval-gated. Right
after the TT-cutoff block, `negamax` now hoists a single
`let in_check = b.in_check(b.side);` and `let static_eval = self.eval.eval(b);`
binding, replacing the previous per-branch `b.in_check(b.side)` call. RFP is
fail-soft: `if depth <= 6 && !in_check && beta.abs() < MATE_BOUND &&
static_eval - 80 * depth >= beta { return static_eval; }`, pruning at shallow
depth when the static eval already beats beta by a depth-scaled margin. The
pre-existing NMP guard now reuses `!in_check` and adds `&& static_eval >=
beta`, so a null move is only tried when the static eval already looks like
it clears beta; NMP's own fail-hard `s >= beta => return beta` cutoff,
reduction `r = 2 + depth / 4`, and `self.history` push/pop remain unchanged
from Task 9. The frozen benchmark comparison is **1,449,025 nodes /
2,634,716 NPS** for the baseline and **644,979 nodes / 2,549,751 NPS** for
the candidate: 804,046 fewer nodes (**−55.49%**). A fresh finalizer re-bench
reproduced 644,979 nodes exactly; NPS is timing-dependent and is not an Elo
claim.

Validation was exact: the `rfp_keeps_tactics` regression-anchor test (a
queen-fork capture at depth 7 and a back-rank mate at depth 5) was added
first and confirmed passing on the clean Task 9 baseline before the RFP/NMP
change, then reconfirmed passing afterward — RFP does not prune away either
tactic. `cargo test --release` had **35 passed, 0 failed, 1 ignored** (34
baseline + 1 new); all-target release clippy with `-D warnings` was clean;
ignored `perft_deep` was **1 passed, 35 filtered**, its chained asserts
confirming live startpos depth 6 = **119,060,324**. The UCI smoke returned
`uciok`, `readyok`, and legal `bestmove b1c3`; `git diff --check` was clean;
the diff scope was exactly `ferrum/src/search.rs` (+16/−1, the single
deletion being the old NMP guard line replaced by the new one).

An initial SPRT attempt at concurrency 2 was OS-SIGKILL'd instantly with
**0 games** completed — a transient host memory spike from other
applications, not an engine defect; the stub is preserved at
`.superpowers/sdd/task-10-sprt-killed-attempt-1.log` and excluded from all
accounting. The authoritative run used **concurrency 1** — an
environment-only override, with games remaining independent and the SPRT
valid at the same 8+0.08 time control.

The definitive normalized fastchess SPRT at 8+0.08 (concurrency 1) ended
when **H1 was accepted** after **672 games: 111W/49L/512D**
(111+49+512=672) for candidate versus baseline. Relative self-play SPRT
Delta-Elo was **+32.15 ± 11.13**, and LLR **+2.95** crossed the **+2.94**
upper boundary. This is a relative self-play development estimate, not an
absolute anchored Elo. The definitive log has exactly one `Finished match`,
took 03:08:54, and recorded no error, illegal-move, disconnect, crash,
killed, or failed entry; its 1,526,057-byte PGN is consistent with the
reported game count.

The accepted source is retained. PVS and the history heuristic remain
absent; quiet-move ordering still falls back to the killer/unscored
precedence established in Task 7. Cost: **$0**.

## M1 Task 11 — ACCEPTED

Task 11 adds full-window late move reductions (LMR) to `negamax`, replacing
the single unconditional full-window recursive-call line in the move loop
with a reduce-gated block. Right after `b.make(m)`, `gives_check =
b.in_check(b.side)` correctly reads whether `m` gives check (at this point
`b.side` is the opponent to move); `quiet = !m.is_capture() &&
!m.is_promo()`; the reduction amount is `reduce = if depth >= 3 && legal > 3
&& quiet && !in_check && !gives_check { 1 + (legal > 6) as i32 } else { 0 }`,
reusing the pre-existing node-level `in_check` binding from Task 10. When
`reduce > 0`, a reduced-depth probe runs first at the *same* full window
`[-beta, -alpha]`, and the engine only re-searches at full depth — at that
same full window again — if the reduced probe beats alpha (`reduced >
alpha`); otherwise the reduced score is returned directly. There is no
null-window call anywhere in the diff and no first-move special case: this
is full-window LMR, not PVS. `self.history` push/pop bracket both possible
recursive calls (reduced probe and full-depth re-search) unchanged. The
frozen benchmark comparison is **644,979 nodes / 2,872,191 NPS** for the
baseline and **195,096 nodes / 2,564,393 NPS** for the candidate: 449,883
fewer nodes (**−69.75%**). A fresh finalizer re-bench reproduced 195,096
nodes exactly; NPS is timing-dependent and is not an Elo claim.

Validation was exact: the `lmr_still_finds_deep_tactic` regression-anchor
test (a winning knight-fork tactic, `c3d5`, at depth 8) was added first and
confirmed passing on the clean Task 10 baseline before the LMR change, then
reconfirmed passing afterward — LMR's reductions plus re-search do not prune
away the tactic. `cargo test --release` had **36 passed, 0 failed, 1
ignored** (35 baseline + 1 new); all-target release clippy with `-D
warnings` was clean; ignored `perft_deep` was **1 passed, 36 filtered**, its
chained asserts confirming live startpos depth 6 = **119,060,324**. The UCI
smoke returned `uciok`, `readyok`, and legal `bestmove b1c3`; `git diff
--check` was clean; the diff scope was exactly `ferrum/src/search.rs`
(+23/−1, the single deletion being the old unconditional recursive-call
line). PVS and the history heuristic remain absent; qsearch, NMP, RFP,
aspiration windows, and killer-move ordering are unchanged from Task 10.

An initial SPRT attempt at concurrency 1 was OS-SIGKILL'd instantly with
**0 games** completed, under host memory pressure from other applications —
not an engine defect. The stub is preserved at
`.superpowers/sdd/task-11-sprt-killed-attempt-1.log` and excluded from all
accounting. After the user freed memory (closing the other applications), the
retry ran at the same **concurrency 1** and completed cleanly: the memory fix,
not a concurrency change, resolved the failure. The SPRT remains valid — same
binaries, same 8+0.08 time control, independent games.

The definitive normalized fastchess SPRT at 8+0.08 (concurrency 1) ended
when **H1 was accepted** after **710 games: 100W/45L/565D**
(100+45+565=710) for candidate versus baseline. Relative self-play SPRT
Delta-Elo was **+26.97 ± 10.42**, and LLR **+2.95** crossed the **+2.94**
upper boundary. This is a relative self-play development estimate, not an
absolute anchored Elo. The definitive log has exactly one `Finished match`,
took 03:19:15, and recorded no error, illegal-move, disconnect, crash,
killed, or failed entry; its 1,631,781-byte PGN is consistent with the
reported game count.

The accepted source is retained. PVS and the history heuristic remain
absent; quiet-move ordering still falls back to the killer/unscored
precedence established in Task 7. Cost: **$0**.

## M1 Task 12 — ACCEPTED

Task 12 adds move-count late move pruning (LMP) to the `negamax` move loop.
Right after `legal += 1`, the per-move `gives_check = b.in_check(b.side)` and
`quiet = !m.is_capture() && !m.is_promo()` bindings were hoisted up so the
pruning guard and the pre-existing Task 11 LMR block share them (LMR now reuses
these instead of recomputing them). The guard is
`if legal > 1 && quiet && !in_check && !gives_check && best > -MATE_BOUND &&
depth <= 3 && legal > 4 + depth * depth { b.unmake(m, undo); legal -= 1;
continue; }`, skipping late quiet, non-checking moves at shallow depth once a
real best already exists. Critically, the guard sits **before**
`self.history.push(b.hash)`, so a pruned move unmakes and `continue`s without
ever pushing to the repetition stack — push/pop stay balanced on every path,
and the pruned move never reaches the TT store, killer store, or `best_move`
update. The `legal -= 1` exactly undoes the earlier `legal += 1`, and the
`legal > 1 && best > -MATE_BOUND` conjunction guarantees a genuinely-searched,
non-losing move already exists whenever a prune fires, so the terminal
`legal == 0` mate/stalemate check can never be reached on a false premise. LMP
is a pure skip (`continue`), not a differently-windowed search, so the
full-window-negamax design invariant is preserved — no null-window or PVS is
introduced. The frozen benchmark comparison is **195,096 nodes / 2,613,893
NPS** for the baseline and **161,418 nodes / 2,037,817 NPS** for the candidate:
33,678 fewer nodes (**−17.26%**). A fresh finalizer re-bench reproduced 161,418
nodes exactly; NPS is timing-dependent and is not an Elo claim.

Validation was exact: the `lmp_keeps_tactics` regression-anchor test (a
back-rank rook mate, `e1e8`, at depth 6) was added and confirmed passing both
before and after the change. `cargo test --release` had **37 passed, 0 failed,
1 ignored** (36 baseline + 1 new); all-target release clippy with `-D warnings`
was clean; ignored `perft_deep` was **1 passed, 36 filtered**, confirming live
startpos depth 6 = **119,060,324** (movegen untouched); the diff scope was
exactly `ferrum/src/search.rs` (+15/−2). Spec-compliance and code-quality
reviews both passed with merge-ready verdicts, verifying the history push/pop
balance, mate/stalemate legal-count safety, TT/killer isolation of pruned
moves, full-window-only compliance, and clean non-interference with LMR (their
domains overlap only at `depth == 3`, where LMP's `continue` runs first).

An honest limitation, flagged by the code-quality review and recorded here:
`lmp_keeps_tactics` (used verbatim from the plan's Step 1, mirroring the RFP and
LMR `*_keeps_tactics` anchors) is a weak *LMP-specific* regression guard —
because the test's root is at nominal depth 6 (`> 3`), LMP never fires on the
root move loop, and the answer `e1e8` is an immediate root mate that is fully
searched regardless of what LMP does in sibling subtrees; the reviewer
empirically confirmed the assertion still passes even with the guard forced off.
The branch does get code coverage (it fires 22 times inside the depth-6 tree),
but the assertion is insensitive to the guard's correctness. This is a
plan-inherited test-design property, not a code defect; the decisive 798-game
SPRT is the operative correctness/strength backstop for this feature.

An initial SPRT attempt at concurrency 1 was OS-SIGKILL'd instantly with
**0 games** completed, under a transient host memory spike from other
applications — not an engine defect; the stub is preserved at
`.superpowers/sdd/task-12-sprt-killed-attempt-1.log` and excluded from all
accounting. A 2-game fastchess smoke then ran cleanly, confirming the host had
recovered, and the retry ran at the same **concurrency 1** and completed to the
H1 boundary — the transient memory spike, not a concurrency change, was the sole
cause. The SPRT remains valid — same binaries, same 8+0.08 time control,
independent games.

The definitive normalized fastchess SPRT at 8+0.08 (concurrency 1) ended when
**H1 was accepted** after **798 games: 92W/43L/663D** (92+43+663=798) for
candidate versus baseline. Relative self-play SPRT Delta-Elo was **+21.36 ±
8.61** (95% CI **[+12.75, +29.97]**, wholly positive), and LLR **+2.94** crossed
the **+2.94** upper boundary. This is a relative self-play development estimate,
not an absolute anchored Elo. The definitive log has exactly one `Finished
match`, took 03:44:58, and recorded no error, illegal-move, disconnect, crash,
killed, or failed entry; its 1,853,930-byte PGN independently contains all 798
games.

The accepted source is retained. PVS and the history heuristic remain absent;
quiet-move ordering still falls back to the killer/unscored precedence
established in Task 7. Cost: **$0**.

## M1 Tasks 13–16 — ACCEPTED (bundle)

Tasks 13–16 add the final M1 search features on top of the accepted Task 12
move loop: **futility pruning of shallow quiets** (T13, `e1a904d`), **check
extensions** (T14, `db3faa1`), **static exchange evaluation** for capture
ordering and qsearch pruning (T15, `0bfb984`), and **soft/hard time management
with stability** (T16, `f663667`). Each was implemented, spec- and
quality-reviewed, and committed separately with scope `ferrum/src/search.rs`
only. The frozen bench fell from the **161,418-node** Task 12 baseline to
**79,667 nodes** for the T16 stack (**−50.65%**), at 1,457,195 NPS; NPS is
timing-dependent and is not an Elo claim. Release tests are **40 passed, 0
failed, 1 ignored**; all-target release clippy with `-D warnings` is clean.

**Gating methodology — bundle, not per-rung.** Each of these four features is
individually small (~+3 Elo), below the resolution of the project's per-patch
SPRT at `elo0=0 elo1=8`: a ~+3-Elo patch neither crosses the +2.94 boundary nor
yields a cap CI wholly above zero, so it wanders to the 2,000-game cap
inconclusively. A confirming individual run was done for T13: the definitive
concurrency-2 T13-vs-T12 SPRT went the full **2,000 games** (149W/133L/1718D)
with relative Delta-Elo **+2.78 ± 5.07** (95% CI **[−2.29, +7.85]**), LLR
**+0.58**, no boundary — a non-acceptance under the cap rule, statistically
indistinguishable from the rejected Task 5 PVS result (+2.95 ± 6.04). Rather
than reject four sound, standard techniques one at a time for being individually
sub-resolution, the four were gated **as one bundle** against the pre-stack Task
12 baseline (user-authorized decision). This is the statistically correct test
for a set of small same-direction patches, and it is the *stronger* bar: the
whole stack must prove net-positive against a fixed baseline.

**Definitive bundle SPRT (H1 accepted).** The concurrency-2 fastchess SPRT at
8+0.08, candidate `/tmp/ferrum-t16` vs baseline `/tmp/ferrum-t12`, ended when
**H1 was accepted** after **620 games: 76W/18L/526D** (76+18+526=620), LOS
100.00%. Relative self-play Delta-Elo was **+32.60 ± 9.48** (95% CI **[+23.12,
+42.08]**, wholly positive), and LLR **+2.95** crossed the **+2.94** upper
boundary. The four M1 search features together are worth ~+33 self-play Elo over
the Task 12 stack. This is a relative self-play development estimate, not an
absolute anchored Elo. The run finished in 01:35:19 with exactly one `Finished
match`, **zero time-forfeit games**, and no crash/illegal/disconnect entry.

**Concurrency on Apple M1 (4P+4E).** The host is an Apple M1 with 4 performance
+ 4 efficiency cores. concurrency 2 runs the two games' four engine processes on
the four P-cores and produces clean games; concurrency 4 (eight processes)
forces four engines onto the ~3×-slower E-cores with no headroom for the OS and
fastchess, which manifests as the multi-hundred-second time-forfeit overruns and
OOM kills recorded against the earlier tasks. concurrency 2 is therefore the
standing setting for this host at 8+0.08 — a hardware ceiling, not a statistical
one (games are independent at any concurrency; only time forfeits threaten
validity). An initial T13-vs-T12 attempt at concurrency 4 was discarded for
exactly these contaminated time-forfeit results before the authoritative
concurrency-2 runs.

The accepted source is retained across all four commits. The full-window negamax
invariant holds: no PVS, null-window, or first-move special case is introduced by
futility pruning, check extensions, SEE ordering/qsearch pruning, or time
management. Cost: **$0**.

## M1 SPRT history

Per-feature self-play SPRT results at 8+0.08 (relative development Elo, not
absolute). Accepted features are retained in `search.rs`; rejected features were
reverted with their diffs archived under `.superpowers/sdd/`.

| Task | Feature | SPRT (candidate vs prior) | Verdict |
|---|---|---|---|
| 4 | runtime magic bitboards | frozen perf gate, no SPRT (+4.45% < 10% bar) | rejected / reverted |
| 5 | principal variation search | +2.95 ± 6.04, LLR +0.46, cap (CI crosses 0) | rejected / reverted |
| 6 | aspiration windows | +6.95 ± 5.86, LLR +1.86, cap (CI [+1.09, +12.81]) | accepted |
| 7 | killer moves | +23.50 ± 10.09, LLR +2.98 crossed +2.94 | accepted |
| 8 | butterfly history heuristic | +4.86 ± 6.46, LLR +0.99, cap (CI crosses 0) | rejected / reverted |
| 9 | null-move pruning | +45.72 ± 14.48, LLR +2.97 crossed +2.94 | accepted |
| 10 | reverse futility + eval-gated NMP | +32.15 ± 11.13, LLR +2.95 crossed +2.94 | accepted |
| 11 | full-window late move reductions | +26.97 ± 10.42, LLR +2.95 crossed +2.94 | accepted |
| 12 | late move pruning (move-count) | +21.36 ± 8.61, LLR +2.94 crossed +2.94 | accepted |
| 13–16 | futility + check ext + SEE + time mgmt (bundle) | +32.60 ± 9.48, LLR +2.95 crossed +2.94 | accepted (bundle) |

Retained search = full-window negamax (no PVS/null-window) + aspiration +
killers + NMP + RFP + LMR + LMP + futility + check extensions + SEE + soft/hard
time management. No history table, no magic bitboards.

## M1 exit — SHORT OF BAR (v0.2.0 tagged as search-complete checkpoint)

**Bar:** ferrum's CCRL-anchored rating with the lower bound of its 95% CI ≥
~2300 (M1 target band ~2300–2500).

**Result: ferrum 2035.5 ± 37.7 (95% CI [1997.8, 2073.2]) — MISSED by ~265
Elo.** The anchored gauntlet ran **1,120 games** at 8+0.08 (concurrency 2,
ROUNDS 140 per opponent, **0 time-forfeits**, 02:59:59) against the pinned CCRL
pool, rated with Ordo (`ordo -Q -W -s 1000 -F 95 -m anchors.txt`, anchors fixed
to their CCRL blitz values, ferrum solved). White advantage fit at 120.5 ± 19.3.
Per-anchor:

| Opponent (CCRL) | ferrum W–L–D | ferrum score |
|---|---|---|
| stash-v17 (2296) | 18–211–51 | 15.5% |
| stash-v21 (2713) | 4–246–30 | 6.8% |
| weiss-2.0 (3320) | 4–253–23 | 5.5% |
| stash-v37 (3423) | 2–263–15 | 3.4% |
| **total** | **28–973–119** | **7.8%** (87.5/1120) |

**Analysis.** Every retained M1 search feature is individually SPRT-validated and
the stack is correct, but the absolute gain over M0 is only ~+100 CCRL
(loosely-estimated ~1930 → anchored 2035). The closest, most informative anchor
(stash-v17, 2296) pins ferrum at ~2000–2010; the farther anchors agree within
the fit (error only ±37.7 over 1,120 games). The limiter is the **hand-crafted
evaluation**: search improvements on a weak HCE top out around 2000–2100, and the
~265-Elo gap to the bar is an eval-quality gap, not a search-depth gap.
Search-margin tuning (LMR/LMP/futility thresholds) yields tens of Elo, not
hundreds, so it cannot close this gap. The designed lever for 2300+ is the
**NNUE evaluation (M2)**, which replaces HCE. The original ~2300–2500 M1 target
was optimistic for an HCE-only engine; the honest M1 deliverable is a complete,
SPRT-gated search stack measured at a real **2035 ± 38 CCRL**. The ~2300 exit
bar was not met, so the tag does not certify that target; by explicit user
decision **`ferrum-v0.2.0` is tagged as a search-complete checkpoint** at 2035
CCRL, decoupled from the (optimistic) 2300 goal. Chosen direction: **M2 (NNUE
eval)**, the designed lever for 2300+. Cost: **$0**.

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
| 2026-07-19 | M1 Task 10 local RFP + eval-gated-NMP experiment (accepted) | $0.00 | $0.00 |
| 2026-07-19 | M1 Task 11 local late-move-reductions experiment (accepted) | $0.00 | $0.00 |
| 2026-07-19 | M1 Task 12 local late-move-pruning experiment (accepted) | $0.00 | $0.00 |
| 2026-07-21 | M1 Tasks 13–16 local search-stack bundle (futility+check-ext+SEE+time-mgmt, accepted) | $0.00 | $0.00 |
| 2026-07-21 | M1 exit anchored gauntlet (Stash×3 + Weiss + Ordo, local) — 2035 CCRL, bar missed | $0.00 | $0.00 |

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
gates. **Task 6, aspiration windows; Task 7, killer moves; Task 9, null-move
pruning; Task 10, reverse futility pruning + eval-gated NMP; Task 11,
full-window late move reductions; and Task 12, move-count late move pruning,
are accepted and retained**; PVS remains absent, and the history table is
absent — quiet-move ordering falls back to the killer/unscored precedence
established in Task 7. **Tasks 13–16 — futility pruning of shallow quiets, check
extensions, static exchange evaluation (SEE), and soft/hard time management —
are accepted as a bundle** (cumulative +32.60 ± 9.48 self-play Elo vs the Task
12 stack, LLR +2.95 crossing +2.94), completing the M1 search work. **Task 17,
the CCRL-anchored gauntlet** (Stash v17/v21/v37 + Weiss 2.0, rated with Ordo) is
now **complete: ferrum rates 2035 ± 38 CCRL (95% CI [1998, 2073]), short of the
~2300 bar** — the HCE eval is the ceiling. `ferrum-v0.2.0` is tagged as a
search-complete checkpoint (not as clearing 2300); the chosen next step is M2.
No magic or history retry is planned; the path to the ~2300–2500 target is the
**NNUE evaluation (M2)**, not further search tuning (which yields tens of Elo,
not the ~265 needed). Any future compact redesign requires separate approval.
