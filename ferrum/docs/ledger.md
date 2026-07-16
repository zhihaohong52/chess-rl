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

Runtime-found magic bitboards (Task 4) and PVS (Task 5) remain rejected and
reverted under their frozen gates. **Task 6, aspiration windows, and Task 7,
killer moves, are accepted and retained**; PVS remains absent. **Task 8,
history heuristic, is next after Task 7 review closure.** The remaining search
stack is null-move, LMR, and better time management, followed by the
CCRL-anchored gauntlet for the first honest absolute rating. No magic retry is
planned; any future compact redesign requires separate approval. Target:
~2300–2500.
