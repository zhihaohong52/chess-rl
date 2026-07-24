# ferrum M6 — gen-2 self-play NNUE — design spec

**Date:** 2026-07-24
**Status:** design, pre-plan
**Baseline:** `ferrum-v0.4.0` (commit `3430cbc`, 2901.5 ± 25.2 CCRL), on branch
`feat/ferrum-m4-search` (local/unmerged; also carries the M5 PVS rejection at
`c111f09`).
**Predecessor levers, all closed:** M3 (eval architecture — saturated at the
ChessBench data ceiling), M4 (search modernization — only move ordering paid),
M5 (PVS — rejected, fast-TC search compute-saturated).

---

## 1. Goal and success criteria

Execute **one proven turn of the self-play flywheel**: build a self-play data
generator, train a gen-2 NNUE on ChessBench + self-play data with *real game
outcomes*, and SPRT-gate it against v0.4.0.

**Success = both of:**
1. A gen-2 net that beats the gen-0 net by a gated margin (SPRT LLR ≥ +2.94 at
   elo0=0, elo1=8, 8+0.08), **and**
2. A reusable, documented datagen → convert → train → gate pipeline that gen-3
   can re-run with a different recipe.

**Honest expectation:** a single self-play generation at this strength typically
yields **~+30–80 Elo**. That likely does **not** clear the 3000 CI-lower-bound
bar on its own — M7+ would be further generations (self-play from the improved
net, accumulating data). M6 is one turn, not the whole climb.

A gated **negative** is still a valid milestone outcome (as M3/M5 were): it would
say self-play data at ~2900 strength does not lift the eval, redirecting effort.
The pipeline is retained regardless.

---

## 2. Why self-play, precisely (the hypothesis)

bullet's training target for a value net is:

```
target = wdl · game_result + (1 − wdl) · sigmoid(score / scale)
```

where `wdl` is a **global scalar weight** on the game-result term (bullet's
`ConstantWDL{value}`), `game_result ∈ {0, 0.5, 1}` is the side-to-move-relative
game outcome, and `score` is the side-to-move-relative label eval.

Gen-0 was trained with **`wdl = 0` (score-only)**. This was forced, not chosen:
ChessBench provides only a Stockfish win% per position, and its "result" is
*derived from that same win%* — so the `game_result` term carried **zero
information independent of the `score` term**. Turning the blend on would have
added noise, not signal (documented in the M3 ledger: "WDL-blending cannot help
… ChessBench's WDL and score targets both derive from the same Stockfish win%").

**Self-play breaks this.** Playing games to completion produces a `game_result`
that is the *actual outcome* of the game the position occurred in — genuinely
independent of the per-position search score. For the first time the blend term
carries real signal (roll-out value), which the score term (a fixed-depth static
lookahead) cannot express.

**Turning that blend on is the entire M6 experiment.** Every other variable is
held fixed so the result is attributable to the data.

---

## 3. Scope boundaries (non-goals)

- **Architecture is frozen** at gen-0's `768 → 512×2 → 1` SCReLU perspective net
  with plain `Chess768` inputs. No king-buckets, no larger hidden layer. The M3
  bucket engine code remains available for a *later* generation, but mixing an
  architecture change into M6 would confound the data result. **Because the
  architecture is identical, ferrum needs no inference-code change** — gen-2 is a
  drop-in `EVALFILE` swap (same `FeNN` v1 header, same forward pass).
- **No search changes.** The v0.4.0 search is the fixed substrate.
- **Not the full 3000 climb.** One generation only.
- **No new engine as labeler.** Positions are labeled by ferrum-v0.4.0's own
  search during self-play — this is bootstrapped self-play, not distillation from
  a stronger teacher. (The gain comes from breadth + real outcomes, not from
  superhuman labels; see §11.)

---

## 4. Pipeline overview

Four stages, each a well-bounded unit with a file/byte interface:

```
[1] datagen          [2] convert            [3] train              [4] gate
ferrum selfplay  →   bullet-utils convert → bullet (A6000)     →   sprt.sh
(N parallel      →   text shards →              gen-2 net       →   gen2 vs gen0
 workers)            bulletformat .data         (2 variants)        8+0.08 SPRT
    │                     │                         │                   │
 FEN|score|result    32-byte records          FeNN gen2.bin        accept → gauntlet
 text shards         (+ 63M ChessBench)                            reject → ledger
```

Stage boundaries are **portable files**: stage 1 emits host-agnostic text; stage
2 emits architecture-independent bulletformat bytes; stage 3 emits a `FeNN` net;
stage 4 consumes two nets. Any stage can run on any host (see §9).

---

## 5. Stage 1 — datagen (`ferrum selfplay` subcommand)

A new `selfplay` subcommand in the ferrum binary. Plays fixed-node self-play
games from diversified openings, filters to quiet positions, and streams
`FEN | score | result` text to a shard file. Reuses the existing
`board`/`movegen`/`search`/`eval` modules unchanged.

### 5.1 Invocation

```
ferrum selfplay --seed <u64> --games <N> --nodes <n> --out <path> [--shard-size <M>]
```

- `--seed` — per-worker PRNG seed (distinct per worker → decorrelated games).
- `--nodes` — fixed nodes per move (default **5,000**). No clock → deterministic
  across hosts and immune to the concurrency-2 time-forfeit ceiling that binds
  8+0.08 SPRTs. Higher = better labels, fewer positions/hour (see §9 table).
- `--out` — shard file prefix; workers write `<out>.<seed>.NNNN.txt`.
- `--shard-size` — positions per shard file (default 1,000,000) so a crash or a
  spot preemption loses only the current shard.

### 5.2 RNG

ferrum is currently deterministic (no RNG). Add a small seeded **xorshift64**
generator, used **only** for opening-move selection. Seeding per worker makes
parallel workers explore different games; a fixed seed makes a single worker's
run reproducible (required by the determinism test, §10).

### 5.3 Opening diversity

For each game: pick a random line from `books/openings_m1.epd`, then play
**`--random-plies` (default 8)** uniformly-random *legal* moves before recording
starts. This decorrelates games so the corpus is not dominated by one opening
tree. Positions during the random-ply prefix are **not** recorded.

### 5.4 Game loop

From the post-opening position, until the game terminates:
1. Search the position at `--nodes` fixed nodes → `(best_move, stm_score)`.
2. **Candidate-record** the position with its `stm_score` (filtering in §5.5
   decides whether it is emitted).
3. Play `best_move`; update the running position.

**Termination** (natural): checkmate, stalemate, insufficient material,
50-move rule, threefold repetition. **Adjudication** (for clean labels + speed —
unlike an SPRT, datagen *wants* decided games ended early):
- **Win/loss adjudication:** `|stm_score| ≥ 1000` cp for **4 consecutive plies**
  → adjudicate the higher side the winner.
- **Draw adjudication:** `|stm_score| ≤ 10` cp for **8 consecutive plies** after
  move 40 → adjudicate draw.

The final outcome `O ∈ {white_win, black_win, draw}` labels every recorded
position from that game.

### 5.5 Quiet-position filter (matches the gen-0 convention)

A candidate position is **dropped** if any of:
- the side to move is **in check**;
- the position's **best move is a capture or gives check** (train on quiet
  positions — a noisy best move means the static eval is not the operative
  signal);
- the score is a **mate score** (`|score| ≥ MATE_BOUND`);
- its **Zobrist key was already emitted** in this worker's run (dedup via an
  in-process `HashSet<u64>`; per-worker dedup is sufficient at pilot scale,
  global cross-shard dedup is a documented optional post-pass).

Expected survival ≈ 55–65% of plies.

### 5.6 Output record format

One line per surviving position:

```
<FEN> | <score_cp> | <wdl>
```

- `<FEN>` — the real board FEN (`board.to_fen()`). bullet's `convert --from text`
  reads **only** placement + side-to-move; castling/ep/clocks are ignored
  (verified in the M2 spike, `nnue/README.md`).
- `<score_cp>` and `<wdl>` are **White-relative** (bullet's text convention: the
  loader mirrors to side-to-move-relative internally using the FEN's STM). So:
  `score_cp = stm_score` if White to move else `−stm_score`; `wdl ∈ {1.0, 0.5,
  0.0}` for White-win / draw / White-loss.

> **Implementation gate (byte-verify before any large run):** the exact
> White-vs-STM convention and the `wdl` token format (`1.0/0.5/0.0` float vs
> `0/1/2` int) MUST be confirmed against `bulletformat`'s `ChessBoard::from_str`
> at the pinned commit `cebc78a` by converting a handful of hand-checked
> positions and reading the packed 32-byte records back (score sign and result
> byte STM-relative as expected). **Proven fallback if it does not verify:** emit
> in the canonical (STM-as-White) frame with a synthetic FEN `<placement> w - - 0
> 1` and STM-relative score/result — the exact encoding gen-0 used and round-trip
> proved in M2.

### 5.7 Parallelism

Datagen is embarrassingly parallel: independent games, no shared state. A
launcher (`tools/datagen.sh`) spawns **P worker processes** (P = core count),
each with a distinct `--seed`, each writing its own shard files. Linear scaling
to any box (§9). Runs under the proven setsid double-fork daemon; a sentinel file
marks completion and a running position count is logged for monitoring.

---

## 6. Stage 2 — convert and corpus assembly

1. **Self-play → bulletformat.** Concatenate the worker text shards and pack:
   ```
   cargo run --release -p bullet-utils -- convert --from text \
     --input selfplay.txt --output selfplay.data --threads <n>
   ```
   32 bytes/position → ~1.3 GB per 40M positions.
2. **ChessBench → bulletformat.** Re-run the existing `tools/chessbench_to_bullet.py`
   over the 63M ChessBench corpus (npz on HF, `james77mill/chessbench-encoded-npz`)
   to reproduce gen-0's training data, **assigning each record a result byte**
   bucketed from its own score (`win% > 0.6 → win`, `< 0.4 → loss`, else `draw`).
   This makes the blend ≈ a no-op on the ChessBench fraction (its result ≈ its
   score-implied result) while the self-play fraction carries real signal.
3. **Interleave.** Produce the mixed training set (bullet reads a list of `.data`
   files sequentially; shuffle-interleave shards so batches mix sources).

### 6.1 The mixed-WDL subtlety (load-bearing)

bullet's `wdl` weight is **global** — it cannot distinguish ChessBench (derived
result) from self-play (real result). The mix is only sound because, for
ChessBench, `game_result ≈ sigmoid(score)` by construction (§2), so the blend
term barely perturbs those records regardless of the weight, while it injects
genuine roll-out signal on the self-play records. The residual perturbation is
the bucketing quantization error (continuous `sigmoid(score)` vs a 3-level
result), which is small. **If this perturbation proves harmful** (pilot SPRT
worse than gen-0 and diagnosis points at the ChessBench fraction), the **clean
fallback is the fine-tune variant** (§7), which sidesteps mixed-WDL entirely.

---

## 7. Stage 3 — training (gen-2, A6000 via bullet)

Trainer: bullet at pinned `cebc78a`, A6000 on ThunderCompute (recipe in
`nnue/README.md` §"Cloud training"). Config mirrors gen-0's `simple.rs`-derived
setup (dual_perspective, `Chess768`, HIDDEN=512, SCReLU, AdamW, QA=255/QB=64/
SCALE=400, `FeNN` v1 export) with **one change: `wdl_scheduler` blend on**.

**De-risk the recipe with two cheap variants** (~$1–2 each; same pattern as M3's
1024-vs-512), SPRT the survivor(s) vs gen-0:

| variant | data | init | `wdl` weight | rationale |
|---|---|---|---|---|
| **V1 mix-from-scratch** (primary) | ChessBench 63M + self-play, interleaved | random | 0.3–0.5 | retains gen-0 breadth, adds self-play signal; best odds of a gated gain |
| **V2 fine-tune** (fallback/clean) | self-play only | **gen-0 weights** | 0.4 | thin-corpus friendly, no mixed-WDL concern; clean attribution of the self-play delta |

Superbatch count tuned to corpus size (gen-0 used 40; the mix is larger). Export
`gen2.bin` with the identical `FeNN` v1 header so ferrum loads it unchanged.

**Delete the instance immediately** on completion (billing = instance lifetime;
snapshot only if a re-train is imminent). Record actual $ + wall-clock in the
ledger.

---

## 8. Stage 4 — SPRT gate

**Net-vs-net** SPRT: candidate = v0.4.0 binary + `gen2.bin`, baseline = v0.4.0
binary + `gen0.bin`. Same binary, **different net per engine** — precedent: M3
gen-1-vs-gen-0.

### 8.1 Tooling change

`tools/sprt.sh` currently loads one `EVALFILE` into **both** engines. Add
**per-engine overrides** `CAND_EVALFILE` / `BASE_EVALFILE` (falling back to the
shared `EVALFILE` when unset — backward compatible). Search-feature SPRTs keep
using the shared form; net SPRTs set the two.

### 8.2 Parameters

- TC 8+0.08, concurrency 2 (the clean-games ceiling on this Apple-M1 host).
- `elo0 = 0, elo1 = 8, α = β = 0.05`. Accept LLR ≥ +2.94; reject ≤ −2.94;
  **inconclusive at the game cap → reject** (pre-committed, as M4 B4 / M5).
- Both nets loaded via the per-engine EVALFILEs (else HCE-vs-HCE — the standard
  trap).

### 8.3 Verdict paths

- **Accept** → run the anchored exit gauntlet (identical methodology to
  M2/M3/M4: 1120 games vs Stash v17/v21/v37 + Weiss 2.0, 8+0.08, cc2, fixed-anchor
  Ordo) for an absolute CCRL number; ledger the result; **tag `ferrum-v0.5.0`**;
  upload the self-play corpus to HF (`james77mill/...`) as the gen-3 seed.
- **Reject** → ledger the negative (Elo point estimate, CI, LLR, games, node
  budget, position count); **keep** the datagen pipeline + tooling; no tag.

---

## 9. Host runbook (deferred choice)

The datagen tool is **host-agnostic**: portable text output, same daemon recipe,
runs on the Mac or any rented x86/ARM Linux box (ARM needs only a one-line
`aarch64` target add — ferrum is std-only Rust; bulletformat output is
architecture-independent). Pick the row at execution time.

Modeled from ferrum's measured **1.10M NPS** at 5,000 nodes/move (±40% until the
first real shard calibrates): ~100–120 positions/sec per M1-P-core-equivalent →
work of **~56 / 111 / 278 M1-P-core-hours** for **20M / 40M / 100M** positions.

| platform | config | eff. cores | 20M pilot | 40M | 100M | ~cost (pilot) |
|---|---|---|---|---|---|---|
| **Local M1** | 4P+4E | ~5.5 | ~10 h | ~20 h | ~50 h | **$0** |
| Oracle A1 spot | 64 OCPU (ARM) | ~35 | ~1.6 h | ~3.2 h | ~8 h | ~$0.5 |
| Alibaba c7a.16xl | 64 vCPU | ~42 | ~1.3 h | ~2.6 h | ~6.6 h | ~$2 / ~$0.5 spot |
| Alibaba c7a.32xl | 128 vCPU | ~83 | ~0.7 h | ~1.3 h | ~3.3 h | ~$2.3 / ~$0.6 spot |
| GCP t2a-48 | 48 vCPU (ARM) | ~26 | ~2.1 h | ~4.3 h | ~11 h | ~$3 / ~$1 spot |

**Default: local M1 for the pilot.** The ~10 h is overnight dead time; renting
collapses it to ~1–2 h but for ~$1–2 buys time that cannot be used (asleep, no
SPRT to run until a net exists) and adds a new-provider ops surface. Renting
earns its keep only at the **100M+ scale-up** (local ~50 h vs rented ~3–7 h),
where a proven-positive pilot justifies the setup; `c7a.32xl` is the wall-clock
winner there and the cost delta is noise. All data-volume decisions are pilot-
first: **~20–40M self-play positions**, then scale only on positive signal.

---

## 10. Testing strategy

Preserve the existing 64-test suite; add datagen-focused tests (crate is
bin-only → `cargo test --bin ferrum`):

- **Filter unit tests:** an in-check position is dropped; a position whose best
  move is a capture is dropped; a position whose best move gives check is
  dropped; a mate-score position is dropped; a quiet position is kept.
- **Determinism:** two `selfplay` runs with the same seed produce byte-identical
  output; different seeds diverge.
- **Result consistency:** within one game, every emitted position's `wdl` is the
  same game outcome mapped through its own side-to-move (White-to-move and
  Black-to-move positions of a White win get `1.0` and `0.0` respectively).
- **Score sanity:** emitted scores are STM-relative and finite; a spot re-eval of
  a sample correlates (not garbage/sign-flipped).
- **Round-trip smoke (the §5.6 gate):** a sample of emitted lines converts via
  `bullet-utils` without error, packed record count == input line count, and the
  packed `score`/`result` bytes are STM-relative as expected.
- **sprt.sh:** `CAND_EVALFILE`/`BASE_EVALFILE` load distinct nets; unset falls
  back to shared `EVALFILE` (a dry `--help`/config-echo check).

---

## 11. Risks and mitigations

1. **Mixed-WDL semantics** (§6.1) — the global `wdl` weight cannot separate real
   from derived results. *Mitigation:* the ChessBench result≈score identity makes
   it near-harmless; V2 fine-tune is the clean fallback if not.
2. **Bootstrap ceiling** — self-play is labeled by ferrum-v0.4.0's own ~2900
   search, so labels cannot exceed its eval horizon. The gain is from **breadth**
   (positions ChessBench under-samples) **+ real outcomes** (roll-out value the
   static score cannot express), not from superhuman labels. A flat pilot would
   be a genuine, informative negative, not a bug.
3. **Convention/sign bugs** at the text↔bulletformat and STM↔White boundaries —
   the highest-probability implementation error. *Mitigation:* the §5.6 byte-
   verify gate before any large run; the M2 canonical-frame fallback.
4. **Datagen throughput off-model** (±40%) — the pilot's first shard calibrates
   positions/sec; adjust `--nodes`/volume before committing a full run.
5. **Self-play draw rate too high** (equal-strength mirror match) — dilutes
   signal. *Mitigation:* the random-ply openings + adjudication; if draws still
   dominate, raise `--random-plies` or widen the opening book.

---

## 12. Cost and budget

- Datagen: ~$0 (local) to ~$4 (rented pilot).
- Training: ~$1–2 per variant on A6000 × up to 2 variants = ~$2–4.
- SPRT + gauntlet: $0 (local).
- **M6 total ~$2–4**, cumulative **~$4–6** of the ~$20–25 approved (hard alerts
  at $10/$20). Well clear.

---

## 13. File manifest

**New:**
- `ferrum/src/selfplay.rs` — datagen game loop, adjudication, filter, emit.
- `ferrum/tools/datagen.sh` — parallel worker launcher (under the setsid daemon).
- `ferrum/nnue/gen2_train.rs` — bullet training config (V1 + V2).

**Modified:**
- `ferrum/src/main.rs` — dispatch the `selfplay` subcommand.
- `ferrum/src/types.rs` (or `selfplay.rs`) — seeded xorshift64 PRNG.
- `ferrum/tools/sprt.sh` — per-engine `CAND_EVALFILE`/`BASE_EVALFILE`.
- `ferrum/tools/chessbench_to_bullet.py` — add result-byte bucketing for the mix.
- `ferrum/nnue/README.md` — gen-2 datagen + training record.
- `ferrum/docs/ledger.md` — M6 section.

**Produced (gitignored / external):** self-play text + `.data` shards, `gen2.bin`,
HF corpus upload.

---

## 14. Exit criteria

- `ferrum selfplay` produces filtered, format-verified shards; datagen tests
  pass; clippy clean.
- The §5.6 round-trip byte-verification passed before the full datagen run.
- gen-2 trained (≥1 variant); SPRT vs gen-0 run to a verdict at the pre-committed
  boundaries.
- **Accept:** anchored gauntlet + Ordo done, ledger updated, `ferrum-v0.5.0`
  tagged, corpus on HF.
- **Reject:** ledger records the negative with full numbers; pipeline retained;
  no tag.
- In all cases: no cloud instance left running; actual spend + wall-clock
  recorded; the protected files (`.claude/settings.local.json`, `config.json`,
  `claudex-gateway-skill/`, `nnue/*.bin`) never staged.
