# ferrum M3 — gen-1 NNUE (king-bucketed) design

**Status:** approved design (brainstormed 2026-07-22). Next: implementation plan
via `writing-plans`, executed subagent-driven + SPRT-gated.

## Goal

Raise ferrum from its gen-0 rating of **2669 ± 25 CCRL** (`ferrum-v0.3.0`) toward
the 3000+ project goal by replacing the plain-768 NNUE evaluation with a
**king-bucketed, larger perspective net**. Realistic target: **~2850–2950 CCRL**
in one generation; 3000+ is expected to need a later gen-2 (bigger net /
self-play data + more search). This is an **eval-only** milestone — no search
changes — so the SPRT and gauntlet measure the net cleanly.

## Baseline (gen-0, for reference)

- Net: `768 → 512×2 → 1`, SCReLU, QA=255 / QB=64 / SCALE=400. Feature set = plain
  side-to-move-perspective 768 (12 piece planes × 64 squares), matching bullet's
  `Chess768`. File format "FeNN" v1 (16-byte header + col-major weights), 789,522 B.
- Trained with `bullet` (pinned `cebc78a093d92cbc87e56cfef049184c225270b0`) on a
  ThunderCompute A6000 over 63M ChessBench positions. Pipeline recorded in
  `nnue/README.md`; integration in `src/nnue.rs` behind the `Eval` trait
  (`EvalKind{Hce,Nnue}`), with the incremental accumulator + grow-only buffer
  pool from M2 Tasks 5–6.

## Data & labeling (decided)

- **Reuse ChessBench** (`james77mill/chessbench-encoded-npz`): 585 train shards ×
  250k ≈ **~100M positions**, labels derived from **Stockfish-16** win% per legal
  move (~3400-quality). Gen-0 used only 63M; gen-1 uses the **full ~100M**.
- **The 77M transformer as a labeler is rejected as dominated**: it is ~2050 Elo,
  far weaker than the Stockfish-16 labels ChessBench already carries. Self-play
  data generation is likewise deferred (a gen-2+ lever; even then Stockfish would
  be the better labeler). Gen-1 changes the *architecture*, not the data source.
- Conversion reuses `tools/chessbench_to_bullet.py` (M2 Task 2), which emits the
  32-byte bullet `ChessBoard` record **including king squares** (`ksq`,
  `opp_ksq`) — exactly what the bucketed input needs. Verify the king-square
  fields are populated correctly for the bucketed trainer (they were unused by
  `Chess768`).
- Use ChessBench **val shards** to compute validation loss for model selection
  across tuning runs.

## Architecture (decided)

- **Feature set:** king-bucketed, horizontally-mirrored 768 — bullet's
  `ChessBucketsMirrored` input. For each perspective: the king square (in that
  perspective's orientation) selects a **bucket**; when the king is on the e–h
  files the board is horizontally mirrored (`sq ^= 7`) so the king always folds
  to the a–d half. The per-piece index is then
  `bucket * 768 + piece_plane * 64 + view_sq`, where `piece_plane ∈ 0..11`
  (own/enemy × {P,N,B,R,Q,K}) and `view_sq` applies the perspective vertical flip
  (`^56` for black-to-move) and the horizontal king mirror. Total input rows =
  `num_buckets * 768`.
- **Bucket count:** tune **4 vs 8** (start 8). The king-square→bucket table is a
  fixed `[usize; 64]` hardcoded **identically** in the trainer config and in
  `ferrum`.
- **Hidden:** **1024** per perspective (`1024×2 → 1`). Activation/quant unchanged
  from gen-0 (SCReLU, QA=255, QB=64, SCALE=400) — proven, and keeps the forward
  pass identical in shape.

**Hard correctness requirement.** The exact bucket table, mirror rule, and index
composition **must be verified byte-for-byte against bullet's
`ChessBucketsMirrored` mapper at the pinned commit** over a battery of positions
(as gen-0 did for `Chess768` — see `nnue/README.md`). Do not hardcode index
constants from memory; derive them from bullet's source and prove equality.

## Net file format v2

Bump "FeNN" header to **version 2** with an added **`num_buckets` u16** field
(and keep hidden_size / QA / QB / SCALE fields). Payload:
`feature_weights[(num_buckets*768) × 1024]` (col-major, matching bullet's export
order — verify), `feature_bias[1024]`, `output_weights[2048]`, `output_bias`.
Sizes grow accordingly (e.g. 8 buckets → ~12.6 MB; nets stay gitignored). The
loader in `src/nnue.rs` **version-dispatches**: v1 → plain-768 path (gen-0 still
loadable), v2 → bucketed path. This lets a single ferrum binary load either net;
the gen-1-vs-gen-0 SPRT then runs one binary with two `EvalFile`s.

## Engine changes (`src/nnue.rs`, `src/board.rs`, `src/search.rs`)

1. **Feature indexing:** implement the bucketed+mirrored index (above), selected
   by net-file version. Keep the plain-768 path for v1.
2. **King-move accumulator refresh:** the M2 incremental accumulator assumes every
   piece move touches ≤4 features. With king buckets that is false when the
   **moving side's king changes bucket or crosses the mirror line** (including
   castling, which moves the king two files) — then *every* feature index for
   that perspective changes, so that **one perspective must be full-refreshed**
   while the opponent perspective still updates incrementally. Design:
   `feature_delta`/make detects a king move for side S; if S's king bucket-or-mirror
   changed, flag perspective S for refresh; `eval` refreshes flagged perspectives
   from scratch before the forward pass. The unmoved perspective and all non-king
   moves stay on the fast incremental path.
3. **Bucket map + mirror helpers:** shared constants/functions used by both the
   full-refresh (`fresh_accumulator`) and incremental paths, so they cannot drift.

## Validation & acceptance

- **Invariant test (extends M2 Task 5's):** incremental accumulator (with
  king-refresh) `==` full-refresh at every node, for **both** perspectives, over
  fixtures that exercise **king moves across buckets, king moves across the mirror
  line, and castling both sides** — plus the existing quiet/capture/ep/promotion
  cases. Runs with a synthetic bucketed net (always-on) and with the real gen-1
  net (`#[ignore]`d, run `--ignored`).
- **Byte-verification test:** ferrum's bucketed feature index equals bullet's
  `ChessBucketsMirrored` over a position battery.
- **SPRT gen-1 vs gen-0** at 8+0.08 (concurrency 2). Given the expected large
  gain, use a wider band (e.g. `elo0=0 elo1=10`); accept per the project rule (LLR
  boundary, or 95% CI wholly > 0 at the 2000-game cap).
- **Anchored gauntlet:** re-run the 1,120-game gauntlet (same 4 CCRL anchors,
  8+0.08, Ordo fixed-anchor) → gen-1 CCRL rating; record ledger row + spend; tag
  **`ferrum-v0.4.0`**.

## Sequencing / decomposition (for the plan)

Ordered so **all engine work + tests land locally ($0) before any cloud spend**,
and cloud spend is gated on green tests:

1. **Format v2 + bucket-map constants** (derive from bullet, byte-verified).
2. **Engine: bucketed indexing + king-refresh accumulator + v2 loader**, validated
   with a *synthetic* bucketed net via the extended invariant + byte-verification
   tests. No training needed yet.
3. **Data:** verify converter emits king squares; convert the full ~100M corpus
   (local, $0).
4. **Train + tune** (cloud, 2–4 runs, ~$8–12): bucket count (4/8) and LR/WDL
   recipe; select best by val loss.
5. **Integrate** the real gen-1 net, run the real-net invariant test, **SPRT vs
   gen-0**, then **anchored gauntlet**, ledger, tag `ferrum-v0.4.0`.

## Risks & mitigations

- **bullet↔ferrum feature-index mismatch** → silently wrong eval that still
  "works." Mitigation: mandatory byte-for-byte verification + the invariant test
  (this is why step 2 precedes any training).
- **King-refresh desync** (esp. castling / mirror-crossing). Mitigation: the
  extended invariant test must include those exact cases; a failure blocks
  progress.
- **1024 hidden ≈ 2× forward-pass cost** (~484k → ~300k nps expected). Still well
  clear of time-forfeits at 8+0.08 given the M2 incremental accumulator; if the
  gauntlet TC gets tight, the deferred **NEON SIMD** (original Task 6) is held in
  reserve as a pure-speed, bit-identical follow-up.
- **Tuning overspend.** Hard cap ~$12 for the gen-1 phase; delete the A6000
  instance immediately after each run (cost discipline; $10/$20 alerts stand).

## Budget

~$8–12 A6000 (authorized), against ~$20–25 total with ~$0.70 spent through M2.
Leaves >$10 of headroom.

## Config invariants (unchanged from M1/M2)

Full-window negamax (no PVS/null-window in the PV/move loop; the NMP scout window
is the one accepted exception). SPRT TC 8+0.08, concurrency 2 (Apple-M1 ceiling).
Batched SPRTs. Terse conventional commits, no trailer. Stage only the specific
files per commit; never `gen0.bin`/`gen1.bin`, `.claude/settings.local.json`,
`config.json`, or `claudex-gateway-skill/`.
