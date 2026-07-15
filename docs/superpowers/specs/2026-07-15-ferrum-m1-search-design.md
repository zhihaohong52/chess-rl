# ferrum M1 — search-strength grind to ~2300–2500 (design)

**Date:** 2026-07-15
**Status:** approved by user (brainstorming session)
**Parent spec:** `2026-07-14-ferrum-nnue-engine-design.md` (this refines that spec's M1 row)

## 1. Context

M0 shipped `ferrum-v0.1.0`: perft-exact bitboard movegen, plain alpha-beta +
quiescence + always-replace TT, MVV-LVA + TT-move ordering, HCE eval, full UCI,
bench, path-scoped CI. Exit match: 40.5% vs Stockfish `UCI_Elo=2000` (est
~1930). The M0 search is **plain alpha-beta — not PVS** (every move searched
full-window) and ordering is TT + MVV-LVA + promotions only; there are no
killers, no history, no null-move, no LMR, and sliders use ray-scan (no magic
bitboards).

M1 adds the community-canonical search-strength features — each as one
SPRT-gated patch — plus magic bitboards and real time management, and stands up
the reusable CCRL-anchored gauntlet that produces honest absolute ratings from
here through M3.

## 2. Goal and exit criterion

**Goal:** ~2300–2500 anchored strength via search engineering alone (eval stays
HCE; NNUE is M2).

**Exit criterion:** the anchored gauntlet (§4.3, ≥1000 games, Ordo) places
ferrum's rating in the ~2300–2500 band with a reported 95% CI. Tag
`ferrum-v0.2.0`. This is the only number reported as "Elo"; per-patch SPRT Elo
is a relative development signal, not an absolute rating.

## 3. Methodology decisions (this session)

- **Per-patch testing = SPRT-lite hybrid.** fastchess SPRT, STC 8s+0.08s,
  bounds `elo0=0 elo1=8`, α=β=0.05, **~2000-game cap** so a patch resolves in
  1–3 h on the 8 GB Mac. Keeps sequential-test early-stopping without multi-day
  grinds. Strict narrow-bound SPRT is deferred to the M2/M3 NNUE work.
- **M1 absolute rating = full CCRL anchor pool, stood up now**, reusable through
  M3 — with the low end extended to ~2300–2500 so that at M1 strength the games
  are informative rather than near-shutouts against 3000+ engines. Net pool span
  ≈ **2300–3100 CCRL blitz**.
- **Compute:** all SPRT and gauntlet games run on the user's Mac. **M1 spend =
  $0** (no ThunderCompute until M2's NNUE training).

## 4. Components

### 4.1 SPRT-lite harness — `ferrum/tools/sprt.sh`

Wraps fastchess to A/B two `cargo build --release` binaries (baseline = last
committed engine, candidate = working tree) as a self-play SPRT:

- STC 8+0.08, `-sprt elo0=0 elo1=8 alpha=0.05 beta=0.05`, `-rounds 1000
  -games 2` (≤2000 games), concurrency 4, `-openings` from the M1 book,
  `-pgnout file=...`, `-repeat`.
- Builds the baseline binary from `HEAD` into a temp target dir and the
  candidate from the working tree, so the script is a single command:
  `tools/sprt.sh` prints the SPRT verdict (accept H1 / accept H0 / cap reached)
  with the running Elo ± estimate.
- The fastchess build in use wants `-pgnout file=X` (key=value), Stockfish at
  `/opt/homebrew/bin/stockfish`, fastchess at
  `/Users/james/Documents/GitHub/fastchess/fastchess` (documented in the
  script header; overridable by env vars).

### 4.2 Expanded opening book

M0's 24-position EPD book is too small and narrow for thousands of games
(repetition, first-move bias). M1 needs ~500–1000 balanced, shallow positions.
Extend `ferrum/tools/gen_openings.py` to emit a larger book (broaden the
mainline set and/or walk a few plies of legal variety from each seed), written
to `ferrum/books/openings_m1.epd`. Used by both `sprt.sh` and `gauntlet.sh`.
Deterministic and regenerable; committed to the repo.

### 4.3 CCRL anchor gauntlet — `ferrum/tools/gauntlet.sh` + `ferrum/bench/anchors/`

- **Pool:** a handful of open-source engines with published CCRL blitz ratings,
  pinned to exact versions, spanning ~2300–3100 so ferrum draws informative
  games at every milestone. Candidates that build cleanly on Apple Silicon:
  CT800 (~2100–2300, low anchor), Fruit 2.1 (~2765, classic reference), Stash
  at a mid and a strong version, Weiss (~3050). The plan pins the final list
  with exact tags/commits and records the source CCRL rating of each.
- **Setup doc:** `ferrum/bench/anchors/README.md` — clone/build commands and
  flags per engine (ARM macOS), the pinned version, and each engine's CCRL
  blitz rating (the Ordo anchor value). Binaries themselves are **not**
  committed (`.gitignore` the built binaries); the doc makes the pool
  reproducible.
- **Gauntlet runner:** `gauntlet.sh` runs ferrum vs each anchor, ≥1000 games
  total, STC, M1 book, PGN out; feeds results to **Ordo** with the anchors'
  CCRL ratings fixed, emitting ferrum's rating + CI. Ordo install/build is
  documented in the same README.

### 4.4 Magic bitboards

Replace ray-scan rook/bishop attacks in `ferrum/src/attacks.rs` with fancy
magic bitboards. Magics are found at init by trial with the existing splitmix64
PRNG (`zobrist.rs`), attack tables cached in `OnceLock` (std-only, no external
deps). Correctness is locked two ways: (1) the perft suite stays exact, and (2)
a **cross-validation test** asserts magic attacks equal the retained ray-scan
attacks for every square across sampled occupancy masks. Gate: perft green +
bench nps materially up (target ≥2× slider nps) + SPRT pass (more depth in fixed
time → Elo).

### 4.5 Search features — one SPRT-gated patch each

Added to `ferrum/src/search.rs` (and small helpers) in dependency order. Each is
its own commit with its own unit test and SPRT verdict:

1. **PVS** — first move full-window, rest null-window `(α, α+1)` with re-search
   on fail-high. Foundation for LMR.
2. **Aspiration windows** — narrow window around the previous iteration's score
   at the ID root; widen on fail-high/low.
3. **Killer moves** — two per ply, ordered after good captures.
4. **History heuristic** — butterfly history for quiet moves, decaying.
5. **Null-move pruning** — reduced-depth null search with a zugzwang guard (not
   in check; side to move has non-pawn material).
6. **Reverse futility / static null-move pruning** — at low depth, prune when
   `static_eval − margin ≥ beta`.
7. **Late move reductions (LMR)** — reduce depth for late quiet moves; re-search
   at full depth if the reduced search beats alpha.
8. **Late move pruning** — at low depth, skip late quiet moves by move count.
9. **Futility pruning** — at low depth, skip quiets that cannot raise alpha.
10. **Check extensions** — extend one ply when the side to move is in check.
11. **SEE (static exchange evaluation)** — order winning vs losing captures and
    prune losing captures in quiescence.

Not every feature is guaranteed to pass SPRT-lite solo; some may need a tuning
pass or be dropped. The ledger records the verdict for each honestly.

### 4.6 Time management

Replace `time/25 + inc/2` in `search.rs` with soft/hard bounds: a soft target
(stop starting new ID iterations past it) and a hard cap (abort mid-search),
derived from wtime/btime/winc/binc, with optional best-move-stability scaling
(spend less when the best move is stable across iterations). Its own SPRT patch.

## 5. Per-patch pipeline (data flow)

For each feature in §4.4–4.6:

1. **TDD:** a unit test pins the mechanic where testable (e.g. PVS returns the
   same best move as plain AB on fixed tactical positions; NMP never fires while
   in check; SEE returns correct signed material for known exchanges).
2. `cargo test` + `cargo clippy --all-targets --release -- -D warnings` green.
3. `cargo run --release -- bench` node count **changes** — the checksum proves
   the patch actually altered the search (or, for magic bitboards, that nps rose
   while the node count for a fixed depth is unchanged).
4. `tools/sprt.sh` → SPRT-lite verdict.
5. **Pass** → commit + ledger row. **Fail** → revert the change (note it in the
   ledger with the measured Elo) and move on.

## 6. Testing strategy

- **Correctness:** perft unchanged across the whole milestone (critical for the
  magic-bitboard patch); per-feature unit tests; bench node checksum as the
  "did-what-it-claimed" guard.
- **Strength (relative):** SPRT-lite per patch, self-play.
- **Strength (absolute):** anchored gauntlet at the milestone exit only
  (≥1000 games, Ordo).
- **CI:** `.github/workflows/ferrum-ci.yml` (clippy + test + bench) green at
  every commit — unchanged from M0.

## 7. Files

- Create: `ferrum/tools/sprt.sh`, `ferrum/tools/gauntlet.sh`,
  `ferrum/bench/anchors/README.md`, `ferrum/books/openings_m1.epd`.
- Modify: `ferrum/src/attacks.rs` (magic bitboards), `ferrum/src/search.rs`
  (all search features + time mgmt), `ferrum/src/movegen.rs` (if slider call
  sites change), `ferrum/tools/gen_openings.py` (bigger book),
  `ferrum/docs/ledger.md` (per-patch rows + M1 exit), `ferrum/.gitignore` (built
  anchor binaries), possibly small helpers (`see`, `history`) as new modules or
  within `search.rs`.

## 8. Risks and mitigations

- **Mac wall-clock** (~12 patches × 1–3 h) → SPRT-lite game cap; background
  overnight jobs; batch runs.
- **Magic-bitboard correctness** → cross-validation test vs retained ray-scan +
  the perft suite.
- **Features that fail or need tuning** → expected; per-patch gating + honest
  ledger; a feature that fails solo may be revisited after LMR/history land
  (interactions matter).
- **Anchor engines building on Apple Silicon** → pin exact versions, document
  flags, keep alternates (the pool needs ~5 engines, not a specific 5).
- **Full pool at ~2400 = many losses vs 3000+** → low-end extension to
  ~2300–2500 so the M1 exit gauntlet has informative games and a usable CI.

## 9. Budget

M1 is local-only: **$0**. ThunderCompute spend begins at M2 (gen-0 NNUE via
bullet on an A6000). Hard alerts at $10 / $20 cumulative remain in force.

## 10. Non-goals (M1)

No NNUE (M2). No lazy SMP / multithreading. No Syzygy probing in search (M3). No
singular extensions / correction history (M3 SPRT grind). No Lichess deployment.
