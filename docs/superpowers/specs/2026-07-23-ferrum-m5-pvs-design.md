# ferrum M5 — Principal Variation Search (design)

**Date:** 2026-07-23
**Status:** approved, pre-implementation
**Baseline:** `ferrum-v0.4.0` (commit `3430cbc`, branch `feat/ferrum-m4-search`), 2901.5 ± 25.2 CCRL
**Target artifact:** `ferrum-v0.5.0` if the gate passes; no tag if it does not

---

## Goal

Replace the full-window search of non-first moves with principal variation search
(PVS): scout later moves with a null window, and pay for a full-window re-search
only when the scout proves the move is actually better than alpha.

This retires the one search constraint ferrum has carried since M0.

## Why now

M4 established that ferrum's search was already well-tuned for fast time control —
three of four bundles (search shaping, singular extensions, correction history) were
flat or negative. The one real gap was move ordering, and closing it was worth
+76.8 ± 18.5 Elo.

PVS is the direct beneficiary of that result. Its value is proportional to move
ordering quality: a scout search is cheap exactly when the first move is usually
best, because then the scout fails low and no re-search is needed. Before M4 B1,
ferrum ordered quiet moves flat — no history at all — so scouting would have
thrashed on re-searches. The M4 result is what makes M5 worth attempting.

## What the invariant actually was

The constraint is narrower than "ferrum never uses a null window". Null-move
pruning at `src/search.rs:389` already searches `[-beta, -beta + 1]`. The M4 spec
formulation is the accurate one:

> the main move loop searches every move full-window, with full-window re-searches

M5 deletes exactly that clause. Null-move pruning, the aspiration window, the
transposition table, and qsearch are untouched.

## Replacement correctness anchor

**Score equivalence between PVS and plain alpha-beta, over a pure search core.**

### Scope of the claim, stated precisely

PVS is score-identical to alpha-beta **only when no depth-reducing heuristic is in
play**. With LMR active, the reduced search returns an approximation; probing that
approximation with a null window rather than a full window legitimately returns a
different value, which in turn changes whether the re-search fires. A
`pvs=on` vs `pvs=off` diff with LMR enabled would therefore disagree for correct
reasons, and would be a useless test.

The equivalence test consequently disables the entire inexactness family on both
sides of the diff: TT cutoffs, null-move pruning, reverse futility pruning, late
move pruning, futility pruning, and LMR.

**What this proves:** the scout / re-search ladder never loses a score. That is the
complete class of bug PVS introduces, and it is the class the old invariant existed
to make impossible.

**What this does not prove:** that LMR-under-PVS is sound. Nothing can prove that —
LMR is a heuristic and reductions are not score-preserving under any windowing
scheme. The existing behavioural tests (`lmr_still_finds_deep_tactic`,
`rfp_keeps_tactics`, `lmp_keeps_tactics`, `futility_keeps_tactics`,
`check_extension_finds_mate_missed_without_it`) remain the guard for that, and must
pass unmodified.

### Why the heuristics stay on both sides in production

Check extensions, qsearch (including its SEE pruning), repetition detection, and the
fifty-move rule are *identically* applied in both shapes, so both modes compute the
same function and equality holds. Only the score-inexact heuristics need disabling.

## Architecture

One new type in `src/search.rs`. No new files.

```rust
/// Search-shape switches. `Searcher::new` and `Searcher::with_nnue` build the
/// production shape; nothing in `uci.rs` can reach these. `heuristics: false`
/// disables everything that makes the search inexact — TT cutoffs, null-move,
/// RFP, LMP, futility, LMR — leaving a pure alpha-beta core in which PVS is
/// provably score-identical to plain full-window search, and diffable.
struct Shape { pvs: bool, heuristics: bool }
```

- `Shape::production()` → `{ pvs: true, heuristics: true }`
- `Shape::pure(pvs)` → `{ pvs, heuristics: false }` — test-only constructor

Stored as `Searcher.shape`. These are **runtime bools, deliberately not
`#[cfg(test)]`-gated**: cfg-gating would mean the test suite exercises different
code than ships, which reintroduces the exact risk the anchor exists to remove. The
cost is one perfectly-predicted branch at each pruning site.

**Acceptance condition on that cost:** measured at the point where the guards exist but
PVS is not yet wired in, `./target/release/ferrum bench` must report a node count
**exactly equal** to the `3430cbc` baseline, with nps inside the run-to-run noise band.
Identical node counts are themselves the proof that adding the guards changed no
behavior; the nps comparison is then a clean like-for-like measure of the branch cost.

## The move loop

Replaces `src/search.rs:443-457`. The `reduce` computation above it is unchanged.

```rust
let score = if !self.shape.pvs || legal == 1 {
    // First legal move establishes the PV — full window, as today.
    -self.negamax(b, depth - 1, -beta, -alpha, ply + 1, Some(cur_pt))
} else {
    // (a) reduced scout, when LMR applies; else fall straight through to (b)
    let mut s = if reduce > 0 {
        -self.negamax(b, depth - 1 - reduce, -alpha - 1, -alpha, ply + 1, Some(cur_pt))
    } else {
        alpha + 1
    };
    // (b) full-depth scout, once the reduced probe beat alpha
    if s > alpha {
        s = -self.negamax(b, depth - 1, -alpha - 1, -alpha, ply + 1, Some(cur_pt));
    }
    // (c) full-window re-search — only fires at PV nodes, where beta > alpha + 1
    if s > alpha && s < beta {
        s = -self.negamax(b, depth - 1, -beta, -alpha, ply + 1, Some(cur_pt));
    }
    s
};
```

Three properties worth noting:

1. **Free at non-PV nodes.** There `beta == alpha + 1`, so `s < beta` in guard (c)
   is unsatisfiable and the full-window re-search never runs.
2. **LMR needs no separate work.** Once later moves are scouted, most of the tree
   becomes null-window nodes, and inside those the pre-existing LMR code
   `[-beta, -alpha]` *was already* a null window. The explicit ladder above only
   changes behaviour at PV nodes. This is why M5 is one bundle, not two.
3. **Window arithmetic is safe.** `MATE = 30_000`; `-alpha - 1` reaches at most
   `±30_001`, and `-MATE - 1` is already in use at `src/search.rs:405`.

The `alpha + 1` sentinel when `reduce == 0` is intentional: it makes guard (b)
unconditionally true so the full-depth scout runs, without duplicating the call.

`alpha` is re-read each iteration and mutates on improvement at
`src/search.rs:468`, as today.

## Guarding the heuristics

Each inexact site gains a `self.shape.heuristics &&` conjunct:

| Site | Line (at `3430cbc`) |
|---|---|
| TT cutoff | `search.rs:360` |
| Reverse futility pruning | `search.rs:375` |
| Null-move pruning | `search.rs:383` |
| Late move pruning | `search.rs:420` |
| Futility pruning | `search.rs:428` |
| LMR (`reduce` computation) | `search.rs:443` |

TT *stores* remain unconditional. With cutoffs disabled the table only influences
move ordering, and a full-window alpha-beta root score is ordering-independent, so
this cannot perturb the diff.

## Testing

Three layers, ordered by what they buy.

### 1. New — equivalence (the anchor)

For each position, assert
`score(Shape::pure(true)) == score(Shape::pure(false))` at a full `[-MATE, MATE]`
window.

Suite: at least six positions covering quiet middlegame, tactical (the existing
knight-fork and back-rank FENs), a king-and-pawn endgame, a position with the side to
move in check, a stalemate-adjacent position, and Kiwipete. The plan pins the exact
FENs. Depth 3–6, scaled to branching factor — necessarily shallow, because with TT
cutoffs disabled the tree grows exponentially and `cargo test` builds in debug by
default. Branchy positions get 3, sparse ones 5–6.

### 2. New — node-count pre-gate

At fixed depth with the production shape, PVS must search **at least 10% fewer nodes**
than the `3430cbc` baseline, summed over a small position set. The threshold is a
smoke test for "the scout is actually firing", not a performance target — a correct
PVS at this ordering quality should comfortably exceed it.

This runs **before** the SPRT is launched. If PVS is not saving nodes, the
implementation is wrong and the SPRT would burn ~400 games measuring a bug. M4
established the value of a cheap behavioural pre-check: at gate B3, two candidate
binaries had byte-identical sizes and were only distinguished by a node-count diff
before an SPRT was spent.

### 3. Unchanged — the existing suite

All current tests in `src/search.rs` must pass **unmodified**: `finds_mate_in_1`,
`takes_free_queen`, `checkmate_overrides_fifty_move_draw`, `qsearch_reports_checkmate`,
`stalemate_is_none`, the four aspiration tests, `killer_stored_and_deduped`,
`rfp_keeps_tactics`, `lmr_still_finds_deep_tactic`, `lmp_keeps_tactics`,
`futility_keeps_tactics`, both move-ordering tests, `see_values`,
`respects_movetime_and_returns_legal`, `check_extension_finds_mate_missed_without_it`,
`beta_cutoff_populates_quiet_history`.

**Editing one of these to make it pass is a red flag, not a fix.** M4 rejected
razoring on exactly such a signal — it pruned a forced quiet mate, and the test that
caught it was right.

Note that `full_window_score()` at `src/search.rs:660` currently calls the
*production* negamax with a wide window, so it is not a pure oracle. It keeps its
present meaning and its four existing callers; the equivalence test uses the new
`Shape::pure` path instead.

## Gate

Single SPRT, identical methodology to M4:

```bash
# run from the ferrum/ directory; EVALFILE must be absolute or ferrum-relative
NET="$(pwd)/nnue/gen0.bin"
EVALFILE="$NET" CONCURRENCY=2 ELO0=0 ELO1=8 ROUNDS=1500 \
  tools/sprt.sh /tmp/ferrum-base /tmp/ferrum-cand
```

- Time control 8+0.08, concurrency 2 (Apple M1 clean ceiling)
- Base is a release build of `3430cbc`; candidate is the PVS build
- `EVALFILE` is mandatory — without it the A/B compares two HCE engines rather than
  the real one
- **Accept** at LLR ≥ +2.94, **reject** at LLR ≤ −2.94
- **Inconclusive at the 3000-game cap → reject.** Pre-committed, as in M4 B4.

### On accept

1. 1120-game anchored gauntlet (140 rounds × 4 anchors: stash-v17 2296, stash-v21
   2713, weiss-2.0 3320, stash-v37 3423)
2. Fixed-anchor Ordo: `ordo -Q -W -s 1000 -n 4 -F 95 -m anchors.txt -p <pgn>`
3. Ledger entry in `ferrum/docs/ledger.md`; archive PGN and Ordo output to
   `ferrum/bench/anchors/results/`
4. Tag `ferrum-v0.5.0`

### On reject

Revert the change, record the negative result in the ledger with the LLR and game
count, no tag. A negative here is a genuine finding: it would mean node savings do
not convert to strength at 8+0.08, which would redirect the remaining effort toward
data rather than search.

## Long-running process operations

SPRT and gauntlet runs must be launched as a detached daemon. Bare `nohup` and the
harness background flag are both reaped on this host. The working recipe is a Python
double-fork + `os.setsid()` daemon, launched from a **solo** Bash call with the
sandbox disabled, watched by polling a sentinel file.

## Non-goals

- **SPSA tuning.** Deferred; tuning should follow PVS, not precede it, since PVS
  changes the tree the parameters operate on.
- **Multithreading / lazy SMP.** Real strength, but CCRL is a single-core list, so it
  would not move the number this project is measured by.
- **Re-gating M4's B2 and B4.** Both near-missed and both plausibly behave differently
  under PVS, but folding them in would destroy attribution for the PVS result itself.
- **Any eval, NNUE, or cloud work.** M3 established the eval lever is saturated at the
  ChessBench data ceiling; breaking it needs new data, which is a separate milestone.
- **Splitting `src/search.rs`.** At 924 lines it is a legitimate candidate, but a small
  diff that an SPRT can cleanly attribute is worth more this milestone.

## Risks

| Risk | Mitigation |
|---|---|
| Re-search condition subtly wrong, losing scores | The equivalence test is precisely this check |
| Runtime `Shape` bools slow the hot loop | `bench` node rate must match `3430cbc` |
| PVS saves nodes but not Elo at 8+0.08 | Node-count pre-gate separates "bug" from "real negative"; the SPRT then answers the Elo question honestly |
| Scout searches pollute the TT and destabilise aspiration | Already-documented benign instability; both aspiration tests tolerate `\|actual − expected\| ≤ 64` |

## Success criterion

SPRT accept at LLR ≥ +2.94 against `3430cbc` at 8+0.08, with the full existing test
suite passing unmodified and the equivalence anchor in place.
