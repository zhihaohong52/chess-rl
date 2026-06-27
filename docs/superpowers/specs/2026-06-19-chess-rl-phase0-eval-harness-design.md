# Chess-RL: Eval/Ablation Harness + Locked Baseline (Phase 0) — Design

**Date:** 2026-06-19
**Branch:** `feat/transformer-rebuild`
**Status:** Approved design, ready for implementation plan.

## Context

A gauge comparison settled the training direction: dense policy-distribution
distillation beats pointwise action-value at our compute budget.

| Model | Training data | Raw top-1 | MCTS-100 |
| --- | --- | --- | --- |
| Old policy-distribution | 62k dense (~29 moves/pos) | 28.9% (1k) / 27.7% (300) | 34.0% (300) |
| New action-value | 1.98M pointwise (1 move/pos) | 25.7% (1k) / 24.3% (300) | 32.3% (300) |

A Codex architecture review (`docs/codex_comments/model_architecture_findings.md`)
corroborates this and proposes a broader program: lock dense policy as the
primary track, add a stable regularization preset, make Smolgen configurable,
enrich the policy head, add board history and light auxiliary heads, and keep
action-value only as a separate Q head later.

Compute economics changed too: ThunderCompute offers A6000 48 GB @ $0.35/hr and
A100 80 GB @ $0.78/hr (per-minute billing, no egress). A $10 budget buys ~28
A6000-hours or ~13 A100-hours — abundant GPU compute relative to the M1. The
bottleneck flips to dense-data generation (CPU-bound Stockfish), which we run
locally and upload.

## Program Roadmap (4 phases)

Each phase is its own spec → plan → implementation cycle. This document fully
specifies **Phase 0 only**.

- **Phase 0 — Foundation (this doc):** eval/ablation harness + locked
  `baseline-v1`. Everything downstream is measured against it.
- **Phase 1 — Dense data + primary training:** harden the Stockfish generator
  (two-stage candidate selection, resume safety, manifest); generate locally;
  dense policy+WDL training as the default; beat the 62k baseline.
- **Phase 2 — Architecture ablations (config-gated, measured by Phase 0):**
  `stable-v1` regularization (dropout/LayerScale), configurable Smolgen, richer
  policy head, compact board history, lightweight auxiliary heads.
- **Phase 3 — GPU scale-up:** winning config + a (configurable, possibly larger)
  model + max dense data → the ~$10 ThunderCompute run(s).

**Sequencing rationale:** a credible ablation program is impossible without a
consistent gate harness, so Phase 0 comes first. Dense generation is CPU-bound
and runs unattended in the background while Phase 0 is built.

## Phase 0 Goal

Ship no model improvements. Build the scaffolding that makes every later change
measurable, reproducible, and apples-to-apples, and that makes
"evaluated with the wrong objective" structurally impossible.

## Components

### A. Preset registry — `src/model/presets.py`

Named model configurations as override dicts layered over the existing `Config`.

- `PRESETS: dict[str, dict]` maps a preset name to architecture-knob overrides.
- `baseline-v1` reproduces today's exact model: `d_model=256, n_layers=8,
  n_heads=8, d_ff=1024, smolgen every layer, transformer_dropout=0.0`. It MUST
  build a model with exactly **10,255,879** parameters.
- `build_model(preset_name) -> (net, resolved_config)`: applies overrides to a
  fresh `Config`, constructs `ChessTransformer`, returns the net and the resolved
  config (so param count and knobs are introspectable).
- `ChessTransformer` is refactored to read **all** architecture knobs from
  config — no hardcoded architecture constants. The **default forward contract is
  unchanged**: `forward(square_tokens[B,64], state_features[B,18]) ->
  (policy_logits[B,1924], wdl_logits[B,3], moves_left[B,1])`.

Rationale: one keystone that makes all later variants config-gated and
baseline-compatible (Codex item #2). Chosen over per-variant config files or a
heavier framework (Hydra) to stay minimal and match the existing `Config`.

### B. Checkpoint metadata + evaluator routing

Make objective mismatches impossible (the 13.3% bug: a policy model searched
with action-value semantics).

- **Sidecar:** each checkpoint save writes `<ckpt>.json` next to the `.pt`:
  `{preset, objective: "policy"|"action_value", train_data, step, git_sha}`.
  `DistillTrainer.fit` and `fit_av` get a small change to emit it.
- **Routing helper — `src/eval/routing.py`:**
  `load_for_eval(ckpt_path, objective=None, preset=None) -> (net, evaluator)`
  reads the sidecar, builds the model from the named preset, loads weights, and
  constructs `TransformerEvaluator(objective=…)`. If the sidecar is absent (e.g.
  the legacy `checkpoints/distill/best.pt`), `objective` and `preset` MUST be
  passed explicitly or it raises — never guesses.
- All gauge/eval entry points route through this single helper.

Chosen: JSON sidecar (human-readable, greppable, decoupled from the tensor file)
over embedding metadata inside the `.pt`.

### C. Gate suite — `src/eval/gates.py`

Computes a fixed battery on pinned datasets, returning a flat metrics dict.

**1. Puzzle gates** (fixed ChessBench puzzle subsets — first 1000 / first 300 of
`data/puzzles.csv`, so numbers chain to today's baseline):
- `raw_top1@1000`, `raw_top1@300` — argmax legal move, no search.
- `mcts100_top1@300` — MCTS-100 via the routed evaluator.

**2. Policy/value metrics** (pinned held-out **dense** val set, encoded once from
the 62k test bag):
- Policy: `top1`, `top3`, `top5`, `policy_ce`, `legal_mass` (softmax mass on
  legal moves).
- Value: `wdl_ce`, `value_sign_acc`, `draw_calibration` (mean predicted draw
  prob vs actual on near-0.5 positions), `mate_in_1_sanity` (a few crafted
  positions where the best move must dominate).

**3. Throughput** (reported, not gated; machine-dependent): `batch_latency_ms`
(positions/sec at batch 256), `mcts_nodes_per_sec`.

Each metric is a small pure function with its own unit test. The suite takes
`(net, evaluator, datasets)`; raw-top1 and throughput are objective-agnostic,
MCTS is objective-aware via the routed evaluator.

**Val-set caveat:** `baseline-v1` was trained on the full 62k test bag, so the
dense policy/value metrics are *upward-biased* for it (train-set leakage). They
become genuinely held-out for Phase 1+ models trained on Stockfish-generated
data. Therefore the **puzzle gates** (from the separate `data/puzzles.csv`
ChessBench puzzle file) are the primary unbiased cross-model comparison; the
dense policy/value metrics are mainly for tracking Phase 1+ models and detecting
regressions, and the baseline's values are recorded as a reference, not a fair
target.

### D. Ablation runner + pinned datasets + frozen baseline

- **Runner — `scripts/ablate.py`:** takes a list of checkpoints (each with a
  sidecar), runs the gate suite on each via the routing helper, emits a markdown
  + CSV table, one row per model:
  `preset | params | raw_top1@1k | raw_top1@300 | mcts100@300 | top3 | top5 |
  policy_ce | wdl_ce | sign_acc | latency`. Output to `docs/ablations/`.
- **Pinned datasets — `src/eval/fixtures.py`:** deterministic puzzle subsets
  (first 300/1000 of `data/puzzles.csv`) and a held-out dense val shard encoded
  once from the 62k test bag to a fixed local path. Fixtures are **regenerated
  deterministically on demand if absent** — not committed to git.
- **Frozen baseline-v1:** copy the current policy model to
  `checkpoints/baseline-v1/best.pt` + sidecar (`preset=baseline-v1,
  objective=policy`), run the suite once, commit its metrics row as
  `docs/ablations/baseline-v1.md` — the reference for every Phase 2 change.

**Persistence policy:** binaries (the baseline `.pt`, the dense val `.npz`,
`data/puzzles.csv`) stay local (already gitignored). Only metrics tables under
`docs/ablations/` are committed.

### E. Testing

- **Metrics:** unit-test each gate fn on synthetic inputs (known top-k,
  hand-computed CE/WDL-CE, calibration edge cases, mate-in-1 sanity).
- **Presets:** `baseline-v1` builds a model with param count == 10,255,879; an
  override preset changes the expected knob and param count.
- **Routing:** sidecar round-trip; correct evaluator objective selected;
  missing-sidecar raises.
- **Runner:** smoke test on 2 tiny presets → well-formed table.
- All new tests CPU-only and fast; the existing ~80-test suite stays green.

## Data Flow

```text
preset name ──> build_model ──> ChessTransformer
checkpoint ──> sidecar ──> load_for_eval ──> (net, routed evaluator)
                                   │
pinned datasets (puzzles, dense val, mate fixtures)
                                   ▼
                            gate suite ──> metrics dict ──> ablate.py ──> docs/ablations/*.md|csv
```

## Error Handling

- Missing checkpoint sidecar → require explicit `--objective`/`--preset`, else
  raise with a clear message.
- Terminal / no-legal-move positions → skipped in gates.
- Puzzle parse failures → skipped (existing behavior).
- Fixtures absent → regenerated deterministically before metrics run.

## Out of Scope (Phase 0)

No architecture improvements, no new heads, no data generation changes, no
training-recipe changes. Those are Phases 1–3.

## Success Criteria

- `baseline-v1` reproduces the current model exactly (param count + gate numbers
  matching today's 28.9% / 34.0%).
- `scripts/ablate.py` produces a comparison table for ≥1 checkpoint.
- Routing prevents objective mismatches (covered by tests).
- Existing test suite remains green; new metric/preset/routing tests pass.
