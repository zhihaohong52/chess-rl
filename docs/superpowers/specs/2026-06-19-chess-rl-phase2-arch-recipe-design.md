# Chess-RL Phase 2: Architecture/Recipe Ablations — Design

**Date:** 2026-06-19
**Branch:** `feat/transformer-rebuild`
**Status:** Approved design, ready for implementation plan.

## Context

Phase 0 built a config-gated eval/ablation harness (`src/model/presets.py`,
`scripts/ablate.py`, gate suite) and locked `baseline-v1` (10,255,879 params; raw
top-1 28.9%, MCTS-100 34.0%). Phase 1 validated dense policy-distribution training
on downloaded data (`prdev/chessbench-full-policy-value` via
`iter_hf_dense`, T=0.1): the `hf-dense-300k` model reaches **raw top-1 0.359
(+7 pts vs baseline)** and recovered mate-in-1, but **MCTS regressed to 0.310**.

A c_puct × simulations sweep diagnosed the regression: "more sims →
monotonically worse" at every c_puct, and no setting recovers MCTS > raw.
Dirichlet noise is already off for eval. **The value head is the bottleneck** —
the sharp, strong policy is undermined in search by a value function that isn't
good enough to back up. The fix is a better value head, not MCTS tuning.

Phase 2 pursues four optimizations, each a config-gated preset ablated via the
Phase 0 harness, to improve convergence/regularization and (priority) fix the
value bottleneck so MCTS beats raw.

## Goal

Produce a `phase2-best` preset that beats both `hf-dense-300k` (raw 0.359) and
`baseline-v1` on **raw top-1 AND MCTS-100**, with the value head strong enough
that **MCTS ≥ raw** (the regression is reversed). Every change is measured, kept
only if it helps.

## Decisions (locked during brainstorming)

- **Run location: hybrid.** Smoke each preset locally on M1/MPS (correctness,
  tiny runs); run the real ~7000-step ablations and the scale-up on a rented
  ThunderCompute A6000 (~$0.02 and ~3 min per 10M-param run — the $10 budget
  covers the whole program).
- **Strategy: incremental stack, value-first.** Add one change to the previous
  keeper, keeping it only if it improves the headline gates:
  `p2-value` → `p2-value-swiglu` → `p2-value-swiglu-drop` → (+EMA flag).
- **Value target: derived (no re-encode).** Use expected score `v = W + 0.5·D`
  from the WDL already stored in the shards.
- **Gating success metric:** value head must flip **MCTS ≥ raw**.

## Components

### 1. Config knobs + presets (`config.py`, `src/model/presets.py`)

New config fields (defaults preserve `baseline-v1` exactly):
- `value_head_type`: `"wdl"` (default) | `"hlgauss"`
- `value_buckets`: int, default 64
- `value_sigma_frac`: HL-Gauss σ as a fraction of bucket width, default 0.75
- `value_loss_weight`: float, default 1.0
- `ffn_type`: `"mlp"` (default) | `"swiglu"`
- `transformer_dropout`: exists (default 0.0)
- `ema_decay`: float, default 0.0 (off)

Presets (incremental):
- `p2-value` = baseline-v1 + `value_head_type=hlgauss` (value_loss_weight probe
  value, e.g. 1.0 then 2.0 if helpful)
- `p2-value-swiglu` = + `ffn_type=swiglu`
- `p2-value-swiglu-drop` = + `transformer_dropout=0.05` (try 0.10 if it helps)
- EMA is a trainer flag applied on top (`--ema-decay 0.999`), gauged on the EMA
  checkpoint; same architecture, so no separate arch preset.

### 2. HL-Gauss distributional value head (`src/model/heads.py`)

- `DistributionalValueHead(d_model, buckets)`: small MLP on the CLS token →
  `buckets` logits over expected-score bins in `[0,1]`, centers
  `c_k = (k+0.5)/buckets`.
- **Soft target:** scalar `v = W + 0.5·D` (from stored WDL). HL-Gauss target =
  `N(v, σ)` integrated per bucket via the normal CDF
  (`p_k = Φ((edge_{k+1}−v)/σ) − Φ((edge_k−v)/σ)`, renormalized over the bin
  range), with `σ = value_sigma_frac · (1/buckets)`.
- **Loss:** `−Σ target_k · log_softmax(logits)_k`, scaled by `value_loss_weight`.
- **Inference value:** `v̂ = Σ c_k · softmax(logits)_k ∈ [0,1]`; MCTS/evaluator
  value = `2·v̂ − 1 ∈ [−1,1]`.

### 3. SwiGLU FFN (`src/model/transformer.py`)

- `SwiGLUFFN(d_model, h)` = `down(silu(gate(x)) ⊙ up(x))` (linears: gate, up
  `d_model→h`; down `h→d_model`). To hold params ≈ the 2-matrix MLP, set
  `h = round(d_ff · 2/3 / 8) · 8`. `EncoderLayer` selects MLP vs SwiGLU by
  `cfg.ffn_type`.

### 4. Dropout (`src/model/transformer.py`)

- `nn.Dropout(cfg.transformer_dropout)` after the attention output projection,
  after the FFN, and on residual adds. Gated so `0.0` is a no-op (baseline path
  unchanged); `.eval()` disables it. Presets set 0.05–0.10.

### 5. EMA of weights (`src/training/distill_trainer.py`, `scripts/distill.py`)

- When `ema_decay > 0`, maintain a shadow EMA of parameters
  (`ema = decay·ema + (1−decay)·param` each step). Validate on EMA weights; save
  `best_ema.pt` (+ sidecar) alongside `best.pt`. `distill.py --ema-decay` flag.
  The gauge may target either checkpoint.

### 6. Harness integration (`src/model/evaluator.py`, `src/eval/gates.py`,
   `src/eval/metrics_core.py`)

- **Evaluator** reads `value_head_type` and `value_buckets` from the **net**
  (which stores them as attributes from `cfg` at construction, so no extra config
  needs threading through); for `hlgauss` it computes value = `2·v̂ − 1` (else the
  existing WDL / action_value paths).
- **Gates/metrics** branch on head type: `value_sign_acc` = sign(`v̂−0.5`) vs
  sign(`v−0.5`); add `value_mae = |v̂−v|`; `wdl_ce` → `value_ce` (HL-Gauss CE)
  for the distributional head. Reported comparably for either head so ablation
  rows line up.
- `routing`/`build_model`/`ablate.py` already build from preset, so they pick up
  the new heads/FFN automatically.

## Ablation Flow

For each preset in the incremental order:
1. Smoke locally (1-step / tiny-shard correctness run; gates smoke).
2. Real run on the GPU: 300k dense set, ~7000 steps, batch (GPU can do ≥1024),
   sidecar with preset + train_data.
3. `scripts/ablate.py` → `docs/ablations/<preset>.md` (committed).
4. Keep the change iff it improves the headline gates (raw and/or MCTS) without
   regressing the other; otherwise drop it and try the next on the prior keeper.
5. Accumulate keepers into `phase2-best`; that preset feeds the scale-up.

## Success Criteria

- `p2-value` flips **MCTS ≥ raw** and ideally MCTS > baseline 0.340.
- Each later change kept only if it improves raw and/or MCTS without regressing.
- `phase2-best` beats `hf-dense-300k` (raw 0.359) and `baseline-v1` on raw AND
  MCTS.
- `baseline-v1` remains bit-identical (defaults unchanged); full test suite green.

## Testing

- HL-Gauss target: soft target sums to 1, peaks at `v`'s bucket, `E[buckets]`
  recovers `v` within a bucket width.
- `DistributionalValueHead`: output shape `[B, buckets]`; expected-value helper
  correct.
- SwiGLU: output shape `[B,T,d_model]`; param count ≈ MLP within a small margin.
- Dropout: changes outputs in `train()`, identical in `eval()`.
- EMA: shadow update math; validation/save uses EMA weights when enabled.
- Evaluator: `hlgauss` value scalar in `[−1,1]`, monotonic in `v̂`.
- Trainer: selects WDL-CE vs HL-Gauss-CE by `value_head_type`.
- Presets build; `baseline-v1` param count still 10,255,879.
- Gate suite runs end-to-end on a `hlgauss` model (smoke).

## Out of Scope

The GPU scale-up run itself (bigger model / more data) — that is the next phase,
informed by `phase2-best`. RMSNorm/QK-norm and other marginal tweaks (only if a
later need appears). No change to the data pipeline (value target is derived).
