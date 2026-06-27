# Chess-RL Phase 1 (Validate): Dense-Data Training on 50k — Design

**Date:** 2026-06-19
**Branch:** `feat/transformer-rebuild`
**Status:** Approved design, ready for implementation plan.

## Context

Phase 0 delivered the eval/ablation harness and locked `baseline-v1` (a
policy-distribution model trained on the 62k ChessBench test bag). Its reference
gate metrics (`docs/ablations/baseline-v1.md`): **raw top-1 28.9% (1k) / 27.7%
(300), MCTS-100 34.0% (300)**.

The gauge ([gauge-actionvalue-vs-dense]) established that dense policy
distillation beats pointwise action-value at our compute budget, so the program
makes dense policy-distribution the primary track. We built a Stockfish dense
generator (`scripts/gen_dense_stockfish.py`) and produced a first dataset:
`data/shards_dense_d10_mpv16/` = **49k train + 1k val** dense positions (depth
10, multipv 16), local and gitignored.

This Phase 1 iteration is **validate-first**: before investing in
generator-hardening + large-scale generation + a paid GPU run (Phase 1b), train
on the existing 50k and confirm our *self-generated* dense data is a teacher
worth scaling.

## Goal

Train `baseline-v1`'s architecture from scratch on the 50k Stockfish dense set
(local MPS) and gauge it against `baseline-v1` on the Phase 0 puzzle gates. The
question answered: **is our depth-10 Stockfish dense data at least as good a
teacher, per position, as the fixed 62k test bag?** If yes, scaling it (Phase
1b) is justified.

We hold the architecture fixed at `baseline-v1` so the only variable is the
training data (architecture changes are Phase 2).

## Components

### 1. `scripts/distill.py` — sidecar + short-run LR schedule (only code change)

The current `distill.py` builds `ChessTransformer(Config)` and calls `fit`
without `meta` (no sidecar) using the default 60k-step LR schedule. Two problems
for this iteration: (a) the gauge (`ablate.py` via `load_for_eval`) needs a
checkpoint sidecar to route the evaluator; (b) a 3000-step run under a
warmup=2000 / total=60000 schedule spends most of the run warming up and barely
decays LR.

Changes (contained to `distill.py`):
- Add args: `--preset` (default `baseline-v1`), `--train-data` (free-text label
  for the sidecar), `--warmup` (default 200), `--lr` (default 3e-4).
- Resolve the preset via `src.model.presets.resolve_config(preset)`; set
  `cfg.distill_lr`, `cfg.distill_warmup_steps = --warmup`,
  `cfg.distill_total_steps = --steps` on it so the cosine schedule matches the
  actual run length.
- Build the net from that resolved config (`ChessTransformer(cfg)`), construct
  `DistillTrainer(net, cfg, mixed_precision=...)`.
- Pass `meta={"preset": preset, "train_data": train_data}` to `fit` so
  `best.pt`/`last.pt` get sidecars (`objective="policy"` is stamped by the
  trainer).

Existing behavior is preserved when the new args are at their defaults except
that a sidecar is now written (which is desired). `--train`/`--val` globs,
`--batch`, `--steps`, `--val-every`, `--mixed-precision` are unchanged.

### 2. Training run (no new code)

Run `distill.py` from scratch on the 50k Stockfish dense set:
- `--train 'data/shards_dense_d10_mpv16/train_*.npz'`
- `--val 'data/shards_dense_d10_mpv16/val_*.npz'`
- `--preset baseline-v1`, `--train-data stockfish-d10-mpv16-50k`
- `--batch 128` (8 GB MPS-safe), `--steps 3000`, `--warmup 200`, `--lr 3e-4`,
  `--val-every 300`
- `--ckpt checkpoints/dense_d10_50k`

Local MPS, monitored (~52 min at ~1 step/s). `best.pt` saved on
`val_policy_loss` improvement, with sidecar. Checkpoints stay local (gitignored).

### 3. Gauge + decision gate (no new code)

Run the Phase 0 harness on the new checkpoint:
```
python scripts/ablate.py --ckpt checkpoints/dense_d10_50k/best.pt \
    --device mps --out docs/ablations/dense-d10-50k.md
```
The sidecar routes it to the policy evaluator. Commit the metrics table.

**Headline comparison vs `baseline-v1`** uses the clean, separate puzzle gates
(`data/puzzles.csv`), NOT the leakage-prone dense val:
- `raw_top1@1000`, `raw_top1@300`, `mcts100_top1@300`.

**Success criterion:** the 50k model matches or beats `baseline-v1` —
`raw_top1@1000 >= ~0.289` and `mcts100_top1@300 >= ~0.340`, allowing ~1–2 pt
noise. Because 49k depth-10 positions ≈ baseline's 62k would mean our
self-generated teacher (which we can scale ~unboundedly, unlike the fixed test
bag) is competitive, this green-lights Phase 1b.

**If it underperforms:** diagnose before spending the $10 — likely causes are
teacher depth too shallow (raise depth/multipv), too few positions (generate
more), or recipe (steps/LR/epochs). Record the outcome and the decision in
`docs/ablations/dense-d10-50k.md` (a short "Decision" note).

## Data Flow

```text
data/shards_dense_d10_mpv16/{train,val}_*.npz
      --> distill.py (preset baseline-v1, batch128, 3000 steps, meta)
      --> checkpoints/dense_d10_50k/best.pt (+ best.json sidecar)
      --> ablate.py (sidecar -> policy evaluator -> run_gates)
      --> docs/ablations/dense-d10-50k.md  (compare to baseline-v1.md)
```

## Testing

- One small test for the `distill.py` change: a tiny `fit` run (CPU, 1 step,
  policy batch) writes a sidecar whose `objective=="policy"` and
  `preset`/`train_data` match what was passed. Mirror
  `tests/training/test_trainer_sidecar.py`.
- Training and gauge are runs with no new logic — verified by their outputs
  (the gate table), not unit tests.
- The full existing suite stays green.

## Out of Scope (this iteration → Phase 1b)

Generator hardening (two-stage candidate gen, resume safety, manifest),
large-scale dense generation, GPU training, any architecture change. Those are
triggered only if this validation succeeds.

## Success Criteria (summary)

- `distill.py` writes a routable sidecar; its small test passes; suite green.
- A trained checkpoint at `checkpoints/dense_d10_50k/best.pt` (+ sidecar).
- A committed `docs/ablations/dense-d10-50k.md` with gate metrics + a one-line
  Decision note comparing to `baseline-v1` and stating go / no-go on Phase 1b.
