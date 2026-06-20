# Chess-RL Phase 3: GPU Scale-Up (77M model on ThunderCompute)

**Date:** 2026-06-20
**Branch:** `feat/transformer-rebuild`

Phases 0–2 are complete. Phase 0 built a config-gated eval/ablation harness
(`src/model/presets.py`, locked `baseline-v1` at 10,255,879 params). Phase 1
validated dense policy distillation on the downloadable `prdev/chessbench-full-policy-value`
dataset. Phase 2 produced `phase2-best` (preset `p2-value-swiglu-drop` + EMA
0.999): HL-Gauss value head + SwiGLU FFN + dropout 0.05, the strongest engine so
far (MCTS 0.850 vs Stockfish skill-1, undefeated; 0.700 vs skill-3). The recipe
is validated; the next lever is **scale on a rented GPU** under a **$10 hard
budget**. A Codex review (`docs/codex_comments/model_architecture_stockfish_target_2026-06-20.md`)
reframes the goal away from "beat Stockfish" and toward establishing that the
system improves with scale, measured by **checkpoint-vs-checkpoint head-to-head
at equal compute** — which this phase adopts as its core gate.

## Goal

Train a **~77M-parameter** model (the user's chosen jump to Codex's "Base 80M"
rung) with the `phase2-best` recipe on ThunderCompute, and measure whether it
**beats the 10M `phase2-best` head-to-head at equal MCTS budget** — not on
puzzles. Produce a firm Elo for the winner against a wider Stockfish ladder.

This is explicitly a **strength-max scale-up** (the user's "original plan"), not
the minimal controlled 30M experiment. We still keep the head-to-head gate
because puzzle top-1 is known (Phase 2) to misjudge engine strength.

## Locked decisions

| Decision | Choice | Why |
| --- | --- | --- |
| GPU | **RTX A6000** ($0.35/hr, ~28 h in budget) | 77M needs <10% of 48 GB VRAM; A100/H100 advantages (VRAM, bandwidth) are unusable at our scale; A6000 gives the most hours + debug slack |
| Model | **`p3-80m`** = d_model 512, n_layers 16, n_heads 16, d_ff 3072 (SwiGLU) → **77.0M** | Within Codex's recommended shape (d_model ≤512, 12–16 layers); head_dim 32; param count verified with the harness |
| Recipe | `phase2-best`: HL-Gauss value (64 buckets) + SwiGLU + dropout 0.05 + EMA 0.999 | The validated Phase-2 stack |
| Data | **~100M dense positions** @ T=0.1, one pass, encoded **on the box** | 294k (Phase 2) and even 10M starve a 77M model; one pass over 100M *unique* positions beats multi-epoch over a small set (Codex: more unique data > more passes); encoding is now fast + parallel |
| Control | **10M `p2-value-swiglu-drop` retrained on the same 100M** | Isolates *scale* from *data* — the clean Codex experiment; cheap on the GPU |
| Core gate | **model-vs-model head-to-head** (77M vs the 10M control) at equal MCTS sims | Codex's success criterion; the gate we currently lack |
| Out of scope | self-play loop | The only path that truly surpasses the teacher, but too big for $10 + M1 — a future phase |

## Honest expectations (data is the binding constraint)

A 77M model is still a **strong imitator**, bounded by Stockfish/label quality
(DeepMind used ~15B positions for 270M). ~100M unique positions strongly feeds
the 77M and gives the scale hypothesis a fair test, but a win over the 10M
control is not guaranteed. Mitigations: one-pass training, dropout 0.05, EMA,
early-stop on val loss. Because the 10M control trains on the **same** 100M, a
null/negative head-to-head is a clean, valid finding — it means *scale didn't
help at this data/compute*, which bounds the distillation ceiling rather than
being a failure.

## Components to build (local, TDD'd, before renting)

All new architecture stays config-gated; **`baseline-v1` must remain
bit-identical** (the `tests/model/test_presets.py` 10,255,879-param lock stays
green). Encoding-pipeline speedups are already landed (commit `73b6b70`).

### 1. `p3-80m` preset — `src/model/presets.py`
Add `"p3-80m": {"d_model": 512, "n_layers": 16, "n_heads": 16, "d_ff": 3072,
"ffn_type": "swiglu", "value_head_type": "hlgauss", "value_buckets": 64,
"transformer_dropout": 0.05}`. Test: `build_model("p3-80m")` constructs and
reports 77.0M params; `baseline-v1` param-lock test still passes.

### 2. CUDA device support — `src/training/distill_trainer.py`, `scripts/distill.py`
`DistillTrainer.__init__` default device becomes
`"cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"`.
Add `--device` to `distill.py` and pass it through. MPS/CPU paths unchanged
(local gauges stay reproducible). Test: trainer honors an explicit `device=`
arg; default selection logic unit-tested with `torch.cuda.is_available` patched.

### 3. Real mixed precision (AMP) — `src/training/distill_trainer.py`
Today `--mixed-precision` is a no-op. Wire `torch.autocast` around the
forward+loss in `train_step`. On CUDA use **bf16** (Ampere-native, no
`GradScaler` needed). On MPS/CPU, AMP is a no-op so existing behavior and
numerics are untouched. Test: a CUDA-gated test (skipped when no CUDA) asserts a
step runs under autocast; a CPU test asserts `mixed_precision=True` is a no-op
(loss finite, weights update).

### 4. Model-vs-model head-to-head — `src/eval/arena.py` + `scripts/arena_eval.py`
Add a `ModelOpponent` (or generalize `play_match` to accept two movers) so two
checkpoints play each other with MCTS at equal simulations, reusing
`play_match`/`elo_diff`. Extend `arena_eval.py` with a `--vs <ckpt>` mode that
loads both nets via `load_for_eval`, plays N games (color-balanced), and reports
W/D/L + score + est. Elo gap. Test: a self-play sanity match (a net vs itself)
scores ≈0.5 over enough games; the opponent interface matches `StockfishOpponent`.

### 5. HF raw-shard download helper — `scripts/download_hf_dense.py`
Small script to fetch N raw shards `train-{NNNNN}-of-01024.msgpack.zst` from the
`prdev/chessbench-full-policy-value` HF resolve URL into `data/raw_hf/`. Used on
the box (fast network). Test: URL pattern is well-formed (no network test).

## Data plan

On the box: download ~29 raw shards (~3.5M positions each, ~72 MB → ~2.1 GB) into
`data/raw_hf/`, then `scripts/preencode.py --source hf_dense --input
'data/raw_hf/train-*.msgpack.zst' --workers <vCPUs> --temperature 0.1
--val-fraction 0.002 --shard-size 250000 --out-dir data/shards_p3_100m`
(uncompressed default). Encoding ~100M is ~35 min in parallel; ~43 GB on disk
(ensure the instance disk ≥ 100 GB). Result: ~99.8M train + ~0.2M val. Both the
77M run and the 10M control train on this same set.

## Training recipe

**77M run:** `scripts/distill.py --preset p3-80m --device cuda --mixed-precision
--train 'data/shards_p3_100m/train_*.npz' --val 'data/shards_p3_100m/val_*.npz'
--batch 1024 --steps ~98000 --warmup 2000 --lr 2e-4 --ema-decay 0.999
--val-every 2000 --ckpt checkpoints/p3_80m`.
- Batch 1024 (VRAM allows far more; tune up if throughput-bound).
- LR 2e-4 (lower than the 10M's 3e-4 for the bigger model); cosine via existing
  scheduler; warmup 2000.
- ~1 pass over 100M (~97.7k steps at batch 1024). Gauge throughput with a
  500-step smoke run first and resize `--steps` to the budget.
- EMA → `best_ema.pt` is the evaluation checkpoint.

**10M control:** same command with `--preset p2-value-swiglu-drop --lr 3e-4
--ckpt checkpoints/p3_10m_ctrl` on the **same** `data/shards_p3_100m`. Trains
~8× faster per step than the 77M, so it's a small fraction of the budget. Take
its `best_ema.pt` for the head-to-head.

## Evaluation plan

1. **Standard gates** — `scripts/ablate.py` on `p3_80m/best_ema.pt`: raw top-1,
   value calibration (MAE/sign-acc), mate-in-1, throughput.
2. **Core gate (head-to-head, scale isolation)** —
   `arena_eval.py --vs checkpoints/p3_10m_ctrl/best_ema.pt` at equal sims
   (e.g. 100), ≥100 color-balanced games. Both models trained on the same 100M,
   so this isolates scale. **Success: 77M scores >0.5 beyond ~1.5 std-err.**
3. **Champion check (secondary head-to-head)** — 77M vs the prior champion
   `checkpoints/p2_full_300k/best_ema.pt` (10M `phase2-best`, trained on 294k),
   same settings — "did we beat the old best?" (confounded by data, hence
   secondary).
4. **Elo ladder** — wider Stockfish arena (skills 1–10, depth ≥4, ≥40 games/skill)
   on the stronger model for a firmer absolute Elo.
5. Record all in `docs/ablations/p3-80m.md` with a Decision section.

## Remote runbook (cost discipline)

Provision only when local code + a smoke plan are ready. Per-minute billing;
**stop the instance whenever idle**.

1. `tnr create` (A6000, disk ≥ 100 GB for ~100M shards) → `tnr status` → `tnr connect <id>`.
2. On the box: clone branch, create venv, `pip install -r requirements.txt`
   (+ torch CUDA build), install Stockfish for arena.
3. `tnr scp` the repo or `git pull`; download raw shards on the box.
4. Parallel-encode → smoke train (500 steps) to gauge throughput/VRAM → size
   `--steps` → full 77M train → 10M control train → eval (head-to-heads + ladder).
5. `tnr scp` checkpoints + ablation md back to local.
6. `tnr delete <id>`.

## Success criteria

- `p3-80m` builds at 77.0M; `baseline-v1` stays bit-identical (param lock green).
- Both trains (77M + 10M control on the same 100M) complete within budget;
  `best_ema.pt` + routable sidecars produced.
- **Primary:** head-to-head 77M vs 10M control over ≥100 games (win OR a clear
  null — both valid scale findings). **Secondary:** 77M vs prior champion
  `p2_full_300k`.
- Firm Elo ladder for the stronger model; results + decision in
  `docs/ablations/p3-80m.md`.
- Total spend < $10 (target ~$3–6 of GPU time: ~$0.3 encode + ~$1–2 (77M) +
  ~$0.3 (10M control) + ~$1–2 head-to-heads/ladder, with buffer for the smoke
  run and debugging).

## Risks

- **Data-starvation / overfit at 77M** — largely addressed at 100M; one-pass
  training + dropout/EMA/early-stop. If val still underfits, the data lever
  (more shards) is available.
- **Encode/disk at 100M** — ~43 GB encoded; provision disk ≥ 100 GB and confirm
  free space before encoding.
- **First CUDA run** — new device + AMP path; de-risk with the 500-step smoke run
  before committing budget.
- **Throughput unknown** — measure before sizing `--steps`; A6000 + bf16 + seq-65
  should be fast, but don't assume.
- **MCTS arena is slow** — head-to-head games dominate eval time; cap games/sims
  sensibly and run on the GPU box.

## Testing strategy

TDD each code unit locally (CPU/MPS): preset param count, device selection, AMP
no-op-on-CPU + CUDA-gated step, head-to-head self-play ≈0.5, download URL shape.
Keep the full suite green and `baseline-v1` bit-identical on every model-touching
task. Real training/eval runs happen on the box (not in CI).
