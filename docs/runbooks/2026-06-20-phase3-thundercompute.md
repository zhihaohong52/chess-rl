# Phase 3 ThunderCompute Runbook (A6000, 77M + 10M control on ~100M)

Spec: `docs/superpowers/specs/2026-06-20-chess-rl-phase3-gpu-scaleup-design.md`.
Plan: `docs/superpowers/plans/2026-06-20-chess-rl-phase3-gpu-scaleup.md`.
Per-minute billing — **stop/delete the instance whenever idle**. Target ~$3–6.

## 1. Provision (local)
```bash
tnr create --gpu a6000 --disk 100          # disk >= 100 GB for ~100M shards (~43 GB)
tnr status                                  # note the instance id
tnr connect <id>                            # opens SSH
```
If `--gpu`/`--disk` flag names differ in your `tnr` version, run `tnr create --help`.

## 2. Environment (on the box)
```bash
git clone https://github.com/zhihaohong52/chess-rl && cd chess-rl
git checkout feat/transformer-rebuild
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # torch will pull the CUDA build on Linux
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
sudo apt-get update && sudo apt-get install -y stockfish   # for the ladder
nproc                                       # number of vCPUs -> --workers below
```

## 3. Data: download + encode (on the box)
```bash
# Verify ONE shard first (pattern guard), then bulk.
python scripts/download_hf_dense.py --start 0 --count 1 --out-dir data/raw_hf
python scripts/download_hf_dense.py --start 1 --count 28 --out-dir data/raw_hf  # ~29 total
df -h .                                      # confirm free space before encoding
python scripts/preencode.py --source hf_dense \
    --input 'data/raw_hf/train-*.msgpack.zst' --out-dir data/shards_p3_100m \
    --temperature 0.1 --val-fraction 0.002 --shard-size 250000 --workers $(nproc)
```

## 4. Smoke test (gauge throughput before committing budget)
```bash
python -u scripts/distill.py --preset p3-80m --device cuda --mixed-precision \
    --train 'data/shards_p3_100m/train_*.npz' --val 'data/shards_p3_100m/val_*.npz' \
    --batch 1024 --steps 500 --warmup 100 --lr 2e-4 --val-every 250 \
    --ckpt checkpoints/smoke
# Note steps/sec and peak VRAM (nvidia-smi). Resize --steps for ~1 pass over 100M
# (~97,700 steps at batch 1024) within budget.
```

## 5. Train 77M (~1 pass)
```bash
python -u scripts/distill.py --preset p3-80m --device cuda --mixed-precision \
    --train 'data/shards_p3_100m/train_*.npz' --val 'data/shards_p3_100m/val_*.npz' \
    --train-data hf_dense_100m --batch 1024 --steps 98000 --warmup 2000 --lr 2e-4 \
    --ema-decay 0.999 --val-every 2000 --ckpt checkpoints/p3_80m
```

## 6. Train 10M control (same 100M data)
```bash
python -u scripts/distill.py --preset p2-value-swiglu-drop --device cuda --mixed-precision \
    --train 'data/shards_p3_100m/train_*.npz' --val 'data/shards_p3_100m/val_*.npz' \
    --train-data hf_dense_100m --batch 1024 --steps 98000 --warmup 2000 --lr 3e-4 \
    --ema-decay 0.999 --val-every 2000 --ckpt checkpoints/p3_10m_ctrl
```

## 7. Evaluate
```bash
# Standard gates (ablate uses --ckpt + --out; checkpoint sidecar routes the evaluator)
python scripts/ablate.py --ckpt checkpoints/p3_80m/best_ema.pt \
    --out docs/ablations/p3-80m.md --device cuda
# Primary head-to-head: scale isolation (both on 100M)
python scripts/arena_eval.py --model checkpoints/p3_80m/best_ema.pt \
    --vs checkpoints/p3_10m_ctrl/best_ema.pt --games 100 --simulations 100 --device cuda
# Secondary: vs prior champion (scp it up first if not present)
python scripts/arena_eval.py --model checkpoints/p3_80m/best_ema.pt \
    --vs checkpoints/p2_full_300k/best_ema.pt --games 100 --simulations 100 --device cuda
# Elo ladder on the stronger model
python scripts/arena_eval.py --model checkpoints/p3_80m/best_ema.pt \
    --skills 1 2 3 4 5 6 7 8 9 10 --games 40 --simulations 100 --depth 4 --device cuda
```

## 8. Record + retrieve + tear down
```bash
# On the box: write docs/ablations/p3-80m.md with results + a Decision section.
# Locally:
tnr scp <id>:chess-rl/checkpoints/p3_80m ./checkpoints/p3_80m
tnr scp <id>:chess-rl/checkpoints/p3_10m_ctrl ./checkpoints/p3_10m_ctrl
tnr scp <id>:chess-rl/docs/ablations/p3-80m.md ./docs/ablations/p3-80m.md
tnr delete <id>                              # stop billing
```
(Exact `tnr scp` direction/syntax: check `tnr scp --help`.)

## Cost guardrails
- Stop/delete the instance during any local-only work.
- If the head-to-head/ladder is eating budget, cut ladder skills or games first.
- Hard ceiling: $10. Re-check `tnr status` / billing between phases.
