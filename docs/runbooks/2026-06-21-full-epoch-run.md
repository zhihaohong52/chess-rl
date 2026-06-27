# Runbook: 77M full-epoch run on the 100M data (vlw=0.3)

**Goal:** train the 77M (`p3-80m`) for ~1 full epoch over the existing 100M dense
positions (p3_80m did only ~0.25 epoch), at `value_loss_weight=0.3`, and test
whether data saturation delivers the data-lever gain (data was the biggest lever:
+223 Elo, see `docs/ablations/p3-80m.md`). Reports the new always-on value
calibration metrics (`val_wdl_ce`, `val_draw_cal`, `val_ece`) every val step.

**Prereq — billing:** the $10 credit is spent. Add credit before starting:
- **100k steps (1.0 epoch):** ~15.3 h A100 + ~0.4 h setup ≈ **~$12.3** → top up to **~$14** for margin.
- **95k steps (0.95 epoch):** ~14.5 h ≈ **~$11.5** → fits **$12**.
A100 prototyping = $0.78/h; measured rate ≈ 0.55 s/step. Hard wall: with no card
on file the instance stops at the credit limit (best_ema is saved every val step,
so a near-final checkpoint survives a cut-off).

## Sequence (driven headless via MCP run_command + inline-flag ssh)

1. **Restore + create** (A6000 if available is cheaper; else A100):
   `create_instance(gpu_type=a100xl, template="chess-rl-p3-100m-data", mode=development, disk_size_gb=120, vcpus=4)`
   Poll `list_instances` until RUNNING. Save the returned key to
   `~/.thunder/keys/<uuid>` (chmod 600).

2. **Sync code** (snapshot is at an older commit):
   ```bash
   cd ~/chess-rl && git pull --ff-only origin feat/transformer-rebuild
   # must include 6c2e6ff (value_ece + always-on calib) and 0258dcf (--value-loss-weight)
   ```

3. **Verify data:** `ls data/shards_p3_100m/train_*.npz | wc -l` → 585 train, 195 val.

4. **Train** (fresh, NOT warm-start — clean schedule over the full epoch):
   ```bash
   nohup bash -c "python -u scripts/distill.py --preset p3-80m --device cuda \
     --mixed-precision --train 'data/shards_p3_100m/train_*.npz' \
     --val 'data/shards_p3_100m/val_*.npz' --train-data hf_dense_100m \
     --batch 1024 --steps 100000 --warmup 2000 --lr 2e-4 --ema-decay 0.999 \
     --val-every 2000 --value-loss-weight 0.3 --stream --num-workers 4 \
     --ckpt checkpoints/p3_80m_1epoch; touch ~/train_1ep.done" > ~/train_1ep.log 2>&1 &
   ```
   Watch `val_top1` (target: clear the vlw=0.3 25k result of 0.506) and the new
   `val_ece` / `val_wdl_ce` (should fall = better-calibrated value as data grows).

5. **Pull checkpoint** (use INLINE ssh/scp flags — zsh does not word-split `$VAR`;
   scp from a `#!/bin/bash` script or inline works, sftp may fail with
   "Connection closed"):
   ```bash
   scp -i ~/.thunder/keys/<uuid> -P <port> -o StrictHostKeyChecking=no \
     -o UserKnownHostsFile=/dev/null \
     ubuntu@<ip>:/home/ubuntu/chess-rl/checkpoints/p3_80m_1epoch/best_ema.pt \
     checkpoints/p3_80m_1epoch/   # + best_ema.json; verify sha256
   ```

6. **Eval locally (free):** head-to-head vs `p3_80m` and vs `p3_80m_vlw03`
   (`arena_eval.py --vs`, 60 games, 50 sims, MPS) + `ablate.py` (now reports
   `value_ece`). Append results to `docs/ablations/p3-80m.md`.

7. **Delete** instance (confirm first) to stop billing; the 100M data + env remain
   in snapshot `chess-rl-p3-100m-data`.

## Success criteria
- `val_top1` > 0.506 (beats the 0.25-epoch + vlw=0.3 model) — expected if
  data-bound.
- `val_ece` / `val_wdl_ce` improve vs p3_80m (better-calibrated value head).
- **Primary:** head-to-head Elo > p3_80m at equal MCTS. If a full epoch does NOT
  beat 0.25 epoch in *games*, the bottleneck has shifted off raw data quantity
  (→ self-play / more unique shards).
