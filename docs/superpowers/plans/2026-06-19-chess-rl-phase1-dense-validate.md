# Phase 1 (Validate): Dense-Data Training on 50k — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train `baseline-v1`'s architecture from scratch on the 50k Stockfish dense set, then gauge it against `baseline-v1` on the Phase 0 puzzle gates to decide whether to scale dense generation (Phase 1b).

**Architecture:** One code change (`scripts/distill.py` gains preset/sidecar/short-run-schedule support so checkpoints are routable by the Phase 0 gauge), then a monitored local MPS training run, then the Phase 0 ablation runner with a go/no-go decision note. No model or harness changes.

**Tech Stack:** Python 3.12, PyTorch 2.12 (MPS), python-chess, numpy, pytest.

**Environment & conventions (read first):**
- Use the off-iCloud venv: `/Users/james/.venvs/chess-rl/bin/python` and `/Users/james/.venvs/chess-rl/bin/pytest`.
- Prefix shell commands with `rtk`, including inside `&&` chains.
- Branch `feat/transformer-rebuild`. Commit after each task.
- Data already present: `data/shards_dense_d10_mpv16/` (`train_*.npz` = 49k, `val_*.npz` = 1k). Do not regenerate it.
- Baseline reference (`docs/ablations/baseline-v1.md`): raw_top1@1000 0.2890, raw_top1@300 0.2767, mcts100_top1@300 0.3400, params 10,255,879.
- Verified interfaces: `src.model.presets.resolve_config(preset) -> SimpleNamespace` (mutable; has all Config attrs incl. `distill_lr`, `distill_warmup_steps`, `distill_total_steps`); `ChessTransformer(cfg)`; `DistillTrainer(net, cfg, mixed_precision=False, device=None)` with `.fit(train_loader, steps, val_loader=None, val_every=1000, ckpt_dir=..., meta=None)` (writes `best.pt`/`last.pt` + JSON sidecar when meta given, stamping `objective="policy"`); `src.data.dataset.make_dataloader(shard_paths, batch_size, policy_size, shuffle, num_workers)`; `src.data.preencode.write_shard(labeled_positions, path)`; `src.data.labeled_position.LabeledPosition`; `src.training.checkpoint_meta.read_sidecar(ckpt)`.

---

## File Structure

- Modify: `scripts/distill.py` — add `--preset`, `--train-data`, `--warmup`, `--lr`; resolve preset config; set short-run LR schedule; build net from preset; pass `meta` to `fit`.
- Create: `tests/training/test_distill_cli.py` — integration test that running `distill.py` writes a routable sidecar.
- Run-only (no code): training run → `checkpoints/dense_d10_50k/`; gauge → `docs/ablations/dense-d10-50k.md` (committed).

---

## Task 1: `distill.py` — preset + sidecar + short-run LR schedule

**Files:**
- Modify: `scripts/distill.py`
- Test: `tests/training/test_distill_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/training/test_distill_cli.py
import sys
import chess

from src.data.labeled_position import LabeledPosition
from src.data.preencode import write_shard
from src.data.targets import winprob_to_wdl
from src.training.checkpoint_meta import read_sidecar


def _tiny_dense_shard(path):
    # one position: e2e4 gets all the policy mass, white-winning value
    lp = LabeledPosition(
        fen=chess.STARTING_FEN,
        policy=[("e2e4", 1.0)],
        wdl=winprob_to_wdl(0.6),
        moves_left=40.0,
    )
    write_shard([lp, lp], path)


def test_distill_cli_writes_routable_sidecar(tmp_path, monkeypatch):
    import scripts.distill as distill

    train = str(tmp_path / "train_0.npz")
    val = str(tmp_path / "val_0.npz")
    _tiny_dense_shard(train)
    _tiny_dense_shard(val)
    ckpt = str(tmp_path / "ck")

    argv = ["distill.py", "--train", train, "--val", val,
            "--preset", "baseline-v1", "--train-data", "unit-test",
            "--batch", "2", "--steps", "1", "--warmup", "1", "--lr", "3e-4",
            "--val-every", "1", "--ckpt", ckpt]
    monkeypatch.setattr(sys, "argv", argv)
    distill.main()

    meta = read_sidecar(str(tmp_path / "ck" / "last.pt"))
    assert meta is not None
    assert meta["objective"] == "policy"
    assert meta["preset"] == "baseline-v1"
    assert meta["train_data"] == "unit-test"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/training/test_distill_cli.py -q`
Expected: FAIL — `distill.py` does not yet accept `--preset` (argparse error / SystemExit), or no sidecar is written.

- [ ] **Step 3: Implement the change**

Replace the body of `scripts/distill.py` from the imports through `main()` with:

```python
#!/usr/bin/env python3
"""Train ChessTransformer by distillation on NPZ shards (PyTorch).

Usage:
    python scripts/distill.py --train 'data/shards_dense_d10_mpv16/train_*.npz' \
           --val 'data/shards_dense_d10_mpv16/val_*.npz' --preset baseline-v1 \
           --train-data stockfish-d10-mpv16-50k --batch 128 --steps 3000 \
           --warmup 200 --lr 3e-4 --val-every 300 --ckpt checkpoints/dense_d10_50k
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.model.presets import resolve_config
from src.model.transformer import ChessTransformer
from src.game.move_encoder import get_move_encoder
from src.data.dataset import make_dataloader
from src.training.distill_trainer import DistillTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="glob for training shards (*.npz)")
    ap.add_argument("--val", default=None, help="glob for validation shards (*.npz)")
    ap.add_argument("--preset", default="baseline-v1", help="model preset name")
    ap.add_argument("--train-data", default="", help="free-text label stored in the sidecar")
    ap.add_argument("--steps", type=int, default=Config.distill_total_steps)
    ap.add_argument("--batch", type=int, default=Config.distill_batch_size)
    ap.add_argument("--warmup", type=int, default=Config.distill_warmup_steps)
    ap.add_argument("--lr", type=float, default=Config.distill_lr)
    ap.add_argument("--ckpt", default="checkpoints/distill")
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--mixed-precision", action="store_true")
    args = ap.parse_args()

    # Resolve the model preset and align the LR schedule with this run's length.
    cfg = resolve_config(args.preset)
    cfg.distill_lr = args.lr
    cfg.distill_warmup_steps = args.warmup
    cfg.distill_total_steps = args.steps

    P = get_move_encoder().policy_size
    train_loader = make_dataloader(
        sorted(glob.glob(args.train)), args.batch, P, shuffle=True
    )
    val_loader = (
        make_dataloader(sorted(glob.glob(args.val)), args.batch, P, shuffle=False)
        if args.val
        else None
    )

    net = ChessTransformer(cfg)
    trainer = DistillTrainer(net, cfg, mixed_precision=args.mixed_precision)
    best = trainer.fit(
        train_loader,
        args.steps,
        val_loader=val_loader,
        val_every=args.val_every,
        ckpt_dir=args.ckpt,
        meta={"preset": args.preset, "train_data": args.train_data},
    )
    print(f"done. best val_policy_loss={best:.4f}; checkpoints in {args.ckpt}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/training/test_distill_cli.py -q`
Expected: PASS (1 passed). (Builds the real model + 1 step on MPS/CPU; ~10–30s.)

- [ ] **Step 5: Run the existing suite to confirm no regression**

Run: `/Users/james/.venvs/chess-rl/bin/pytest -q`
Expected: all pass (the prior 111 + this new test), 1 skipped, 2 deselected.

- [ ] **Step 6: Commit**

```bash
rtk git add scripts/distill.py tests/training/test_distill_cli.py && \
rtk git commit -m "feat(phase1): distill.py preset + sidecar + short-run LR schedule"
```

---

## Task 2: Train on the 50k Stockfish dense set (monitored run)

**Files:** none (produces local `checkpoints/dense_d10_50k/` — gitignored).

This is an execution task, not a pytest task. The training is long (~52 min on MPS); run it monitored so val progress streams and `best.pt` is saved on improvement.

- [ ] **Step 1: Sanity-check the data is present**

Run: `ls data/shards_dense_d10_mpv16/train_*.npz | wc -l && ls data/shards_dense_d10_mpv16/val_*.npz | wc -l`
Expected: a nonzero train shard count and at least 1 val shard. If zero, STOP — the dataset is missing (it should already exist from Phase 0 generation).

- [ ] **Step 2: Launch the monitored training run**

Run (batch 128 is the 8 GB MPS-safe size; do NOT raise it):
```bash
/Users/james/.venvs/chess-rl/bin/python scripts/distill.py \
  --train 'data/shards_dense_d10_mpv16/train_*.npz' \
  --val 'data/shards_dense_d10_mpv16/val_*.npz' \
  --preset baseline-v1 --train-data stockfish-d10-mpv16-50k \
  --batch 128 --steps 3000 --warmup 200 --lr 3e-4 --val-every 300 \
  --ckpt checkpoints/dense_d10_50k
```
Stream the `step N loss ... {val_policy_loss, val_value_sign_acc, val_top1}` lines. Watch that `val_policy_loss` trends down and `val_top1` trends up. Expected end state: a printed `done. best val_policy_loss=...` line and `checkpoints/dense_d10_50k/best.pt` + `best.json` on disk.

- [ ] **Step 3: Confirm the checkpoint + routable sidecar exist**

Run: `ls -lh checkpoints/dense_d10_50k/best.pt && cat checkpoints/dense_d10_50k/best.json`
Expected: a ~39 MB `best.pt` and a sidecar JSON with `"preset": "baseline-v1"`, `"objective": "policy"`, `"train_data": "stockfish-d10-mpv16-50k"`.

- [ ] **Step 4: Confirm the checkpoint is NOT staged for commit**

Run: `rtk git status --porcelain checkpoints/`
Expected: empty output (checkpoints are gitignored via `*.pt`). If anything under `checkpoints/` appears, do NOT commit it.

No commit in this task (only a local binary was produced).

---

## Task 3: Gauge vs baseline-v1 + record the go/no-go decision

**Files:**
- Create (committed): `docs/ablations/dense-d10-50k.md` and `docs/ablations/dense-d10-50k.csv`

- [ ] **Step 1: Run the Phase 0 ablation runner on the new checkpoint (MPS)**

Run:
```bash
/Users/james/.venvs/chess-rl/bin/python scripts/ablate.py \
  --ckpt checkpoints/dense_d10_50k/best.pt --device mps \
  --out docs/ablations/dense-d10-50k.md
```
The sidecar routes it to the policy evaluator. This runs the full gates (raw top-1 on 1000 puzzles, MCTS-100 on 300, dense val metrics, throughput) — several minutes. Expected: prints the table and writes `docs/ablations/dense-d10-50k.md` + `.csv`.

- [ ] **Step 2: Compare to baseline-v1 and decide**

Read both tables:
```bash
rtk read docs/ablations/baseline-v1.md && rtk read docs/ablations/dense-d10-50k.md
```
Compare the clean puzzle gates: `raw_top1@1000`, `raw_top1@300`, `mcts100_top1@300`.
- **GO (green-light Phase 1b)** if the 50k model matches or beats baseline within ~1–2 pts: `raw_top1@1000 >= ~0.27` and `mcts100_top1@300 >= ~0.33` (i.e., our self-generated depth-10 teacher is competitive with the fixed 62k test bag despite fewer positions).
- **NO-GO (diagnose first)** if it is clearly worse (e.g., `raw_top1@1000 < ~0.26`): the likely fixes are deeper Stockfish (raise `--depth`/`--multipv-cap` in generation), more positions, or recipe (steps/LR). Do not spend the GPU budget yet.

- [ ] **Step 3: Append a Decision note to the metrics doc**

Append a short section to `docs/ablations/dense-d10-50k.md` (use the Edit/Write tool, do not hand-wave) stating: the comparison numbers (new vs baseline for the three puzzle gates), the verdict (GO / NO-GO), and the one-line rationale. Example shape (fill with the ACTUAL observed numbers — do not invent):

```markdown
## Decision (2026-06-19)

| gate | baseline-v1 | dense-d10-50k |
| --- | --- | --- |
| raw_top1@1000 | 0.2890 | <observed> |
| raw_top1@300 | 0.2767 | <observed> |
| mcts100_top1@300 | 0.3400 | <observed> |

**Verdict: <GO | NO-GO>.** <one-line rationale referencing the numbers, e.g.
"49k depth-10 dense ≈ baseline's 62k on raw top-1 and MCTS, so the self-generated
teacher is competitive and scalable — green-light Phase 1b (harden generator +
large-scale generation + GPU run)." OR "raw top-1 fell to X (< baseline 0.289);
raise teacher depth / position count before scaling.">
```

- [ ] **Step 4: Verify only metrics/docs are staged, then commit**

Run: `rtk git status --porcelain`
Expected: only `docs/ablations/dense-d10-50k.md` and `docs/ablations/dense-d10-50k.csv` are new; no `.pt`/`.npz`/`data/` entries. Then:
```bash
rtk git add docs/ablations/dense-d10-50k.md docs/ablations/dense-d10-50k.csv && \
rtk git commit -m "feat(phase1): gauge dense-d10-50k vs baseline-v1 + decision note" && \
rtk git push
```

---

## Self-Review Checklist (completed by plan author)

- **Spec coverage:** distill.py preset+sidecar+schedule (Task 1, with its test); training run on the 50k set with the exact recipe (Task 2); gauge via ablate.py + committed metrics + go/no-go decision note (Task 3). All spec components covered.
- **Placeholder scan:** the Decision-note template explicitly says to fill in ACTUAL observed numbers (not a placeholder left in code/results) — acceptable because the numbers are produced by the run, not knowable at plan time.
- **Type consistency:** `resolve_config` returns a mutable namespace with `distill_lr/warmup_steps/total_steps`; `DistillTrainer(net, cfg)` reads those at init; `fit(..., meta=...)` writes the sidecar; `read_sidecar` reads it; `ablate.py`/`load_for_eval` consume `preset`+`objective` from it. Signatures consistent across tasks.
- **Persistence policy:** only `docs/ablations/*` committed; `.pt` checkpoints gitignored (verified in Task 2 step 4 and Task 3 step 4).
