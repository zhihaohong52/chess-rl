# Phase 1 Plan 3 — Distillation Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the `ChessTransformer` by supervised distillation on pre-encoded shards — policy KL + WDL cross-entropy + moves-left Huber — with validation metrics, checkpointing, a CLI, budget go/no-go gates, and a rented-GPU runbook.

**Architecture:** A `DistillTrainer` wraps the Plan 1 network and a Plan 2 `tf.data` dataset; an AdamW optimizer with warmup→cosine LR drives a mixed-precision `train_step`; validation tracks policy KL, value-sign accuracy, and top-1 move match; the best checkpoint by validation policy-loss is kept.

**Tech Stack:** TensorFlow/Keras (AdamW, mixed precision), NumPy, pytest. Depends on Plan 1 (`ChessTransformer`) and Plan 2 (`make_dataset`).

**Contracts:** `docs/superpowers/plans/_interface-contracts.md` — "Losses". Tests use tiny synthetic batches (no real shards, no GPU).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/training/distill_losses.py` (new) | Policy/value/moves-left losses + combined total |
| `src/training/distill_trainer.py` (new) | `DistillTrainer`: optimizer, `train_step`, `fit`, validation, checkpoints |
| `src/training/distill_metrics.py` (new) | Validation metrics (KL, value-sign acc, top-1 match) |
| `scripts/distill.py` (new) | CLI: shards → trained checkpoint |
| `scripts/gate_smoke.py` (new) | Go/no-go gate: loss decreases on a synthetic batch |
| `tests/training/` (new) | Unit tests |

All shell commands prefixed with `rtk`. Use `.venv/bin/python` and `.venv/bin/python -m pytest`.

---

### Task 1: Losses

**Files:**
- Create: `src/training/distill_losses.py`
- Test: `tests/training/__init__.py`, `tests/training/test_losses.py`

- [ ] **Step 1: Write the failing test**

`tests/training/__init__.py`: (empty)

`tests/training/test_losses.py`:
```python
import numpy as np
import tensorflow as tf
from src.training.distill_losses import policy_loss, value_loss, moves_left_loss, total_loss


def test_policy_loss_lower_when_aligned():
    target = tf.constant([[0.0, 1.0, 0.0]])
    good = tf.constant([[0.0, 10.0, 0.0]])
    bad = tf.constant([[10.0, 0.0, 0.0]])
    assert float(policy_loss(good, target)) < float(policy_loss(bad, target))


def test_value_loss_lower_when_aligned():
    target = tf.constant([[1.0, 0.0, 0.0]])  # win
    good = tf.constant([[10.0, 0.0, 0.0]])
    bad = tf.constant([[0.0, 0.0, 10.0]])
    assert float(value_loss(good, target)) < float(value_loss(bad, target))


def test_total_loss_combines_parts():
    pol = tf.constant([[0.0, 5.0, 0.0]]); pol_t = tf.constant([[0.0, 1.0, 0.0]])
    wdl = tf.constant([[5.0, 0.0, 0.0]]); wdl_t = tf.constant([[1.0, 0.0, 0.0]])
    ml = tf.constant([[40.0]]); ml_t = tf.constant([[42.0]])
    loss, parts = total_loss(pol, wdl, ml, pol_t, wdl_t, ml_t)
    assert set(parts.keys()) == {"policy", "value", "moves_left"}
    assert float(loss) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk .venv/bin/python -m pytest tests/training/test_losses.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.training.distill_losses'`.

- [ ] **Step 3: Write minimal implementation**

`src/training/distill_losses.py`:
```python
"""Distillation losses: policy KL/CE, WDL cross-entropy, moves-left Huber."""

import tensorflow as tf

_huber = tf.keras.losses.Huber()


def policy_loss(policy_logits, policy_target):
    """Soft-target cross-entropy over the move distribution."""
    ce = tf.nn.softmax_cross_entropy_with_logits(labels=policy_target, logits=policy_logits)
    return tf.reduce_mean(ce)


def value_loss(wdl_logits, wdl_target):
    """Softmax cross-entropy over {W,D,L}."""
    ce = tf.nn.softmax_cross_entropy_with_logits(labels=wdl_target, logits=wdl_logits)
    return tf.reduce_mean(ce)


def moves_left_loss(moves_left_pred, moves_left_target):
    return _huber(moves_left_target, moves_left_pred)


def total_loss(policy_logits, wdl_logits, moves_left_pred,
               policy_target, wdl_target, moves_left_target,
               value_weight=1.0, moves_left_weight=0.1):
    p = policy_loss(policy_logits, policy_target)
    v = value_loss(wdl_logits, wdl_target)
    m = moves_left_loss(moves_left_pred, moves_left_target)
    total = p + value_weight * v + moves_left_weight * m
    return total, {"policy": p, "value": v, "moves_left": m}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk .venv/bin/python -m pytest tests/training/test_losses.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
rtk git add src/training/distill_losses.py tests/training/__init__.py tests/training/test_losses.py && rtk git commit -m "feat: distillation losses"
```

---

### Task 2: Validation metrics

**Files:**
- Create: `src/training/distill_metrics.py`
- Test: `tests/training/test_metrics.py`

- [ ] **Step 1: Write the failing test**

`tests/training/test_metrics.py`:
```python
import numpy as np
import tensorflow as tf
from src.training.distill_metrics import value_sign_accuracy, top1_move_match


def test_value_sign_accuracy():
    wdl_logits = tf.constant([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]])  # win, loss
    wdl_target = tf.constant([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert value_sign_accuracy(wdl_logits, wdl_target) == 1.0


def test_top1_move_match():
    pol = tf.constant([[0.1, 9.0, 0.2], [3.0, 0.0, 0.0]])
    tgt = tf.constant([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    assert top1_move_match(pol, tgt) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk .venv/bin/python -m pytest tests/training/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.training.distill_metrics'`.

- [ ] **Step 3: Write minimal implementation**

`src/training/distill_metrics.py`:
```python
"""Validation metrics for distillation."""

import tensorflow as tf


def value_sign_accuracy(wdl_logits, wdl_target):
    """Fraction where sign(P(W)-P(L)) matches sign(target_W - target_L)."""
    p = tf.nn.softmax(wdl_logits, axis=-1)
    pred = tf.sign(p[:, 0] - p[:, 2])
    true = tf.sign(wdl_target[:, 0] - wdl_target[:, 2])
    return float(tf.reduce_mean(tf.cast(tf.equal(pred, true), tf.float32)))


def top1_move_match(policy_logits, policy_target):
    """Fraction where argmax(pred) == argmax(target)."""
    pred = tf.argmax(policy_logits, axis=-1)
    true = tf.argmax(policy_target, axis=-1)
    return float(tf.reduce_mean(tf.cast(tf.equal(pred, true), tf.float32)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk .venv/bin/python -m pytest tests/training/test_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/training/distill_metrics.py tests/training/test_metrics.py && rtk git commit -m "feat: distillation validation metrics"
```

---

### Task 3: DistillTrainer — train_step decreases loss

**Files:**
- Create: `src/training/distill_trainer.py`
- Test: `tests/training/test_trainer.py`

- [ ] **Step 1: Write the failing test**

`tests/training/test_trainer.py`:
```python
import numpy as np
import tensorflow as tf
import chess
from config import Config
from src.model.transformer import ChessTransformer
from src.game.token_encoder import encode_batch
from src.game.move_encoder import get_move_encoder
from src.training.distill_trainer import DistillTrainer


def _tiny_batch():
    P = get_move_encoder().policy_size
    boards = [chess.Board(), chess.Board()]
    sq, sf = encode_batch(boards, [0, 0])
    e2e4 = get_move_encoder().encode(chess.Move.from_uci("e2e4"))
    pol = np.zeros((2, P), dtype=np.float32); pol[:, e2e4] = 1.0
    wdl = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    ml = np.array([[40.0], [40.0]], dtype=np.float32)
    return (tf.constant(sq), tf.constant(sf)), (tf.constant(pol), tf.constant(wdl), tf.constant(ml))


def test_train_step_decreases_loss():
    net = ChessTransformer(Config)
    trainer = DistillTrainer(net, Config)
    inputs, targets = _tiny_batch()
    first = float(trainer.train_step(inputs, targets)[0])
    for _ in range(30):
        last = float(trainer.train_step(inputs, targets)[0])
    assert last < first * 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk .venv/bin/python -m pytest tests/training/test_trainer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.training.distill_trainer'`.

- [ ] **Step 3: Write minimal implementation**

`src/training/distill_trainer.py`:
```python
"""Supervised distillation trainer for ChessTransformer."""

import os
import tensorflow as tf
from tensorflow import keras

from src.training.distill_losses import total_loss
from src.training.distill_metrics import value_sign_accuracy, top1_move_match


class DistillTrainer:
    def __init__(self, net, cfg, mixed_precision: bool = False):
        self.net = net
        self.cfg = cfg
        if mixed_precision:
            keras.mixed_precision.set_global_policy("mixed_bfloat16")
        lr = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.0,
            decay_steps=cfg.distill_total_steps,
            warmup_target=cfg.distill_lr,
            warmup_steps=cfg.distill_warmup_steps,
            alpha=0.05,
        )
        self.opt = keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4, clipnorm=1.0)

    @tf.function(reduce_retracing=True)
    def train_step(self, inputs, targets):
        sq, sf = inputs
        pol_t, wdl_t, ml_t = targets
        with tf.GradientTape() as tape:
            pol, wdl, ml = self.net(sq, sf, training=True)
            loss, parts = total_loss(pol, wdl, ml, pol_t, wdl_t, ml_t)
        grads = tape.gradient(loss, self.net.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.net.trainable_variables))
        return loss, parts

    def evaluate(self, val_ds, max_batches: int = 50):
        ps, vs, ts, n = 0.0, 0.0, 0.0, 0
        for i, (inputs, targets) in enumerate(val_ds):
            if i >= max_batches:
                break
            sq, sf = inputs
            pol_t, wdl_t, ml_t = targets
            pol, wdl, ml = self.net(sq, sf, training=False)
            _, parts = total_loss(pol, wdl, ml, pol_t, wdl_t, ml_t)
            ps += float(parts["policy"]); n += 1
            vs += value_sign_accuracy(wdl, wdl_t)
            ts += top1_move_match(pol, pol_t)
        n = max(n, 1)
        return {"val_policy_loss": ps / n, "val_value_sign_acc": vs / n, "val_top1": ts / n}

    def fit(self, train_ds, steps: int, val_ds=None, val_every: int = 1000,
            ckpt_dir: str = "checkpoints/distill"):
        os.makedirs(ckpt_dir, exist_ok=True)
        best = float("inf")
        it = iter(train_ds)
        for step in range(1, steps + 1):
            try:
                inputs, targets = next(it)
            except StopIteration:
                it = iter(train_ds)
                inputs, targets = next(it)
            loss, parts = self.train_step(inputs, targets)
            if val_ds is not None and step % val_every == 0:
                metrics = self.evaluate(val_ds)
                print(f"step {step} loss {float(loss):.4f} {metrics}")
                if metrics["val_policy_loss"] < best:
                    best = metrics["val_policy_loss"]
                    self.net.save_weights(os.path.join(ckpt_dir, "best.weights.h5"))
        self.net.save_weights(os.path.join(ckpt_dir, "last.weights.h5"))
        return best
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk .venv/bin/python -m pytest tests/training/test_trainer.py -v`
Expected: PASS. (If loss doesn't drop 10%, raise the step count to 60 or lower `Config.distill_warmup_steps` for the test — the warmup may keep LR near 0 for the first steps.)

- [ ] **Step 5: Commit**

```bash
rtk git add src/training/distill_trainer.py tests/training/test_trainer.py && rtk git commit -m "feat: DistillTrainer (train_step, evaluate, fit)"
```

---

### Task 4: Checkpoint save/load round-trip

**Files:**
- Test: `tests/training/test_checkpoint.py`

- [ ] **Step 1: Write the failing test**

`tests/training/test_checkpoint.py`:
```python
import numpy as np
import tensorflow as tf
import chess
from config import Config
from src.model.transformer import ChessTransformer
from src.game.token_encoder import encode_batch


def test_checkpoint_roundtrip(tmp_path):
    net = ChessTransformer(Config)
    sq, sf = encode_batch([chess.Board()], [0])
    p1, _, _ = net(tf.constant(sq), tf.constant(sf))
    path = str(tmp_path / "ck.weights.h5")
    net.save_weights(path)
    net2 = ChessTransformer(Config)
    net2(tf.constant(sq), tf.constant(sf))
    net2.load_weights(path)
    p2, _, _ = net2(tf.constant(sq), tf.constant(sf))
    assert np.allclose(p1.numpy(), p2.numpy(), atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails (or passes immediately)**

Run: `rtk .venv/bin/python -m pytest tests/training/test_checkpoint.py -v`
Expected: PASS (the network's Keras `save_weights`/`load_weights` already supports this; this test guards the trainer's checkpoint contract). If it fails on a shape/build error, ensure `net2` is built by a forward pass before `load_weights`.

- [ ] **Step 3: (No new code expected)** If failing, the only valid fix is building before load (already in the test) — otherwise the contract holds.

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk .venv/bin/python -m pytest tests/training/test_checkpoint.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add tests/training/test_checkpoint.py && rtk git commit -m "test: distillation checkpoint round-trip"
```

---

### Task 5: Training CLI

**Files:**
- Create: `scripts/distill.py`

- [ ] **Step 1: Write the CLI**

`scripts/distill.py`:
```python
#!/usr/bin/env python3
"""Train ChessTransformer by distillation on TFRecord shards.

Usage: python scripts/distill.py --train 'data/shards/train_*.tfrecord' \
       --val 'data/shards/val_*.tfrecord' --steps 60000 --batch 1024 \
       --ckpt checkpoints/distill --mixed-precision
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.model.transformer import ChessTransformer
from src.game.move_encoder import get_move_encoder
from src.data.dataset import make_dataset
from src.training.distill_trainer import DistillTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="glob for training shards")
    ap.add_argument("--val", default=None, help="glob for validation shards")
    ap.add_argument("--steps", type=int, default=Config.distill_total_steps)
    ap.add_argument("--batch", type=int, default=Config.distill_batch_size)
    ap.add_argument("--ckpt", default="checkpoints/distill")
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--mixed-precision", action="store_true")
    args = ap.parse_args()

    P = get_move_encoder().policy_size
    train_ds = make_dataset(sorted(glob.glob(args.train)), args.batch, P, shuffle=True)
    val_ds = (make_dataset(sorted(glob.glob(args.val)), args.batch, P, shuffle=False)
              if args.val else None)

    net = ChessTransformer(Config)
    trainer = DistillTrainer(net, Config, mixed_precision=args.mixed_precision)
    best = trainer.fit(train_ds, args.steps, val_ds=val_ds,
                       val_every=args.val_every, ckpt_dir=args.ckpt)
    print(f"done. best val_policy_loss={best:.4f}; checkpoints in {args.ckpt}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the CLI argument parsing**

Run: `rtk .venv/bin/python scripts/distill.py --help`
Expected: prints usage with `--train`, `--val`, `--steps`, `--mixed-precision`.

- [ ] **Step 3: Commit**

```bash
rtk git add scripts/distill.py && rtk git commit -m "feat: distillation training CLI"
```

---

### Task 6: Go/no-go smoke gate

**Files:**
- Create: `scripts/gate_smoke.py`
- Test: `tests/training/test_gate_smoke.py`

- [ ] **Step 1: Write the failing test**

`tests/training/test_gate_smoke.py`:
```python
from scripts.gate_smoke import run_smoke_gate


def test_smoke_gate_passes_on_synthetic_batch():
    ok, info = run_smoke_gate(steps=30)
    assert ok is True
    assert info["final_loss"] < info["initial_loss"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk .venv/bin/python -m pytest tests/training/test_gate_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.gate_smoke'`.

- [ ] **Step 3: Write minimal implementation**

`scripts/gate_smoke.py`:
```python
#!/usr/bin/env python3
"""Go/no-go gate: confirm the loss decreases on a synthetic batch before paid training."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
import chess
from config import Config
from src.model.transformer import ChessTransformer
from src.game.token_encoder import encode_batch
from src.game.move_encoder import get_move_encoder
from src.training.distill_trainer import DistillTrainer


def run_smoke_gate(steps: int = 30):
    P = get_move_encoder().policy_size
    sq, sf = encode_batch([chess.Board(), chess.Board()], [0, 0])
    e2e4 = get_move_encoder().encode(chess.Move.from_uci("e2e4"))
    pol = np.zeros((2, P), dtype=np.float32); pol[:, e2e4] = 1.0
    wdl = np.array([[1.0, 0.0, 0.0]] * 2, dtype=np.float32)
    ml = np.array([[40.0]] * 2, dtype=np.float32)
    inputs = (tf.constant(sq), tf.constant(sf))
    targets = (tf.constant(pol), tf.constant(wdl), tf.constant(ml))

    trainer = DistillTrainer(ChessTransformer(Config), Config)
    initial = float(trainer.train_step(inputs, targets)[0])
    final = initial
    for _ in range(steps):
        final = float(trainer.train_step(inputs, targets)[0])
    return final < initial, {"initial_loss": initial, "final_loss": final}


if __name__ == "__main__":
    ok, info = run_smoke_gate()
    print(info)
    sys.exit(0 if ok else 1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk .venv/bin/python -m pytest tests/training/test_gate_smoke.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add scripts/gate_smoke.py tests/training/test_gate_smoke.py && rtk git commit -m "feat: go/no-go smoke gate"
```

---

### Task 7: Document the rented-GPU runbook in this plan

**Files:**
- Modify: this plan file (append the runbook section below — already included; no code).

- [ ] **Step 1: Confirm the runbook section (below) is present and accurate; commit any edits.**

```bash
rtk git add docs/superpowers/plans/2026-06-17-chess-rl-phase1-plan3-distill.md && rtk git commit -m "docs: rented-GPU distillation runbook" --allow-empty
```

---

## Rented-GPU Runbook (not executed by tests)

Run only after **all four budget go/no-go gates pass locally on the M1**:

1. **Smoke gate:** `rtk .venv/bin/python scripts/gate_smoke.py` exits 0 (loss decreased).
2. **Data throughput:** Plan 2's `tests/data/test_throughput.py` passes (≥200 pos/sec pre-encode), and a `make_dataset(...)` loop sustains the target positions/sec.
3. **Smolgen A/B:** train two short runs (smolgen on vs a `Config.smolgen_hidden=0` ablation) on a small slice; keep smolgen only if its `val_top1`/`val_policy_loss` is ≥ the baseline.
4. **M1 inference:** Plan 4's latency benchmark meets the sims/sec target (or a fallback is chosen).

Then:

1. Pre-encode a bounded ChessBench slice on the M1: `rtk .venv/bin/python scripts/preencode.py --source chessbench --input <bag> --out-dir data/shards --shard-size 100000` (split into `train_*`/`val_*`). Target ~10–50M positions, ~a few–10 GB.
2. Rent a spot **RTX 4090** (~$0.30–0.40/hr; budget ~25–30h ≈ $9–11). Upload the packed shards (compact int8) to the box / cloud storage.
3. Run: `python scripts/distill.py --train 'data/shards/train_*.tfrecord' --val 'data/shards/val_*.tfrecord' --steps 60000 --batch 1024 --mixed-precision --ckpt checkpoints/distill`.
4. Checkpoint `best.weights.h5` to cloud storage every validation; on spot preemption, relaunch — `fit` resumes from the dataset (re-load `best.weights.h5` first if continuing).
5. Download `best.weights.h5`; evaluate with Plan 5's harness. Stop when the budget (~$10) or the strength target (~2000–2500 with MCTS) is hit.

---

## Self-Review (completed by plan author)

- **Spec coverage (§5 losses, §8 gates):** policy KL/soft-CE + WDL CE + moves-left Huber with weights `1.0`/`0.1` ✓; warmup→cosine AdamW + mixed precision ✓; validation metrics (KL, value-sign, top-1) ✓; best-checkpoint selection ✓; CLI ✓; smoke gate + runbook covering all four budget gates ✓.
- **Contract consistency:** loss formulation and weights match `_interface-contracts.md`; consumes Plan 1 `ChessTransformer.call(sq, sf, training=...)` → `(policy_logits, wdl_logits, moves_left)` and Plan 2 `make_dataset(paths, batch, policy_size, shuffle)` exactly.
- **Placeholders:** none — all code complete; the runbook is intentionally prose (operational steps), not code.
- **Naming consistency:** `policy_loss/value_loss/moves_left_loss/total_loss`, `DistillTrainer.train_step/evaluate/fit`, `run_smoke_gate` consistent across tasks/tests.

## Notes for the implementer
- `total_loss` returns `(scalar, parts_dict)`; `train_step` returns the same shape — keep that 2-tuple stable (Plan-3 CLI and gate depend on it).
- Mixed precision is off by default so unit tests are numerically stable; enable `--mixed-precision` only on the GPU run.
