# Phase 0: Eval/Ablation Harness + Locked Baseline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a config-driven eval/ablation harness and lock the current model as `baseline-v1`, so every future architecture change is measured apples-to-apples and objective-mismatch bugs become impossible.

**Architecture:** A preset registry layers override dicts over the existing `Config` to build named model variants; checkpoints carry a JSON sidecar (`preset`, `objective`) and a routing helper builds the correct evaluator from it; a gate suite computes a fixed metric battery on pinned datasets; an ablation runner emits a comparison table. No model behavior changes in Phase 0.

**Tech Stack:** Python 3.12, PyTorch 2.12 (MPS), python-chess, numpy, pytest.

**Environment & conventions (read first):**
- Use the off-iCloud venv: all Python via `/Users/james/.venvs/chess-rl/bin/python` and `/Users/james/.venvs/chess-rl/bin/pytest`.
- Prefix shell commands with `rtk` (e.g. `rtk git commit`), including inside `&&` chains.
- Work on branch `feat/transformer-rebuild`. Commit after every task.
- The current model has **exactly 10,255,879 parameters** — this is the `baseline-v1` lock value.
- `Config` (in `config.py`) is a class with class-level attributes; `ChessTransformer(cfg)` reads all architecture knobs from `cfg` (`d_model, n_heads, d_ff, n_layers, smolgen_compress, smolgen_hidden, smolgen_gen, state_dim`). It accepts either the class or an instance/namespace.
- Tests must pass `device="cpu"` where a device is needed, and stay fast.

---

## File Structure

- Create `src/model/presets.py` — preset registry + `build_model`.
- Create `src/training/checkpoint_meta.py` — sidecar read/write + git sha.
- Modify `src/training/distill_trainer.py` — optional `meta` arg on `fit`/`fit_av` to emit sidecars.
- Create `src/eval/routing.py` — `load_for_eval` (sidecar → net + correct evaluator).
- Create `src/eval/metrics_core.py` — pure metric functions (policy/value/calibration).
- Create `src/eval/fixtures.py` — pinned puzzle subsets, dense val shard (regenerated), mate-in-1 positions.
- Create `src/eval/gates.py` — run-level gate functions + `run_gates`.
- Create `scripts/ablate.py` — ablation runner (markdown + CSV).
- Create `scripts/freeze_baseline.py` — copy baseline ckpt + write sidecar.
- Tests under `tests/model/`, `tests/training/`, `tests/eval/`.
- Outputs committed only under `docs/ablations/`.

---

## Task 1: Preset registry + `build_model`

**Files:**
- Create: `src/model/presets.py`
- Test: `tests/model/test_presets.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model/test_presets.py
import pytest
from src.model.presets import PRESETS, resolve_config, build_model

BASELINE_PARAMS = 10_255_879


def test_baseline_v1_is_registered():
    assert "baseline-v1" in PRESETS


def test_resolve_config_applies_overrides():
    cfg = resolve_config("baseline-v1")
    # baseline-v1 keeps the current Config values
    assert cfg.d_model == 256
    assert cfg.n_layers == 8
    assert cfg.transformer_dropout == 0.0


def test_build_baseline_v1_param_count_is_locked():
    net, cfg = build_model("baseline-v1")
    n = sum(p.numel() for p in net.parameters())
    assert n == BASELINE_PARAMS, f"baseline-v1 drifted: {n} != {BASELINE_PARAMS}"


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        resolve_config("does-not-exist")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/model/test_presets.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.model.presets'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/model/presets.py
"""Named model presets layered over Config, plus a model builder.

A preset is a dict of architecture-knob overrides applied on top of the base
Config. `baseline-v1` is the empty-override preset: it reproduces the current
model exactly (10,255,879 params). Future architecture variants (Phase 2) are
added here as new presets; the model stays config-driven and the default forward
contract is unchanged.
"""

from types import SimpleNamespace

from config import Config
from src.model.transformer import ChessTransformer

# name -> dict of Config attribute overrides
PRESETS = {
    "baseline-v1": {},
}


def resolve_config(preset_name: str) -> SimpleNamespace:
    """Return a config namespace = Config defaults with the preset's overrides."""
    if preset_name not in PRESETS:
        raise KeyError(
            f"unknown preset {preset_name!r}; known: {sorted(PRESETS)}"
        )
    base = {k: v for k, v in vars(Config).items() if not k.startswith("__")}
    base.update(PRESETS[preset_name])
    return SimpleNamespace(**base)


def build_model(preset_name: str):
    """Build a ChessTransformer for the named preset.

    Returns (net, resolved_config).
    """
    cfg = resolve_config(preset_name)
    net = ChessTransformer(cfg)
    return net, cfg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/model/test_presets.py -q`
Expected: PASS (4 passed). If the param-count test fails, do NOT change the number — investigate why `ChessTransformer(resolve_config("baseline-v1"))` differs from `ChessTransformer(Config)`.

- [ ] **Step 5: Commit**

```bash
rtk git add src/model/presets.py tests/model/test_presets.py && \
rtk git commit -m "feat(phase0): preset registry + build_model with locked baseline-v1"
```

---

## Task 2: Checkpoint sidecar (read/write + git sha)

**Files:**
- Create: `src/training/checkpoint_meta.py`
- Test: `tests/training/test_checkpoint_meta.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/training/test_checkpoint_meta.py
from src.training.checkpoint_meta import sidecar_path, write_sidecar, read_sidecar


def test_sidecar_path_swaps_extension(tmp_path):
    p = str(tmp_path / "best.pt")
    assert sidecar_path(p) == str(tmp_path / "best.json")


def test_write_then_read_roundtrip(tmp_path):
    ckpt = str(tmp_path / "best.pt")
    write_sidecar(ckpt, {"preset": "baseline-v1", "objective": "policy",
                         "train_data": "test-bag-62k"})
    meta = read_sidecar(ckpt)
    assert meta["preset"] == "baseline-v1"
    assert meta["objective"] == "policy"
    assert meta["train_data"] == "test-bag-62k"
    assert "git_sha" in meta  # auto-added


def test_read_missing_returns_none(tmp_path):
    assert read_sidecar(str(tmp_path / "nope.pt")) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/training/test_checkpoint_meta.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.training.checkpoint_meta'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/training/checkpoint_meta.py
"""JSON sidecar for checkpoints: records how a model must be evaluated.

The sidecar lives next to the .pt (best.pt -> best.json) and carries at least
{preset, objective}. The eval routing helper reads it so a model is always
searched with the matching evaluator (policy vs action_value).
"""

import json
import os
import subprocess


def sidecar_path(ckpt_path: str) -> str:
    return os.path.splitext(ckpt_path)[0] + ".json"


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_sidecar(ckpt_path: str, meta: dict) -> str:
    """Write <ckpt>.json with meta (git_sha auto-added if absent). Returns path."""
    data = dict(meta)
    data.setdefault("git_sha", _git_sha())
    path = sidecar_path(ckpt_path)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    return path


def read_sidecar(ckpt_path: str):
    """Return the sidecar dict, or None if it does not exist."""
    path = sidecar_path(ckpt_path)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/training/test_checkpoint_meta.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
rtk git add src/training/checkpoint_meta.py tests/training/test_checkpoint_meta.py && \
rtk git commit -m "feat(phase0): checkpoint JSON sidecar (preset/objective metadata)"
```

---

## Task 3: Emit sidecars from the trainers

**Files:**
- Modify: `src/training/distill_trainer.py` (the `fit` method at lines ~152-171 and `fit_av` method)
- Test: `tests/training/test_trainer_sidecar.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/training/test_trainer_sidecar.py
import numpy as np
import torch
import chess

from config import Config
from src.model.transformer import ChessTransformer
from src.game.token_encoder import encode_batch
from src.game.move_encoder import get_move_encoder
from src.game.orientation import to_canonical_move
from src.training.distill_trainer import DistillTrainer
from src.training.checkpoint_meta import read_sidecar


def _av_batch():
    me = get_move_encoder()
    boards = [chess.Board(), chess.Board()]
    sq, sf = encode_batch(boards, [0, 0])
    a = me.encode(to_canonical_move(chess.Move.from_uci("e2e4"), chess.WHITE))
    inputs = (torch.tensor(sq, dtype=torch.long), torch.tensor(sf, dtype=torch.float32))
    targets = (torch.tensor([a, a], dtype=torch.long),
               torch.tensor([0.9, 0.9], dtype=torch.float32))
    return inputs, targets


def test_fit_av_writes_sidecar(tmp_path):
    net = ChessTransformer(Config)
    trainer = DistillTrainer(net, Config, device="cpu")
    loader = [_av_batch()]
    ckpt_dir = str(tmp_path / "ck")
    trainer.fit_av(loader, steps=1, val_loader=loader, val_every=1,
                   ckpt_dir=ckpt_dir, meta={"preset": "baseline-v1",
                                            "train_data": "unit"})
    meta = read_sidecar(str(tmp_path / "ck" / "last.pt"))
    assert meta is not None
    assert meta["objective"] == "action_value"
    assert meta["preset"] == "baseline-v1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/training/test_trainer_sidecar.py -q`
Expected: FAIL — `fit_av()` got an unexpected keyword argument `meta`.

- [ ] **Step 3: Write minimal implementation**

Add the import near the top of `src/training/distill_trainer.py` (after the existing imports on lines 6-11):

```python
from src.training.checkpoint_meta import write_sidecar
```

Add this helper method inside `DistillTrainer` (place it just above `def evaluate`):

```python
    def _save_ckpt(self, ckpt_dir, name, objective, meta):
        import os as _os
        path = _os.path.join(ckpt_dir, name)
        torch.save(self.net.state_dict(), path)
        if meta is not None:
            full = dict(meta)
            full["objective"] = objective
            write_sidecar(path, full)
```

Change `fit`'s signature and its two `torch.save(... best.pt/last.pt ...)` calls. Replace the body from the `for step` loop's checkpoint save through the final save:

```python
    def fit(self, train_loader, steps: int, val_loader=None, val_every: int = 1000,
            ckpt_dir: str = "checkpoints/distill", meta=None):
        os.makedirs(ckpt_dir, exist_ok=True)
        best = float("inf")
        it = iter(train_loader)
        for step in range(1, steps + 1):
            try:
                inputs, targets = next(it)
            except StopIteration:
                it = iter(train_loader)
                inputs, targets = next(it)
            loss, parts = self.train_step(inputs, targets)
            if val_loader is not None and step % val_every == 0:
                metrics = self.evaluate(val_loader)
                print(f"step {step} loss {loss:.4f} {metrics}")
                if metrics["val_policy_loss"] < best:
                    best = metrics["val_policy_loss"]
                    self._save_ckpt(ckpt_dir, "best.pt", "policy", meta)
        self._save_ckpt(ckpt_dir, "last.pt", "policy", meta)
        return best
```

Change `fit_av` likewise — add `meta=None` to its signature and replace its two saves (the `best.pt` save inside the val branch and the final `last.pt` save) with:

```python
                if metrics["val_av_bce"] < best:
                    best = metrics["val_av_bce"]
                    self._save_ckpt(ckpt_dir, "best.pt", "action_value", meta)
```

and at the end:

```python
        self._save_ckpt(ckpt_dir, "last.pt", "action_value", meta)
        return best
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/training/test_trainer_sidecar.py tests/training/test_actionvalue_trainer.py tests/training/test_trainer.py -q`
Expected: PASS (existing trainer tests still green; new sidecar test passes).

- [ ] **Step 5: Commit**

```bash
rtk git add src/training/distill_trainer.py tests/training/test_trainer_sidecar.py && \
rtk git commit -m "feat(phase0): trainers emit checkpoint sidecars (optional meta)"
```

---

## Task 4: Evaluator routing

**Files:**
- Create: `src/eval/routing.py`
- Test: `tests/eval/test_routing.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_routing.py
import pytest
import torch

from src.model.presets import build_model
from src.training.checkpoint_meta import write_sidecar
from src.eval.routing import load_for_eval


def _save_baseline(tmp_path, objective):
    net, _ = build_model("baseline-v1")
    ckpt = str(tmp_path / "m.pt")
    torch.save(net.state_dict(), ckpt)
    write_sidecar(ckpt, {"preset": "baseline-v1", "objective": objective})
    return ckpt


def test_routing_uses_sidecar_objective(tmp_path):
    ckpt = _save_baseline(tmp_path, "action_value")
    net, ev = load_for_eval(ckpt, device="cpu")
    assert ev.objective == "action_value"


def test_routing_missing_sidecar_requires_explicit(tmp_path):
    net, _ = build_model("baseline-v1")
    ckpt = str(tmp_path / "bare.pt")
    torch.save(net.state_dict(), ckpt)
    with pytest.raises(ValueError):
        load_for_eval(ckpt, device="cpu")  # no sidecar, no explicit args
    # explicit override works
    net2, ev2 = load_for_eval(ckpt, objective="policy", preset="baseline-v1",
                              device="cpu")
    assert ev2.objective == "policy"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/eval/test_routing.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.routing'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/eval/routing.py
"""Load a checkpoint for evaluation with the CORRECT evaluator.

Reads the checkpoint sidecar (preset, objective) and builds the matching
TransformerEvaluator. Prevents objective mismatches (e.g. searching a policy
model with action_value semantics). If no sidecar exists, the caller MUST pass
objective and preset explicitly.
"""

import torch

from src.model.presets import build_model
from src.model.evaluator import TransformerEvaluator
from src.training.checkpoint_meta import read_sidecar


def load_for_eval(ckpt_path, objective=None, preset=None, device=None,
                  use_fp16=False):
    """Return (net, evaluator) ready for gates.

    objective/preset override the sidecar; if there is no sidecar they are
    required.
    """
    meta = read_sidecar(ckpt_path) or {}
    preset = preset or meta.get("preset")
    objective = objective or meta.get("objective")
    if preset is None or objective is None:
        raise ValueError(
            f"no sidecar for {ckpt_path} and preset/objective not given; "
            "pass preset= and objective= explicitly"
        )
    net, _cfg = build_model(preset)
    net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    evaluator = TransformerEvaluator(net, use_fp16=use_fp16, device=device,
                                     objective=objective)
    return net, evaluator
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/eval/test_routing.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
rtk git add src/eval/routing.py tests/eval/test_routing.py && \
rtk git commit -m "feat(phase0): evaluator routing from checkpoint sidecar"
```

---

## Task 5: Core metric functions (policy/value/calibration)

**Files:**
- Create: `src/eval/metrics_core.py`
- Test: `tests/eval/test_metrics_core.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_metrics_core.py
import math
import torch

from src.eval.metrics_core import (
    policy_topk_match, policy_cross_entropy, legal_mass,
    wdl_cross_entropy, value_sign_acc, draw_calibration,
)


def test_policy_topk_match_top1_and_top3():
    # target's best move is index 2; model ranks index 2 third-highest
    logits = torch.tensor([[3.0, 2.0, 1.5, 0.0]])
    target = torch.zeros(1, 4); target[0, 2] = 1.0
    assert policy_topk_match(logits, target, k=1) == 0.0
    assert policy_topk_match(logits, target, k=3) == 1.0


def test_policy_cross_entropy_matches_manual():
    logits = torch.tensor([[0.0, 0.0]])         # softmax -> [0.5, 0.5]
    target = torch.tensor([[1.0, 0.0]])
    ce = policy_cross_entropy(logits, target)
    assert abs(ce - math.log(2)) < 1e-5


def test_legal_mass_sums_softmax_over_supported_moves():
    logits = torch.tensor([[10.0, 10.0, -10.0]])  # mass ~ on idx 0,1
    target = torch.tensor([[0.5, 0.5, 0.0]])      # legal support = {0,1}
    assert legal_mass(logits, target) > 0.99


def test_wdl_cross_entropy_and_sign_acc():
    wdl_logits = torch.tensor([[5.0, 0.0, -5.0]])  # predicts Win
    wdl_target = torch.tensor([[1.0, 0.0, 0.0]])
    assert wdl_cross_entropy(wdl_logits, wdl_target) < 0.05
    assert value_sign_acc(wdl_logits, wdl_target) == 1.0


def test_draw_calibration_reports_mean_draw_prob():
    wdl_logits = torch.tensor([[0.0, 10.0, 0.0], [0.0, 10.0, 0.0]])  # ~all draw
    wdl_target = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    assert draw_calibration(wdl_logits, wdl_target) > 0.99
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/eval/test_metrics_core.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.metrics_core'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/eval/metrics_core.py
"""Pure tensor metric functions for the gate suite. No I/O, no model building."""

import torch
import torch.nn.functional as F


def policy_topk_match(pol_logits, target_probs, k: int) -> float:
    """Fraction of rows where the target's argmax move is in the model's top-k."""
    tgt_best = target_probs.argmax(dim=1)                 # [B]
    topk = pol_logits.topk(k, dim=1).indices              # [B,k]
    hit = (topk == tgt_best.unsqueeze(1)).any(dim=1)      # [B]
    return float(hit.float().mean())


def policy_cross_entropy(pol_logits, target_probs) -> float:
    """Mean soft cross-entropy -sum(target * log_softmax(logits))."""
    logp = F.log_softmax(pol_logits, dim=1)
    return float(-(target_probs * logp).sum(dim=1).mean())


def legal_mass(pol_logits, target_probs) -> float:
    """Mean softmax probability the model puts on moves with target support>0."""
    p = F.softmax(pol_logits, dim=1)
    support = (target_probs > 0).float()
    return float((p * support).sum(dim=1).mean())


def wdl_cross_entropy(wdl_logits, wdl_target) -> float:
    logp = F.log_softmax(wdl_logits, dim=1)
    return float(-(wdl_target * logp).sum(dim=1).mean())


def value_sign_acc(wdl_logits, wdl_target) -> float:
    """Agreement of sign(P(W)-P(L)) between prediction and target."""
    p = F.softmax(wdl_logits, dim=1)
    pred = torch.sign(p[:, 0] - p[:, 2])
    tgt = torch.sign(wdl_target[:, 0] - wdl_target[:, 2])
    return float((pred == tgt).float().mean())


def draw_calibration(wdl_logits, wdl_target) -> float:
    """Mean predicted draw probability (compare against target draw rate)."""
    p = F.softmax(wdl_logits, dim=1)
    return float(p[:, 1].mean())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/eval/test_metrics_core.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
rtk git add src/eval/metrics_core.py tests/eval/test_metrics_core.py && \
rtk git commit -m "feat(phase0): core policy/value/calibration metric functions"
```

---

## Task 6: Pinned fixtures

**Files:**
- Create: `src/eval/fixtures.py`
- Test: `tests/eval/test_fixtures.py`

Note: the dense val shard is regenerated locally from `data/test/action_value_data.bag` (the 62k dense test bag) and is NOT committed. The test that needs it is marked `slow` so the fast suite never depends on the bag.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_fixtures.py
import os
import chess
import pytest

from src.eval.fixtures import mate_in_one_positions, load_gate_puzzles, PUZZLE_CSV


def test_mate_in_one_positions_are_legal_and_correct():
    positions = mate_in_one_positions()
    assert len(positions) >= 3
    for fen, uci in positions:
        board = chess.Board(fen)
        mv = chess.Move.from_uci(uci)
        assert mv in board.legal_moves
        board.push(mv)
        assert board.is_checkmate()


@pytest.mark.skipif(not os.path.exists(PUZZLE_CSV),
                    reason="puzzles.csv not present locally")
def test_load_gate_puzzles_returns_requested_count():
    pz = load_gate_puzzles(50)
    assert len(pz) == 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/eval/test_fixtures.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.fixtures'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/eval/fixtures.py
"""Pinned evaluation datasets. Binaries stay local (gitignored); only metrics
are committed. The dense val shard is regenerated deterministically on demand.
"""

import os

from src.eval.puzzles import load_chessbench_puzzles

PUZZLE_CSV = "data/puzzles.csv"
TEST_BAG = "data/test/action_value_data.bag"
DENSE_VAL_PATH = "data/fixtures/dense_val.npz"
DENSE_VAL_SIZE = 5000


def load_gate_puzzles(n: int):
    """First n ChessBench puzzles (deterministic: file order)."""
    return load_chessbench_puzzles(path=PUZZLE_CSV, max_puzzles=n)


def ensure_dense_val(n: int = DENSE_VAL_SIZE) -> str:
    """Regenerate the pinned dense val shard from the 62k test bag if absent.

    Deterministic: the first n grouped positions of the test bag. Returns the
    shard path.
    """
    if os.path.exists(DENSE_VAL_PATH):
        return DENSE_VAL_PATH
    import itertools
    from src.data.chessbench import iter_chessbench
    from src.data.preencode import write_shard
    if not os.path.exists(TEST_BAG):
        raise FileNotFoundError(
            f"{TEST_BAG} not found; download it with scripts/download_chessbench.py --test"
        )
    os.makedirs(os.path.dirname(DENSE_VAL_PATH), exist_ok=True)
    positions = list(itertools.islice(iter_chessbench(TEST_BAG), n))
    write_shard(positions, DENSE_VAL_PATH)
    return DENSE_VAL_PATH


def mate_in_one_positions():
    """(fen, best_uci) where best_uci delivers immediate checkmate."""
    return [
        # Back-rank mate: white rook to e8 is mate.
        ("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1", "e1e8"),
        # Queen mate: Qh7#.
        ("6k1/5p1p/6p1/8/8/8/5Q2/6K1 w - - 0 1", "f2f7"),
        # Black back-rank: rook to a1 is mate.
        ("4r1k1/8/8/8/8/8/5PPP/6K1 b - - 0 1", "e8e1"),
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/eval/test_fixtures.py -q`
Expected: PASS (1 passed, possibly 1 skipped if `data/puzzles.csv` is absent — it is present in this repo, so expect 2 passed).

If `test_mate_in_one_positions_are_legal_and_correct` fails on a specific FEN, fix that FEN until `board.push(mv); board.is_checkmate()` is True — do not weaken the assertion.

- [ ] **Step 5: Commit**

```bash
rtk git add src/eval/fixtures.py tests/eval/test_fixtures.py && \
rtk git commit -m "feat(phase0): pinned eval fixtures (puzzles, dense val, mate-in-1)"
```

---

## Task 7: Run-level gate functions + `run_gates`

**Files:**
- Create: `src/eval/gates.py`
- Test: `tests/eval/test_gates.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_gates.py
import torch

from src.model.presets import build_model
from src.model.evaluator import TransformerEvaluator
from src.eval.gates import raw_top1, mate_in_one_acc, throughput, run_gates
from src.eval.fixtures import load_gate_puzzles, mate_in_one_positions


def test_raw_top1_runs_on_small_puzzle_set():
    net, _ = build_model("baseline-v1")
    pz = load_gate_puzzles(20)
    acc = raw_top1(net, "cpu", pz)
    assert 0.0 <= acc <= 1.0


def test_mate_in_one_acc_in_range():
    net, _ = build_model("baseline-v1")
    acc = mate_in_one_acc(net, "cpu", mate_in_one_positions())
    assert 0.0 <= acc <= 1.0


def test_throughput_reports_positive_latency():
    net, _ = build_model("baseline-v1")
    out = throughput(net, "cpu", batch=8, iters=2)
    assert out["batch_latency_ms"] > 0.0


def test_run_gates_smoke_returns_expected_keys():
    net, _ = build_model("baseline-v1")
    ev = TransformerEvaluator(net, device="cpu", objective="policy")
    out = run_gates(net, ev, "cpu",
                    puzzle_counts=(20,), mcts_puzzles=0, dense_val=False,
                    throughput_batch=8, throughput_iters=2)
    assert "raw_top1@20" in out
    assert "mate_in_1" in out
    assert "batch_latency_ms" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/eval/test_gates.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.gates'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/eval/gates.py
"""Run-level gate suite: puzzles, MCTS, dense policy/value, throughput.

Objective-agnostic gates (raw top-1, throughput) take a net; objective-aware
gates (MCTS) take a routed evaluator. run_gates assembles a flat metrics dict.
"""

import time

import chess
import numpy as np
import torch

from config import Config
from src.game.token_encoder import encode_position
from src.game.orientation import to_canonical_move
from src.game.move_encoder import get_move_encoder
from src.eval import metrics_core as mc


def raw_top1(net, device, puzzles) -> float:
    """Argmax legal-move accuracy (no search). Objective-agnostic."""
    me = get_move_encoder()
    net.eval().to(device)
    if not puzzles:
        return 0.0
    correct = 0
    for pz in puzzles:
        board = chess.Board(pz.fen)
        legal = list(board.legal_moves)
        if not legal:
            continue
        sq, sf = encode_position(board, 0)
        sq_t = torch.tensor(sq[None], dtype=torch.long, device=device)
        sf_t = torch.tensor(sf[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            pol, _, _ = net.predict_batch(sq_t, sf_t)
        pol = pol[0].cpu().numpy()
        idxs = [me.encode(to_canonical_move(mv, board.turn)) for mv in legal]
        best = legal[int(np.argmax(pol[idxs]))]
        if best == pz.solution_moves[0]:
            correct += 1
    return correct / len(puzzles)


def mcts_top1(evaluator, puzzles, simulations: int) -> float:
    """MCTS-N argmax accuracy using the routed evaluator."""
    from src.mcts.batched_mcts import BatchedMCTS
    if not puzzles or simulations <= 0:
        return 0.0
    mcts = BatchedMCTS(evaluator, Config, num_simulations=simulations)
    correct = 0
    for pz in puzzles:
        mcts.reset()
        mv = mcts.choose_move(chess.Board(pz.fen), temperature=0.0)
        if mv == pz.solution_moves[0]:
            correct += 1
    return correct / len(puzzles)


def mate_in_one_acc(net, device, positions) -> float:
    """Fraction of mate-in-1 positions where argmax legal move is the mate."""
    me = get_move_encoder()
    net.eval().to(device)
    if not positions:
        return 0.0
    correct = 0
    for fen, uci in positions:
        board = chess.Board(fen)
        legal = list(board.legal_moves)
        sq, sf = encode_position(board, 0)
        sq_t = torch.tensor(sq[None], dtype=torch.long, device=device)
        sf_t = torch.tensor(sf[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            pol, _, _ = net.predict_batch(sq_t, sf_t)
        pol = pol[0].cpu().numpy()
        idxs = [me.encode(to_canonical_move(mv, board.turn)) for mv in legal]
        best = legal[int(np.argmax(pol[idxs]))]
        if best == chess.Move.from_uci(uci):
            correct += 1
    return correct / len(positions)


def policy_value_metrics(net, device, val_loader, max_batches: int = 50) -> dict:
    """top1/3/5, policy CE, legal mass, WDL CE, value sign acc, draw cal."""
    net.eval().to(device)
    acc = {k: 0.0 for k in ("top1", "top3", "top5", "policy_ce", "legal_mass",
                            "wdl_ce", "value_sign_acc", "draw_cal")}
    n = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            if i >= max_batches:
                break
            (sq, sf), (pol_t, wdl_t, _ml) = inputs, targets
            sq = sq.to(device); sf = sf.to(device)
            pol_t = pol_t.to(device); wdl_t = wdl_t.to(device)
            pol, wdl, _ = net(sq, sf)
            acc["top1"] += mc.policy_topk_match(pol, pol_t, 1)
            acc["top3"] += mc.policy_topk_match(pol, pol_t, 3)
            acc["top5"] += mc.policy_topk_match(pol, pol_t, 5)
            acc["policy_ce"] += mc.policy_cross_entropy(pol, pol_t)
            acc["legal_mass"] += mc.legal_mass(pol, pol_t)
            acc["wdl_ce"] += mc.wdl_cross_entropy(wdl, wdl_t)
            acc["value_sign_acc"] += mc.value_sign_acc(wdl, wdl_t)
            acc["draw_cal"] += mc.draw_calibration(wdl, wdl_t)
            n += 1
    n = max(n, 1)
    return {k: v / n for k, v in acc.items()}


def throughput(net, device, batch: int = 256, iters: int = 5) -> dict:
    """Forward latency (ms/batch) and positions/sec at the given batch size."""
    net.eval().to(device)
    sq = torch.zeros(batch, 64, dtype=torch.long, device=device)
    sf = torch.zeros(batch, 18, dtype=torch.float32, device=device)
    with torch.no_grad():
        net.predict_batch(sq, sf)  # warmup
        t0 = time.perf_counter()
        for _ in range(iters):
            net.predict_batch(sq, sf)
        dt = (time.perf_counter() - t0) / iters
    return {"batch_latency_ms": dt * 1000.0, "positions_per_sec": batch / dt}


def run_gates(net, evaluator, device, puzzle_counts=(300, 1000),
              mcts_puzzles=300, mcts_sims=100, dense_val=True,
              throughput_batch=256, throughput_iters=5) -> dict:
    """Assemble the full gate dict. Set dense_val=False / mcts_puzzles=0 to skip
    the heavy parts in fast smoke runs."""
    from src.eval.fixtures import (load_gate_puzzles, mate_in_one_positions,
                                   ensure_dense_val)
    out = {}
    maxc = max(puzzle_counts) if puzzle_counts else 0
    all_pz = load_gate_puzzles(maxc) if maxc else []
    for c in puzzle_counts:
        out[f"raw_top1@{c}"] = raw_top1(net, device, all_pz[:c])
    if mcts_puzzles:
        out[f"mcts{mcts_sims}_top1@{mcts_puzzles}"] = mcts_top1(
            evaluator, all_pz[:mcts_puzzles], mcts_sims)
    out["mate_in_1"] = mate_in_one_acc(net, device, mate_in_one_positions())
    if dense_val:
        from src.data.dataset import make_dataloader
        path = ensure_dense_val()
        loader = make_dataloader([path], batch_size=256,
                                 policy_size=get_move_encoder().policy_size,
                                 shuffle=False)
        out.update(policy_value_metrics(net, device, loader))
    out.update(throughput(net, device, batch=throughput_batch,
                          iters=throughput_iters))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/eval/test_gates.py -q`
Expected: PASS (4 passed). These run on CPU with an untrained model — values are unconstrained, tests only check ranges/keys.

- [ ] **Step 5: Commit**

```bash
rtk git add src/eval/gates.py tests/eval/test_gates.py && \
rtk git commit -m "feat(phase0): gate suite (puzzles, mcts, dense policy/value, throughput)"
```

---

## Task 8: Ablation runner

**Files:**
- Create: `scripts/ablate.py`
- Test: `tests/eval/test_ablate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_ablate.py
import torch

from src.model.presets import build_model
from src.training.checkpoint_meta import write_sidecar
from scripts.ablate import run_row, to_markdown, to_csv


def _baseline_ckpt(tmp_path):
    net, _ = build_model("baseline-v1")
    ckpt = str(tmp_path / "m.pt")
    torch.save(net.state_dict(), ckpt)
    write_sidecar(ckpt, {"preset": "baseline-v1", "objective": "policy"})
    return ckpt


def test_run_row_has_params_and_metric(tmp_path):
    ckpt = _baseline_ckpt(tmp_path)
    row = run_row(ckpt, device="cpu",
                  gate_kwargs=dict(puzzle_counts=(20,), mcts_puzzles=0,
                                   dense_val=False, throughput_batch=8,
                                   throughput_iters=2))
    assert row["preset"] == "baseline-v1"
    assert row["params"] == 10_255_879
    assert "raw_top1@20" in row


def test_table_formatters_roundtrip():
    rows = [{"preset": "baseline-v1", "params": 10255879, "raw_top1@20": 0.25}]
    md = to_markdown(rows)
    csv = to_csv(rows)
    assert "baseline-v1" in md and "params" in md
    assert "preset,params,raw_top1@20" in csv
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/eval/test_ablate.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.ablate'` (ensure `scripts/__init__.py` exists; if not, create an empty one and add it in this task's commit).

- [ ] **Step 3: Write minimal implementation**

First ensure `scripts` is importable:

```bash
test -f scripts/__init__.py || touch scripts/__init__.py
```

```python
# scripts/ablate.py
"""Run the gate suite over one or more checkpoints and emit a comparison table.

Usage:
  python scripts/ablate.py --ckpt checkpoints/baseline-v1/best.pt \
      --out docs/ablations/run.md
Each checkpoint must have a sidecar (or pass --objective/--preset for bare ones).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.model.presets import resolve_config, PRESETS
from src.eval.routing import load_for_eval
from src.eval.gates import run_gates
from src.training.checkpoint_meta import read_sidecar


def _param_count(preset):
    from src.model.presets import build_model
    net, _ = build_model(preset)
    return sum(p.numel() for p in net.parameters())


def run_row(ckpt_path, device="cpu", objective=None, preset=None,
            gate_kwargs=None) -> dict:
    gate_kwargs = gate_kwargs or {}
    meta = read_sidecar(ckpt_path) or {}
    preset = preset or meta.get("preset")
    net, ev = load_for_eval(ckpt_path, objective=objective, preset=preset,
                            device=device)
    row = {"preset": preset, "params": sum(p.numel() for p in net.parameters())}
    row.update(run_gates(net, ev, device, **gate_kwargs))
    return row


def _columns(rows):
    cols = ["preset", "params"]
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    return cols


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def to_markdown(rows) -> str:
    cols = _columns(rows)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep] + body) + "\n"


def to_csv(rows) -> str:
    cols = _columns(rows)
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(_fmt(r.get(c, "")) for c in cols))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="output .md path (.csv written alongside)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--objective", default=None)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--fast", action="store_true",
                    help="skip MCTS+dense val for a quick run")
    args = ap.parse_args()

    gate_kwargs = (dict(puzzle_counts=(300, 1000), mcts_puzzles=0, dense_val=False)
                   if args.fast else {})
    rows = [run_row(c, device=args.device, objective=args.objective,
                    preset=args.preset, gate_kwargs=gate_kwargs)
            for c in args.ckpt]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(to_markdown(rows))
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    with open(csv_path, "w") as fh:
        fh.write(to_csv(rows))
    print(f"wrote {args.out} and {csv_path}")
    print(to_markdown(rows))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/eval/test_ablate.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
rtk git add scripts/ablate.py scripts/__init__.py tests/eval/test_ablate.py && \
rtk git commit -m "feat(phase0): ablation runner (markdown + csv table)"
```

---

## Task 9: Freeze baseline-v1 and record its metrics

**Files:**
- Create: `scripts/freeze_baseline.py`
- Create (committed): `docs/ablations/baseline-v1.md` and `docs/ablations/baseline-v1.csv`
- Local only (gitignored): `checkpoints/baseline-v1/best.pt` + `best.json`

This task produces the reference row. It runs the real (slow) gate suite on the existing trained model and is not a pytest test.

- [ ] **Step 1: Write the freeze script**

```python
# scripts/freeze_baseline.py
"""Copy the current trained policy model to checkpoints/baseline-v1/ and write
its sidecar (preset=baseline-v1, objective=policy). The .pt stays local."""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.checkpoint_meta import write_sidecar

SRC = "checkpoints/distill/best.pt"
DST_DIR = "checkpoints/baseline-v1"
DST = os.path.join(DST_DIR, "best.pt")


def main():
    if not os.path.exists(SRC):
        raise SystemExit(f"{SRC} not found; cannot freeze baseline")
    os.makedirs(DST_DIR, exist_ok=True)
    shutil.copyfile(SRC, DST)
    write_sidecar(DST, {"preset": "baseline-v1", "objective": "policy",
                        "train_data": "chessbench-test-bag-62k"})
    print(f"froze baseline-v1 -> {DST} (+ sidecar)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the freeze script**

Run: `/Users/james/.venvs/chess-rl/bin/python scripts/freeze_baseline.py`
Expected: prints `froze baseline-v1 -> checkpoints/baseline-v1/best.pt (+ sidecar)`. Confirm `checkpoints/baseline-v1/best.json` contains `"objective": "policy"`.

- [ ] **Step 3: Run the full gate suite into the committed metrics table**

Run:
```bash
/Users/james/.venvs/chess-rl/bin/python scripts/ablate.py \
  --ckpt checkpoints/baseline-v1/best.pt \
  --out docs/ablations/baseline-v1.md
```
Expected: writes `docs/ablations/baseline-v1.md` + `.csv` and prints the table. Sanity-check the numbers against the known baseline: `raw_top1@1000 ≈ 0.289`, `raw_top1@300 ≈ 0.277`, `mcts100_top1@300 ≈ 0.34`. If `raw_top1@1000` is far from ~0.289, STOP — the harness or routing is wrong; do not "fix" by adjusting expectations.

- [ ] **Step 4: Verify .gitignore keeps binaries local**

Run: `rtk git status --porcelain`
Expected: `docs/ablations/baseline-v1.md` and `.csv` appear as new; `checkpoints/baseline-v1/best.pt` does NOT appear (covered by `*.pt` ignore). If the `.pt` or `dense_val.npz` appears, add the appropriate ignore rule before committing.

- [ ] **Step 5: Commit (metrics only)**

```bash
rtk git add scripts/freeze_baseline.py docs/ablations/baseline-v1.md docs/ablations/baseline-v1.csv && \
rtk git commit -m "feat(phase0): freeze baseline-v1 + record reference gate metrics"
```

---

## Task 10: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the complete fast test suite**

Run: `/Users/james/.venvs/chess-rl/bin/pytest -q`
Expected: all tests pass (existing ~80 + the new Phase 0 tests), no errors. `slow`-marked tests are deselected by `pytest.ini`.

- [ ] **Step 2: Run a fast end-to-end ablation smoke**

Run:
```bash
/Users/james/.venvs/chess-rl/bin/python scripts/ablate.py \
  --ckpt checkpoints/baseline-v1/best.pt --fast --out /tmp/ablate_fast.md
```
Expected: prints a table with `preset`, `params=10255879`, `raw_top1@300`, `raw_top1@1000`; completes in well under a minute.

- [ ] **Step 3: Push the branch**

```bash
rtk git push
```
Expected: branch `feat/transformer-rebuild` updated on origin.

---

## Self-Review Checklist (completed by plan author)

- **Spec coverage:** preset registry (T1), sidecar+routing (T2–T4), gate suite incl. policy/value/throughput/mate sanity (T5–T7), pinned fixtures regenerated-not-committed (T6), ablation runner md+csv (T8), frozen baseline + metrics-only commit (T9), suite green (T10). All spec sections A–E covered.
- **Objective-mismatch guard:** routing (T4) raises without sidecar/explicit args; `run_gates` MCTS uses the routed evaluator — covered by tests.
- **Baseline lock:** exact param count 10,255,879 asserted (T1) and re-checked in the ablation row (T8).
- **Persistence policy:** only `docs/ablations/*` committed; `.pt`/`.npz` rely on existing `*.pt`/`*.npz` gitignore (verified in T9 step 4).
- **Type consistency:** `build_model`→`(net,cfg)`, `load_for_eval`→`(net,evaluator)`, evaluator exposes `.objective`, metric fns take `(logits,target)` and return float, `run_gates` kwargs match `scripts/ablate.py` usage.
