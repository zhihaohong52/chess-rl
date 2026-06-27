# Chess-RL Phase 3 GPU Scale-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the local, TDD-tested code needed to train a 77M transformer on a rented A6000 and measure it head-to-head against a 10M control, then document the exact ThunderCompute run.

**Architecture:** Five small, config-gated code units (new preset, CUDA device path, bf16 AMP, model-vs-model arena mode, HF shard downloader), each built and tested locally on CPU/MPS. The actual GPU training/eval is a documented runbook executed on the box — not in CI. `baseline-v1` must stay bit-identical throughout.

**Tech Stack:** PyTorch (MPS local / CUDA on box), python-chess, NumPy, ThunderCompute `tnr` CLI, Stockfish 18.

**Conventions (every task):**
- Interpreter: `/Users/james/.venvs/chess-rl/bin/python`; tests: `/Users/james/.venvs/chess-rl/bin/python -m pytest`.
- Prefix shell commands with `rtk` (e.g. `rtk git commit`).
- Branch `feat/transformer-rebuild`. Commit after each task.
- The encoding pipeline is already optimized and committed (`73b6b70`) — do not re-touch it.
- After every model-touching task, the baseline param lock (`tests/model/test_presets.py::test_build_baseline_v1_param_count_is_locked`) must stay green.

---

### Task 1: `p3-80m` preset (77M)

**Files:**
- Modify: `src/model/presets.py` (the `PRESETS` dict, after `p2-value-swiglu-drop`)
- Test: `tests/model/test_presets.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/model/test_presets.py`:

```python
def test_p3_80m_is_registered_and_locked():
    assert "p3-80m" in PRESETS
    net, cfg = build_model("p3-80m")
    n = sum(p.numel() for p in net.parameters())
    assert n == 76_982_212, f"p3-80m drifted: {n} != 76982212"
    assert cfg.d_model == 512
    assert cfg.n_layers == 16
    assert cfg.n_heads == 16
    assert cfg.d_ff == 3072
    assert cfg.ffn_type == "swiglu"
    assert cfg.value_head_type == "hlgauss"
    assert cfg.value_buckets == 64
    assert cfg.transformer_dropout == 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/james/.venvs/chess-rl/bin/python -m pytest tests/model/test_presets.py::test_p3_80m_is_registered_and_locked -v`
Expected: FAIL — `ValueError: unknown preset 'p3-80m'`.

- [ ] **Step 3: Add the preset**

In `src/model/presets.py`, add this entry to `PRESETS` immediately after the `"p2-value-swiglu-drop"` entry:

```python
    "p3-80m": {"d_model": 512, "n_layers": 16, "n_heads": 16, "d_ff": 3072,
               "ffn_type": "swiglu", "value_head_type": "hlgauss",
               "value_buckets": 64, "transformer_dropout": 0.05},
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/james/.venvs/chess-rl/bin/python -m pytest tests/model/test_presets.py -v`
Expected: PASS — including `test_build_baseline_v1_param_count_is_locked` (baseline still 10,255,879) and the new `p3-80m` test (76,982,212).

- [ ] **Step 5: Commit**

```bash
rtk git add src/model/presets.py tests/model/test_presets.py
rtk git commit -m "feat(phase3): p3-80m preset (77M, SwiGLU+HL-Gauss+dropout), param-locked"
```

---

### Task 2: CUDA device selection + `--device` flag

**Files:**
- Modify: `src/training/distill_trainer.py` (add `_default_device()`, use it in `__init__`)
- Modify: `scripts/distill.py` (add `--device`, pass through)
- Test: `tests/training/test_device_selection.py` (create)

Context: `DistillTrainer.__init__(self, net, cfg, mixed_precision=False, device=None)` currently sets `device = "mps" if torch.backends.mps.is_available() else "cpu"` when `device is None`. On a CUDA box that silently falls back to CPU. We add a `_default_device()` helper that prefers CUDA, and a `--device` passthrough. MPS/CPU behavior is unchanged.

- [ ] **Step 1: Write the failing test**

Create `tests/training/test_device_selection.py`:

```python
from src.training.distill_trainer import _default_device


def test_default_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert _default_device() == "cuda"


def test_default_device_mps_when_no_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    assert _default_device() == "mps"


def test_default_device_cpu_fallback(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    assert _default_device() == "cpu"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/james/.venvs/chess-rl/bin/python -m pytest tests/training/test_device_selection.py -v`
Expected: FAIL — `ImportError: cannot import name '_default_device'`.

- [ ] **Step 3: Implement the helper and use it**

In `src/training/distill_trainer.py`, add this module-level function above the `DistillTrainer` class:

```python
def _default_device():
    """Best available device: CUDA (box) > MPS (local M1) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

Then in `DistillTrainer.__init__`, replace the existing device-defaulting line:

```python
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
```

with:

```python
        if device is None:
            device = _default_device()
```

- [ ] **Step 4: Add `--device` to `scripts/distill.py`**

In `scripts/distill.py`, add this argument next to the other `ap.add_argument` calls:

```python
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (default: auto)")
```

and pass it to the trainer by changing:

```python
    trainer = DistillTrainer(net, cfg, mixed_precision=args.mixed_precision)
```

to:

```python
    trainer = DistillTrainer(net, cfg, mixed_precision=args.mixed_precision,
                             device=args.device)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `/Users/james/.venvs/chess-rl/bin/python -m pytest tests/training/test_device_selection.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
rtk git add src/training/distill_trainer.py scripts/distill.py tests/training/test_device_selection.py
rtk git commit -m "feat(phase3): CUDA-aware default device + --device flag (mps/cpu unchanged)"
```

---

### Task 3: Real bf16 AMP (CUDA-only autocast)

**Files:**
- Modify: `src/training/distill_trainer.py` (`train_step` forward+loss under autocast; add `import contextlib`)
- Test: `tests/training/test_amp_noop.py` (create)

Context: `--mixed-precision` is currently a no-op. We wrap the forward+loss in `train_step` with `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` ONLY when `mixed_precision and device == "cuda"`. bf16 needs no `GradScaler`. On MPS/CPU it stays a no-op, so local numerics and baseline reproducibility are untouched. The current `train_step` body is:

```python
    def train_step(self, inputs, targets):
        (sq, sf), (pol_t, wdl_t, ml_t) = self._to_device(inputs, targets)
        self.net.train()
        self.opt.zero_grad()
        pol, wdl, ml = self.net(sq, sf)
        loss, parts = total_loss(
            pol, wdl, ml, pol_t, wdl_t, ml_t,
            value_weight=getattr(self.cfg, "value_loss_weight", 1.0),
            value_head_type=getattr(self.net, "value_head_type", "wdl"),
            value_buckets=getattr(self.net, "value_buckets", 64),
            value_sigma_frac=getattr(self.cfg, "value_sigma_frac", 0.75),
        )
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        self._update_ema()
        self.scheduler.step()
        return float(loss.detach()), {k: float(v.detach()) for k, v in parts.items()}
```

- [ ] **Step 1: Write the failing test**

Create `tests/training/test_amp_noop.py`:

```python
import copy
import torch

from src.model.presets import build_model
from src.training.distill_trainer import DistillTrainer


def _one_batch(P):
    sq = torch.zeros((4, 64), dtype=torch.long)
    sf = torch.zeros((4, 18), dtype=torch.float32)
    pol_t = torch.zeros((4, P), dtype=torch.float32)
    pol_t[:, 0] = 1.0
    wdl_t = torch.tensor([[1.0, 0.0, 0.0]] * 4, dtype=torch.float32)
    ml_t = torch.zeros((4,), dtype=torch.float32)
    return (sq, sf), (pol_t, wdl_t, ml_t)


def test_amp_is_noop_on_cpu():
    """On CPU, mixed_precision=True must not change numerics vs False."""
    net, cfg = build_model("baseline-v1")
    P = net.policy_head_size if hasattr(net, "policy_head_size") else cfg.policy_size
    inputs, targets = _one_batch(cfg.policy_size)
    init_state = copy.deepcopy(net.state_dict())

    t_off = DistillTrainer(net, cfg, mixed_precision=False, device="cpu")
    loss_off, _ = t_off.train_step(inputs, targets)

    net.load_state_dict(init_state)  # reset weights + optimizer-independent compare
    t_on = DistillTrainer(net, cfg, mixed_precision=True, device="cpu")
    loss_on, _ = t_on.train_step(inputs, targets)

    assert abs(loss_off - loss_on) < 1e-6, (loss_off, loss_on)


def test_amp_cuda_step_runs():
    """bf16 autocast step runs and produces a finite loss (box-only)."""
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("no CUDA")
    net, cfg = build_model("p3-80m")
    inputs, targets = _one_batch(cfg.policy_size)
    t = DistillTrainer(net, cfg, mixed_precision=True, device="cuda")
    loss, _ = t.train_step(inputs, targets)
    assert loss == loss and loss < float("inf")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/james/.venvs/chess-rl/bin/python -m pytest tests/training/test_amp_noop.py::test_amp_is_noop_on_cpu -v`
Expected: PASS already (AMP is currently a global no-op), OR error if `policy_size` access differs. If it already passes, that is fine — it guards the no-op invariant we must preserve. Proceed to implement the CUDA path without breaking it.

(Note: this test's value is as a regression guard. The behavioural change is CUDA-only and is covered by `test_amp_cuda_step_runs`, which runs on the box.)

- [ ] **Step 3: Implement CUDA-only autocast**

In `src/training/distill_trainer.py`, add at the top with the other imports:

```python
import contextlib
```

Then change the forward+loss region of `train_step` from the plain forward to an autocast-guarded one:

```python
        self.opt.zero_grad()
        use_amp = self.mixed_precision and self.device == "cuda"
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_amp else contextlib.nullcontext()
        )
        with amp_ctx:
            pol, wdl, ml = self.net(sq, sf)
            loss, parts = total_loss(
                pol, wdl, ml, pol_t, wdl_t, ml_t,
                value_weight=getattr(self.cfg, "value_loss_weight", 1.0),
                value_head_type=getattr(self.net, "value_head_type", "wdl"),
                value_buckets=getattr(self.net, "value_buckets", 64),
                value_sigma_frac=getattr(self.cfg, "value_sigma_frac", 0.75),
            )
        loss.backward()
```

(bf16 needs no GradScaler; `backward()`/`step()` stay outside the autocast context, which is correct.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/james/.venvs/chess-rl/bin/python -m pytest tests/training/test_amp_noop.py -v`
Expected: `test_amp_is_noop_on_cpu` PASS; `test_amp_cuda_step_runs` SKIPPED (no CUDA locally).

- [ ] **Step 5: Run the trainer suite to confirm no regression**

Run: `/Users/james/.venvs/chess-rl/bin/python -m pytest tests/training -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
rtk git add src/training/distill_trainer.py tests/training/test_amp_noop.py
rtk git commit -m "feat(phase3): real bf16 AMP in train_step (CUDA-only; no-op on mps/cpu)"
```

---

### Task 4: Model-vs-model head-to-head arena

**Files:**
- Modify: `scripts/arena_eval.py` (factor mover builders; add `--vs` mode)
- Test: `tests/eval/test_arena_headtohead.py` (create)

Context: `src/eval/arena.py::play_match(engine, opponent, num_games, max_moves)` already takes two move-producer callables, so no change to `arena.py` is needed. We refactor `arena_eval.py`'s inline movers into reusable builders and add a `--vs CKPT` mode that loads a second checkpoint via `load_for_eval` and plays the two models against each other with MCTS at equal simulations.

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_arena_headtohead.py`:

```python
import chess

from src.eval.arena import play_match
from src.model.presets import build_model
from src.game.move_encoder import get_move_encoder
import scripts.arena_eval as ae


def _first_legal(board):
    return next(iter(board.legal_moves), None)


def test_self_play_scores_half_over_even_games():
    """A deterministic mover vs itself nets exactly 0.5 over an even game count
    (color alternation makes the fixed game's result cancel). This guards the
    head-to-head scoring the --vs mode relies on."""
    res = play_match(_first_legal, _first_legal, num_games=4, max_moves=40)
    assert res.total == 4
    assert res.score == 0.5


def test_build_raw_mover_returns_legal_move():
    net, cfg = build_model("baseline-v1")
    me = get_move_encoder()
    mover = ae.build_raw_mover(net, "cpu", me)
    move = mover(chess.Board())
    assert move in chess.Board().legal_moves
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/james/.venvs/chess-rl/bin/python -m pytest tests/eval/test_arena_headtohead.py -v`
Expected: FAIL — `AttributeError: module 'scripts.arena_eval' has no attribute 'build_raw_mover'`.

- [ ] **Step 3: Refactor movers into builders and add `--vs`**

Replace the body of `scripts/arena_eval.py` with this (keeps the existing Stockfish ladder; adds builders + `--vs`):

```python
#!/usr/bin/env python3
"""Arena eval: RAW vs MCTS vs Stockfish, and model-vs-model head-to-head.

Stockfish ladder (default):
  python scripts/arena_eval.py --model checkpoints/p3_80m/best_ema.pt \
      --skills 1 3 5 --games 40 --simulations 100 --depth 4

Head-to-head (two checkpoints, MCTS at equal sims):
  python scripts/arena_eval.py --model checkpoints/p3_80m/best_ema.pt \
      --vs checkpoints/p3_10m_ctrl/best_ema.pt --games 100 --simulations 100
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.eval.routing import load_for_eval
from src.eval.gates import _best_legal_move
from src.eval.arena import play_match
from src.eval.elo import elo_diff
from src.eval.stockfish_opponent import StockfishOpponent, stockfish_available
from src.mcts.batched_mcts import BatchedMCTS
from src.game.move_encoder import get_move_encoder


def build_raw_mover(net, device, me):
    """Move-producer: greedy best legal move from the raw policy."""
    return lambda board: _best_legal_move(net, device, board, me)


def build_mcts_mover(evaluator, simulations):
    """Move-producer: MCTS at fixed simulations, greedy (temperature 0)."""
    mcts = BatchedMCTS(evaluator, Config, num_simulations=simulations)

    def mover(board):
        mcts.reset()
        return mcts.choose_move(board, temperature=0.0)

    return mover


def run_head_to_head(model, vs, games, simulations, device, max_moves):
    net_a, ev_a = load_for_eval(model, device=device)
    net_b, ev_b = load_for_eval(vs, device=device)
    mover_a = build_mcts_mover(ev_a, simulations)
    mover_b = build_mcts_mover(ev_b, simulations)
    res = play_match(engine=mover_a, opponent=mover_b,
                     num_games=games, max_moves=max_moves)
    gap = elo_diff(res.score, games=res.total)
    print(f"head-to-head (MCTS {simulations} sims): {os.path.basename(model)} "
          f"vs {os.path.basename(vs)}", flush=True)
    print(f"  W/D/L {res.wins}/{res.draws}/{res.losses}  score {res.score:.3f}  "
          f"estEloGap {gap:+.0f}", flush=True)
    return res


def run_stockfish_ladder(model, skills, games, simulations, depth, device, max_moves):
    net, ev = load_for_eval(model, device=device)
    me = get_move_encoder()
    raw_mover = build_raw_mover(net, device, me)
    mcts_mover = build_mcts_mover(ev, simulations)
    print(f"model={model}  games/skill={games}  sims={simulations}  sf_depth={depth}",
          flush=True)
    print(f"{'mover':>6} {'skill':>5} {'W/D/L':>10} {'score':>6} {'estElo':>7}",
          flush=True)
    for skill in skills:
        for name, mover in (("raw", raw_mover), ("mcts", mcts_mover)):
            with StockfishOpponent(skill_level=skill, depth=depth) as opp:
                res = play_match(engine=mover, opponent=opp,
                                 num_games=games, max_moves=max_moves)
                est = opp.approximate_elo + elo_diff(res.score, games=res.total)
                print(f"{name:>6} {skill:5d} {res.wins:>3}/{res.draws}/{res.losses:<3} "
                      f"{res.score:6.3f} {est:7.0f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/p3_80m/best_ema.pt")
    ap.add_argument("--vs", default=None,
                    help="second checkpoint for model-vs-model head-to-head")
    ap.add_argument("--skills", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--simulations", type=int, default=100)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--max-moves", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.vs:
        run_head_to_head(args.model, args.vs, args.games, args.simulations,
                         args.device, args.max_moves)
        return 0

    if not stockfish_available():
        print("Stockfish not found; cannot run ladder.", file=sys.stderr)
        return 1
    run_stockfish_ladder(args.model, args.skills, args.games, args.simulations,
                         args.depth, args.device, args.max_moves)
    return 0


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/james/.venvs/chess-rl/bin/python -m pytest tests/eval/test_arena_headtohead.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
rtk git add scripts/arena_eval.py tests/eval/test_arena_headtohead.py
rtk git commit -m "feat(phase3): model-vs-model head-to-head (--vs) + reusable mover builders"
```

---

### Task 5: HF raw-shard downloader

**Files:**
- Create: `scripts/download_hf_dense.py`
- Test: `tests/data/test_download_hf_dense.py` (create)

Context: To get ~100M positions on the box we download ~29 raw shards from the
`prdev/chessbench-full-policy-value` HF dataset. The repo's files are named
`train-NNNNN-of-01024.msgpack.zst` and served from the HF `resolve/main` URL.
We make the URL/filename pure functions (testable without network) and a thin
`curl` driver for the box.

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_download_hf_dense.py`:

```python
from scripts.download_hf_dense import shard_filename, shard_url


def test_shard_filename_zero_padded_of_1024():
    assert shard_filename(0) == "train-00000-of-01024.msgpack.zst"
    assert shard_filename(7) == "train-00007-of-01024.msgpack.zst"
    assert shard_filename(1023) == "train-01023-of-01024.msgpack.zst"


def test_shard_url_points_at_hf_resolve_main():
    url = shard_url(0)
    assert url == (
        "https://huggingface.co/datasets/prdev/chessbench-full-policy-value/"
        "resolve/main/train-00000-of-01024.msgpack.zst"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/james/.venvs/chess-rl/bin/python -m pytest tests/data/test_download_hf_dense.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.download_hf_dense'`.

- [ ] **Step 3: Implement the downloader**

Create `scripts/download_hf_dense.py`:

```python
#!/usr/bin/env python3
"""Download raw dense shards from prdev/chessbench-full-policy-value (HF).

Files are `train-NNNNN-of-01024.msgpack.zst`. Run on the GPU box (fast network):

  python scripts/download_hf_dense.py --start 0 --count 29 --out-dir data/raw_hf

Verify one shard first (the dataset's exact filename pattern can change):

  python scripts/download_hf_dense.py --start 0 --count 1 --out-dir data/raw_hf
"""
import argparse
import os
import subprocess
import sys

REPO = "prdev/chessbench-full-policy-value"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
NUM_SHARDS = 1024


def shard_filename(idx: int) -> str:
    return f"train-{idx:05d}-of-{NUM_SHARDS:05d}.msgpack.zst"


def shard_url(idx: int) -> str:
    return f"{BASE}/{shard_filename(idx)}"


def download_one(idx: int, out_dir: str) -> str:
    dest = os.path.join(out_dir, shard_filename(idx))
    if os.path.exists(dest):
        print(f"exists, skipping {dest}", flush=True)
        return dest
    url = shard_url(idx)
    print(f"downloading {url}", flush=True)
    rc = subprocess.call(
        ["curl", "-fL", "--retry", "3", "--create-dirs", "-o", dest, url]
    )
    if rc != 0:
        raise RuntimeError(f"curl failed ({rc}) for {url}")
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--out-dir", default="data/raw_hf")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for idx in range(args.start, min(args.start + args.count, NUM_SHARDS)):
        download_one(idx, args.out_dir)
    print(f"done: {args.count} shard(s) into {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/james/.venvs/chess-rl/bin/python -m pytest tests/data/test_download_hf_dense.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
rtk git add scripts/download_hf_dense.py tests/data/test_download_hf_dense.py
rtk git commit -m "feat(phase3): HF raw-shard downloader (URL helpers + curl driver)"
```

---

### Task 6: Full local suite green

**Files:** none (verification gate before the remote run).

- [ ] **Step 1: Run the whole suite**

Run: `/Users/james/.venvs/chess-rl/bin/python -m pytest -q`
Expected: all pass (the Phase-2 baseline 141 + new Phase-3 tests; only CUDA-gated AMP test skipped). If anything fails, fix before proceeding — do not rent the GPU on a red suite.

- [ ] **Step 2: Confirm baseline lock explicitly**

Run: `/Users/james/.venvs/chess-rl/bin/python -m pytest tests/model/test_presets.py -q`
Expected: PASS — baseline-v1 still 10,255,879.

- [ ] **Step 3: Push the branch**

```bash
rtk git push
```

---

### Task 7: ThunderCompute runbook (documentation)

**Files:**
- Create: `docs/runbooks/2026-06-20-phase3-thundercompute.md`

This is a documentation task (no code/tests). The verification is: every command is copy-pasteable, paths match the spec, and the cost-discipline notes are present. After the real run, the operator fills in measured throughput and links the ablation doc.

- [ ] **Step 1: Write the runbook**

Create `docs/runbooks/2026-06-20-phase3-thundercompute.md` with exactly this content:

````markdown
# Phase 3 ThunderCompute Runbook (A6000, 77M + 10M control on ~100M)

Spec: `docs/superpowers/specs/2026-06-20-chess-rl-phase3-gpu-scaleup-design.md`.
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
# Standard gates
python scripts/ablate.py --model checkpoints/p3_80m/best_ema.pt   # writes/append docs/ablations
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
````

- [ ] **Step 2: Verify the runbook**

Read it top to bottom: confirm every path matches the spec (`data/shards_p3_100m`, `checkpoints/p3_80m`, `checkpoints/p3_10m_ctrl`, `checkpoints/p2_full_300k`), the preset names exist, and the cost-discipline notes are present.

- [ ] **Step 3: Commit**

```bash
rtk git add docs/runbooks/2026-06-20-phase3-thundercompute.md
rtk git commit -m "docs(phase3): ThunderCompute A6000 runbook (download->encode->train->eval)"
rtk git push
```

---

## Plan self-review

**Spec coverage:**
- `p3-80m` preset + 77M lock → Task 1. ✅
- CUDA device path + `--device` → Task 2. ✅
- Real bf16 AMP → Task 3. ✅
- Model-vs-model head-to-head (`--vs`) → Task 4. ✅
- HF shard downloader → Task 5. ✅
- Encoding pipeline → already done (`73b6b70`), out of plan scope. ✅
- Data plan / training recipe / two head-to-heads / ladder / scp / delete → Task 7 runbook. ✅
- `baseline-v1` bit-identical → asserted in Tasks 1 & 6. ✅

**Placeholder scan:** none — all code blocks are complete; the only deliberately-deferred values are runbook `<id>` (instance id) and post-run measured throughput, which cannot exist before the run.

**Type/name consistency:** `build_raw_mover(net, device, me)` and `build_mcts_mover(evaluator, simulations)` are defined in Task 4 and used consistently in `arena_eval.py`. `_default_device()` defined and used in Task 2, tested in Task 2. Preset name `p3-80m`, control preset `p2-value-swiglu-drop`, checkpoints `p3_80m`/`p3_10m_ctrl`/`p2_full_300k`, shard dir `data/shards_p3_100m` consistent across Tasks 1/4/7. `shard_filename`/`shard_url` defined and tested in Task 5.
