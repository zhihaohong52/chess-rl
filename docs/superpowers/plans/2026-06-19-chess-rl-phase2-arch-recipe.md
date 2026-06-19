# Phase 2: Architecture/Recipe Ablations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four config-gated optimizations (HL-Gauss value head [priority], SwiGLU FFN, dropout, EMA) so an incrementally-stacked `phase2-best` preset beats `hf-dense-300k`/`baseline-v1` on raw top-1 AND MCTS, fixing the value-head bottleneck (MCTS ≥ raw).

**Architecture:** Every change is gated by a config flag whose default reproduces `baseline-v1` bit-identically (10,255,879 params, WDL value head, GELU MLP FFN, dropout 0.0, no EMA). The model branches on `cfg.value_head_type` and `cfg.ffn_type`; the trainer/evaluator/gates branch on the net's `value_head_type`. Ablated via the Phase 0 harness.

**Tech Stack:** Python 3.12, PyTorch 2.12 (MPS local / CUDA on rented GPU), numpy, pytest.

**Environment & conventions (read first):**
- Use the off-iCloud venv: `/Users/james/.venvs/chess-rl/bin/python` and `.../bin/pytest`.
- Prefix shell commands with `rtk`. Branch `feat/transformer-rebuild`. Commit per task.
- **`baseline-v1` MUST stay bit-identical**: defaults `value_head_type="wdl"`, `ffn_type="mlp"`, `transformer_dropout=0.0`, `ema_decay=0.0`. `nn.Dropout(0.0)` and the unchanged `nn.Sequential(Linear, GELU, Linear)` MLP add no params and consume no init RNG, so param count and init are preserved. The test `tests/model/test_presets.py::test_build_baseline_v1_param_count_is_locked` (==10,255,879) guards this — keep it green at every task.
- Confirmed current interfaces:
  - `ValueHead(d_model, hidden=128)`: `d1=Linear(d,128)`, `d2=Linear(128,3)`, `forward(cls)=d2(relu(d1(cls)))`.
  - `MovesLeftHead` similar with `d2→1`, softplus.
  - `EncoderLayer(cfg, shared_smolgen_out)`: `ln1, attn=BiasedMHA, smolgen, ln2, ffn=nn.Sequential(Linear(d,d_ff),GELU(),Linear(d_ff,d))`; `forward`: `h=ln1(x); bias=pad(smolgen(h[:,1:])); x=x+attn(h,bias); x=x+ffn(ln2(x))`.
  - `ChessTransformer.__init__` builds `value_head=ValueHead(d)`; `forward(square_tokens,state_features)` returns `(policy_head(sq_out), value_head(cls_out), moves_left_head(cls_out))`; `predict_batch` = eval no-grad forward.
  - `distill_losses.total_loss(policy_logits, wdl_logits, moves_left_pred, policy_target, wdl_target, moves_left_target, value_weight=1.0, moves_left_weight=0.1)` → `(total, {"policy","value","moves_left"})`.
  - `DistillTrainer.train_step` calls `total_loss`; `evaluate` uses `value_sign_accuracy(wdl, wdl_t)` from `distill_metrics`; both unpack `pol, wdl, ml = self.net(sq, sf)`.
  - `evaluator.evaluate_batch` computes value: policy → `wdl[i,0]-wdl[i,2]`; action_value → `2*q.max()-1`.
  - `presets.resolve_config` returns a `SimpleNamespace` from `vars(Config)`; `build_model(preset)` → `(net, cfg)`.

---

## File Structure

- Create `src/model/value_dist.py` — pure HL-Gauss helpers (`bucket_centers`, `hl_gauss_target`, `expected_value`).
- Modify `config.py` — add `value_head_type, value_buckets, value_sigma_frac, value_loss_weight, ffn_type, ema_decay`.
- Modify `src/model/heads.py` — add `DistributionalValueHead`.
- Modify `src/model/transformer.py` — value-head branch in `ChessTransformer`; `SwiGLUFFN` + `ffn_type`/dropout in `EncoderLayer`; store `value_head_type`/`value_buckets` attrs.
- Modify `src/training/distill_losses.py` — value-head-aware value loss + `total_loss` branch.
- Modify `src/training/distill_trainer.py` — pass value-head info to loss; value-head-aware val metrics; EMA.
- Modify `scripts/distill.py` — `--ema-decay`.
- Modify `src/model/evaluator.py` — hlgauss value.
- Modify `src/eval/gates.py` + `src/eval/metrics_core.py` — value metrics per head type.
- Modify `src/model/presets.py` — `p2-*` presets.
- Tests under `tests/model/`, `tests/training/`, `tests/eval/`.

---

## Task 1: Config knobs + HL-Gauss math (`src/model/value_dist.py`)

**Files:**
- Modify: `config.py`
- Create: `src/model/value_dist.py`
- Test: `tests/model/test_value_dist.py`

- [ ] **Step 1: Add config fields.** In `config.py`, in the transformer section (after `transformer_dropout = 0.0`), add:

```python
    # Phase 2 value head / recipe knobs (defaults preserve baseline-v1)
    value_head_type = "wdl"      # "wdl" | "hlgauss"
    value_buckets = 64
    value_sigma_frac = 0.75      # HL-Gauss sigma as fraction of a bucket width
    value_loss_weight = 1.0
    ffn_type = "mlp"             # "mlp" | "swiglu"
    ema_decay = 0.0              # 0 = off
```

- [ ] **Step 2: Write the failing test**

```python
# tests/model/test_value_dist.py
import math
import torch
from src.model.value_dist import bucket_centers, hl_gauss_target, expected_value


def test_bucket_centers():
    c = bucket_centers(4)
    assert torch.allclose(c, torch.tensor([0.125, 0.375, 0.625, 0.875]))


def test_hl_gauss_target_sums_to_one_and_peaks_at_v():
    v = torch.tensor([0.2, 0.8])
    t = hl_gauss_target(v, buckets=64, sigma_frac=0.75)
    assert t.shape == (2, 64)
    assert torch.allclose(t.sum(dim=1), torch.ones(2), atol=1e-5)
    # peak bucket ~ floor(v*64)
    assert t[0].argmax().item() in (12, 13)   # 0.2*64=12.8
    assert t[1].argmax().item() in (50, 51)   # 0.8*64=51.2


def test_expected_value_recovers_v():
    v = torch.tensor([0.3, 0.6, 0.9])
    t = hl_gauss_target(v, buckets=128, sigma_frac=0.75)
    vhat = expected_value(torch.log(t + 1e-12))   # logits ~ log target
    assert torch.allclose(vhat, v, atol=1.0 / 128 * 2)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/model/test_value_dist.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.model.value_dist'`.

- [ ] **Step 4: Implement**

```python
# src/model/value_dist.py
"""HL-Gauss distributional-value helpers (pure tensor functions).

Value is expected score in [0,1]. A scalar target v is turned into a soft
histogram over `buckets` bins via a Gaussian CDF (HL-Gauss); the head's expected
value is sum(bucket_center * softmax(logits)).
"""
import math
import torch


def bucket_centers(buckets: int, device=None) -> torch.Tensor:
    idx = torch.arange(buckets, dtype=torch.float32, device=device)
    return (idx + 0.5) / buckets


def _ndtr(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def hl_gauss_target(v: torch.Tensor, buckets: int, sigma_frac: float = 0.75) -> torch.Tensor:
    """v: [B] in [0,1] -> soft target [B, buckets] (rows sum to 1)."""
    device = v.device
    edges = torch.linspace(0.0, 1.0, buckets + 1, device=device)  # [K+1]
    sigma = sigma_frac * (1.0 / buckets)
    vv = v.clamp(0.0, 1.0).unsqueeze(1)                            # [B,1]
    cdf = _ndtr((edges.unsqueeze(0) - vv) / sigma)                 # [B,K+1]
    probs = cdf[:, 1:] - cdf[:, :-1]                               # [B,K]
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)


def expected_value(value_logits: torch.Tensor) -> torch.Tensor:
    """value_logits [B,K] -> vhat [B] in [0,1]."""
    centers = bucket_centers(value_logits.shape[-1], device=value_logits.device)
    return (torch.softmax(value_logits, dim=-1) * centers).sum(dim=-1)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/model/test_value_dist.py -q`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
rtk git add config.py src/model/value_dist.py tests/model/test_value_dist.py && \
rtk git commit -m "feat(phase2): config knobs + HL-Gauss value helpers"
```

---

## Task 2: DistributionalValueHead (`src/model/heads.py`)

**Files:**
- Modify: `src/model/heads.py`
- Test: `tests/model/test_dist_value_head.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model/test_dist_value_head.py
import torch
from src.model.heads import DistributionalValueHead


def test_dist_value_head_shape():
    head = DistributionalValueHead(256, 64)
    out = head(torch.randn(4, 256))
    assert out.shape == (4, 64)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/model/test_dist_value_head.py -q`
Expected: FAIL with `ImportError: cannot import name 'DistributionalValueHead'`.

- [ ] **Step 3: Implement** — add to `src/model/heads.py` after `ValueHead`:

```python
class DistributionalValueHead(nn.Module):
    """CLS -> logits over `buckets` expected-score bins in [0,1] (HL-Gauss)."""

    def __init__(self, d_model, buckets, hidden=128):
        super().__init__()
        self.d1 = nn.Linear(d_model, hidden)
        self.d2 = nn.Linear(hidden, buckets)

    def forward(self, cls):
        return self.d2(F.relu(self.d1(cls)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/model/test_dist_value_head.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
rtk git add src/model/heads.py tests/model/test_dist_value_head.py && \
rtk git commit -m "feat(phase2): DistributionalValueHead"
```

---

## Task 3: Value-head branch in ChessTransformer (`src/model/transformer.py`)

**Files:**
- Modify: `src/model/transformer.py`
- Test: `tests/model/test_transformer_valuehead.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model/test_transformer_valuehead.py
import torch
from src.model.presets import build_model, resolve_config, PRESETS
from src.model.transformer import ChessTransformer


def test_baseline_wdl_unchanged():
    net, _ = build_model("baseline-v1")
    assert net.value_head_type == "wdl"
    sq = torch.zeros(2, 64, dtype=torch.long)
    sf = torch.zeros(2, 18, dtype=torch.float32)
    pol, val, ml = net(sq, sf)
    assert val.shape == (2, 3)


def test_hlgauss_value_output_shape():
    cfg = resolve_config("baseline-v1")
    cfg.value_head_type = "hlgauss"
    cfg.value_buckets = 64
    net = ChessTransformer(cfg)
    assert net.value_head_type == "hlgauss" and net.value_buckets == 64
    sq = torch.zeros(2, 64, dtype=torch.long)
    sf = torch.zeros(2, 18, dtype=torch.float32)
    _, val, _ = net(sq, sf)
    assert val.shape == (2, 64)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/model/test_transformer_valuehead.py -q`
Expected: FAIL with `AttributeError: 'ChessTransformer' object has no attribute 'value_head_type'`.

- [ ] **Step 3: Implement** — in `ChessTransformer.__init__`, replace the line `self.value_head = ValueHead(d)` with:

```python
        self.value_head_type = getattr(cfg, "value_head_type", "wdl")
        self.value_buckets = getattr(cfg, "value_buckets", 64)
        if self.value_head_type == "hlgauss":
            self.value_head = DistributionalValueHead(d, self.value_buckets)
        else:
            self.value_head = ValueHead(d)
```

Add `DistributionalValueHead` to the import from `src.model.heads`:

```python
from src.model.heads import PolicyHead, ValueHead, MovesLeftHead, DistributionalValueHead
```

Update the `forward` docstring's return line to: `Returns (policy_logits[B,P], value_out[B,3]|[B,K], moves_left[B,1]).` (forward body is unchanged — it already returns `self.value_head(cls_out)`.)

**Important:** keep the `else` branch building `ValueHead(d)` exactly as before so `baseline-v1` param count and init are unchanged.

- [ ] **Step 4: Run tests**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/model/test_transformer_valuehead.py tests/model/test_presets.py -q`
Expected: PASS — both new tests AND the baseline param-count lock (10,255,879) still green.

- [ ] **Step 5: Commit**

```bash
rtk git add src/model/transformer.py tests/model/test_transformer_valuehead.py && \
rtk git commit -m "feat(phase2): config-gated value-head branch (wdl|hlgauss)"
```

---

## Task 4: Value-head-aware loss (`src/training/distill_losses.py`)

**Files:**
- Modify: `src/training/distill_losses.py`
- Test: `tests/training/test_value_loss.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/training/test_value_loss.py
import torch
from src.training.distill_losses import total_loss
from src.game.move_encoder import get_move_encoder


def _common():
    P = get_move_encoder().policy_size
    pol = torch.randn(2, P); pol_t = torch.softmax(torch.randn(2, P), dim=-1)
    ml = torch.zeros(2, 1); ml_t = torch.zeros(2, 1)
    wdl_t = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
    return P, pol, pol_t, ml, ml_t, wdl_t


def test_total_loss_wdl_default():
    P, pol, pol_t, ml, ml_t, wdl_t = _common()
    wdl = torch.randn(2, 3)
    loss, parts = total_loss(pol, wdl, ml, pol_t, wdl_t, ml_t)
    assert set(parts) == {"policy", "value", "moves_left"}
    assert torch.isfinite(loss)


def test_total_loss_hlgauss_branch():
    P, pol, pol_t, ml, ml_t, wdl_t = _common()
    K = 64
    val = torch.randn(2, K)
    loss, parts = total_loss(pol, val, ml, pol_t, wdl_t, ml_t,
                             value_head_type="hlgauss", value_buckets=K,
                             value_sigma_frac=0.75)
    assert torch.isfinite(loss) and torch.isfinite(parts["value"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/training/test_value_loss.py -q`
Expected: FAIL — `total_loss()` got an unexpected keyword `value_head_type`.

- [ ] **Step 3: Implement** — in `src/training/distill_losses.py`, add a distributional value loss and branch `total_loss`:

```python
def value_loss_hlgauss(value_logits, wdl_target, buckets, sigma_frac):
    """HL-Gauss CE: target scalar = expected score W + 0.5*D (from WDL target)."""
    from src.model.value_dist import hl_gauss_target
    v = wdl_target[:, 0] + 0.5 * wdl_target[:, 1]
    target = hl_gauss_target(v, buckets, sigma_frac)
    return -(target * F.log_softmax(value_logits, dim=-1)).sum(dim=-1).mean()
```

Replace `total_loss` with:

```python
def total_loss(policy_logits, value_out, moves_left_pred,
               policy_target, wdl_target, moves_left_target,
               value_weight=1.0, moves_left_weight=0.1,
               value_head_type="wdl", value_buckets=64, value_sigma_frac=0.75):
    p = policy_loss(policy_logits, policy_target)
    if value_head_type == "hlgauss":
        v = value_loss_hlgauss(value_out, wdl_target, value_buckets, value_sigma_frac)
    else:
        v = value_loss(value_out, wdl_target)
    m = moves_left_loss(moves_left_pred, moves_left_target)
    total = p + value_weight * v + moves_left_weight * m
    return total, {"policy": p, "value": v, "moves_left": m}
```

- [ ] **Step 4: Run tests**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/training/test_value_loss.py tests/training/test_trainer.py -q`
Expected: PASS — new tests AND existing trainer tests (which call `total_loss` with defaults) still green.

- [ ] **Step 5: Commit**

```bash
rtk git add src/training/distill_losses.py tests/training/test_value_loss.py && \
rtk git commit -m "feat(phase2): value-head-aware total_loss (wdl|hlgauss)"
```

---

## Task 5: Trainer wiring (loss + val metrics) (`src/training/distill_trainer.py`)

**Files:**
- Modify: `src/training/distill_trainer.py`
- Test: `tests/training/test_trainer_hlgauss.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/training/test_trainer_hlgauss.py
import numpy as np, torch, chess
from src.model.presets import resolve_config
from src.model.transformer import ChessTransformer
from src.game.token_encoder import encode_batch
from src.game.move_encoder import get_move_encoder
from src.training.distill_trainer import DistillTrainer


def _batch():
    P = get_move_encoder().policy_size
    boards = [chess.Board(), chess.Board()]
    sq, sf = encode_batch(boards, [0, 0])
    pol = np.zeros((2, P), dtype=np.float32); pol[:, 0] = 1.0
    wdl = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]], dtype=np.float32)
    ml = np.array([[40.0], [40.0]], dtype=np.float32)
    inputs = (torch.tensor(sq, dtype=torch.long), torch.tensor(sf, dtype=torch.float32))
    targets = (torch.tensor(pol), torch.tensor(wdl), torch.tensor(ml))
    return inputs, targets


def test_trainer_hlgauss_step_and_eval():
    cfg = resolve_config("baseline-v1")
    cfg.value_head_type = "hlgauss"; cfg.value_buckets = 64
    net = ChessTransformer(cfg)
    trainer = DistillTrainer(net, cfg, device="cpu")
    inputs, targets = _batch()
    loss, parts = trainer.train_step(inputs, targets)
    assert np.isfinite(loss) and np.isfinite(parts["value"])
    metrics = trainer.evaluate([(inputs, targets)], max_batches=1)
    assert np.isfinite(metrics["val_value_sign_acc"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/training/test_trainer_hlgauss.py -q`
Expected: FAIL (value loss shape mismatch: WDL-CE applied to a [B,64] output, or sign-acc errors).

- [ ] **Step 3: Implement.** In `DistillTrainer.train_step`, where it calls `total_loss(...)`, pass the value-head info (read from the net + cfg). Replace the `total_loss(pol, wdl, ml, pol_t, wdl_t, ml_t)` call with:

```python
        loss_t, parts = total_loss(
            pol, wdl, ml, pol_t, wdl_t, ml_t,
            value_weight=getattr(self.cfg, "value_loss_weight", 1.0),
            value_head_type=getattr(self.net, "value_head_type", "wdl"),
            value_buckets=getattr(self.net, "value_buckets", 64),
            value_sigma_frac=getattr(self.cfg, "value_sigma_frac", 0.75),
        )
```

(Keep the variable name `wdl` for the net's value output even though it is `[B,K]` for hlgauss — it is just the second forward output.)

In `DistillTrainer.evaluate`, replace the `vs += value_sign_accuracy(wdl, wdl_t)` line with a value-head-aware sign accuracy:

```python
                if getattr(self.net, "value_head_type", "wdl") == "hlgauss":
                    from src.model.value_dist import expected_value
                    vhat = expected_value(wdl)                       # [B] in [0,1]
                    v_tgt = wdl_t[:, 0] + 0.5 * wdl_t[:, 1]
                    vs += float(((vhat > 0.5) == (v_tgt > 0.5)).float().mean())
                else:
                    vs += value_sign_accuracy(wdl, wdl_t)
```

- [ ] **Step 4: Run tests**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/training/test_trainer_hlgauss.py tests/training/test_trainer.py tests/training/test_distill_cli.py -q`
Expected: PASS — hlgauss path works AND the existing WDL trainer/CLI tests still green.

- [ ] **Step 5: Commit**

```bash
rtk git add src/training/distill_trainer.py tests/training/test_trainer_hlgauss.py && \
rtk git commit -m "feat(phase2): trainer wires hlgauss loss + val sign-acc"
```

---

## Task 6: Evaluator + gates value-head branch (`src/model/evaluator.py`, `src/eval/gates.py`)

**Files:**
- Modify: `src/model/evaluator.py`, `src/eval/gates.py`
- Test: `tests/eval/test_hlgauss_eval.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_hlgauss_eval.py
import chess
from src.model.presets import resolve_config
from src.model.transformer import ChessTransformer
from src.model.evaluator import TransformerEvaluator
from src.eval.gates import policy_value_metrics
from src.data.dataset import make_dataloader
from src.eval.fixtures import ensure_dense_val
from src.game.move_encoder import get_move_encoder


def _hlgauss_net():
    cfg = resolve_config("baseline-v1")
    cfg.value_head_type = "hlgauss"; cfg.value_buckets = 64
    return ChessTransformer(cfg)


def test_evaluator_hlgauss_value_in_range():
    net = _hlgauss_net()
    ev = TransformerEvaluator(net, device="cpu", objective="policy")
    policy, value = ev.evaluate(chess.Board())
    assert -1.0 <= value <= 1.0
    assert abs(sum(policy.values()) - 1.0) < 1e-4


def test_gates_policy_value_metrics_hlgauss_smoke(tmp_path):
    import pytest, os
    if not os.path.exists("data/test/action_value_data.bag"):
        pytest.skip("test bag absent")
    net = _hlgauss_net()
    path = ensure_dense_val()
    P = get_move_encoder().policy_size
    loader = make_dataloader([path], batch_size=64, policy_size=P, shuffle=False)
    m = policy_value_metrics(net, "cpu", loader, max_batches=1)
    assert "value_sign_acc" in m and 0.0 <= m["value_sign_acc"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/eval/test_hlgauss_eval.py -q`
Expected: FAIL (evaluator treats the [B,64] output as WDL → wrong value / index error).

- [ ] **Step 3: Implement.** In `src/model/evaluator.py`, in the `evaluate_batch` policy branch, compute value per head type. Find the policy-objective block that sets `value = float(wdl[i, 0] - wdl[i, 2])` and replace it with:

```python
                if getattr(self.net, "value_head_type", "wdl") == "hlgauss":
                    from src.model.value_dist import bucket_centers
                    import numpy as _np
                    logits_v = wdl[i]                       # [K]
                    p = _np.exp(logits_v - logits_v.max()); p = p / p.sum()
                    centers = bucket_centers(len(p)).numpy()
                    vhat = float((p * centers).sum())
                    value = 2.0 * vhat - 1.0
                else:
                    value = float(wdl[i, 0] - wdl[i, 2])  # P(W) - P(L)
```

(Here `wdl` is the net's second output array for row `i`; for hlgauss it is the `[K]` bucket logits. Keep the existing `action_value` branch untouched.)

In `src/eval/gates.py`, make `policy_value_metrics` value metrics head-aware. Replace the value-metric accumulation lines (`acc["wdl_ce"] += ...`, `acc["value_sign_acc"] += ...`, `acc["draw_cal"] += ...`) with:

```python
            if getattr(net, "value_head_type", "wdl") == "hlgauss":
                from src.model.value_dist import expected_value, hl_gauss_target
                import torch.nn.functional as _F
                v_tgt = wdl_t[:, 0] + 0.5 * wdl_t[:, 1]
                vhat = expected_value(wdl)
                tgt = hl_gauss_target(v_tgt, wdl.shape[-1],
                                      getattr(net.cfg, "value_sigma_frac", 0.75))
                acc["wdl_ce"] += float(-(tgt * _F.log_softmax(wdl, dim=-1)).sum(-1).mean())
                acc["value_sign_acc"] += float(((vhat > 0.5) == (v_tgt > 0.5)).float().mean())
                acc["draw_cal"] += float((vhat - v_tgt).abs().mean())
            else:
                acc["wdl_ce"] += mc.wdl_cross_entropy(wdl, wdl_t)
                acc["value_sign_acc"] += mc.value_sign_acc(wdl, wdl_t)
                acc["draw_cal"] += mc.draw_calibration(wdl, wdl_t)
```

(`wdl` here is the net's value output `pol, wdl, _ = net(sq, sf)`; for hlgauss it's `[B,K]`. The `wdl_ce`/`draw_cal` keys are reused so the ablation table columns line up; their meaning for hlgauss is value-CE / value-MAE, documented in the ablation note.)

- [ ] **Step 4: Run tests**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/eval/test_hlgauss_eval.py tests/eval/test_gates.py tests/model/test_evaluator.py -q`
Expected: PASS — hlgauss eval works AND existing WDL eval/gates green.

- [ ] **Step 5: Commit**

```bash
rtk git add src/model/evaluator.py src/eval/gates.py tests/eval/test_hlgauss_eval.py && \
rtk git commit -m "feat(phase2): evaluator + gates value-head branch (hlgauss)"
```

---

## Task 7: SwiGLU FFN (`src/model/transformer.py`)

**Files:**
- Modify: `src/model/transformer.py`
- Test: `tests/model/test_swiglu.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model/test_swiglu.py
import torch
from src.model.transformer import SwiGLUFFN
from src.model.presets import resolve_config
from src.model.transformer import ChessTransformer


def test_swiglu_shape_and_param_parity():
    d, d_ff = 256, 1024
    sw = SwiGLUFFN(d, d_ff)
    x = torch.randn(2, 5, d)
    assert sw(x).shape == (2, 5, d)
    mlp_params = 2 * d * d_ff + d_ff + d            # Linear(d,d_ff)+Linear(d_ff,d) incl. biases
    sw_params = sum(p.numel() for p in sw.parameters())
    assert abs(sw_params - mlp_params) / mlp_params < 0.10   # within 10%


def test_swiglu_preset_builds():
    cfg = resolve_config("baseline-v1")
    cfg.ffn_type = "swiglu"
    net = ChessTransformer(cfg)
    sq = torch.zeros(2, 64, dtype=torch.long); sf = torch.zeros(2, 18)
    pol, val, ml = net(sq, sf)
    assert pol.shape[0] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/model/test_swiglu.py -q`
Expected: FAIL with `ImportError: cannot import name 'SwiGLUFFN'`.

- [ ] **Step 3: Implement.** In `src/model/transformer.py`, add the module (after `BiasedMHA`):

```python
class SwiGLUFFN(nn.Module):
    """Gated FFN: down(silu(gate(x)) * up(x)). h ~ 2/3 d_ff to match MLP params."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        h = max(8, round(d_ff * 2 / 3 / 8) * 8)
        self.gate = nn.Linear(d_model, h)
        self.up = nn.Linear(d_model, h)
        self.down = nn.Linear(h, d_model)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

In `EncoderLayer.__init__`, replace the `self.ffn = nn.Sequential(...)` block with:

```python
        if getattr(cfg, "ffn_type", "mlp") == "swiglu":
            self.ffn = SwiGLUFFN(cfg.d_model, cfg.d_ff)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(),
                nn.Linear(cfg.d_ff, cfg.d_model),
            )
```

- [ ] **Step 4: Run tests**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/model/test_swiglu.py tests/model/test_presets.py -q`
Expected: PASS — SwiGLU works AND baseline-v1 param count still 10,255,879 (mlp branch unchanged).

- [ ] **Step 5: Commit**

```bash
rtk git add src/model/transformer.py tests/model/test_swiglu.py && \
rtk git commit -m "feat(phase2): config-gated SwiGLU FFN"
```

---

## Task 8: Dropout in EncoderLayer (`src/model/transformer.py`)

**Files:**
- Modify: `src/model/transformer.py`
- Test: `tests/model/test_dropout.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model/test_dropout.py
import torch
from src.model.presets import resolve_config
from src.model.transformer import ChessTransformer


def test_dropout_changes_train_not_eval():
    cfg = resolve_config("baseline-v1")
    cfg.transformer_dropout = 0.5
    net = ChessTransformer(cfg)
    sq = torch.zeros(2, 64, dtype=torch.long); sf = torch.zeros(2, 18)
    net.train()
    torch.manual_seed(0); a = net(sq, sf)[0]
    torch.manual_seed(1); b = net(sq, sf)[0]
    assert not torch.allclose(a, b)        # dropout randomizes in train
    net.eval()
    c = net(sq, sf)[0]; d = net(sq, sf)[0]
    assert torch.allclose(c, d)            # deterministic in eval
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/model/test_dropout.py -q`
Expected: FAIL (no dropout yet → train outputs identical across seeds → `not allclose` is False).

- [ ] **Step 3: Implement.** In `EncoderLayer.__init__`, add after the ffn block:

```python
        self.drop = nn.Dropout(getattr(cfg, "transformer_dropout", 0.0))
```

In `EncoderLayer.forward`, wrap the sublayer outputs:

```python
    def forward(self, x):
        h = self.ln1(x)
        bias64 = self.smolgen(h[:, 1:, :])
        bias = F.pad(bias64, (1, 0, 1, 0))
        x = x + self.drop(self.attn(h, bias))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x
```

- [ ] **Step 4: Run tests**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/model/test_dropout.py tests/model/test_presets.py -q`
Expected: PASS — dropout works AND baseline-v1 param count still 10,255,879 (`Dropout(0.0)` adds no params, no init RNG).

- [ ] **Step 5: Commit**

```bash
rtk git add src/model/transformer.py tests/model/test_dropout.py && \
rtk git commit -m "feat(phase2): config-gated transformer dropout"
```

---

## Task 9: EMA of weights (`src/training/distill_trainer.py`, `scripts/distill.py`)

**Files:**
- Modify: `src/training/distill_trainer.py`, `scripts/distill.py`
- Test: `tests/training/test_ema.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/training/test_ema.py
import torch
from src.model.presets import resolve_config
from src.model.transformer import ChessTransformer
from src.training.distill_trainer import DistillTrainer
from src.training.checkpoint_meta import read_sidecar
import numpy as np, chess
from src.game.token_encoder import encode_batch
from src.game.move_encoder import get_move_encoder


def _batch():
    P = get_move_encoder().policy_size
    sq, sf = encode_batch([chess.Board(), chess.Board()], [0, 0])
    pol = np.zeros((2, P), dtype=np.float32); pol[:, 0] = 1.0
    wdl = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32)
    ml = np.array([[40.0], [40.0]], dtype=np.float32)
    return ((torch.tensor(sq, dtype=torch.long), torch.tensor(sf, dtype=torch.float32)),
            (torch.tensor(pol), torch.tensor(wdl), torch.tensor(ml)))


def test_ema_saves_ema_checkpoint(tmp_path):
    cfg = resolve_config("baseline-v1")
    cfg.ema_decay = 0.9
    net = ChessTransformer(cfg)
    trainer = DistillTrainer(net, cfg, device="cpu")
    loader = [_batch()]
    ckpt = str(tmp_path / "ck")
    trainer.fit(loader, steps=2, val_loader=loader, val_every=1, ckpt_dir=ckpt,
                meta={"preset": "baseline-v1", "train_data": "unit"})
    import os
    assert os.path.exists(os.path.join(ckpt, "best_ema.pt"))
    assert read_sidecar(os.path.join(ckpt, "best_ema.pt"))["objective"] == "policy"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/training/test_ema.py -q`
Expected: FAIL (no `best_ema.pt` written).

- [ ] **Step 3: Implement.** In `DistillTrainer.__init__`, after the optimizer/scheduler setup, add EMA state:

```python
        self.ema_decay = getattr(cfg, "ema_decay", 0.0)
        self._ema = None
        if self.ema_decay > 0:
            self._ema = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
```

Add an EMA update method and call it from `train_step` (after `self.opt.step()` in BOTH `train_step` and `train_step_av`):

```python
    def _update_ema(self):
        if self._ema is None:
            return
        d = self.ema_decay
        sd = self.net.state_dict()
        for k, v in sd.items():
            if v.dtype.is_floating_point:
                self._ema[k].mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                self._ema[k].copy_(v)
```

Add `self._update_ema()` immediately after `self.opt.step()` in `train_step` (the policy path) — locate `self.opt.step()` and add the call after it.

In `DistillTrainer.fit`, after computing `metrics` in the val branch and saving `best.pt`, also evaluate + save the EMA model when enabled. Replace the val-branch body with:

```python
            if val_loader is not None and step % val_every == 0:
                metrics = self.evaluate(val_loader)
                print(f"step {step} loss {loss:.4f} {metrics}")
                if metrics["val_policy_loss"] < best:
                    best = metrics["val_policy_loss"]
                    self._save_ckpt(ckpt_dir, "best.pt", "policy", meta)
                if self._ema is not None:
                    self._save_ema(ckpt_dir, "best_ema.pt", "policy", meta, val_loader)
```

Add the EMA save+eval helper (loads EMA weights into a temporary state, evaluates, saves, restores):

```python
    def _save_ema(self, ckpt_dir, name, objective, meta, val_loader):
        live = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        self.net.load_state_dict(self._ema)
        ema_metrics = self.evaluate(val_loader)
        best = getattr(self, "_ema_best", float("inf"))
        if ema_metrics["val_policy_loss"] < best:
            self._ema_best = ema_metrics["val_policy_loss"]
            self._save_ckpt(ckpt_dir, name, objective, meta)
        self.net.load_state_dict(live)
```

In `scripts/distill.py`, add the flag and set it on cfg before building the trainer. Add to the argparser:

```python
    ap.add_argument("--ema-decay", type=float, default=0.0)
```

and after `cfg.distill_total_steps = args.steps` add:

```python
    cfg.ema_decay = args.ema_decay
```

- [ ] **Step 4: Run tests**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/training/test_ema.py tests/training/test_trainer.py tests/training/test_distill_cli.py -q`
Expected: PASS — EMA saves `best_ema.pt`; existing trainer/CLI tests green (EMA off by default → no behavior change).

- [ ] **Step 5: Commit**

```bash
rtk git add src/training/distill_trainer.py scripts/distill.py tests/training/test_ema.py && \
rtk git commit -m "feat(phase2): EMA weights (--ema-decay) + best_ema.pt"
```

---

## Task 10: Phase 2 presets + full-suite verification (`src/model/presets.py`)

**Files:**
- Modify: `src/model/presets.py`
- Test: `tests/model/test_p2_presets.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model/test_p2_presets.py
import torch
from src.model.presets import PRESETS, build_model


def test_p2_presets_registered():
    for name in ("p2-value", "p2-value-swiglu", "p2-value-swiglu-drop"):
        assert name in PRESETS


def test_p2_value_builds_hlgauss():
    net, cfg = build_model("p2-value")
    assert net.value_head_type == "hlgauss"
    sq = torch.zeros(2, 64, dtype=torch.long); sf = torch.zeros(2, 18)
    _, val, _ = net(sq, sf)
    assert val.shape == (2, cfg.value_buckets)


def test_p2_swiglu_drop_flags():
    _, cfg = build_model("p2-value-swiglu-drop")
    assert cfg.ffn_type == "swiglu"
    assert cfg.transformer_dropout > 0.0
    assert cfg.value_head_type == "hlgauss"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/james/.venvs/chess-rl/bin/pytest tests/model/test_p2_presets.py -q`
Expected: FAIL (presets not registered).

- [ ] **Step 3: Implement** — in `src/model/presets.py`, extend `PRESETS`:

```python
PRESETS = {
    "baseline-v1": {},
    "p2-value": {"value_head_type": "hlgauss", "value_buckets": 64},
    "p2-value-swiglu": {"value_head_type": "hlgauss", "value_buckets": 64,
                        "ffn_type": "swiglu"},
    "p2-value-swiglu-drop": {"value_head_type": "hlgauss", "value_buckets": 64,
                             "ffn_type": "swiglu", "transformer_dropout": 0.05},
}
```

- [ ] **Step 4: Run the full suite**

Run: `/Users/james/.venvs/chess-rl/bin/pytest -q`
Expected: all pass (existing + all Phase 2 tests), 1 skipped, 2 deselected. Confirm `baseline-v1` param lock (10,255,879) still green.

- [ ] **Step 5: Commit**

```bash
rtk git add src/model/presets.py tests/model/test_p2_presets.py && \
rtk git commit -m "feat(phase2): p2-value / p2-value-swiglu / p2-value-swiglu-drop presets"
```

---

## Task 11: Local smoke of each preset (hybrid step — no commit)

**Files:** none (validation only)

- [ ] **Step 1: Smoke-train each preset 2 steps on the 300k shards (CPU/MPS), confirm no crash + sidecar.**

```bash
for p in p2-value p2-value-swiglu p2-value-swiglu-drop; do
  /Users/james/.venvs/chess-rl/bin/python scripts/distill.py \
    --train 'data/shards_hf_dense_300k/train_*.npz' \
    --val 'data/shards_hf_dense_300k/val_*.npz' \
    --preset "$p" --train-data smoke --batch 16 --steps 2 --warmup 1 \
    --val-every 1 --ckpt "/tmp/smoke_$p" && echo "OK $p"
done
```
Expected: each prints `done. best val_policy_loss=...` and `OK <preset>`. If a preset crashes, STOP and fix before GPU runs.

- [ ] **Step 2: Smoke the gauge on one hlgauss checkpoint.**

```bash
/Users/james/.venvs/chess-rl/bin/python scripts/ablate.py --ckpt /tmp/smoke_p2-value/best.pt \
  --device cpu --fast --out /tmp/smoke_ablate.md && rtk read /tmp/smoke_ablate.md
```
Expected: a well-formed table row for the hlgauss model (no crash). This validates the evaluator/gates hlgauss path end-to-end.

- [ ] **Step 3: Report readiness.** Confirm all three presets smoke-train and the gauge runs on an hlgauss checkpoint. The real ablation runs (300k, ~7000 steps, EMA on the value-best preset) happen on the rented GPU per the spec's ablation flow — that is the next phase of work, not this plan.

---

## Self-Review Checklist (completed by plan author)

- **Spec coverage:** config knobs (T1), HL-Gauss math (T1) + head (T2) + forward branch (T3) + loss (T4) + trainer (T5) + evaluator/gates (T6); SwiGLU (T7); dropout (T8); EMA (T9); presets (T10); local smoke (T11). All spec components covered.
- **Baseline preservation:** every model-touching task re-runs `test_presets.py` (param lock 10,255,879); defaults keep wdl/mlp/dropout-0/ema-off.
- **Placeholder scan:** no TBD/placeholders; all code blocks complete; value target `v=W+0.5D` derived (no data change).
- **Type consistency:** `total_loss(...value_head_type,value_buckets,value_sigma_frac)` (T4) matches the trainer call (T5); net exposes `value_head_type`/`value_buckets` (T3) consumed by trainer (T5), evaluator + gates (T6); `expected_value`/`hl_gauss_target`/`bucket_centers` signatures consistent across T1/T5/T6; `_save_ckpt` (existing) reused by `_save_ema` (T9).
- **Hybrid execution:** T11 smokes locally; real ablations on GPU are out of this plan's scope (spec's ablation flow).
