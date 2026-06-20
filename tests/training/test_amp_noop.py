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
    inputs, targets = _one_batch(cfg.policy_size)
    init_state = copy.deepcopy(net.state_dict())

    t_off = DistillTrainer(net, cfg, mixed_precision=False, device="cpu")
    loss_off, _ = t_off.train_step(inputs, targets)

    net.load_state_dict(init_state)
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
