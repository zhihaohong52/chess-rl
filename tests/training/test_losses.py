import torch
from src.training.distill_losses import policy_loss, value_loss, moves_left_loss, total_loss


def test_policy_loss_lower_when_aligned():
    target = torch.tensor([[0.0, 1.0, 0.0]])
    good = torch.tensor([[0.0, 10.0, 0.0]])
    bad = torch.tensor([[10.0, 0.0, 0.0]])
    assert float(policy_loss(good, target)) < float(policy_loss(bad, target))


def test_value_loss_lower_when_aligned():
    target = torch.tensor([[1.0, 0.0, 0.0]])  # win
    good = torch.tensor([[10.0, 0.0, 0.0]])
    bad = torch.tensor([[0.0, 0.0, 10.0]])
    assert float(value_loss(good, target)) < float(value_loss(bad, target))


def test_total_loss_combines_parts():
    pol = torch.tensor([[0.0, 5.0, 0.0]]); pol_t = torch.tensor([[0.0, 1.0, 0.0]])
    wdl = torch.tensor([[5.0, 0.0, 0.0]]); wdl_t = torch.tensor([[1.0, 0.0, 0.0]])
    ml = torch.tensor([[40.0]]); ml_t = torch.tensor([[42.0]])
    loss, parts = total_loss(pol, wdl, ml, pol_t, wdl_t, ml_t)
    assert set(parts.keys()) == {"policy", "value", "moves_left"}
    assert float(loss) > 0
