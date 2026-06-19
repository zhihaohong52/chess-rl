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
