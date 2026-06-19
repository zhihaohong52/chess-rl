"""Distillation losses: policy KL/CE, WDL cross-entropy, moves-left Huber (PyTorch)."""

import torch
import torch.nn.functional as F


def policy_loss(policy_logits, policy_target):
    """Soft-target cross-entropy over the move distribution."""
    return -(policy_target * F.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()


def value_loss(wdl_logits, wdl_target):
    """Softmax cross-entropy over {W,D,L}."""
    return -(wdl_target * F.log_softmax(wdl_logits, dim=-1)).sum(dim=-1).mean()


def moves_left_loss(moves_left_pred, moves_left_target):
    return F.smooth_l1_loss(moves_left_pred, moves_left_target)


def value_loss_hlgauss(value_logits, wdl_target, buckets, sigma_frac):
    """HL-Gauss CE: target scalar = expected score W + 0.5*D (from WDL target)."""
    from src.model.value_dist import hl_gauss_target
    v = wdl_target[:, 0] + 0.5 * wdl_target[:, 1]
    target = hl_gauss_target(v, buckets, sigma_frac)
    return -(target * F.log_softmax(value_logits, dim=-1)).sum(dim=-1).mean()


def total_loss(policy_logits, value_out, moves_left_pred,
               policy_target, wdl_target, moves_left_target,
               value_weight=1.0, moves_left_weight=0.1,
               value_head_type="wdl", value_buckets=64, value_sigma_frac=0.75):
    p = policy_loss(policy_logits, policy_target)
    if value_head_type == "hlgauss":
        v = value_loss_hlgauss(value_out, wdl_target, value_buckets, value_sigma_frac)
    elif value_head_type == "wdl":
        v = value_loss(value_out, wdl_target)
    else:
        raise ValueError(
            f"Unknown value_head_type: {value_head_type!r}; expected 'wdl' or 'hlgauss'"
        )
    m = moves_left_loss(moves_left_pred, moves_left_target)
    total = p + value_weight * v + moves_left_weight * m
    return total, {"policy": p, "value": v, "moves_left": m}
