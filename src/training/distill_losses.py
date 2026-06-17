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


def total_loss(policy_logits, wdl_logits, moves_left_pred,
               policy_target, wdl_target, moves_left_target,
               value_weight=1.0, moves_left_weight=0.1):
    p = policy_loss(policy_logits, policy_target)
    v = value_loss(wdl_logits, wdl_target)
    m = moves_left_loss(moves_left_pred, moves_left_target)
    total = p + value_weight * v + moves_left_weight * m
    return total, {"policy": p, "value": v, "moves_left": m}
