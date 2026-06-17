"""Validation metrics for distillation (PyTorch)."""

import torch
import torch.nn.functional as F


def value_sign_accuracy(wdl_logits, wdl_target) -> float:
    """Fraction where sign(P(W)-P(L)) matches sign(target_W - target_L)."""
    p = F.softmax(wdl_logits, dim=-1)
    pred = torch.sign(p[:, 0] - p[:, 2])
    true = torch.sign(wdl_target[:, 0] - wdl_target[:, 2])
    return float((pred == true).float().mean())


def top1_move_match(policy_logits, policy_target) -> float:
    """Fraction where argmax(pred) == argmax(target)."""
    pred = torch.argmax(policy_logits, dim=-1)
    true = torch.argmax(policy_target, dim=-1)
    return float((pred == true).float().mean())
