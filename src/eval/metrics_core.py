"""Pure tensor metric functions for the gate suite. No I/O, no model building."""

import torch
import torch.nn.functional as F


def policy_topk_match(pol_logits, target_probs, k: int) -> float:
    """Fraction of rows where the target's argmax move is in the model's top-k."""
    k = min(k, pol_logits.shape[1])
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
    """Mean soft cross-entropy of WDL prediction vs target distribution."""
    logp = F.log_softmax(wdl_logits, dim=1)
    return float(-(wdl_target * logp).sum(dim=1).mean())


def value_sign_acc(wdl_logits, wdl_target) -> float:
    """Agreement of sign(P(W)-P(L)) between prediction and target."""
    p = F.softmax(wdl_logits, dim=1)
    pred = torch.sign(p[:, 0] - p[:, 2])
    tgt = torch.sign(wdl_target[:, 0] - wdl_target[:, 2])
    return float((pred == tgt).float().mean())


def draw_calibration(wdl_logits, wdl_target) -> float:
    """Absolute gap between mean predicted draw prob and mean target draw prob
    (0 = perfectly calibrated on the draw class)."""
    p = F.softmax(wdl_logits, dim=1)
    pred_draw = p[:, 1].mean()
    target_draw = wdl_target[:, 1].mean()
    return float((pred_draw - target_draw).abs())
