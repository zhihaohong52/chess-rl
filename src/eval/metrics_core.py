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


def value_ece(pred_score, target_score, n_bins: int = 15) -> float:
    """Expected calibration error of the value head's expected-score estimate.

    Both inputs are expected scores (W + 0.5*D) in [0, 1] — predicted vs actual.
    Positions are bucketed by predicted score into `n_bins` equal-width bins; for
    each non-empty bin we take |mean(pred) - mean(actual)|; the result is the
    sample-weighted mean of those gaps (a reliability-diagram ECE; 0 = perfectly
    calibrated). This is the calibration-bucket metric MCTS leaf values need —
    sign accuracy and a single draw-class gap both miss bin-wise miscalibration.
    """
    pred = pred_score.detach().reshape(-1).float()
    target = target_score.detach().reshape(-1).float()
    n = pred.numel()
    if n == 0:
        return 0.0
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=pred.device)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        # the last bin is closed on the right so pred == 1.0 is counted
        in_bin = (pred >= lo) & (pred <= hi) if b == n_bins - 1 else (pred >= lo) & (pred < hi)
        cnt = int(in_bin.sum())
        if cnt == 0:
            continue
        gap = (pred[in_bin].mean() - target[in_bin].mean()).abs()
        ece += (cnt / n) * float(gap)
    return ece
