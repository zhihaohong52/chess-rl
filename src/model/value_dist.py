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
    edges = torch.linspace(0.0, 1.0, buckets + 1, device=device)
    sigma = sigma_frac * (1.0 / buckets)
    vv = v.clamp(0.0, 1.0).unsqueeze(1)
    cdf = _ndtr((edges.unsqueeze(0) - vv) / sigma)
    probs = cdf[:, 1:] - cdf[:, :-1]
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)


def expected_value(value_logits: torch.Tensor) -> torch.Tensor:
    """value_logits [B,K] -> vhat [B] in [0,1]."""
    centers = bucket_centers(value_logits.shape[-1], device=value_logits.device)
    return (torch.softmax(value_logits, dim=-1) * centers).sum(dim=-1)
