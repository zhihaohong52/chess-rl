"""Pure label-math helpers for distillation targets."""

import numpy as np


def cp_to_winprob(cp: float) -> float:
    """Centipawn eval -> win probability in (0,1) via logistic (400cp ~ 0.76)."""
    return 1.0 / (1.0 + 10.0 ** (-cp / 400.0))


def winprob_to_wdl(wp: float, draw_scale: float = 2.0):
    """Map a win probability to (W, D, L). Draw mass peaks near wp=0.5.

    D = 1 - |2*wp - 1|**draw_scale shaped, then W/L split the remainder by wp.
    """
    decisiveness = abs(2.0 * wp - 1.0) ** draw_scale
    d = 1.0 - decisiveness
    rem = 1.0 - d
    w = rem * wp
    l = rem * (1.0 - wp)
    s = w + d + l
    return (w / s, d / s, l / s)


def scores_to_policy(moves, cps, temperature: float = 1.0):
    """Softmax over per-move centipawn scores -> [(move, prob), ...]."""
    arr = np.array(cps, dtype=np.float32)
    denom = max(1e-3, 400.0 * max(temperature, 1e-3))
    logits = arr / denom
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs = probs / probs.sum()
    return list(zip(moves, probs.tolist()))
