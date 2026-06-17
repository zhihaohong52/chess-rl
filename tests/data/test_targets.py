import numpy as np
from src.data.targets import cp_to_winprob, winprob_to_wdl, scores_to_policy


def test_cp_to_winprob_monotonic_and_centered():
    assert abs(cp_to_winprob(0) - 0.5) < 1e-6
    assert cp_to_winprob(400) > 0.8
    assert cp_to_winprob(-400) < 0.2


def test_winprob_to_wdl_sums_to_one():
    w, d, l = winprob_to_wdl(0.5)
    assert abs(w + d + l - 1.0) < 1e-6
    assert d > 0 and abs(w - l) < 1e-6  # symmetric at 0.5


def test_scores_to_policy_softmax():
    moves = ["e2e4", "d2d4", "a2a3"]
    cps = [50, 40, -200]
    pol = scores_to_policy(moves, cps, temperature=1.0)
    assert abs(sum(p for _, p in pol) - 1.0) < 1e-6
    best = max(pol, key=lambda mp: mp[1])
    assert best[0] == "e2e4"
