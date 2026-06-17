# tests/eval/test_elo.py
"""Tests for Elo math utilities."""
import math
import pytest
from src.eval.elo import elo_diff, expected_score, gauntlet_elo


def test_expected_score_equal_elo():
    # Equal opponents -> expected score = 0.5
    assert expected_score(0) == pytest.approx(0.5, abs=1e-9)


def test_expected_score_positive_diff():
    # +200 Elo advantage -> expected score ≈ 0.7597
    result = expected_score(200)
    assert result == pytest.approx(0.7597, abs=0.001)


def test_expected_score_negative_diff():
    # Symmetric: score(-d) = 1 - score(d)
    assert expected_score(-200) == pytest.approx(1 - expected_score(200), abs=1e-9)


def test_elo_diff_known_score():
    # Score of 0.76 corresponds to roughly +200 Elo
    diff = elo_diff(0.76, games=1)
    assert diff == pytest.approx(198.0, abs=5.0)


def test_elo_diff_draw():
    # Score of 0.5 -> 0 Elo difference
    assert elo_diff(0.5, games=1) == pytest.approx(0.0, abs=1e-6)


def test_elo_diff_perfect_score_clamps():
    # Perfect score (1.0) should return large finite value, not inf/nan
    diff = elo_diff(1.0, games=10)
    assert math.isfinite(diff)
    assert diff > 600.0


def test_elo_diff_zero_score_clamps():
    # Zero score should return large negative finite value
    diff = elo_diff(0.0, games=10)
    assert math.isfinite(diff)
    assert diff < -600.0


def test_gauntlet_elo_single_opponent():
    # One opponent, known score -> consistent with elo_diff
    result = gauntlet_elo([(0.76, 100, 0)])
    expected = elo_diff(0.76, games=100)
    assert result == pytest.approx(expected, abs=1.0)


def test_gauntlet_elo_multiple_opponents():
    # Two opponents at different base Elos; result is weighted average
    result = gauntlet_elo([(0.76, 50, 0), (0.60, 50, 200)])
    assert isinstance(result, float)
    assert math.isfinite(result)
    assert 100.0 < result < 400.0
