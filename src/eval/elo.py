# src/eval/elo.py
"""Elo rating utilities: difference from match score and gauntlet aggregation."""

import math
from typing import List, Tuple

# Logistic Elo model: E(d) = 1 / (1 + 10^(-d/400))
_LOG10 = math.log(10)
_EPSILON = 1e-6  # clamp to avoid log(0)
_CLAMP_DIFF = 800.0  # max returnable Elo difference


def expected_score(elo_diff: float) -> float:
    """Expected score for the engine when it has `elo_diff` Elo advantage.

    Args:
        elo_diff: Elo rating of engine minus Elo rating of opponent.

    Returns:
        Expected score in [0, 1] (0.5 = equal).
    """
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))


def elo_diff(score: float, games: int = 1) -> float:  # noqa: ARG001 (games kept for API clarity)
    """Compute Elo difference from observed match score using logistic model.

    Args:
        score: Observed score fraction (wins + 0.5*draws) / total_games, in [0, 1].
        games: Number of games played (unused in logistic formula but kept for
               future BayesElo / LOS extensions).

    Returns:
        Elo difference (engine Elo minus opponent Elo). Clamped to ±_CLAMP_DIFF
        at extreme scores to avoid infinity.
    """
    # Clamp score away from 0 and 1 to avoid log(0)
    s = max(_EPSILON, min(1.0 - _EPSILON, score))
    # Logistic inverse: d = 400 * log10(s / (1 - s))
    diff = 400.0 * math.log10(s / (1.0 - s))
    return max(-_CLAMP_DIFF, min(_CLAMP_DIFF, diff))


def gauntlet_elo(
    results: List[Tuple[float, int, float]],
) -> float:
    """Estimate engine Elo from gauntlet results against opponents at known Elos.

    Uses weighted average of per-opponent Elo estimates (weighted by game count).

    Args:
        results: List of (score, games, opponent_elo) tuples.
                 score is the fraction in [0,1]; games is total games played;
                 opponent_elo is the known Elo of that opponent (default 0 if
                 unknown, which returns relative Elo difference).

    Returns:
        Estimated engine Elo (or Elo difference from a base of 0 if opponent
        Elos are all 0).
    """
    if not results:
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0
    for score, games, opponent_elo in results:
        weight = float(games)
        diff = elo_diff(score, games=games)
        engine_elo = opponent_elo + diff
        weighted_sum += weight * engine_elo
        total_weight += weight

    if total_weight == 0.0:
        return 0.0
    return weighted_sum / total_weight
