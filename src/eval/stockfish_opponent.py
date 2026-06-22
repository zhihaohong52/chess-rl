# src/eval/stockfish_opponent.py
"""Stockfish opponent wrapper for arena matches.

Gates on Stockfish binary availability. Use `stockfish_available()` before
constructing `StockfishOpponent` in scripts; unit tests that need the binary
are skipped when it is absent.
"""

from __future__ import annotations

import shutil
import sys
import os
from typing import Optional

import chess

# Paths to check for the Stockfish binary (same list as StockfishEvaluator)
_CANDIDATE_PATHS = [
    "stockfish",
    "/usr/local/bin/stockfish",
    "/usr/bin/stockfish",
    "/opt/homebrew/bin/stockfish",
    "/opt/local/bin/stockfish",
]


def stockfish_available() -> bool:
    """Return True if a Stockfish binary is findable on this system."""
    for candidate in _CANDIDATE_PATHS:
        if os.path.exists(candidate):
            return True
        found = shutil.which(candidate)
        if found:
            return True
    return False


class StockfishOpponent:
    """Move-producer backed by Stockfish at a fixed skill level and depth.

    This is a callable `(chess.Board) -> chess.Move` suitable for use as the
    `opponent` argument of `play_match()`.

    Skill level maps to UCI_Elo via Stockfish's UCI_LimitStrength / UCI_Elo
    options (1=800 Elo ladder rung, 20=full strength).

    Args:
        skill_level: Stockfish skill level 0–20.
        depth: Search depth cap (passed as `time_limit`-equivalent via depth).
        stockfish_path: Override binary path; auto-detected if None.

    Raises:
        RuntimeError: If Stockfish binary cannot be found.
    """

    # Approximate Elo for each skill level 0-20 (rough ladder for Elo reporting)
    SKILL_TO_ELO = {
        0: 800, 1: 900, 2: 1000, 3: 1100, 4: 1200,
        5: 1300, 6: 1400, 7: 1500, 8: 1600, 9: 1700,
        10: 1800, 11: 1900, 12: 2000, 13: 2100, 14: 2200,
        15: 2300, 16: 2400, 17: 2500, 18: 2600, 19: 2700,
        20: 3200,
    }

    def __init__(
        self,
        skill_level: int = 5,
        depth: int = 5,
        stockfish_path: Optional[str] = None,
    ) -> None:
        if not stockfish_available():
            raise RuntimeError(
                "Stockfish binary not found. Install Stockfish and ensure it is "
                "on PATH, or pass stockfish_path explicitly."
            )
        # Import here to avoid hard dependency at module load time
        # (StockfishEvaluator raises RuntimeError in __init__ if binary absent,
        # which we have already checked above).
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        ))))
        from src.training.stockfish_evaluator import StockfishEvaluator  # noqa: PLC0415

        self.skill_level = skill_level
        self.depth = depth
        # Clamp Elo to Stockfish's UCI_Elo minimum, which varies by build
        # (1320 on many, 1350 on SF 14.1). Start at 1320; if the engine rejects
        # it ("...to be at least N"), parse N and retry so any build works.
        target_elo = max(1320, self.SKILL_TO_ELO.get(skill_level, 1500))
        try:
            self._evaluator = StockfishEvaluator(
                stockfish_path=stockfish_path,
                depth=depth,
                limit_strength=True,
                elo=target_elo,
            )
        except Exception as exc:  # noqa: BLE001 - retry once at the engine's min/max
            import re  # noqa: PLC0415
            m_min = re.search(r"at least (\d+)", str(exc))
            m_max = re.search(r"at most (\d+)", str(exc))
            if m_min:
                target_elo = max(int(m_min.group(1)), target_elo)
            elif m_max:
                target_elo = min(int(m_max.group(1)), target_elo)
            else:
                raise
            self._evaluator = StockfishEvaluator(
                stockfish_path=stockfish_path,
                depth=depth,
                limit_strength=True,
                elo=target_elo,
            )
        # Verify the strength limit was actually applied; otherwise the ladder Elo
        # labels are meaningless (Stockfish would run at full strength silently).
        opts = self._evaluator.engine.options
        self.strength_limited = ("UCI_LimitStrength" in opts and "UCI_Elo" in opts)
        if not self.strength_limited:
            print(
                "warning: this Stockfish build lacks UCI_LimitStrength/UCI_Elo; "
                "arena Elo labels will be inaccurate (running near full strength).",
                file=sys.stderr,
            )

    @property
    def approximate_elo(self) -> float:
        """Approximate Elo for this skill level (for gauntlet_elo reporting)."""
        return float(self.SKILL_TO_ELO.get(self.skill_level, 1500))

    def __call__(self, board: chess.Board) -> Optional[chess.Move]:
        """Return Stockfish's best move for the given position."""
        return self._evaluator.get_best_move(board)

    def close(self) -> None:
        """Shut down the Stockfish engine process."""
        self._evaluator.close()

    def __enter__(self) -> "StockfishOpponent":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
