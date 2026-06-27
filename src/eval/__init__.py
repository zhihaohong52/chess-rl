# src/eval/__init__.py
"""Evaluation harness: arena, puzzles, Elo utilities, Stockfish opponent."""

from .arena import play_match, MatchResult
from .elo import elo_diff, expected_score, gauntlet_elo
from .puzzles import load_puzzles, puzzle_accuracy, Puzzle
from .stockfish_opponent import StockfishOpponent, stockfish_available

__all__ = [
    "play_match",
    "MatchResult",
    "elo_diff",
    "expected_score",
    "gauntlet_elo",
    "load_puzzles",
    "puzzle_accuracy",
    "Puzzle",
    "StockfishOpponent",
    "stockfish_available",
]
