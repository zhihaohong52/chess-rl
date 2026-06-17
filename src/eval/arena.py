# src/eval/arena.py
"""Arena: play a match between two move-producer callables and return a score."""

from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import chess


# A move-producer is any callable that receives a chess.Board and returns a
# chess.Move (or None to resign).
MoveProducer = Callable[[chess.Board], Optional[chess.Move]]


@dataclasses.dataclass
class MatchResult:
    """Aggregated result of a multi-game match from the engine's perspective."""

    wins: int
    draws: int
    losses: int
    total: int

    @property
    def score(self) -> float:
        """Fraction score: (wins + 0.5*draws) / total. Returns 0.5 if total=0."""
        if self.total == 0:
            return 0.5
        return (self.wins + 0.5 * self.draws) / self.total


def _play_single_game(
    white: MoveProducer,
    black: MoveProducer,
    max_moves: int,
) -> float:
    """Play one game; return result from White's perspective: 1.0/0.5/0.0."""
    board = chess.Board()
    move_count = 0

    while not board.is_game_over() and move_count < max_moves:
        mover = white if board.turn == chess.WHITE else black
        move = mover(board)
        if move is None:
            # Resignation
            return 0.0 if board.turn == chess.WHITE else 1.0
        if move not in board.legal_moves:
            # Illegal move counts as resignation
            return 0.0 if board.turn == chess.WHITE else 1.0
        board.push(move)
        move_count += 1

    if not board.is_game_over():
        # max_moves reached — count as draw
        return 0.5

    outcome = board.outcome()
    if outcome is None:
        return 0.5
    if outcome.winner is None:
        return 0.5
    return 1.0 if outcome.winner == chess.WHITE else 0.0


def play_match(
    engine: MoveProducer,
    opponent: MoveProducer,
    num_games: int = 10,
    max_moves: int = 300,
) -> MatchResult:
    """Play `num_games` games, alternating colours each game.

    The engine plays White in even-indexed games (0, 2, …) and Black in
    odd-indexed games (1, 3, …).

    Args:
        engine: The move-producer under evaluation.
        opponent: The move-producer it plays against.
        num_games: Total number of games to play.
        max_moves: Hard cap on plies per game; game is counted as draw if hit.

    Returns:
        MatchResult with wins/draws/losses from the engine's perspective.
    """
    wins = draws = losses = 0

    for game_idx in range(num_games):
        engine_is_white = (game_idx % 2 == 0)
        white = engine if engine_is_white else opponent
        black = opponent if engine_is_white else engine

        white_score = _play_single_game(white, black, max_moves)

        engine_score = white_score if engine_is_white else (1.0 - white_score)

        if engine_score == 1.0:
            wins += 1
        elif engine_score == 0.5:
            draws += 1
        else:
            losses += 1

    return MatchResult(wins=wins, draws=draws, losses=losses, total=num_games)
