# tests/eval/test_arena.py
"""Tests for the arena match runner."""
import chess
import pytest
from src.eval.arena import play_match, MatchResult


# ---------------------------------------------------------------------------
# Deterministic stub move-producers
# ---------------------------------------------------------------------------

def _always_first_legal(board: chess.Board) -> chess.Move:
    """Always picks the first legal move (sorted by UCI for determinism)."""
    return sorted(board.legal_moves, key=lambda m: m.uci())[0]


def _always_last_legal(board: chess.Board) -> chess.Move:
    """Always picks the last legal move (sorted by UCI for determinism)."""
    return sorted(board.legal_moves, key=lambda m: m.uci())[-1]


def _resign_immediately(board: chess.Board) -> chess.Move:
    """Returns None to signal resignation."""
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_match_result_fields():
    r = MatchResult(wins=3, draws=1, losses=1, total=5)
    assert r.wins == 3
    assert r.draws == 1
    assert r.losses == 1
    assert r.total == 5


def test_match_result_score():
    r = MatchResult(wins=3, draws=2, losses=1, total=6)
    expected = (3 + 0.5 * 2) / 6
    assert r.score == pytest.approx(expected, abs=1e-9)


def test_match_result_score_zero_games():
    r = MatchResult(wins=0, draws=0, losses=0, total=0)
    assert r.score == pytest.approx(0.5, abs=1e-9)


def test_arena_runs_and_returns_result():
    """A short match between two deterministic bots returns a valid MatchResult."""
    result = play_match(
        engine=_always_first_legal,
        opponent=_always_last_legal,
        num_games=2,
        max_moves=50,
    )
    assert isinstance(result, MatchResult)
    assert result.total == 2
    assert result.wins + result.draws + result.losses == 2
    assert 0.0 <= result.score <= 1.0


def test_arena_alternates_colours():
    """With 2 games the engine plays White in game 0 and Black in game 1."""
    colours_seen: list = []

    def recording_engine(board: chess.Board) -> chess.Move:
        colours_seen.append(board.turn)  # True=White
        return sorted(board.legal_moves, key=lambda m: m.uci())[0]

    play_match(
        engine=recording_engine,
        opponent=_always_last_legal,
        num_games=2,
        max_moves=2,
    )
    # Engine should have seen at least one White turn and one Black turn
    assert chess.WHITE in colours_seen
    assert chess.BLACK in colours_seen


def test_arena_max_moves_terminates():
    """max_moves prevents infinite games; result is counted as draw."""
    result = play_match(
        engine=_always_first_legal,
        opponent=_always_first_legal,
        num_games=4,
        max_moves=10,
    )
    assert result.total == 4


def test_arena_score_in_valid_range():
    result = play_match(
        engine=_always_first_legal,
        opponent=_always_last_legal,
        num_games=4,
        max_moves=30,
    )
    assert 0.0 <= result.score <= 1.0
