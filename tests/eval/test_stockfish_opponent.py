# tests/eval/test_stockfish_opponent.py
"""Tests for StockfishOpponent — gated on binary availability."""
import pytest
import chess
from src.eval.stockfish_opponent import StockfishOpponent, stockfish_available


def test_stockfish_available_returns_bool():
    result = stockfish_available()
    assert isinstance(result, bool)


@pytest.mark.skipif(not stockfish_available(), reason="Stockfish binary not found")
def test_stockfish_opponent_returns_legal_move():
    opp = StockfishOpponent(skill_level=1, depth=1)
    try:
        board = chess.Board()
        move = opp(board)
        assert move in board.legal_moves
    finally:
        opp.close()


@pytest.mark.skipif(not stockfish_available(), reason="Stockfish binary not found")
def test_stockfish_opponent_context_manager():
    with StockfishOpponent(skill_level=5, depth=2) as opp:
        board = chess.Board()
        move = opp(board)
        assert move in board.legal_moves


def test_stockfish_opponent_unavailable_raises(monkeypatch):
    """When binary not found, constructing StockfishOpponent raises RuntimeError."""
    # Patch stockfish_available to False
    import src.eval.stockfish_opponent as module
    monkeypatch.setattr(module, "stockfish_available", lambda: False)
    with pytest.raises(RuntimeError, match="Stockfish"):
        StockfishOpponent(skill_level=1, depth=1)
