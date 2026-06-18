import os
import chess
import pytest

from src.eval.fixtures import mate_in_one_positions, load_gate_puzzles, PUZZLE_CSV


def test_mate_in_one_positions_are_legal_and_correct():
    positions = mate_in_one_positions()
    assert len(positions) >= 3
    for fen, uci in positions:
        board = chess.Board(fen)
        mv = chess.Move.from_uci(uci)
        assert mv in board.legal_moves
        board.push(mv)
        assert board.is_checkmate()


@pytest.mark.skipif(not os.path.exists(PUZZLE_CSV),
                    reason="puzzles.csv not present locally")
def test_load_gate_puzzles_returns_requested_count():
    pz = load_gate_puzzles(50)
    assert len(pz) == 50
