import chess
from uci import UCIEngine


def test_uci_engine_returns_legal_bestmove():
    eng = UCIEngine(model_path=None, num_simulations=16)
    eng.position(["startpos"])
    uci_move = eng.get_best_move_uci()
    assert uci_move != "0000"
    assert chess.Move.from_uci(uci_move) in chess.Board().legal_moves


def test_uci_position_with_moves_then_bestmove():
    eng = UCIEngine(model_path=None, num_simulations=16)
    eng.position(["startpos", "moves", "e2e4", "e7e5"])
    uci_move = eng.get_best_move_uci()
    board = chess.Board(); board.push_uci("e2e4"); board.push_uci("e7e5")
    assert chess.Move.from_uci(uci_move) in board.legal_moves
