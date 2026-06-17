import chess
from config import Config
from src.model.transformer import ChessTransformer
from src.model.evaluator import TransformerEvaluator


def test_evaluate_legal_only_and_normalized():
    ev = TransformerEvaluator(ChessTransformer(Config))
    board = chess.Board()
    policy, value = ev.evaluate(board)
    legal = set(board.legal_moves)
    assert set(policy.keys()) == legal
    assert abs(sum(policy.values()) - 1.0) < 1e-4
    assert -1.0 <= value <= 1.0


def test_evaluate_batch_matches_single():
    ev = TransformerEvaluator(ChessTransformer(Config))
    b1 = chess.Board()
    b2 = chess.Board(); b2.push_uci("e2e4")
    out = ev.evaluate_batch([b1, b2], [0, 0])
    assert len(out) == 2
    for (policy, value), b in zip(out, [b1, b2]):
        assert set(policy.keys()) == set(b.legal_moves)
        assert -1.0 <= value <= 1.0
