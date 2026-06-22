import chess

from src.model.presets import build_model
from src.model.evaluator import TransformerEvaluator
from src.engine.hybrid_mover import build_hybrid_mover
from scripts.arena_eval import build_mcts_mover


def _evaluator():
    net, _ = build_model("baseline-v1")
    return TransformerEvaluator(net, device="cpu")


class CountingEvaluator:
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def evaluate(self, board):
        self.calls += 1
        return self.inner.evaluate(board)

    def evaluate_batch(self, boards, idxs):
        self.calls += 1
        return self.inner.evaluate_batch(boards, idxs)


class FakeBook:
    def __init__(self, move):
        self.move = move

    def lookup(self, board, **kw):
        return self.move


class FakeTB:
    def __init__(self, dtz_move=None, value=None):
        self.dtz_move = dtz_move
        self.value = value

    def best_dtz_move(self, board):
        return self.dtz_move

    def probe_value(self, board):
        return self.value


def test_book_move_takes_priority():
    ev = CountingEvaluator(_evaluator())
    e2e4 = chess.Move.from_uci("e2e4")
    mover = build_hybrid_mover(ev, 8, book=FakeBook(e2e4),
                               tablebase=FakeTB(dtz_move=chess.Move.from_uci("d2d4")))
    assert mover(chess.Board()) == e2e4
    assert ev.calls == 0  # neither TB-root nor MCTS consulted


def test_tb_root_move_when_out_of_book():
    ev = CountingEvaluator(_evaluator())
    d2d4 = chess.Move.from_uci("d2d4")
    mover = build_hybrid_mover(ev, 8, book=FakeBook(None), tablebase=FakeTB(dtz_move=d2d4))
    assert mover(chess.Board()) == d2d4
    assert ev.calls == 0  # TB root short-circuits MCTS


def test_mcts_fallback_when_no_book_or_tb_hit():
    ev = CountingEvaluator(_evaluator())
    mover = build_hybrid_mover(ev, 8, book=FakeBook(None), tablebase=FakeTB(dtz_move=None))
    move = mover(chess.Board())
    assert move in chess.Board().legal_moves
    assert ev.calls > 0  # MCTS ran


def test_equivalent_to_plain_mcts_when_disabled():
    # book=None, tablebase=None -> identical move to build_mcts_mover (deterministic)
    ev = _evaluator()
    board = chess.Board()
    plain = build_mcts_mover(ev, 16)(board.copy())
    hybrid = build_hybrid_mover(ev, 16)(board.copy())
    assert plain == hybrid
