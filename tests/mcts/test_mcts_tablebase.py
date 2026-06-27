import chess

from config import Config
from src.model.presets import build_model
from src.model.evaluator import TransformerEvaluator
from src.mcts.batched_mcts import BatchedMCTS

KQK = "8/8/8/4k3/8/8/3QK3/8 w - - 0 1"  # 3 men, all children stay in TB scope


def _evaluator():
    net, _ = build_model("baseline-v1")
    return TransformerEvaluator(net, device="cpu")


class CountingEvaluator:
    """Wraps a real evaluator and counts how many leaf boards the NN batch-evaluates."""

    def __init__(self, inner):
        self.inner = inner
        self.batch_boards_seen = 0

    def evaluate(self, board):
        return self.inner.evaluate(board)

    def evaluate_batch(self, boards, idxs):
        self.batch_boards_seen += len(boards)
        return self.inner.evaluate_batch(boards, idxs)


class ScopedFakeTB:
    """Minimal tablebase stand-in: returns `value` for ≤max_pieces positions, else None."""

    def __init__(self, value, max_pieces=5):
        self.value = value
        self.max_pieces = max_pieces

    def probe_value(self, board):
        if chess.popcount(board.occupied) > self.max_pieces:
            return None
        return self.value


def test_tb_hit_skips_nn_and_backs_up_winning_sign():
    ev = CountingEvaluator(_evaluator())
    # every child leaf: side-to-move (opponent) is lost -> we are winning
    mcts = BatchedMCTS(ev, Config, num_simulations=32, tablebase=ScopedFakeTB(-1.0))
    root = mcts.search(chess.Board(KQK))
    assert ev.batch_boards_seen == 0   # all leaves TB-covered: NN never batch-evaluated
    assert root.value > 0              # negamax: opponent-lost leaves -> winning root


def test_tb_hit_backs_up_losing_sign():
    ev = CountingEvaluator(_evaluator())
    mcts = BatchedMCTS(ev, Config, num_simulations=32, tablebase=ScopedFakeTB(1.0))
    root = mcts.search(chess.Board(KQK))
    assert root.value < 0              # opponent-winning leaves -> losing root


def test_out_of_scope_falls_back_to_nn():
    ev = CountingEvaluator(_evaluator())
    # 32-man start position: out of TB scope -> probe returns None -> NN used
    mcts = BatchedMCTS(ev, Config, num_simulations=16, tablebase=ScopedFakeTB(-1.0))
    mcts.search(chess.Board())
    assert ev.batch_boards_seen > 0


def test_tablebase_none_uses_nn():
    ev = CountingEvaluator(_evaluator())
    mcts = BatchedMCTS(ev, Config, num_simulations=16, tablebase=None)
    mcts.search(chess.Board(KQK))
    assert ev.batch_boards_seen > 0    # default path unchanged: NN evaluates leaves
