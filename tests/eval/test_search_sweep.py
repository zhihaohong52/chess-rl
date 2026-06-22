import chess

from config import Config
from src.model.presets import build_model
from src.model.evaluator import TransformerEvaluator
from src.mcts.batched_mcts import BatchedMCTS
import scripts.search_sweep as ss


def _evaluator():
    net, _ = build_model("baseline-v1")
    return TransformerEvaluator(net, device="cpu")


def test_batchedmcts_c_puct_override_sticks():
    # mcts_mover overrides c_puct after construction; that must take effect
    # (BatchedMCTS reads self.c_puct at select_child time).
    m = BatchedMCTS(_evaluator(), Config, num_simulations=8)
    m.c_puct = 3.0
    assert m.c_puct == 3.0


def test_mcts_mover_returns_legal_move_at_custom_c_puct():
    mover = ss.mcts_mover(_evaluator(), sims=8, c_puct=3.0)
    move = mover(chess.Board())
    assert move in chess.Board().legal_moves
