import chess
from config import Config
from src.model.transformer import ChessTransformer
from src.model.evaluator import TransformerEvaluator
from src.mcts.batched_mcts import BatchedMCTS


def _mcts(sims):
    ev = TransformerEvaluator(ChessTransformer(Config))
    return BatchedMCTS(ev, Config, num_simulations=sims, batch_size=8)


def test_never_returns_illegal_move():
    mcts = _mcts(16)
    for fen in [chess.STARTING_FEN,
                "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1"]:
        b = chess.Board(fen)
        mv = mcts.choose_move(b, temperature=0.0)
        mcts.reset()
        assert mv in b.legal_moves


def test_mate_in_one_is_found_and_value_positive():
    # White: Re1-e8 is checkmate (back rank).
    b = chess.Board("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1")
    mcts = _mcts(400)
    root = mcts.search(b)
    mv = mcts.choose_move(b, temperature=0.0)
    assert mv == chess.Move.from_uci("e1e8")
    assert root.value > 0.0  # winning for side to move
