import time
import pytest
import chess
from config import Config
from src.model.transformer import ChessTransformer
from src.model.evaluator import TransformerEvaluator
from src.mcts.batched_mcts import BatchedMCTS


@pytest.mark.slow
def test_sims_per_second():
    mcts = BatchedMCTS(TransformerEvaluator(ChessTransformer(Config)),
                       Config, num_simulations=256, batch_size=16)
    b = chess.Board()
    mcts.search(b)  # warmup (graph trace)
    mcts.reset()
    t0 = time.time()
    mcts.search(b)
    sps = 256 / (time.time() - t0)
    print(f"MCTS throughput: {sps:.0f} sims/sec")
    assert sps > 50  # minimum for playable blitz on M1; gate #4
