import numpy as np
import torch
import chess

from config import Config
from src.model.transformer import ChessTransformer
from src.game.token_encoder import encode_batch
from src.game.move_encoder import get_move_encoder
from src.game.orientation import to_canonical_move
from src.training.distill_trainer import DistillTrainer


def _av_batch():
    me = get_move_encoder()
    boards = [chess.Board(), chess.Board()]
    sq, sf = encode_batch(boards, [0, 0])
    a = me.encode(to_canonical_move(chess.Move.from_uci("e2e4"), chess.WHITE))
    inputs = (torch.tensor(sq, dtype=torch.long), torch.tensor(sf, dtype=torch.float32))
    targets = (
        torch.tensor([a, a], dtype=torch.long),
        torch.tensor([0.9, 0.9], dtype=torch.float32),
    )
    return inputs, targets


def test_train_step_av_decreases_loss():
    net = ChessTransformer(Config)
    trainer = DistillTrainer(net, Config)
    inputs, targets = _av_batch()
    first = trainer.train_step_av(inputs, targets)
    assert np.isfinite(first)
    for _ in range(30):
        last = trainer.train_step_av(inputs, targets)
    assert last < first * 0.9


def test_evaluate_av_keys():
    net = ChessTransformer(Config)
    trainer = DistillTrainer(net, Config)
    inputs, targets = _av_batch()
    loader = [(inputs, targets)]
    metrics = trainer.evaluate_av(loader, max_batches=1)
    assert set(metrics) == {"val_av_bce", "val_av_mae"}
    assert np.isfinite(metrics["val_av_bce"]) and np.isfinite(metrics["val_av_mae"])
