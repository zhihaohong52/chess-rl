import numpy as np
import torch
import chess
from config import Config
from src.model.transformer import ChessTransformer
from src.game.token_encoder import encode_batch
from src.game.move_encoder import get_move_encoder
from src.training.distill_trainer import DistillTrainer


def _tiny_batch():
    P = get_move_encoder().policy_size
    boards = [chess.Board(), chess.Board()]
    sq, sf = encode_batch(boards, [0, 0])
    e2e4 = get_move_encoder().encode(chess.Move.from_uci("e2e4"))
    pol = np.zeros((2, P), dtype=np.float32); pol[:, e2e4] = 1.0
    wdl = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    ml = np.array([[40.0], [40.0]], dtype=np.float32)
    inputs = (
        torch.tensor(sq, dtype=torch.long),
        torch.tensor(sf, dtype=torch.float32),
    )
    targets = (
        torch.tensor(pol, dtype=torch.float32),
        torch.tensor(wdl, dtype=torch.float32),
        torch.tensor(ml, dtype=torch.float32),
    )
    return inputs, targets


def test_train_step_decreases_loss():
    net = ChessTransformer(Config)
    trainer = DistillTrainer(net, Config)
    inputs, targets = _tiny_batch()
    first = trainer.train_step(inputs, targets)[0]
    for _ in range(30):
        last = trainer.train_step(inputs, targets)[0]
    assert last < first * 0.9
