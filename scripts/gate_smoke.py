#!/usr/bin/env python3
"""Go/no-go gate: confirm the loss decreases on a synthetic batch before paid training."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import chess
from config import Config
from src.model.transformer import ChessTransformer
from src.game.token_encoder import encode_batch
from src.game.move_encoder import get_move_encoder
from src.training.distill_trainer import DistillTrainer


def run_smoke_gate(steps: int = 30):
    P = get_move_encoder().policy_size
    sq, sf = encode_batch([chess.Board(), chess.Board()], [0, 0])
    e2e4 = get_move_encoder().encode(chess.Move.from_uci("e2e4"))
    pol = np.zeros((2, P), dtype=np.float32)
    pol[:, e2e4] = 1.0
    wdl = np.array([[1.0, 0.0, 0.0]] * 2, dtype=np.float32)
    ml = np.array([[40.0]] * 2, dtype=np.float32)

    inputs = (
        torch.tensor(sq, dtype=torch.long),
        torch.tensor(sf, dtype=torch.float32),
    )
    targets = (
        torch.tensor(pol, dtype=torch.float32),
        torch.tensor(wdl, dtype=torch.float32),
        torch.tensor(ml, dtype=torch.float32),
    )

    trainer = DistillTrainer(ChessTransformer(Config), Config)
    initial = trainer.train_step(inputs, targets)[0]
    final = initial
    for _ in range(steps):
        final = trainer.train_step(inputs, targets)[0]
    return final < initial, {"initial_loss": initial, "final_loss": final}


if __name__ == "__main__":
    ok, info = run_smoke_gate()
    print(info)
    sys.exit(0 if ok else 1)
