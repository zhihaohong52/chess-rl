import numpy as np
import torch
import chess

from config import Config
from src.model.transformer import ChessTransformer
from src.game.token_encoder import encode_batch
from src.game.move_encoder import get_move_encoder
from src.game.orientation import to_canonical_move
from src.training.distill_trainer import DistillTrainer
from src.training.checkpoint_meta import read_sidecar


def _av_batch():
    me = get_move_encoder()
    boards = [chess.Board(), chess.Board()]
    sq, sf = encode_batch(boards, [0, 0])
    a = me.encode(to_canonical_move(chess.Move.from_uci("e2e4"), chess.WHITE))
    inputs = (torch.tensor(sq, dtype=torch.long), torch.tensor(sf, dtype=torch.float32))
    targets = (torch.tensor([a, a], dtype=torch.long),
               torch.tensor([0.9, 0.9], dtype=torch.float32))
    return inputs, targets


def test_fit_av_writes_sidecar(tmp_path):
    net = ChessTransformer(Config)
    trainer = DistillTrainer(net, Config, device="cpu")
    loader = [_av_batch()]
    ckpt_dir = str(tmp_path / "ck")
    trainer.fit_av(loader, steps=1, val_loader=loader, val_every=1,
                   ckpt_dir=ckpt_dir, meta={"preset": "baseline-v1",
                                            "train_data": "unit"})
    meta = read_sidecar(str(tmp_path / "ck" / "last.pt"))
    assert meta is not None
    assert meta["objective"] == "action_value"
    assert meta["preset"] == "baseline-v1"
    best_meta = read_sidecar(str(tmp_path / "ck" / "best.pt"))
    assert best_meta is not None and best_meta["objective"] == "action_value"


def _policy_batch():
    me = get_move_encoder()
    boards = [chess.Board(), chess.Board()]
    sq, sf = encode_batch(boards, [0, 0])
    P = me.policy_size
    e2e4 = me.encode(to_canonical_move(chess.Move.from_uci("e2e4"), chess.WHITE))
    pol = np.zeros((2, P), dtype=np.float32); pol[:, e2e4] = 1.0
    wdl = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    ml = np.array([[40.0], [40.0]], dtype=np.float32)
    inputs = (torch.tensor(sq, dtype=torch.long), torch.tensor(sf, dtype=torch.float32))
    targets = (torch.tensor(pol), torch.tensor(wdl), torch.tensor(ml))
    return inputs, targets


def test_fit_writes_policy_sidecar(tmp_path):
    net = ChessTransformer(Config)
    trainer = DistillTrainer(net, Config, device="cpu")
    loader = [_policy_batch()]
    ckpt_dir = str(tmp_path / "ck")
    trainer.fit(loader, steps=1, val_loader=loader, val_every=1,
                ckpt_dir=ckpt_dir, meta={"preset": "baseline-v1", "train_data": "unit"})
    meta = read_sidecar(str(tmp_path / "ck" / "last.pt"))
    assert meta is not None and meta["objective"] == "policy"
