import os, numpy as np, torch, chess
from src.model.presets import resolve_config
from src.model.transformer import ChessTransformer
from src.training.distill_trainer import DistillTrainer
from src.training.checkpoint_meta import read_sidecar
from src.game.token_encoder import encode_batch
from src.game.move_encoder import get_move_encoder


def _batch():
    P = get_move_encoder().policy_size
    sq, sf = encode_batch([chess.Board(), chess.Board()], [0, 0])
    pol = np.zeros((2, P), dtype=np.float32); pol[:, 0] = 1.0
    wdl = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32)
    ml = np.array([[40.0], [40.0]], dtype=np.float32)
    return ((torch.tensor(sq, dtype=torch.long), torch.tensor(sf, dtype=torch.float32)),
            (torch.tensor(pol), torch.tensor(wdl), torch.tensor(ml)))


def test_ema_saves_ema_checkpoint(tmp_path):
    cfg = resolve_config("baseline-v1")
    cfg.ema_decay = 0.9
    net = ChessTransformer(cfg)
    trainer = DistillTrainer(net, cfg, device="cpu")
    loader = [_batch()]
    ckpt = str(tmp_path / "ck")
    trainer.fit(loader, steps=2, val_loader=loader, val_every=1, ckpt_dir=ckpt,
                meta={"preset": "baseline-v1", "train_data": "unit"})
    assert os.path.exists(os.path.join(ckpt, "best_ema.pt"))
    assert read_sidecar(os.path.join(ckpt, "best_ema.pt"))["objective"] == "policy"
