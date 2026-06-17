#!/usr/bin/env python3
"""Train ChessTransformer by distillation on NPZ shards (PyTorch).

Usage:
    python scripts/distill.py --train 'data/shards/train_*.npz' \
           --val 'data/shards/val_*.npz' --steps 60000 --batch 1024 \
           --ckpt checkpoints/distill --mixed-precision
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.model.transformer import ChessTransformer
from src.game.move_encoder import get_move_encoder
from src.data.dataset import make_dataloader
from src.training.distill_trainer import DistillTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="glob for training shards (*.npz)")
    ap.add_argument("--val", default=None, help="glob for validation shards (*.npz)")
    ap.add_argument("--steps", type=int, default=Config.distill_total_steps)
    ap.add_argument("--batch", type=int, default=Config.distill_batch_size)
    ap.add_argument("--ckpt", default="checkpoints/distill")
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--mixed-precision", action="store_true")
    args = ap.parse_args()

    P = get_move_encoder().policy_size
    train_loader = make_dataloader(
        sorted(glob.glob(args.train)), args.batch, P, shuffle=True
    )
    val_loader = (
        make_dataloader(sorted(glob.glob(args.val)), args.batch, P, shuffle=False)
        if args.val
        else None
    )

    net = ChessTransformer(Config)
    trainer = DistillTrainer(net, Config, mixed_precision=args.mixed_precision)
    best = trainer.fit(
        train_loader,
        args.steps,
        val_loader=val_loader,
        val_every=args.val_every,
        ckpt_dir=args.ckpt,
    )
    print(f"done. best val_policy_loss={best:.4f}; checkpoints in {args.ckpt}")


if __name__ == "__main__":
    main()
