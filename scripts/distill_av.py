#!/usr/bin/env python3
"""Train ChessTransformer with the action-value objective (BCE on sampled move).

Usage:
  python scripts/distill_av.py --shards-dir data/shards_av --steps 3000 \
      --batch-size 512 --val-every 300 --ckpt-dir checkpoints/distill_av
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import Config
from src.model.transformer import ChessTransformer
from src.training.distill_trainer import DistillTrainer
from src.data.dataset import make_av_dataloader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-dir", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--val-every", type=int, default=300)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ckpt-dir", default="checkpoints/distill_av")
    ap.add_argument("--resume", default=None, help="optional .pt to warm-start from")
    ap.add_argument("--max-train-shards", type=int, default=0,
                    help="cap number of train shards loaded into RAM (0 = all)")
    args = ap.parse_args()

    cfg = Config()
    cfg.distill_lr = args.lr
    cfg.distill_warmup_steps = args.warmup
    cfg.distill_total_steps = args.steps

    train_shards = sorted(glob.glob(os.path.join(args.shards_dir, "train_*.npz")))
    val_shards = sorted(glob.glob(os.path.join(args.shards_dir, "val_*.npz")))
    if args.max_train_shards:
        train_shards = train_shards[:args.max_train_shards]
    if not train_shards:
        print(f"no train shards in {args.shards_dir}", file=sys.stderr)
        return 1
    print(f"train shards: {len(train_shards)}  val shards: {len(val_shards)}", flush=True)

    net = ChessTransformer(cfg)
    if args.resume:
        net.load_state_dict(torch.load(args.resume, map_location="cpu"))
        print(f"warm-started from {args.resume}", flush=True)

    trainer = DistillTrainer(net, cfg)
    print(f"device: {trainer.device}", flush=True)

    train_loader = make_av_dataloader(train_shards, batch_size=args.batch_size, shuffle=True)
    val_loader = (make_av_dataloader(val_shards, batch_size=args.batch_size, shuffle=False)
                  if val_shards else None)

    best = trainer.fit_av(
        train_loader, steps=args.steps, val_loader=val_loader,
        val_every=args.val_every, ckpt_dir=args.ckpt_dir, log_every=args.log_every,
    )
    print(f"done. best val_av_bce={best:.4f}  ckpts in {args.ckpt_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
