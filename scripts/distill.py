#!/usr/bin/env python3
"""Train ChessTransformer by distillation on NPZ shards (PyTorch).

Usage:
    python scripts/distill.py --train 'data/shards_dense_d10_mpv16/train_*.npz' \
           --val 'data/shards_dense_d10_mpv16/val_*.npz' --preset baseline-v1 \
           --train-data stockfish-d10-mpv16-50k --batch 128 --steps 3000 \
           --warmup 200 --lr 3e-4 --val-every 300 --ckpt checkpoints/dense_d10_50k
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.model.presets import resolve_config
from src.model.transformer import ChessTransformer
from src.game.move_encoder import get_move_encoder
from src.data.dataset import make_dataloader, make_stream_dataloader
from src.training.distill_trainer import DistillTrainer


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="glob for training shards (*.npz)")
    ap.add_argument("--val", default=None, help="glob for validation shards (*.npz)")
    ap.add_argument("--preset", default="baseline-v1", help="model preset name")
    ap.add_argument("--train-data", default="", help="free-text label stored in the sidecar")
    ap.add_argument("--steps", type=int, default=Config.distill_total_steps)
    ap.add_argument("--batch", type=int, default=Config.distill_batch_size)
    ap.add_argument("--warmup", type=int, default=Config.distill_warmup_steps)
    ap.add_argument("--lr", type=float, default=Config.distill_lr)
    ap.add_argument("--ckpt", default="checkpoints/distill")
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--mixed-precision", action="store_true")
    ap.add_argument("--ema-decay", type=float, default=0.0)
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (default: auto)")
    ap.add_argument("--value-loss-weight", type=float, default=None,
                    help="override value_loss_weight (default: preset/config value, "
                         "1.0); e.g. 0.3 down-weights the value head to free policy "
                         "capacity")
    ap.add_argument("--stream", action="store_true",
                    help="stream train shards one-at-a-time (bounded memory; "
                         "required for datasets too large to fit in RAM)")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader worker processes for the train loader")
    return ap


def resolve_run_config(args):
    """Resolve the preset Config and apply this run's CLI overrides: the LR
    schedule (lr/warmup/steps), EMA decay, and the optional value-loss-weight.
    Returns the mutated Config so callers (and tests) can inspect it."""
    cfg = resolve_config(args.preset)
    cfg.distill_lr = args.lr
    cfg.distill_warmup_steps = args.warmup
    cfg.distill_total_steps = args.steps
    cfg.ema_decay = args.ema_decay
    if args.value_loss_weight is not None:
        cfg.value_loss_weight = args.value_loss_weight
    return cfg


def main():
    args = build_parser().parse_args()
    cfg = resolve_run_config(args)

    P = get_move_encoder().policy_size
    train_shards = sorted(glob.glob(args.train))
    if args.stream:
        train_loader = make_stream_dataloader(
            train_shards, args.batch, P, shuffle=True, num_workers=args.num_workers
        )
    else:
        train_loader = make_dataloader(
            train_shards, args.batch, P, shuffle=True, num_workers=args.num_workers
        )
    # Val set is small; eager loading is fine (and evaluate() only reads 50 batches).
    val_loader = (
        make_dataloader(sorted(glob.glob(args.val)), args.batch, P, shuffle=False)
        if args.val
        else None
    )

    net = ChessTransformer(cfg)
    trainer = DistillTrainer(net, cfg, mixed_precision=args.mixed_precision,
                             device=args.device)
    best = trainer.fit(
        train_loader,
        args.steps,
        val_loader=val_loader,
        val_every=args.val_every,
        ckpt_dir=args.ckpt,
        meta={"preset": args.preset, "train_data": args.train_data},
    )
    print(f"done. best val_policy_loss={best:.4f}; checkpoints in {args.ckpt}")


if __name__ == "__main__":
    main()
