#!/usr/bin/env python3
"""Pre-encode a source dataset into npz shards (with an optional train/val split).

Usage:
  python scripts/preencode.py --source chessbench \
      --input data/test/action_value_data.bag --out-dir data/shards \
      --max-positions 20000 --val-fraction 0.1 --shard-size 5000
"""
import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.chessbench import iter_kaggle_csv, iter_chessbench
from src.data.preencode import write_shard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["kaggle", "chessbench"], required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-size", type=int, default=100000)
    ap.add_argument("--max-positions", type=int, default=0,
                    help="cap total positions emitted (0 = all)")
    ap.add_argument("--val-fraction", type=float, default=0.0,
                    help="deterministic fraction routed to val_* shards (0 = train only)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    src = iter_kaggle_csv(args.input) if args.source == "kaggle" else iter_chessbench(args.input)
    if args.max_positions:
        src = itertools.islice(src, args.max_positions)

    val_every = int(round(1.0 / args.val_fraction)) if args.val_fraction > 0 else 0
    buffers = {"train": [], "val": []}
    idx = {"train": 0, "val": 0}
    counts = {"train": 0, "val": 0}

    def flush(split):
        if not buffers[split]:
            return
        path = os.path.join(args.out_dir, f"{split}_{idx[split]:05d}.npz")
        counts[split] += write_shard(buffers[split], path)
        print(f"wrote {path} ({counts[split]} {split})")
        idx[split] += 1
        buffers[split] = []

    for i, lp in enumerate(src):
        split = "val" if (val_every and i % val_every == 0) else "train"
        buffers[split].append(lp)
        if len(buffers[split]) >= args.shard_size:
            flush(split)
    flush("train")
    flush("val")
    print(f"done: {counts['train']} train, {counts['val']} val positions")


if __name__ == "__main__":
    main()
