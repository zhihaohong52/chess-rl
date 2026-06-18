#!/usr/bin/env python3
"""Pre-encode ChessBench action_value .bag(s) into action-value npz shards.

Unlike scripts/preencode.py (policy-distribution targets, requires grouped data),
this streams pointwise (state, sampled action, win%) samples — the native form of
the ChessBench train shards.

Usage:
  python scripts/preencode_av.py --input data/train/action_value-00000-of-02148_data.bag \
      --out-dir data/shards_av --max-positions 2000000 --val-fraction 0.01 \
      --shard-size 100000
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.chessbench import iter_chessbench_actionvalue
from src.data.preencode import write_av_shard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, nargs="+",
                    help="one or more action_value .bag files")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-size", type=int, default=100000)
    ap.add_argument("--max-positions", type=int, default=0,
                    help="cap total samples emitted (0 = all)")
    ap.add_argument("--val-fraction", type=float, default=0.0,
                    help="deterministic fraction routed to val_* shards (0 = train only)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cap = args.max_positions or None
    src = iter_chessbench_actionvalue(args.input, max_positions=cap)

    val_every = int(round(1.0 / args.val_fraction)) if args.val_fraction > 0 else 0
    buffers = {"train": [], "val": []}
    idx = {"train": 0, "val": 0}
    counts = {"train": 0, "val": 0}

    def flush(split):
        if not buffers[split]:
            return
        path = os.path.join(args.out_dir, f"{split}_{idx[split]:05d}.npz")
        counts[split] += write_av_shard(buffers[split], path)
        print(f"wrote {path} ({counts[split]} {split})", flush=True)
        idx[split] += 1
        buffers[split] = []

    for i, sample in enumerate(src):
        split = "val" if (val_every and i % val_every == 0) else "train"
        buffers[split].append(sample)
        if len(buffers[split]) >= args.shard_size:
            flush(split)
    flush("train")
    flush("val")
    print(f"done: {counts['train']} train, {counts['val']} val samples", flush=True)


if __name__ == "__main__":
    main()
