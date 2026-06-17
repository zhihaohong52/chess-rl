#!/usr/bin/env python3
"""Pre-encode a source dataset into npz shards.

Usage: python scripts/preencode.py --source kaggle --input data/chessData.csv \
       --out-dir data/shards --shard-size 100000
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.chessbench import iter_kaggle_csv, iter_chessbench
from src.data.preencode import write_shard


def _chunked(it, size):
    chunk = []
    for x in it:
        chunk.append(x)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["kaggle", "chessbench"], required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-size", type=int, default=100000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    src = iter_kaggle_csv(args.input) if args.source == "kaggle" else iter_chessbench(args.input)
    total = 0
    for i, chunk in enumerate(_chunked(src, args.shard_size)):
        path = os.path.join(args.out_dir, f"shard_{i:05d}.npz")
        total += write_shard(chunk, path)
        print(f"wrote {path} ({total} positions)")
    print(f"done: {total} positions")


if __name__ == "__main__":
    main()
