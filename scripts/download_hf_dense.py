#!/usr/bin/env python3
"""Download raw dense shards from prdev/chessbench-full-policy-value (HF).

Files are `train-NNNNN-of-01024.msgpack.zst`. Run on the GPU box (fast network):

  python scripts/download_hf_dense.py --start 0 --count 29 --out-dir data/raw_hf

Verify one shard first (the dataset's exact filename pattern can change):

  python scripts/download_hf_dense.py --start 0 --count 1 --out-dir data/raw_hf
"""
import argparse
import os
import subprocess

REPO = "prdev/chessbench-full-policy-value"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
NUM_SHARDS = 1024


def shard_filename(idx: int) -> str:
    return f"train-{idx:05d}-of-{NUM_SHARDS:05d}.msgpack.zst"


def shard_url(idx: int) -> str:
    return f"{BASE}/{shard_filename(idx)}"


def download_one(idx: int, out_dir: str) -> str:
    dest = os.path.join(out_dir, shard_filename(idx))
    if os.path.exists(dest):
        print(f"exists, skipping {dest}", flush=True)
        return dest
    url = shard_url(idx)
    tmp = dest + ".part"
    print(f"downloading {url}", flush=True)
    rc = subprocess.call(
        ["curl", "-fL", "--retry", "3", "--create-dirs", "-o", tmp, url]
    )
    if rc != 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"curl failed ({rc}) for {url}")
    os.replace(tmp, dest)  # atomic; a partial .part is never seen as a finished shard
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--out-dir", default="data/raw_hf")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    indices = range(args.start, min(args.start + args.count, NUM_SHARDS))
    for idx in indices:
        download_one(idx, args.out_dir)
    print(f"done: {len(indices)} shard(s) into {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
