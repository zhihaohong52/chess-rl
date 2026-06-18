#!/usr/bin/env python3
"""Download a slice of DeepMind ChessBench (action_value) + puzzles over HTTPS.

Files live under https://storage.googleapis.com/searchless_chess/data/ .
Defaults to the 141 MB test action_value bag (a good first run). The full train
set is 2148 shards x ~1.2 GB (~1.1 TB) -- fetch only what you need.

Usage:
  python scripts/download_chessbench.py                 # test bag (141 MB)
  python scripts/download_chessbench.py --puzzles       # + puzzles.csv (4.5 MB)
  python scripts/download_chessbench.py --train-shards 1  # first train shard (~1.2 GB)
"""

import argparse
import os
import shutil
import subprocess

BASE = "https://storage.googleapis.com/searchless_chess/data"


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"exists, skip: {dest}")
        return
    print(f"downloading {url}\n  -> {dest}")
    # Prefer system curl: the python.org macOS build often lacks a CA bundle, so
    # urllib raises CERTIFICATE_VERIFY_FAILED; curl uses the OS cert store.
    if shutil.which("curl"):
        rc = subprocess.call(
            ["curl", "-fL", "--retry", "3", "--create-dirs", "-o", dest, url]
        )
        if rc != 0:
            raise RuntimeError(f"curl failed ({rc}) for {url}")
        return
    import ssl
    import urllib.request
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(url, context=ctx) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--test", action="store_true",
                    help="download the 141MB test action_value bag (default)")
    ap.add_argument("--puzzles", action="store_true",
                    help="download puzzles.csv (4.5MB)")
    ap.add_argument("--train-shards", type=int, default=0,
                    help="download the first N train action_value shards (~1.2GB each)")
    args = ap.parse_args()

    do_test = args.test or (not args.puzzles and args.train_shards == 0)
    if do_test:
        _download(f"{BASE}/test/action_value_data.bag",
                  os.path.join(args.out_dir, "test", "action_value_data.bag"))
    if args.puzzles:
        _download(f"{BASE}/puzzles.csv", os.path.join(args.out_dir, "puzzles.csv"))
    for i in range(args.train_shards):
        name = f"action_value-{i:05d}-of-02148_data.bag"
        _download(f"{BASE}/train/{name}", os.path.join(args.out_dir, "train", name))
    print("done.")


if __name__ == "__main__":
    main()
