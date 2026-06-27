#!/usr/bin/env python3
"""Pre-encode a source dataset into npz shards (optional train/val split, parallel).

Usage (single file or glob):
  python scripts/preencode.py --source hf_dense \
      --input 'data/raw_hf/train-*.msgpack.zst' --out-dir data/shards_hf_dense \
      --temperature 0.1 --val-fraction 0.02 --workers 8

With --workers > 1 and a glob matching multiple files, each input file is encoded
by its own process (read + encode + write end-to-end in parallel), which is the
right granularity because each zstd+msgpack stream is inherently sequential.
In that mode --max-positions caps positions PER FILE. With a single file or
--workers 1, all inputs stream through one process and --max-positions caps the
total. Shards are written uncompressed by default (~2x faster end-to-end, ~2x
larger, read-compatible); pass --compress when the shards will be uploaded.
"""
import argparse
import glob
import itertools
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.chessbench import iter_kaggle_csv, iter_chessbench, iter_hf_dense
from src.data.preencode import write_shard


def _make_source(source, path, temperature):
    if source == "kaggle":
        return iter_kaggle_csv(path)
    if source == "hf_dense":
        return iter_hf_dense(path, temperature=temperature)
    return iter_chessbench(path, temperature=temperature)


def encode_stream(src, out_dir, prefix, shard_size, val_fraction, max_positions,
                  compress):
    """Encode a LabeledPosition stream into npz shards. Returns (train, val) counts.

    Shards are named ``{split}_{prefix}{idx:05d}.npz``; the default empty prefix
    reproduces the historical ``train_00000.npz`` layout. Positions are routed to
    a deterministic 1-in-N validation split when ``val_fraction > 0``.
    """
    if max_positions:
        src = itertools.islice(src, max_positions)
    val_every = int(round(1.0 / val_fraction)) if val_fraction > 0 else 0
    buffers = {"train": [], "val": []}
    idx = {"train": 0, "val": 0}
    counts = {"train": 0, "val": 0}

    def flush(split):
        if not buffers[split]:
            return
        path = os.path.join(out_dir, f"{split}_{prefix}{idx[split]:05d}.npz")
        counts[split] += write_shard(buffers[split], path, compress=compress)
        idx[split] += 1
        buffers[split] = []

    for i, lp in enumerate(src):
        split = "val" if (val_every and i % val_every == 0) else "train"
        buffers[split].append(lp)
        if len(buffers[split]) >= shard_size:
            flush(split)
    flush("train")
    flush("val")
    return counts["train"], counts["val"]


def _worker(task):
    """Encode one input file into its own (namespaced) shards. Picklable for Pool."""
    (source, path, file_idx, out_dir, shard_size, val_fraction, temperature,
     max_positions, compress) = task
    src = _make_source(source, path, temperature)
    return encode_stream(src, out_dir, f"{file_idx:02d}_", shard_size,
                         val_fraction, max_positions, compress)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["kaggle", "chessbench", "hf_dense"], required=True)
    ap.add_argument("--input", required=True, help="file path or glob")
    ap.add_argument("--temperature", type=float, default=0.1,
                    help="policy softmax temperature for chessbench/hf_dense sources")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-size", type=int, default=100000)
    ap.add_argument("--max-positions", type=int, default=0,
                    help="cap positions (total in single-process; per-file with --workers>1)")
    ap.add_argument("--val-fraction", type=float, default=0.0,
                    help="deterministic fraction routed to val_* shards (0 = train only)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel processes; effective only with a multi-file glob")
    comp = ap.add_mutually_exclusive_group()
    comp.add_argument("--compress", dest="compress", action="store_true",
                      help="zlib-compress shards (~2x smaller, ~2x slower write)")
    comp.add_argument("--no-compress", dest="compress", action="store_false")
    ap.set_defaults(compress=False)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    inputs = sorted(glob.glob(args.input)) or [args.input]

    if args.workers > 1 and len(inputs) > 1:
        tasks = [
            (args.source, p, i, args.out_dir, args.shard_size, args.val_fraction,
             args.temperature, args.max_positions, args.compress)
            for i, p in enumerate(inputs)
        ]
        with Pool(min(args.workers, len(tasks))) as pool:
            results = pool.map(_worker, tasks)
        train = sum(r[0] for r in results)
        val = sum(r[1] for r in results)
    else:
        src = itertools.chain.from_iterable(
            _make_source(args.source, p, args.temperature) for p in inputs
        )
        train, val = encode_stream(src, args.out_dir, "", args.shard_size,
                                   args.val_fraction, args.max_positions, args.compress)

    print(f"done: {train} train, {val} val positions across {len(inputs)} file(s)")


if __name__ == "__main__":
    main()
