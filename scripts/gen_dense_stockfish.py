#!/usr/bin/env python3
"""Generate DENSE policy-distribution data with Stockfish multipv.

The ChessBench train shards are pointwise (1 move/FEN); dense per-position
distributions are far more sample-efficient for distillation (see the gauge:
dense 62k beat action-value 1.98M). DeepMind ships no dense train data, so we
regenerate it locally: for each FEN, Stockfish multipv evaluates all legal moves
-> win% per move -> the SAME LabeledPosition format the proven pipeline consumes
(_build_position). Output npz shards load via src.data.dataset.make_dataloader.

Parallelism: one single-threaded Stockfish process per worker. The work is
CPU-bound INSIDE Stockfish (a subprocess), so this scales across cores on stock
CPython regardless of the GIL.

Usage:
  python scripts/gen_dense_stockfish.py --fen-bag data/train/action_value-00000-of-02148_data.bag \
      --max-positions 100000 --depth 12 --workers 6 --out-dir data/shards_dense \
      --val-fraction 0.02 --shard-size 20000
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import chess.engine

from config import Config
from src.data.bagz import read_records
from src.data.chessbench import decode_action_value, _build_position
from src.data.targets import cp_to_winprob
from src.data.preencode import write_shard


def collect_fens(bag_path, n):
    """Stream unique, non-terminal FENs from an action_value bag (cap at n)."""
    seen = set()
    out = []
    for record in read_records(bag_path):
        fen, _move, _win = decode_action_value(record)
        if fen in seen:
            continue
        seen.add(fen)
        board = chess.Board(fen)
        if board.is_game_over() or not any(board.legal_moves):
            continue
        out.append(fen)
        if len(out) >= n:
            break
    return out


def _score_to_cp(score):
    if score.is_mate():
        m = score.mate()
        return 10000 if (m is not None and m > 0) else -10000
    cp = score.score()
    return int(cp) if cp is not None else 0


def worker(args):
    (wid, fens, start_index, depth, multipv_cap, temperature,
     out_dir, shard_size, val_every, stockfish_path) = args

    from src.training.stockfish_evaluator import StockfishEvaluator
    ev = StockfishEvaluator(stockfish_path=stockfish_path, depth=depth, time_limit=0.0)
    try:
        ev.engine.configure({"Threads": 1, "Hash": 64})
    except Exception:
        pass

    buffers = {"train": [], "val": []}
    idx = {"train": 0, "val": 0}
    counts = {"train": 0, "val": 0}

    def flush(split):
        if not buffers[split]:
            return
        path = os.path.join(out_dir, f"{split}_w{wid}_{idx[split]:05d}.npz")
        counts[split] += write_shard(buffers[split], path)
        idx[split] += 1
        buffers[split] = []

    for j, fen in enumerate(fens):
        gi = start_index + j
        board = chess.Board(fen)
        legal = list(board.legal_moves)
        if not legal:
            continue
        mpv = len(legal) if not multipv_cap else min(len(legal), multipv_cap)
        try:
            infos = ev.engine.analyse(board, chess.engine.Limit(depth=depth), multipv=mpv)
        except Exception:
            continue
        if isinstance(infos, dict):
            infos = [infos]
        moves, wins = [], []
        for info in infos:
            pv = info.get("pv")
            if not pv or "score" not in info:
                continue
            cp = _score_to_cp(info["score"].relative)
            moves.append(pv[0].uci())
            wins.append(cp_to_winprob(cp))
        if not moves:
            continue
        lp = _build_position(fen, moves, wins, temperature)
        split = "val" if (val_every and gi % val_every == 0) else "train"
        buffers[split].append(lp)
        if len(buffers[split]) >= shard_size:
            flush(split)

    flush("train")
    flush("val")
    ev.close()
    return counts["train"], counts["val"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen-bag", required=True)
    ap.add_argument("--max-positions", type=int, required=True)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--multipv-cap", type=int, default=0, help="0 = all legal moves")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-size", type=int, default=20000)
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--temperature", type=float, default=None,
                    help="policy softmax temperature (default Config.distill_policy_temperature)")
    ap.add_argument("--stockfish-path", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    temperature = (args.temperature if args.temperature is not None
                   else getattr(Config, "distill_policy_temperature", 1.0))
    val_every = int(round(1.0 / args.val_fraction)) if args.val_fraction > 0 else 0

    print(f"collecting up to {args.max_positions} unique FENs from {args.fen_bag} ...", flush=True)
    t0 = time.time()
    fens = collect_fens(args.fen_bag, args.max_positions)
    print(f"  got {len(fens)} FENs in {time.time()-t0:.1f}s", flush=True)

    # contiguous chunks so each worker can compute global index for the val split
    W = args.workers
    chunks = []
    base = 0
    per = (len(fens) + W - 1) // W
    for wid in range(W):
        chunk = fens[base:base + per]
        if chunk:
            chunks.append((wid, chunk, base, args.depth, args.multipv_cap, temperature,
                           args.out_dir, args.shard_size, val_every, args.stockfish_path))
        base += per

    print(f"generating with {len(chunks)} workers, depth={args.depth}, "
          f"multipv_cap={args.multipv_cap or 'all'}, T={temperature} ...", flush=True)
    t0 = time.time()
    with mp.Pool(len(chunks)) as pool:
        results = pool.map(worker, chunks)
    dt = time.time() - t0
    tr = sum(r[0] for r in results)
    va = sum(r[1] for r in results)
    rate = (tr + va) / dt if dt > 0 else 0
    print(f"done: {tr} train, {va} val positions in {dt:.0f}s "
          f"({rate:.1f} pos/s, ~{int(rate*3600)} pos/hr)", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
