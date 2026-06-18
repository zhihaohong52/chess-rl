#!/usr/bin/env python3
"""Gauge an action-value checkpoint on ChessBench puzzles.

Reports raw-policy top-1 (argmax legal Q, no search) and optionally MCTS top-1.
Directly comparable to the prior policy-distribution gauge (raw / MCTS top-1).

Usage:
  python scripts/gauge_av.py --model checkpoints/distill_av/best.pt \
      --puzzle-csv data/puzzles.csv --max-puzzles 600 --simulations 100
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import numpy as np
import torch

from config import Config
from src.model.transformer import ChessTransformer
from src.game.token_encoder import encode_position
from src.game.orientation import to_canonical_move
from src.game.move_encoder import get_move_encoder
from src.eval.puzzles import load_chessbench_puzzles


def raw_top1(net, device, puzzles):
    me = get_move_encoder()
    net.eval()
    correct = 0
    for pz in puzzles:
        board = chess.Board(pz.fen)
        legal = list(board.legal_moves)
        if not legal:
            continue
        sq, sf = encode_position(board, 0)
        sq_t = torch.tensor(sq[None], dtype=torch.long, device=device)
        sf_t = torch.tensor(sf[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            pol, _, _ = net.predict_batch(sq_t, sf_t)
        pol = pol[0].cpu().numpy()
        idxs = [me.encode(to_canonical_move(mv, board.turn)) for mv in legal]
        best = legal[int(np.argmax(pol[idxs]))]
        if best == pz.solution_moves[0]:
            correct += 1
    return correct / len(puzzles)


def mcts_top1(net, puzzles, simulations):
    from src.model.evaluator import TransformerEvaluator
    from src.mcts.batched_mcts import BatchedMCTS
    cfg = Config()
    ev = TransformerEvaluator(net, objective="action_value")
    mcts = BatchedMCTS(ev, cfg, num_simulations=simulations)
    correct = 0
    for pz in puzzles:
        board = chess.Board(pz.fen)
        mcts.reset()
        mv = mcts.choose_move(board, temperature=0.0)
        if mv == pz.solution_moves[0]:
            correct += 1
    return correct / len(puzzles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/distill_av/best.pt")
    ap.add_argument("--puzzle-csv", default="data/puzzles.csv")
    ap.add_argument("--max-puzzles", type=int, default=600)
    ap.add_argument("--simulations", type=int, default=100,
                    help="MCTS sims; 0 to skip MCTS gauge")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = Config()
    net = ChessTransformer(cfg).to(device)
    net.load_state_dict(torch.load(args.model, map_location="cpu"))

    puzzles = load_chessbench_puzzles(path=args.puzzle_csv, max_puzzles=args.max_puzzles)
    print(f"puzzles: {len(puzzles)}", flush=True)

    r = raw_top1(net, device, puzzles)
    print(f"raw policy top-1: {r:.3f} ({int(r*len(puzzles))}/{len(puzzles)})", flush=True)

    if args.simulations > 0:
        m = mcts_top1(net, puzzles, args.simulations)
        print(f"MCTS-{args.simulations} top-1: {m:.3f} ({int(m*len(puzzles))}/{len(puzzles)})",
              flush=True)


if __name__ == "__main__":
    main()
