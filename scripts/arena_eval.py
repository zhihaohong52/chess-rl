#!/usr/bin/env python3
"""Arena eval: compare RAW policy vs MCTS for the same model, by playing each
against Stockfish over games. Answers "does search help in PLAY?" (vs the
puzzle metric). Routes the evaluator by the checkpoint sidecar.

Usage:
  python scripts/arena_eval.py --model checkpoints/p2_value_300k/best.pt \
      --skills 1 3 --games 12 --simulations 100 --depth 4
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess

from config import Config
from src.eval.routing import load_for_eval
from src.eval.gates import _best_legal_move
from src.eval.arena import play_match
from src.eval.elo import elo_diff
from src.eval.stockfish_opponent import StockfishOpponent, stockfish_available
from src.game.move_encoder import get_move_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/p2_value_300k/best.pt")
    ap.add_argument("--skills", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--simulations", type=int, default=100)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--max-moves", type=int, default=200)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    if not stockfish_available():
        print("Stockfish not found; cannot run arena.", file=sys.stderr)
        return 1

    net, ev = load_for_eval(args.model, device=args.device)
    me = get_move_encoder()

    def raw_mover(board):
        return _best_legal_move(net, args.device, board, me)

    from src.mcts.batched_mcts import BatchedMCTS
    mcts = BatchedMCTS(ev, Config, num_simulations=args.simulations)

    def mcts_mover(board):
        mcts.reset()
        return mcts.choose_move(board, temperature=0.0)

    print(f"model={args.model}  games/skill={args.games}  sims={args.simulations}  "
          f"sf_depth={args.depth}", flush=True)
    print(f"{'mover':>6} {'skill':>5} {'W/D/L':>10} {'score':>6} {'estElo':>7}", flush=True)

    for skill in args.skills:
        for name, mover in (("raw", raw_mover), ("mcts", mcts_mover)):
            with StockfishOpponent(skill_level=skill, depth=args.depth) as opp:
                res = play_match(engine=mover, opponent=opp,
                                 num_games=args.games, max_moves=args.max_moves)
                est = opp.approximate_elo + elo_diff(res.score, games=res.total)
                print(f"{name:>6} {skill:5d} {res.wins:>3}/{res.draws}/{res.losses:<3} "
                      f"{res.score:6.3f} {est:7.0f}", flush=True)


if __name__ == "__main__":
    main()
