#!/usr/bin/env python3
"""Search sweep: how much does MCTS strength scale with simulations and c_puct?

Self-play head-to-head (identical weights, different search) isolates the search
lever — no Stockfish calibration needed. A score > 0.5 for the higher-sim side is
Elo we leave on the table by under-searching.

  python scripts/search_sweep.py --model checkpoints/p3_80m_1epoch/best_ema.pt --device mps
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.eval.routing import load_for_eval
from src.eval.elo import elo_diff
from src.mcts.batched_mcts import BatchedMCTS
from scripts.arena_eval import head_to_head_openings


def mcts_mover(evaluator, sims, c_puct):
    """Greedy (temp 0, no noise) MCTS mover at a given sim count and c_puct."""
    mcts = BatchedMCTS(evaluator, Config, num_simulations=sims)
    mcts.c_puct = c_puct  # read at select_child time, so override-after-construct works

    def mover(board):
        mcts.reset()
        return mcts.choose_move(board, temperature=0.0)

    return mover


def match(ev, a, b, games, max_moves, seed, plies, label):
    """Self-play head-to-head; a/b are (sims, c_puct). Returns A's score."""
    mover_a = mcts_mover(ev, *a)
    mover_b = mcts_mover(ev, *b)
    w, d, l = head_to_head_openings(mover_a, mover_b, games, max_moves,
                                    seed=seed, opening_plies=plies)
    tot = w + d + l
    sc = (w + 0.5 * d) / tot if tot else 0.5
    gap = elo_diff(sc, games=tot)
    print(f"SWEEP {label}: A{a} vs B{b}  W/D/L {w}/{d}/{l}  score {sc:.3f}  "
          f"Elo {gap:+.0f}", flush=True)
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/p3_80m_1epoch/best_ema.pt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--max-moves", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plies", type=int, default=8)
    args = ap.parse_args()

    _, ev = load_for_eval(args.model, device=args.device)
    cp = Config.c_puct
    print(f"model={args.model} device={args.device} games={args.games} "
          f"default_c_puct={cp}", flush=True)

    # 1) sims ladder vs the 50-sim baseline we've always evaluated at.
    print(f"== sims ladder (vs 50-sim baseline, c_puct={cp}) ==", flush=True)
    for sims in (200, 400):
        match(ev, (sims, cp), (50, cp), args.games, args.max_moves,
              args.seed, args.plies, f"sims{sims}_vs_50")

    # 2) c_puct sweep at fixed 200 sims (vs the default), now that the value head
    #    is well-calibrated (the 10M-era "search hurts" finding may be stale).
    print(f"== c_puct sweep (200 sims, vs default {cp}) ==", flush=True)
    for c in (1.0, 3.0):
        match(ev, (200, c), (200, cp), args.games, args.max_moves,
              args.seed, args.plies, f"cpuct{c}_vs_{cp}")

    print("SWEEP DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
