#!/usr/bin/env python3
"""Arena eval: RAW vs MCTS vs Stockfish, and model-vs-model head-to-head.

Stockfish ladder (default):
  python scripts/arena_eval.py --model checkpoints/p3_80m/best_ema.pt \
      --skills 1 3 5 --games 40 --simulations 100 --depth 4

Head-to-head (two checkpoints, MCTS at equal sims):
  python scripts/arena_eval.py --model checkpoints/p3_80m/best_ema.pt \
      --vs checkpoints/p3_10m_ctrl/best_ema.pt --games 100 --simulations 100
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.eval.routing import load_for_eval
from src.eval.gates import _best_legal_move
from src.eval.arena import play_match
from src.eval.elo import elo_diff
from src.eval.stockfish_opponent import StockfishOpponent, stockfish_available
from src.mcts.batched_mcts import BatchedMCTS
from src.game.move_encoder import get_move_encoder


def build_raw_mover(net, device, me):
    """Move-producer: greedy best legal move from the raw policy."""
    return lambda board: _best_legal_move(net, device, board, me)


def build_mcts_mover(evaluator, simulations):
    """Move-producer: MCTS at fixed simulations, greedy (temperature 0)."""
    mcts = BatchedMCTS(evaluator, Config, num_simulations=simulations)

    def mover(board):
        mcts.reset()
        return mcts.choose_move(board, temperature=0.0)

    return mover


def run_head_to_head(model, vs, games, simulations, device, max_moves):
    if games % 2 != 0:
        print(f"warning: --games {games} is odd; color split is uneven and the "
              f"score carries a small first-mover bias. Use an even count.",
              file=sys.stderr, flush=True)
    net_a, ev_a = load_for_eval(model, device=device)
    net_b, ev_b = load_for_eval(vs, device=device)
    mover_a = build_mcts_mover(ev_a, simulations)
    mover_b = build_mcts_mover(ev_b, simulations)
    res = play_match(engine=mover_a, opponent=mover_b,
                     num_games=games, max_moves=max_moves)
    gap = elo_diff(res.score, games=res.total)
    print(f"head-to-head (MCTS {simulations} sims): {os.path.basename(model)} "
          f"vs {os.path.basename(vs)}", flush=True)
    print(f"  W/D/L {res.wins}/{res.draws}/{res.losses}  score {res.score:.3f}  "
          f"estEloGap {gap:+.0f}", flush=True)
    return res


def run_stockfish_ladder(model, skills, games, simulations, depth, device, max_moves):
    net, ev = load_for_eval(model, device=device)
    me = get_move_encoder()
    raw_mover = build_raw_mover(net, device, me)
    mcts_mover = build_mcts_mover(ev, simulations)
    print(f"model={model}  games/skill={games}  sims={simulations}  sf_depth={depth}",
          flush=True)
    print(f"{'mover':>6} {'skill':>5} {'W/D/L':>10} {'score':>6} {'estElo':>7}",
          flush=True)
    for skill in skills:
        for name, mover in (("raw", raw_mover), ("mcts", mcts_mover)):
            with StockfishOpponent(skill_level=skill, depth=depth) as opp:
                res = play_match(engine=mover, opponent=opp,
                                 num_games=games, max_moves=max_moves)
                est = opp.approximate_elo + elo_diff(res.score, games=res.total)
                print(f"{name:>6} {skill:5d} {res.wins:>3}/{res.draws}/{res.losses:<3} "
                      f"{res.score:6.3f} {est:7.0f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/p3_80m/best_ema.pt")
    ap.add_argument("--vs", default=None,
                    help="second checkpoint for model-vs-model head-to-head")
    ap.add_argument("--skills", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--simulations", type=int, default=100)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--max-moves", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.vs:
        run_head_to_head(args.model, args.vs, args.games, args.simulations,
                         args.device, args.max_moves)
        return 0

    if not stockfish_available():
        print("Stockfish not found; cannot run ladder.", file=sys.stderr)
        return 1
    run_stockfish_ladder(args.model, args.skills, args.games, args.simulations,
                         args.depth, args.device, args.max_moves)
    return 0


if __name__ == "__main__":
    sys.exit(main())
