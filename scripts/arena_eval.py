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
import random
import sys

import chess

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


def _random_opening(rng, plies):
    """A non-terminal board reached by `plies` uniform-random legal moves.

    Gives two deterministic (temperature-0) MCTS engines a varied start so a
    multi-game match isn't just the same line repeated. Falls back toward the
    start position if a random walk happens to end the game early.
    """
    board = chess.Board()
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves or board.is_game_over():
            break
        board.push(rng.choice(moves))
    return chess.Board() if board.is_game_over() else board


def _play_game_from(start, white_mover, black_mover, max_moves):
    """Play one game from `start`; return White's result (1.0/0.5/0.0)."""
    board = start.copy()
    moves = 0
    while not board.is_game_over() and moves < max_moves:
        mv = (white_mover if board.turn == chess.WHITE else black_mover)(board)
        if mv is None or mv not in board.legal_moves:
            return 0.0 if board.turn == chess.WHITE else 1.0
        board.push(mv)
        moves += 1
    if not board.is_game_over():
        return 0.5
    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        return 0.5
    return 1.0 if outcome.winner == chess.WHITE else 0.0


def head_to_head_openings(mover_a, mover_b, games, max_moves, seed=0, opening_plies=8):
    """Match mover_a vs mover_b over paired games from random openings.

    Each opening is played twice (A as White, then B as White) so colours are
    balanced. Returns (wins, draws, losses) from mover_a's perspective. With
    deterministic engines, distinct openings are what create distinct games.
    """
    rng = random.Random(seed)
    wins = draws = losses = 0
    n_pairs = max(1, games // 2)
    for pair in range(n_pairs):
        opening = _random_opening(rng, opening_plies)
        a_white = _play_game_from(opening, mover_a, mover_b, max_moves)        # A's score
        b_white = _play_game_from(opening, mover_b, mover_a, max_moves)        # White=B
        for a_score in (a_white, 1.0 - b_white):
            if a_score == 1.0:
                wins += 1
            elif a_score == 0.5:
                draws += 1
            else:
                losses += 1
        if (pair + 1) % 5 == 0:
            done = (pair + 1) * 2
            print(f"  ...{done}/{2 * n_pairs} games  W/D/L {wins}/{draws}/{losses}",
                  flush=True)
    return wins, draws, losses


def run_head_to_head(model, vs, games, simulations, device, max_moves,
                     seed=0, opening_plies=8):
    net_a, ev_a = load_for_eval(model, device=device)
    net_b, ev_b = load_for_eval(vs, device=device)
    mover_a = build_mcts_mover(ev_a, simulations)
    mover_b = build_mcts_mover(ev_b, simulations)
    wins, draws, losses = head_to_head_openings(
        mover_a, mover_b, games, max_moves, seed=seed, opening_plies=opening_plies)
    total = wins + draws + losses
    score = (wins + 0.5 * draws) / total if total else 0.5
    gap = elo_diff(score, games=total)
    print(f"head-to-head (MCTS {simulations} sims, {opening_plies}-ply random "
          f"openings): {os.path.basename(model)} vs {os.path.basename(vs)}", flush=True)
    print(f"  W/D/L {wins}/{draws}/{losses}  score {score:.3f}  "
          f"estEloGap {gap:+.0f}", flush=True)
    return wins, draws, losses


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
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for head-to-head openings")
    ap.add_argument("--opening-plies", type=int, default=8,
                    help="random plies to vary head-to-head openings")
    args = ap.parse_args()

    if args.vs:
        run_head_to_head(args.model, args.vs, args.games, args.simulations,
                         args.device, args.max_moves, seed=args.seed,
                         opening_plies=args.opening_plies)
        return 0

    if not stockfish_available():
        print("Stockfish not found; cannot run ladder.", file=sys.stderr)
        return 1
    run_stockfish_ladder(args.model, args.skills, args.games, args.simulations,
                         args.depth, args.device, args.max_moves)
    return 0


if __name__ == "__main__":
    sys.exit(main())
