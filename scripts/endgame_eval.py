#!/usr/bin/env python3
"""Endgame-isolation A/B: tablebase player vs net+MCTS, from ≤5-man starts.

Side A = hybrid mover (Syzygy DTZ-optimal in ≤max-piece positions, MCTS above).
Side B = plain net+MCTS, no tablebase. Both load the SAME checkpoint, so the
only difference between them is endgame knowledge. Every game starts from a
≤5-man position, so side A plays a perfect tablebase move on move one — this
isolates the endgame Elo edge of the TB over the bare net, with no middlegame
noise diluting the signal (unlike --tb-one-side from random openings, where the
TB almost never fires because near-start search never reaches ≤5-man leaves).

Each start position is played twice (A as White, then B as White) so colours
are balanced. From a won-for-the-mover position, A converts perfectly; B (the
net) only scores if it can also mate within the move budget against A's
tablebase-perfect defence — so A's net edge is exactly the set of theoretical
wins the net fails to convert in time. Drawn positions are controls: both should
hold, contributing draws.

Usage:
  python scripts/endgame_eval.py --model checkpoints/p3_80m_1epoch/best_ema.pt \
      --syzygy data/syzygy --simulations 200 --device mps
"""
import argparse
import os
import sys

import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.routing import load_for_eval
from src.eval.elo import elo_diff
from src.engine.syzygy_tb import SyzygyTablebase, syzygy_available
from src.engine.hybrid_mover import build_hybrid_mover
from scripts.arena_eval import build_mcts_mover, _play_game_from


# Canonical ≤5-man endgames. A mix of clean wins (where conversion skill — and
# thus the TB edge — shows up) and known draws (controls both sides should hold).
# Each is validated at runtime and skipped if illegal / terminal / out of scope,
# and labelled by what the tablebase says for the side to move, so the report is
# self-documenting rather than trusting these hand-written FENs blindly.
ENDGAME_FENS = [
    ("4k3/8/8/8/8/8/8/3QK3 w - - 0 1",       "KQ vs K"),
    ("4k3/8/8/8/8/8/8/3RK3 w - - 0 1",       "KR vs K"),
    ("4k3/8/8/8/8/8/8/1NB1K3 w - - 0 1",     "KBN vs K (hard mate)"),
    ("4k3/8/8/8/8/8/8/2B1KB2 w - - 0 1",     "KBB vs K"),
    ("4k3/8/8/8/8/8/8/1N2K1N1 w - - 0 1",    "KNN vs K (drawn)"),
    ("4k3/8/8/8/4P3/4K3/8/8 w - - 0 1",      "KP vs K (e-pawn)"),
    ("7k/8/8/8/8/8/7P/7K w - - 0 1",         "KP vs K (rook pawn, drawn)"),
    ("4k3/8/8/8/8/5r2/8/3QK3 w - - 0 1",     "KQ vs KR"),
    ("4k3/8/8/8/8/8/2p5/3QK3 w - - 0 1",     "KQ vs KP"),
    ("4k3/8/8/8/8/5b2/8/3RK3 w - - 0 1",     "KR vs KB (holdable)"),
    ("4k3/8/8/8/8/5n2/8/3RK3 w - - 0 1",     "KR vs KN (holdable)"),
    ("4k3/8/8/8/4p3/8/8/3RK3 w - - 0 1",     "KR vs KP"),
]

_WDL_NAME = {2: "win", 1: "cursed-win", 0: "draw", -1: "blessed-loss", -2: "loss"}


def _validate_positions(tb, max_pieces):
    """Keep the legal, non-terminal, in-scope FENs; label each by TB WDL (STM)."""
    kept = []
    for fen, name in ENDGAME_FENS:
        try:
            board = chess.Board(fen)
        except ValueError:
            print(f"  skip {name}: unparseable FEN", flush=True)
            continue
        if not board.is_valid() or board.is_game_over():
            print(f"  skip {name}: illegal or already terminal", flush=True)
            continue
        if chess.popcount(board.occupied) > max_pieces:
            print(f"  skip {name}: >{max_pieces} men", flush=True)
            continue
        wdl = tb._tb.get_wdl(board)
        tag = _WDL_NAME.get(wdl, "?") if wdl is not None else "no-table"
        if wdl is None:
            print(f"  skip {name}: missing table", flush=True)
            continue
        kept.append((board, name, tag))
    return kept


def run(model, syzygy_dir, simulations, max_pieces, max_moves, device):
    tb = SyzygyTablebase(syzygy_dir, max_pieces=max_pieces)
    print("validating endgame suite:", flush=True)
    positions = _validate_positions(tb, max_pieces)
    if not positions:
        print("no usable positions — check the tablebase directory.", file=sys.stderr)
        return 1
    wins_label = sum(1 for _, _, t in positions if t in ("win", "loss"))
    print(f"  using {len(positions)} positions ({wins_label} decisive, "
          f"{len(positions) - wins_label} drawn controls)\n", flush=True)

    _, ev = load_for_eval(model, device=device)
    mover_a = build_hybrid_mover(ev, simulations, tablebase=tb)   # TB-on
    mover_b = build_mcts_mover(ev, simulations)                   # TB-off (plain net+MCTS)

    print(f"model={os.path.basename(model)}  sims={simulations}  "
          f"max_moves={max_moves}  (A=TB+MCTS  B=plain MCTS)\n", flush=True)
    wins = draws = losses = 0
    for board, name, tag in positions:
        a_white = _play_game_from(board, mover_a, mover_b, max_moves)   # A's score
        b_white = _play_game_from(board, mover_b, mover_a, max_moves)   # White=B
        for a_score in (a_white, 1.0 - b_white):
            if a_score == 1.0:
                wins += 1
            elif a_score == 0.5:
                draws += 1
            else:
                losses += 1
        print(f"  {name:<26} [{tag:<5}]  A_white={a_white:.1f}  "
              f"A_black={1.0 - b_white:.1f}", flush=True)

    total = wins + draws + losses
    score = (wins + 0.5 * draws) / total if total else 0.5
    gap = elo_diff(score, games=total)
    print(f"\nendgame A/B (TB-on vs TB-off, same checkpoint):", flush=True)
    print(f"  W/D/L {wins}/{draws}/{losses}  score {score:.3f}  "
          f"estEloGap(A) {gap:+.0f}  over {total} games", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/p3_80m_1epoch/best_ema.pt")
    ap.add_argument("--syzygy", default="data/syzygy",
                    help="path to a Syzygy tablebase directory")
    ap.add_argument("--simulations", type=int, default=200)
    ap.add_argument("--max-pieces", type=int, default=5)
    ap.add_argument("--max-moves", type=int, default=120,
                    help="ply cap; failure to mate by here counts as a draw (~50-move budget)")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    if not syzygy_available(args.syzygy):
        print(f"no Syzygy tables in {args.syzygy}; cannot run endgame eval.",
              file=sys.stderr)
        return 1
    return run(args.model, args.syzygy, args.simulations, args.max_pieces,
               args.max_moves, args.device)


if __name__ == "__main__":
    sys.exit(main())
