#!/usr/bin/env python3
"""Full-game endgame arena: does the Syzygy layer help the engine *convert*?

The plain head-to-head arena (`arena_eval.py --tb-one-side`) starts every game
from an ~8-ply random opening, i.e. ~30 men. Those games hit the move cap as
draws long before they simplify to <=5 men, so the tablebase layer almost never
fires and the measured edge is ~0 (pure dilution). That tells us nothing about
whether the TB layer is *useful*, only that it's rarely *reached*.

This harness instead seeds each game from a real 6-8 man endgame and plays it to
completion. The engines must navigate the position *down into* the <=5-man bag,
where side A's `best_dtz_move` takes over. It is the honest middle ground between:
  - `endgame_eval.py`  : starts AT the <=5-man boundary (TB fires immediately), and
  - `arena_eval.py`    : starts at ~30 men (TB effectively never fires).

A = hybrid mover (book->TB->MCTS), B = plain net+MCTS, SAME checkpoint, so any
gap is attributable to the book/TB layer alone. Each position is played both
colours (A as White, then B as White) and A's score is `(a_white + (1-b_white))`.
Decisive positions test conversion; symmetric ones are drawn controls that
should stay ~0.5 and catch the TB layer doing harm.

    python scripts/endgame_arena.py --model checkpoints/p3_80m_1epoch/best_ema.pt \
        --syzygy data/syzygy --simulations 64 --device mps --max-moves 100
"""
import argparse
import os
import sys

import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.routing import load_for_eval
from src.eval.elo import elo_diff
from src.engine.syzygy_tb import SyzygyTablebase, syzygy_available
from src.engine.online_tb import OnlineSyzygyTablebase
from scripts.arena_eval import build_hybrid_mover, build_mcts_mover, _play_game_from

# (fen, label, expect) — `expect` is a human annotation for the report only; the
# scoring never uses it. White to move throughout; the pairing supplies both
# colours. Decisive = stronger side up material that must promote / mate by
# simplifying into the <=5-man bag; control = symmetric, should hold ~0.5.
POOL = [
    # --- decisive: pawn majorities promote -> KQ vs K (TB mate) ---
    ("8/4p3/8/4k3/8/4P3/3PKP2/8 w - - 0 1",   "KPPP vs KP  (up 2P)",   "win"),
    ("4k3/4p3/8/8/8/4P3/3P1P2/4K3 w - - 0 1", "KPPP vs KP  (up 2P, b)","win"),
    # --- decisive: piece up, simplifies to KR / KQ vs K ---
    ("4k3/8/8/8/8/4PP2/3QK3/3q4 w - - 0 1",   "KQPP vs KQ  (up 2P)",   "win"),
    ("4k3/8/8/8/8/4PP2/4K3/r6R w - - 0 1",    "KRPP vs KR  (up 2P)",   "win"),
    ("4k3/8/8/8/8/4P3/3PKP2/r6R w - - 0 1",   "KRPPP vs KR (up 3P)",   "win"),
    ("4k3/4p3/8/8/8/8/3PKP2/R6r w - - 0 1",   "KRPP vs KRP (up 1P)",   "win"),
    # --- balanced controls: should hold ~0.5, catch TB-layer harm ---
    ("4k3/3p4/4p3/8/4P3/3P4/4K3/8 w - - 0 1", "KPP vs KPP  (sym)",     "draw"),
    ("6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",   "K3P vs K3P  (sym)",     "draw"),
    ("4k3/4p3/8/8/8/8/4P3/R3K2r w - - 0 1",   "KRP vs KRP  (sym)",     "draw"),
]


def validated(pool):
    """Drop any position that isn't legal, in [6,8] men, and still in play."""
    kept = []
    for fen, label, expect in pool:
        try:
            b = chess.Board(fen)
        except ValueError:
            print(f"  skip (bad FEN): {label}", file=sys.stderr)
            continue
        men = chess.popcount(b.occupied)
        if b.is_valid() and not b.is_game_over() and 6 <= men <= 8:
            kept.append((b, label, expect, men))
        else:
            print(f"  skip ({men}men valid={b.is_valid()}): {label}", file=sys.stderr)
    return kept


def _bucket(score):
    return "win " if score == 1.0 else ("draw" if score == 0.5 else "loss")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/p3_80m_1epoch/best_ema.pt")
    ap.add_argument("--syzygy", default="data/syzygy")
    ap.add_argument("--simulations", type=int, default=64)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-moves", type=int, default=100)
    ap.add_argument("--max-pieces", type=int, default=5)
    ap.add_argument("--online-tb", action="store_true",
                    help="extend the root TB oracle to <=7 men via the Lichess API")
    ap.add_argument("--online-max-pieces", type=int, default=7)
    ap.add_argument("--only", default=None,
                    help="comma-separated substrings; keep only matching position labels")
    args = ap.parse_args()

    if not syzygy_available(args.syzygy):
        print(f"no syzygy tables in {args.syzygy}; aborting", file=sys.stderr)
        return 1
    tb = SyzygyTablebase(args.syzygy, max_pieces=args.max_pieces)
    if args.online_tb:
        tb = OnlineSyzygyTablebase(local=tb, max_pieces=args.online_max_pieces)
    positions = validated(POOL)
    if args.only:
        wanted = [s.strip().lower() for s in args.only.split(",") if s.strip()]
        positions = [p for p in positions if any(w in p[1].lower() for w in wanted)]

    net, ev = load_for_eval(args.model, device=args.device)
    mover_a = build_hybrid_mover(ev, args.simulations, book=None, tablebase=tb)  # book->TB->MCTS
    mover_b = build_mcts_mover(ev, args.simulations)                            # plain net+MCTS

    tb_kind = (f"online<=({args.online_max_pieces})+local<=({args.max_pieces})"
               if args.online_tb else f"local<=({args.max_pieces})")
    print(f"endgame arena: {os.path.basename(args.model)}  sims={args.simulations}  "
          f"max_moves={args.max_moves}  (A=TB[{tb_kind}]+MCTS  B=plain MCTS)", flush=True)
    print(f"  {len(positions)} positions, each played both colours "
          f"({2*len(positions)} games)\n", flush=True)

    wins = draws = losses = 0
    for board, label, expect, men in positions:
        a_white = _play_game_from(board, mover_a, mover_b, args.max_moves)
        b_white = _play_game_from(board, mover_b, mover_a, args.max_moves)
        a_black = 1.0 - b_white
        for s in (a_white, a_black):
            if s == 1.0:
                wins += 1
            elif s == 0.5:
                draws += 1
            else:
                losses += 1
        print(f"  {label:<24} [{expect:<4}] {men}men  "
              f"A_white={a_white:.1f} ({_bucket(a_white)})  "
              f"A_black={a_black:.1f} ({_bucket(a_black)})", flush=True)

    total = wins + draws + losses
    score = (wins + 0.5 * draws) / total if total else 0.5
    gap = elo_diff(score, games=total)
    print(f"\nendgame arena (full games from 6-8 men, TB-on vs TB-off, same ckpt):",
          flush=True)
    print(f"  W/D/L {wins}/{draws}/{losses}  score {score:.3f}  "
          f"estEloGap(A) {gap:+.0f}  over {total} games", flush=True)
    if isinstance(tb, OnlineSyzygyTablebase):
        print(f"  online TB: {tb.hits}/{tb.calls} network probes returned a move",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
