#!/usr/bin/env python3
"""Verify a candidate fix for best_dtz_move WITHOUT touching the shipped file.

Current key:  (our_wdl, child_dtz)            -> maximizes child_dtz toward 0.
  Fails on pawn endings: across a zeroing boundary DTZ magnitudes aren't
  comparable, so "soonest opponent zeroing" (a harmless forced pawn push) outranks
  the winning capture (which leads to a pawnless mate with larger DTZ). The engine
  herds the king instead of grabbing the pawn -> repetition/stalemate draw.

Candidate key: (our_wdl, win_zeroing, child_dtz)
  For a winning move, prefer one that ZEROES (capture / pawn move) and keeps the
  win -> resets the 50-move counter and guarantees material progress; tie-break by
  DTZ as before. Pawnless wins (no zeroing moves) fall straight through to the DTZ
  tie-break, i.e. unchanged from today (KQvK still mates fastest).
"""
import os
import sys

import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.syzygy_tb import SyzygyTablebase

FENS = [
    ("4k3/8/8/8/8/8/8/3QK3 w - - 0 1",   "KQ vs K  (3-man, control)"),
    ("4k3/8/8/8/8/8/8/3RK3 w - - 0 1",   "KR vs K  (3-man, control)"),
    ("4k3/8/8/8/8/5r2/8/3QK3 w - - 0 1", "KQ vs KR (4-man)"),
    ("4k3/8/8/8/8/8/2p5/3QK3 w - - 0 1", "KQ vs KP (4-man)"),
    ("4k3/8/8/8/4p3/8/8/3RK3 w - - 0 1", "KR vs KP (4-man)"),
]
MAX_PLY = 120


def candidate_move(tb, board):
    """best_dtz_move with the (our_wdl, win_zeroing, child_dtz) key."""
    if not tb.in_scope(board):
        return None
    best_move = None
    best_key = None
    for move in board.legal_moves:
        zeroing = board.is_zeroing(move)
        board.push(move)
        try:
            child_wdl = tb._tb.get_wdl(board)
            child_dtz = tb._tb.get_dtz(board)
        finally:
            board.pop()
        if child_wdl is None:
            return None
        our_wdl = -child_wdl
        win_zeroing = 1 if (our_wdl >= 2 and zeroing) else 0
        key = (our_wdl, win_zeroing, child_dtz if child_dtz is not None else 0)
        if best_key is None or key > best_key:
            best_key = key
            best_move = move
    return best_move


def play_line(tb, board, pick):
    b = board.copy()
    plies = 0
    while plies < MAX_PLY and not b.is_game_over(claim_draw=True):
        if not tb.in_scope(b):
            return f"left scope after {plies} plies"
        mv = pick(tb, b)
        if mv is None:
            return f"None at ply {plies}"
        b.push(mv)
        plies += 1
    if b.is_checkmate():
        return f"CONVERTED: mate in {plies} plies"
    if b.is_game_over(claim_draw=True):
        return f"draw ({b.result(claim_draw=True)}) after {plies} plies"
    return f"NOT converted within {MAX_PLY} plies"


def main():
    syz = sys.argv[1] if len(sys.argv) > 1 else "data/syzygy"
    tb = SyzygyTablebase(syz, max_pieces=5)
    for fen, label in FENS:
        board = chess.Board(fen)
        cur = play_line(tb, board, lambda t, b: t.best_dtz_move(b))
        fix = play_line(tb, board, candidate_move)
        print(f"{label:<28} current: {cur:<34} candidate: {fix}")


if __name__ == "__main__":
    sys.exit(main())
