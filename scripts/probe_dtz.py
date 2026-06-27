#!/usr/bin/env python3
"""Diagnose why best_dtz_move under-converts certain 4-man wins.

For each decisive FEN we:
  1. print root scope / WDL / DTZ,
  2. call best_dtz_move and report what it returns,
  3. walk every legal move and show child popcount / WDL / DTZ, catching
     exceptions per-move so we see exactly which child trips the
     `child_wdl is None -> return None` bail-out (or raises),
  4. play a pure-TB line (best_dtz_move every ply it is in scope) and report
     whether the position is actually converted within the ply budget.
"""
import os
import sys

import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.syzygy_tb import SyzygyTablebase

FENS = [
    ("4k3/8/8/8/8/8/8/3QK3 w - - 0 1",   "KQ vs K  (3-man, works)"),
    ("4k3/8/8/8/8/5r2/8/3QK3 w - - 0 1", "KQ vs KR (4-man, anomaly)"),
    ("4k3/8/8/8/8/8/2p5/3QK3 w - - 0 1", "KQ vs KP (4-man, anomaly)"),
    ("4k3/8/8/8/4p3/8/8/3RK3 w - - 0 1", "KR vs KP (4-man, anomaly)"),
]

MAX_PLY = 120


def _safe(fn, board):
    try:
        return fn(board), None
    except Exception as e:  # noqa: BLE001 - we want the class name for diagnosis
        return None, f"{type(e).__name__}: {e}"


def dump_children(tb, board):
    print("    legal-move children (popcount / wdl / dtz):")
    for move in board.legal_moves:
        san = board.san(move)
        board.push(move)
        pc = chess.popcount(board.occupied)
        wdl, werr = _safe(tb._tb.get_wdl, board)
        dtz, derr = _safe(tb._tb.get_dtz, board)
        board.pop()
        flag = ""
        if wdl is None:
            flag = "  <-- wdl None (forces best_dtz_move -> None)"
        print(f"      {san:<7} men={pc}  wdl={wdl!s:<5} {werr or ''}"
              f"  dtz={dtz!s:<6} {derr or ''}{flag}")


def play_tb_line(tb, board):
    """Follow best_dtz_move while in scope; report conversion."""
    b = board.copy()
    plies = 0
    while plies < MAX_PLY and not b.is_game_over(claim_draw=True):
        if not tb.in_scope(b):
            return f"left scope after {plies} plies (>{tb.max_pieces} men?)"
        mv = tb.best_dtz_move(b)
        if mv is None:
            return f"best_dtz_move returned None at ply {plies}: {b.fen()}"
        b.push(mv)
        plies += 1
    if b.is_checkmate():
        return f"CONVERTED: mate in {plies} plies"
    if b.is_game_over(claim_draw=True):
        return f"draw ({b.result(claim_draw=True)}) after {plies} plies"
    return f"NOT converted within {MAX_PLY} plies (still {b.fen()})"


def main():
    syz = sys.argv[1] if len(sys.argv) > 1 else "data/syzygy"
    tb = SyzygyTablebase(syz, max_pieces=5)
    for fen, label in FENS:
        board = chess.Board(fen)
        print(f"\n=== {label} ===")
        print(f"    fen: {fen}")
        print(f"    in_scope={tb.in_scope(board)}  men={chess.popcount(board.occupied)}")
        rw, rwe = _safe(tb._tb.get_wdl, board)
        rd, rde = _safe(tb._tb.get_dtz, board)
        print(f"    root wdl={rw} {rwe or ''}   root dtz={rd} {rde or ''}")
        bm, bme = _safe(tb.best_dtz_move, board)
        if bme:
            print(f"    best_dtz_move RAISED: {bme}")
        else:
            print(f"    best_dtz_move -> {board.san(bm) if bm else None}")
        if bm is None:
            dump_children(tb, board)
        print(f"    pure-TB play-out: {play_tb_line(tb, board)}")


if __name__ == "__main__":
    sys.exit(main())
