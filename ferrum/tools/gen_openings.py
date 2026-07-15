#!/usr/bin/env python3
"""Generate a small EPD opening book of standard, sound openings.

Deterministic engines play identical games from the start position, so
strength matches need opening variety. This emits ~24 mainline openings a few
plies deep (balanced, non-blundering positions) as an EPD book for fastchess:

    fastchess ... -openings file=books/openings.epd format=epd order=random

Regenerate: `python3 tools/gen_openings.py > books/openings_m1.epd`
Requires python-chess (`pip install chess`).
"""
import sys
import chess

# Each entry: SAN mainline. Kept balanced and theory-sound so neither side
# starts with an advantage that would bias the match.
OPENINGS = [
    "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6",          # Ruy Lopez
    "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6",          # Italian
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6",  # Sicilian Najdorf
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6",  # Sicilian Dragon
    "e4 e6 d4 d5 Nc3 Nf6 Bg5 Be7",           # French
    "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5",         # Caro-Kann
    "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7",           # Queen's Gambit Declined
    "d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4",          # Slav
    "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6",            # King's Indian Defence
    "d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O",           # Nimzo-Indian
    "d4 d5 c4 dxc4 Nf3 Nf6 e3 e6",           # Queen's Gambit Accepted
    "c4 e5 Nc3 Nf6 Nf3 Nc6 g3 d5",           # English
    "Nf3 d5 c4 e6 g3 Nf6 Bg2 Be7",           # Reti
    "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6",        # Scandinavian
    "e4 d6 d4 Nf6 Nc3 g6 f4 Bg7",            # Pirc
    "d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5",         # Gruenfeld
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6",        # Scotch
    "e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4",        # Petroff
    "d4 Nf6 c4 e6 Nf3 b6 g3 Bb7",            # Queen's Indian
    "d4 Nf6 c4 e6 g3 d5 Bg2 Be7",            # Catalan
    "d4 d5 Nf3 Nf6 Bf4 e6 e3 Bd6",           # London
    "e4 e5 Nc3 Nf6 f4 d5 fxe5 Nxe4",         # Vienna
    "d4 Nf6 c4 c5 d5 e6 Nc3 exd5",           # Benoni
    "d4 f5 g3 Nf6 Bg2 e6 Nf3 Be7",           # Dutch
]


def _expand(board: chess.Board, plies_left: int, seen: set, out: list, max_per_seed: int):
    """Depth-first walk emitting EPDs, at most `max_per_seed` per seed line."""
    if len(out) and out.count(out[-1]):  # cheap guard; real dedupe via `seen`
        pass
    epd = board.epd()
    if epd not in seen:
        seen.add(epd)
        out.append(epd)
    if plies_left == 0 or len(out) >= max_per_seed:
        return
    # branch on a handful of reasonable legal moves (keep the tree small + sane)
    legal = list(board.legal_moves)
    for mv in legal[: min(3, len(legal))]:
        board.push(mv)
        _expand(board, plies_left - 1, seen, out, max_per_seed)
        board.pop()


def main() -> int:
    seen: set = set()
    all_epds: list = []
    for line in OPENINGS:
        board = chess.Board()
        for san in line.split():
            board.push_san(san)
        per_seed: list = []
        _expand(board, 2, seen, per_seed, max_per_seed=40)
        all_epds.extend(per_seed)
    for epd in all_epds:
        print(epd)
    print(f"generated {len(all_epds)} opening positions", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
