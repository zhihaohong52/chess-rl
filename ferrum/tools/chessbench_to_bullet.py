#!/usr/bin/env python3
"""Convert ChessBench encoded npz shards -> bullet's 32-byte `ChessBoard` format.

Schema source of truth: `ferrum/nnue/README.md` (Task 1 spike). Summary of the
facts this module implements against (do not re-derive, see that file):

- Each npz shard has `square_tokens [N,64] int8` (canonical side-to-move board:
  0=empty, 1..6=mover's own {P,N,B,R,Q,K}, 7..12=opponent's; square index is
  `rank*8+file`, already vertically mirrored `sq^56` when the real side to
  move is Black, so it always reads as "mover is White"), and `wdl [N,3]
  float32` (side-to-move POV, W/D/L via `winprob_to_wdl(wp, draw_scale=2.0)`).
  `legal_indices`/`legal_probs`/`counts` are a ragged CSR policy target keyed
  by this repo's `MoveEncoder` (see `src/game/move_encoder.py`), already
  encoded in the SAME canonical frame as `square_tokens` (verified:
  `encode_uci_canonical` mirrors from/to squares before the table lookup when
  the real turn isn't White, so `MoveEncoder.decode(idx)` returns a
  `chess.Move` whose `.from_square`/`.to_square` line up directly with
  `square_tokens` indices, no extra mirroring needed).
- The value target is recovered from `wdl` by an exact closed-form inversion
  (see `wdl_to_winprob`/`winprob_to_cp` below), verified in the README against
  `src/data/targets.py::winprob_to_wdl`'s fixed `draw_scale=2.0`.
- bullet's `ChessBoard` is a 32-byte little-endian struct (verified against
  `bulletformat` source directly, `crates/chess.rs`, both the `from_raw` and
  `FromStr` impls, at tag `v1.8.0`/pinned commit): `occ:u64` (STM-relative
  occupancy) `pcs:[u8;16]` (4-bit code per occupied square in ascending
  `occ`-bit order, 2/byte, LOW nibble first: `pcs[i/2] |= code << (4*(i&1))`)
  `score:i16` (STM-relative cp) `result:u8` (0=loss,1=draw,2=win) `ksq:u8`
  (mover king square, no extra mirror) `opp_ksq:u8` (opponent king square
  **in the opponent's own perspective**, which reduces to a single
  `enemy_king_square_in_our_frame ^ 56` -- confirmed against
  `ChessBoard::from_raw`'s `opp_ksq = (bbs[1] & bbs[7]).trailing_zeros() ^ 56`)
  `extra:[u8;3]` reserved zero. `bulletformat::BulletFormat::write_to_bin`
  writes these structs back-to-back with NO header/padding between records
  (confirmed by reading `bulletformat`'s `lib.rs`), so bullet's
  `DirectSequentialDataLoader` expects a flat `count * 32`-byte file -- which
  is exactly what this module's `--output` produces.

Two production paths, both supported here (see README "Trainer" section):
  A. `--emit-text PATH` additionally writes a `<FEN>|<score>|<result>` text
     intermediate (synthetic FEN `<placement from square_tokens> w - - 0 1`,
     score/result already side-to-move-relative -- no extra flip needed
     because the synthetic FEN's side-to-move field is always "w", so
     `bulletformat::ChessBoard::from_str`'s internal STM flip never fires).
     Pack it with bullet's own tool:
       cargo run --release --package bullet-utils -- convert --from text \
           --input sample.txt --output sample_from_text.bin --threads 8
  B. `--output PATH` packs the 32-byte struct directly in Python
     (`pack_bullet_record`). This is the primary/default path; A is provided
     for the cross-check the README recommends before the 100M-row cloud run.

Quiet-position filtering (design-spec data filter, all three implemented):
  1. in-check: drop positions where the mover (`square_tokens`' "White") is in
     check, reconstructed via `chess.BaseBoard` + `is_attacked_by`.
  2. best-move-is-capture-or-gives-check: the best move is
     `argmax(legal_probs)` for that position; decode its `legal_indices` entry
     via this repo's `MoveEncoder.decode` (already canonical-frame, see
     above), then decide capture/gives-check by mechanically applying the
     move to a `chess.BaseBoard` copy (`apply_move_on_board`). Castling
     (king moves 2 files on its own rank) and en-passant (diagonal pawn move
     onto an empty square) are detected geometrically from `square_tokens`
     occupancy alone, since castling rights / the en-passant target square
     are intentionally NOT reconstructed here (`state_features` is out of
     scope per the Task 2 spec, and both special cases are recognizable from
     board geometry in any real, reachable game position). This is a known,
     deliberate approximation: it would only misfire on a contrived,
     non-game-reachable board where a 2-file king slide or an empty-square
     diagonal pawn move isn't actually castling/en-passant -- doesn't happen
     in real ChessBench games.
  3. dedupe: within each shard (not globally -- 100M keys would be a real
     memory cost, and the plan only asks for within-shard dedup), keyed by
     the raw `square_tokens` row bytes.
  Plus: `cp` is clamped to +/-`--cp-clamp` (default 3000; the design spec
  gives no exact number, this is a conservative "no forced-mate-scale outlier
  dominates the loss" bound, chosen the same order of magnitude as typical
  NNUE cp scales and easily overridable per-run).

IMPORTANT interpretation note on Step 2's smoke-test invariant: the plan's
literal assertion ("output record count equals input position count, byte
length == count*32") is a check on the *packer*, and is only true when the
quiet filters are OFF (with them on, the whole point is that some positions
are dropped). `main()` supports `--no-filter-check`, `--no-filter-quiet-move`,
`--no-dedupe` for exactly this reason: run once with all three off to validate
the base 1:1 pack/unpack round-trip, and once with the (default-on) filters to
see the real production drop-rate. See the smoke-test script output recorded
in the M2 task report for both numbers.

This streams: each shard's arrays are loaded into memory one shard at a time
(a few hundred thousand positions), never the full ~100M-position corpus.
"""

import argparse
import glob
import math
import os
import struct
import sys

import chess
import numpy as np

# repo root is two levels up from ferrum/tools/ (this file lives at
# ferrum/tools/chessbench_to_bullet.py); insert it so `src.*` resolves
# regardless of the caller's cwd, mirroring scripts/preencode.py's convention.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.game.move_encoder import get_move_encoder

DEFAULT_CP_CLAMP = 3000

# --- 32-byte bulletformat::ChessBoard, little-endian, no implicit padding ---
#   occ:u64  pcs:[u8;16]  score:i16  result:u8  ksq:u8  opp_ksq:u8  extra:[u8;3]
_RECORD_STRUCT = struct.Struct("<Q16shBBB3x")
assert _RECORD_STRUCT.size == 32, _RECORD_STRUCT.size

_OWN_FEN_CHARS = "_PNBRQK"     # index [1..6] -> own (mover) piece char
_ENEMY_FEN_CHARS = "_pnbrqk"   # index [1..6] -> enemy piece char
_RESULT_STR = {0: "0.0", 1: "0.5", 2: "1.0"}


# ---------------------------------------------------------------------------
# Value target: wdl -> win probability -> centipawns (exact inversion, README)
# ---------------------------------------------------------------------------

def wdl_to_winprob(w: float, d: float, l: float) -> float:
    """Invert `src/data/targets.py::winprob_to_wdl(wp, draw_scale=2.0)` exactly."""
    sign = 1.0 if w >= l else -1.0
    wp = 0.5 + sign * math.sqrt(max(0.0, 1.0 - d)) / 2.0
    return min(max(wp, 1e-6), 1.0 - 1e-6)


def winprob_to_cp(wp: float) -> int:
    """Inverse logistic: win probability -> side-to-move-relative centipawns."""
    return round(400.0 * math.log10(wp / (1.0 - wp)))


def clamp_cp(cp: int, limit: int = DEFAULT_CP_CLAMP) -> int:
    return max(-limit, min(limit, cp))


def result_byte_from_winprob(wp: float, eps: float = 1e-3) -> int:
    """0=loss, 1=draw, 2=win (bullet convention); unused by gen-0's score-only loss."""
    if wp > 0.5 + eps:
        return 2
    if wp < 0.5 - eps:
        return 0
    return 1


# ---------------------------------------------------------------------------
# Board reconstruction (piece placement only -- no castling rights/ep square)
# ---------------------------------------------------------------------------

def build_base_board(square_tokens_row) -> chess.BaseBoard:
    """Reconstruct the canonical (mover-as-White) position's piece placement.

    `square_tokens_row` is one row of the npz `square_tokens` array (64 int8).
    Square index `rank*8+file` matches python-chess's convention directly.
    """
    board = chess.BaseBoard(None)  # None board_fen -> empty
    for sq in range(64):
        tok = int(square_tokens_row[sq])
        if tok == 0:
            continue
        color = chess.WHITE if tok <= 6 else chess.BLACK
        piece_type = tok if tok <= 6 else tok - 6
        board.set_piece_at(sq, chess.Piece(piece_type, color))
    return board


def is_in_check(board: chess.BaseBoard) -> bool:
    """Is the mover (canonical White) in check?"""
    king_sq = board.king(chess.WHITE)
    return king_sq is not None and board.is_attacked_by(chess.BLACK, king_sq)


def best_move_from_policy(legal_indices_slice, legal_probs_slice):
    """Decode `argmax(legal_probs)`'s move (already canonical-frame). None if empty."""
    if len(legal_probs_slice) == 0:
        return None
    best = int(np.argmax(legal_probs_slice))
    idx = int(legal_indices_slice[best])
    return get_move_encoder().decode(idx)


def apply_move_on_board(board: chess.BaseBoard, move: chess.Move):
    """Mechanically apply `move` (mover is always White) to a COPY of `board`.

    Handles captures, promotions, en-passant, and castling geometrically (no
    rights/ep-square tracked -- see module docstring). Returns
    `(new_board, is_capture)`.
    """
    b = board.copy()
    moving_pt = b.piece_type_at(move.from_square)
    captured_pt = b.piece_type_at(move.to_square)
    is_capture = captured_pt is not None

    is_en_passant = (
        moving_pt == chess.PAWN
        and chess.square_file(move.from_square) != chess.square_file(move.to_square)
        and captured_pt is None
    )
    if is_en_passant:
        captured_sq = chess.square(
            chess.square_file(move.to_square), chess.square_rank(move.from_square)
        )
        b.remove_piece_at(captured_sq)
        is_capture = True

    is_castling = (
        moving_pt == chess.KING
        and chess.square_rank(move.from_square) == chess.square_rank(move.to_square)
        and abs(chess.square_file(move.from_square) - chess.square_file(move.to_square)) == 2
    )

    b.remove_piece_at(move.from_square)
    new_pt = move.promotion if move.promotion else moving_pt
    # The repo's MoveEncoder encodes queen promotions as ordinary from/to moves
    # (promotion=None), so a pawn reaching the last rank without a flag is a queen
    # promotion — restore it, else a checking queen-promo is misread as a quiet
    # pawn push and a non-quiet position is wrongly kept in the training corpus.
    if moving_pt == chess.PAWN and not move.promotion and chess.square_rank(move.to_square) == 7:
        new_pt = chess.QUEEN
    b.set_piece_at(move.to_square, chess.Piece(new_pt, chess.WHITE))

    if is_castling:
        kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
        rank = chess.square_rank(move.from_square)
        rook_from = chess.square(7 if kingside else 0, rank)
        rook_to = chess.square(5 if kingside else 3, rank)
        b.remove_piece_at(rook_from)
        b.set_piece_at(rook_to, chess.Piece(chess.ROOK, chess.WHITE))

    return b, is_capture


def move_is_capture_or_gives_check(board: chess.BaseBoard, move: chess.Move) -> bool:
    new_board, is_capture = apply_move_on_board(board, move)
    if is_capture:
        return True
    opp_king = new_board.king(chess.BLACK)
    return opp_king is not None and new_board.is_attacked_by(chess.WHITE, opp_king)


# ---------------------------------------------------------------------------
# bulletformat::ChessBoard packing (path B) + a synthetic FEN line (path A)
# ---------------------------------------------------------------------------

def pack_bullet_record(square_tokens_row, cp: int, result: int) -> bytes:
    """Pack one position into bullet's 32-byte `ChessBoard` record (path B)."""
    occ = 0
    nibbles = []
    own_ksq = None
    enemy_ksq = None
    for sq in range(64):
        tok = int(square_tokens_row[sq])
        if tok == 0:
            continue
        occ |= 1 << sq
        is_enemy = 1 if tok > 6 else 0
        piece_type0 = (tok - 1) % 6  # 0=P .. 5=K (bullet convention)
        nibbles.append((is_enemy << 3) | piece_type0)
        if tok == 6:
            own_ksq = sq
        elif tok == 12:
            enemy_ksq = sq
    if own_ksq is None or enemy_ksq is None:
        raise ValueError("position missing a king; cannot form a bullet record")
    if len(nibbles) > 32:
        raise ValueError(f"{len(nibbles)} occupied squares > 32; not a legal chess position")

    pcs = bytearray(16)
    for i, nib in enumerate(nibbles):
        pcs[i // 2] |= nib << (4 * (i & 1))
    opp_ksq = enemy_ksq ^ 56  # opponent's own-perspective king square (see docstring)

    return _RECORD_STRUCT.pack(occ, bytes(pcs), cp, result, own_ksq, opp_ksq)


def unpack_bullet_record(rec: bytes) -> dict:
    """Inverse of `pack_bullet_record`, for round-trip verification/debugging."""
    occ, pcs, score, result, ksq, opp_ksq = _RECORD_STRUCT.unpack(rec)
    pieces = {}
    i = 0
    occ2 = occ
    while occ2:
        sq = (occ2 & -occ2).bit_length() - 1
        nib = (pcs[i // 2] >> (4 * (i & 1))) & 0x0F
        pieces[sq] = ((nib >> 3) & 1, nib & 0x07)  # (is_enemy, piece_type0)
        occ2 &= occ2 - 1
        i += 1
    return {"occ": occ, "pieces": pieces, "score": score, "result": result,
            "ksq": ksq, "opp_ksq": opp_ksq}


def square_tokens_to_fen(square_tokens_row) -> str:
    """Synthetic `<placement> w - - 0 1` FEN from a canonical-frame board row."""
    ranks = []
    for rank in range(7, -1, -1):
        empty = 0
        chars = []
        for file in range(8):
            tok = int(square_tokens_row[rank * 8 + file])
            if tok == 0:
                empty += 1
                continue
            if empty:
                chars.append(str(empty))
                empty = 0
            chars.append(_OWN_FEN_CHARS[tok] if tok <= 6 else _ENEMY_FEN_CHARS[tok - 6])
        if empty:
            chars.append(str(empty))
        ranks.append("".join(chars))
    return "/".join(ranks) + " w - - 0 1"


# ---------------------------------------------------------------------------
# Streaming shard -> record conversion
# ---------------------------------------------------------------------------

def iter_bullet_records(npz_path, cp_clamp=DEFAULT_CP_CLAMP, dedupe=True,
                        filter_check=True, filter_quiet_move=True, stats=None):
    """Yield `(record_bytes, square_tokens_row, cp, result)` for one shard.

    Loads one shard's arrays into memory (a few hundred thousand positions at
    100k-shard scale) -- never the whole corpus. `stats`, if given, is a dict
    with keys total/kept/dropped_dup/dropped_in_check/dropped_noisy_move that
    gets incremented in place.
    """
    d = np.load(npz_path)
    sq_all = d["square_tokens"]
    wdl_all = d["wdl"]
    legal_indices = d["legal_indices"]
    legal_probs = d["legal_probs"]
    counts = d["counts"]

    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    seen = set() if dedupe else None

    for i in range(sq_all.shape[0]):
        if stats is not None:
            stats["total"] += 1
        row = sq_all[i]

        if dedupe:
            key = row.tobytes()
            if key in seen:
                if stats is not None:
                    stats["dropped_dup"] += 1
                continue
            seen.add(key)

        board = None
        if filter_check or filter_quiet_move:
            board = build_base_board(row)

        if filter_check and is_in_check(board):
            if stats is not None:
                stats["dropped_in_check"] += 1
            continue

        if filter_quiet_move:
            lo, hi = offsets[i], offsets[i + 1]
            move = best_move_from_policy(legal_indices[lo:hi], legal_probs[lo:hi])
            if move is not None and move_is_capture_or_gives_check(board, move):
                if stats is not None:
                    stats["dropped_noisy_move"] += 1
                continue

        w, dd, l = (float(x) for x in wdl_all[i])
        wp = wdl_to_winprob(w, dd, l)
        cp = clamp_cp(winprob_to_cp(wp), cp_clamp)
        result = result_byte_from_winprob(wp)

        if stats is not None:
            stats["kept"] += 1
        yield pack_bullet_record(row, cp, result), row, cp, result


def new_stats() -> dict:
    return {"total": 0, "kept": 0, "dropped_dup": 0, "dropped_in_check": 0,
            "dropped_noisy_move": 0}


def convert(input_glob, output_path, cp_clamp=DEFAULT_CP_CLAMP, dedupe=True,
           filter_check=True, filter_quiet_move=True, text_out_path=None,
           limit=None):
    """Stream every shard matching `input_glob` into one flat bullet `.bin`.

    Optionally also writes the path-A text intermediate to `text_out_path`.
    `limit` caps total emitted records (debug/smoke convenience only).
    """
    paths = sorted(glob.glob(input_glob))
    if not paths:
        raise SystemExit(f"no shards matched {input_glob!r}")

    stats = new_stats()
    text_fh = open(text_out_path, "w") if text_out_path else None
    emitted = 0
    try:
        with open(output_path, "wb") as out_bin:
            for shard_path in paths:
                for rec, row, cp, result in iter_bullet_records(
                    shard_path, cp_clamp, dedupe, filter_check, filter_quiet_move, stats
                ):
                    out_bin.write(rec)
                    if text_fh is not None:
                        text_fh.write(
                            f"{square_tokens_to_fen(row)}|{cp}|{_RESULT_STR[result]}\n"
                        )
                    emitted += 1
                    if limit is not None and emitted >= limit:
                        return stats, len(paths)
    finally:
        if text_fh is not None:
            text_fh.close()
    return stats, len(paths)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="glob of input npz shards")
    ap.add_argument("--output", required=True, help="output flat .bin path (32B/record)")
    ap.add_argument("--cp-clamp", type=int, default=DEFAULT_CP_CLAMP)
    ap.add_argument("--emit-text", default=None,
                    help="also write a path-A '<FEN>|<score>|<result>' text intermediate")
    ap.add_argument("--no-filter-check", action="store_true",
                    help="disable the in-check quiet filter")
    ap.add_argument("--no-filter-quiet-move", action="store_true",
                    help="disable the best-move capture/gives-check quiet filter")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="disable within-shard dedup by square_tokens bytes")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap total emitted records (debug/smoke only)")
    args = ap.parse_args()

    stats, n_shards = convert(
        args.input, args.output,
        cp_clamp=args.cp_clamp,
        dedupe=not args.no_dedupe,
        filter_check=not args.no_filter_check,
        filter_quiet_move=not args.no_filter_quiet_move,
        text_out_path=args.emit_text,
        limit=args.limit,
    )
    print(
        f"done: {stats['kept']} kept / {stats['total']} read across {n_shards} shard(s) "
        f"(dropped: dup={stats['dropped_dup']}, in_check={stats['dropped_in_check']}, "
        f"noisy_move={stats['dropped_noisy_move']})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
