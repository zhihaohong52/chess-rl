"""Source readers that emit LabeledPosition records.

- iter_kaggle_csv: a value-only SMOKE source (FEN + centipawn eval). Policy is set
  uniform over legal moves purely to exercise the pipeline shape; NOT for real
  training.
- iter_chessbench: the real training source — DeepMind ChessBench action_value
  `.bag` files (records are (fen, uci_move, win_prob) triples). All legal moves of
  a position are generated per-state and stored contiguously, so we group
  consecutive same-FEN records into one position: policy = softmax(win/T) over its
  moves, value/WDL from the best move's win%. Download with
  `scripts/download_chessbench.py`.
"""

import csv
import struct
import sys

import chess
import numpy as np

from src.data.labeled_position import LabeledPosition
from src.data.targets import cp_to_winprob, winprob_to_wdl
from src.data.bagz import read_records

# moves-left is not available from action_value (no game trajectory); emit a
# neutral constant so the small (0.1-weight) moves-left head still has a target.
_DEFAULT_MOVES_LEFT = 40.0


# ---------------------------------------------------------------------------
# Kaggle CSV smoke source (value-only)
# ---------------------------------------------------------------------------

def _parse_cp(text: str) -> float:
    text = text.strip().replace("+", "")
    if text.startswith("#"):  # mate score, e.g. "#+3"
        return 10000.0 if "-" not in text else -10000.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def iter_kaggle_csv(path: str):
    """Yield value-only LabeledPositions from a Kaggle chessData.csv (FEN,Evaluation)."""
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fen = row["FEN"].strip()
            board = chess.Board(fen)
            legal = list(board.legal_moves)
            if not legal:
                continue
            cp = _parse_cp(row["Evaluation"])
            wp = cp_to_winprob(cp)
            uniform = 1.0 / len(legal)
            policy = [(m.uci(), uniform) for m in legal]
            yield LabeledPosition(
                fen=fen, policy=policy, wdl=winprob_to_wdl(wp), moves_left=60.0
            )


# ---------------------------------------------------------------------------
# ChessBench action_value record codec (Apache Beam TupleCoder wire format)
#   TupleCoder(StrUtf8Coder, StrUtf8Coder, FloatCoder), nested:
#   [varint len(fen)][fen][varint len(move)][move][8-byte big-endian double]
# ---------------------------------------------------------------------------

def _read_varint(buf, pos):
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _write_varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def decode_action_value(record: bytes):
    """Decode one ChessBench action_value record -> (fen, uci_move, win_prob)."""
    pos = 0
    n, pos = _read_varint(record, pos)
    fen = record[pos:pos + n].decode("utf-8")
    pos += n
    n, pos = _read_varint(record, pos)
    move = record[pos:pos + n].decode("utf-8")
    pos += n
    (win,) = struct.unpack(">d", record[pos:pos + 8])
    return fen, move, win


def encode_action_value(fen: str, move: str, win_prob: float) -> bytes:
    """Inverse of decode_action_value (for tests/tooling)."""
    fb = fen.encode("utf-8")
    mb = move.encode("utf-8")
    return (_write_varint(len(fb)) + fb
            + _write_varint(len(mb)) + mb
            + struct.pack(">d", float(win_prob)))


def _build_position(fen, moves, wins, temperature):
    w = np.asarray(wins, dtype=np.float64)
    logits = w / max(temperature, 1e-6)
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    policy = [(m, float(p)) for m, p in zip(moves, probs)]
    wdl = winprob_to_wdl(float(w.max()))  # position value ~ best move's win%
    return LabeledPosition(fen=fen, policy=policy, wdl=wdl,
                           moves_left=_DEFAULT_MOVES_LEFT)


def iter_hf_dense(paths, temperature: float = 0.1, max_positions=None):
    """Yield dense LabeledPositions from prdev/chessbench-full-policy-value shards.

    Each shard is a zstd-compressed stream of msgpack records:
        {"fen": str, "moves": {uci: {"win_prob": float in [0,1], "mate": ...}}}
    `win_prob` is from the side-to-move perspective (a forced mate by the mover is
    1.0), matching our canonical orientation and the ChessBench action_value
    semantics. We build a dense policy = softmax(win/temperature) over ALL provided
    legal moves and value/WDL from the best move's win%. Use a sharp temperature
    (default 0.1) so decisive best moves dominate the target.
    """
    import zstandard  # lazy: only this reader needs them
    import msgpack

    if isinstance(paths, str):
        paths = [paths]
    dctx = zstandard.ZstdDecompressor()
    emitted = 0
    for path in paths:
        with open(path, "rb") as fh:
            reader = dctx.stream_reader(fh)
            unpacker = msgpack.Unpacker(reader, raw=False)
            for rec in unpacker:
                mvs = rec.get("moves")
                if not mvs:
                    continue
                moves = list(mvs.keys())
                wins = [(v["win_prob"] if isinstance(v, dict) else float(v))
                        for v in mvs.values()]
                yield _build_position(rec["fen"], moves, wins, temperature)
                emitted += 1
                if max_positions is not None and emitted >= max_positions:
                    return


def iter_chessbench_actionvalue(paths, max_positions=None):
    """Yield raw (fen, uci_move, win_prob) action-value samples — NO grouping.

    The ChessBench *train* shards are pointwise and globally shuffled: each FEN
    appears ~once per file (one randomly-sampled legal move), so grouping by FEN
    is pointless. The native objective for this data is action-value regression:
    predict win_prob for the single (state, action) sample. `win_prob` is from the
    side-to-move's perspective, matching our canonical (side-to-move) orientation.

    Streams record-by-record (constant memory, no dict), so it scales to the full
    multi-GB shards. `max_positions` caps how many samples are emitted.
    """
    if isinstance(paths, str):
        paths = [paths]
    emitted = 0
    for path in paths:
        for record in read_records(path):
            fen, move, win = decode_action_value(record)
            yield fen, move, float(win)
            emitted += 1
            if max_positions is not None and emitted >= max_positions:
                return


def iter_chessbench(paths, temperature: float = 0.1, max_positions=None):
    """Yield LabeledPositions from ChessBench action_value .bag file(s).

    Records are pointwise (fen, move, win_prob) and SHUFFLED within a file, but
    every legal move of a position is present in the same file (~29 moves/position
    on the test set). We group by FEN with an in-memory dict per file (giving
    complete per-position distributions), then emit: policy = softmax(win/T) over
    the moves, value/WDL from the best move's win%.

    Memory scales with one file's record count — fine for the 141 MB test bag; for
    the ~1.2 GB train shards ensure adequate RAM (or pre-split). `max_positions`
    caps how many positions are emitted (the file is still read fully to complete
    the groups).
    """
    if isinstance(paths, str):
        paths = [paths]

    emitted = 0
    for path in paths:
        groups: dict = {}
        order: list = []
        for record in read_records(path):
            fen, move, win = decode_action_value(record)
            g = groups.get(fen)
            if g is None:
                g = ([], [])
                groups[fen] = g
                order.append(fen)
            g[0].append(sys.intern(move))  # ~1924 unique moves -> share objects
            g[1].append(win)
        for fen in order:
            moves, wins = groups[fen]
            yield _build_position(fen, moves, wins, temperature)
            emitted += 1
            if max_positions is not None and emitted >= max_positions:
                return
