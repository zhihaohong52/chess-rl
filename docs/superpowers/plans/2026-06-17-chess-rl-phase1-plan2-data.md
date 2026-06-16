# Phase 1 Plan 2 — Data Pipeline (Slice, Pre-encode, Loader) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Stockfish-labeled chess positions into pre-encoded TFRecord shards and a `tf.data` loader that yields training batches matching the frozen shard contract.

**Architecture:** A `LabeledPosition` (FEN + per-move policy + WDL + moves-left) is the universal input. Source readers (ChessBench action-value; a Kaggle CSV smoke source) emit `LabeledPosition`s; `preencode` canonicalizes each via Plan 1's `token_encoder`/`orientation`/`move_encoder` and writes TFRecord shards; `dataset` reads shards and scatters the sparse policy into a dense `[P]` target.

**Tech Stack:** TensorFlow `tf.data`/TFRecord, python-chess, NumPy, pytest. Depends on Plan 1 (`src/game/token_encoder.py`, `orientation.py`, `move_encoder.py`).

**Contracts:** `docs/superpowers/plans/_interface-contracts.md` — "Data shard format". Tests use inline fixtures only (no network, no Stockfish).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/data/__init__.py` (new) | Package marker |
| `src/data/targets.py` (new) | Pure label math: centipawn/win% → WDL, move scores → policy |
| `src/data/labeled_position.py` (new) | `LabeledPosition` dataclass (the universal input record) |
| `src/data/preencode.py` (new) | `LabeledPosition` → canonical example dict → TFRecord (serialize + write_shard) |
| `src/data/dataset.py` (new) | TFRecord → `tf.data` batches (scatter sparse policy → dense `[P]`) |
| `src/data/chessbench.py` (new) | Source readers: ChessBench (.bag, documented) + Kaggle CSV (smoke) |
| `tests/data/` (new) | Unit tests + a tiny committed CSV fixture |

All shell commands prefixed with `rtk`. Use `.venv/bin/python` and `.venv/bin/python -m pytest`.

---

### Task 1: Label math (`targets.py`)

**Files:**
- Create: `src/data/__init__.py`, `src/data/targets.py`
- Test: `tests/data/__init__.py`, `tests/data/test_targets.py`

- [ ] **Step 1: Write the failing test**

`tests/data/__init__.py`: (empty)

`tests/data/test_targets.py`:
```python
import numpy as np
from src.data.targets import cp_to_winprob, winprob_to_wdl, scores_to_policy


def test_cp_to_winprob_monotonic_and_centered():
    assert abs(cp_to_winprob(0) - 0.5) < 1e-6
    assert cp_to_winprob(400) > 0.8
    assert cp_to_winprob(-400) < 0.2


def test_winprob_to_wdl_sums_to_one():
    w, d, l = winprob_to_wdl(0.5)
    assert abs(w + d + l - 1.0) < 1e-6
    assert d > 0 and abs(w - l) < 1e-6  # symmetric at 0.5


def test_scores_to_policy_softmax():
    moves = ["e2e4", "d2d4", "a2a3"]
    cps = [50, 40, -200]
    pol = scores_to_policy(moves, cps, temperature=1.0)
    assert abs(sum(p for _, p in pol) - 1.0) < 1e-6
    best = max(pol, key=lambda mp: mp[1])
    assert best[0] == "e2e4"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk .venv/bin/python -m pytest tests/data/test_targets.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.targets'`.

- [ ] **Step 3: Write minimal implementation**

`src/data/__init__.py`: (empty)

`src/data/targets.py`:
```python
"""Pure label-math helpers for distillation targets."""

import numpy as np


def cp_to_winprob(cp: float) -> float:
    """Centipawn eval -> win probability in (0,1) via logistic (400cp ~ 0.76)."""
    return 1.0 / (1.0 + 10.0 ** (-cp / 400.0))


def winprob_to_wdl(wp: float, draw_scale: float = 2.0):
    """Map a win probability to (W, D, L). Draw mass peaks near wp=0.5.

    D = 1 - |2*wp - 1|**draw_scale shaped, then W/L split the remainder by wp.
    """
    decisiveness = abs(2.0 * wp - 1.0) ** draw_scale
    d = 1.0 - decisiveness
    rem = 1.0 - d
    w = rem * wp
    l = rem * (1.0 - wp)
    s = w + d + l
    return (w / s, d / s, l / s)


def scores_to_policy(moves, cps, temperature: float = 1.0):
    """Softmax over per-move centipawn scores -> [(move, prob), ...]."""
    arr = np.array(cps, dtype=np.float32)
    denom = max(1e-3, 400.0 * max(temperature, 1e-3))
    logits = arr / denom
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs = probs / probs.sum()
    return list(zip(moves, probs.tolist()))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk .venv/bin/python -m pytest tests/data/test_targets.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
rtk git add src/data/__init__.py src/data/targets.py tests/data/__init__.py tests/data/test_targets.py && rtk git commit -m "feat: distillation label math"
```

---

### Task 2: `LabeledPosition` record

**Files:**
- Create: `src/data/labeled_position.py`
- Test: `tests/data/test_labeled_position.py`

- [ ] **Step 1: Write the failing test**

`tests/data/test_labeled_position.py`:
```python
from src.data.labeled_position import LabeledPosition


def test_labeled_position_fields():
    lp = LabeledPosition(
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        policy=[("e2e4", 0.6), ("d2d4", 0.4)],
        wdl=(0.5, 0.4, 0.1),
        moves_left=80.0,
        repetition_count=0,
    )
    assert lp.fen.startswith("rnbqkbnr")
    assert abs(sum(p for _, p in lp.policy) - 1.0) < 1e-6
    assert len(lp.wdl) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk .venv/bin/python -m pytest tests/data/test_labeled_position.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.labeled_position'`.

- [ ] **Step 3: Write minimal implementation**

`src/data/labeled_position.py`:
```python
"""Universal labeled-position record consumed by the pre-encoder."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class LabeledPosition:
    fen: str
    policy: List[Tuple[str, float]]   # (uci_move, prob) over legal moves, sums to 1
    wdl: Tuple[float, float, float]   # (win, draw, loss) from side-to-move POV
    moves_left: float                 # plies to game end (>= 0)
    repetition_count: int = 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk .venv/bin/python -m pytest tests/data/test_labeled_position.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/data/labeled_position.py tests/data/test_labeled_position.py && rtk git commit -m "feat: LabeledPosition record"
```

---

### Task 3: Pre-encode one position → example dict (canonical)

**Files:**
- Create: `src/data/preencode.py`
- Test: `tests/data/test_preencode.py`

- [ ] **Step 1: Write the failing test**

`tests/data/test_preencode.py`:
```python
import numpy as np
import chess
from src.data.labeled_position import LabeledPosition
from src.data.preencode import encode_example
from src.game.move_encoder import get_move_encoder
from src.game.orientation import to_canonical_move


def test_encode_example_canonical_white_to_move():
    lp = LabeledPosition(
        fen=chess.STARTING_FEN,
        policy=[("e2e4", 0.7), ("d2d4", 0.3)],
        wdl=(0.5, 0.4, 0.1),
        moves_left=80.0,
    )
    ex = encode_example(lp)
    assert ex["square_tokens"].shape == (64,) and ex["square_tokens"].dtype == np.int8
    assert ex["state_features"].shape == (18,)
    assert ex["wdl"].shape == (3,) and abs(ex["wdl"].sum() - 1.0) < 1e-6
    # white to move -> canonical == real; e2e4 index present with prob 0.7
    me = get_move_encoder()
    e2e4_idx = me.encode(chess.Move.from_uci("e2e4"))
    pairs = dict(zip(ex["legal_indices"].tolist(), ex["legal_probs"].tolist()))
    assert abs(pairs[e2e4_idx] - 0.7) < 1e-6


def test_encode_example_black_to_move_mirrors_moves():
    b = chess.Board()
    b.push_uci("e2e4")  # black to move
    lp = LabeledPosition(fen=b.fen(), policy=[("e7e5", 1.0)], wdl=(0.4, 0.4, 0.2), moves_left=78.0)
    ex = encode_example(lp)
    me = get_move_encoder()
    expected = me.encode(to_canonical_move(chess.Move.from_uci("e7e5"), chess.BLACK))
    assert ex["legal_indices"].tolist() == [expected]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk .venv/bin/python -m pytest tests/data/test_preencode.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.preencode'`.

- [ ] **Step 3: Write minimal implementation**

`src/data/preencode.py`:
```python
"""Pre-encode LabeledPosition -> canonical example dict and TFRecord shards."""

import chess
import numpy as np
import tensorflow as tf

from src.game.token_encoder import encode_position
from src.game.orientation import to_canonical_move
from src.game.move_encoder import get_move_encoder


def encode_example(lp) -> dict:
    """Convert a LabeledPosition into the canonical numpy example dict."""
    board = chess.Board(lp.fen)
    me = get_move_encoder()
    sq, sf = encode_position(board, lp.repetition_count)

    indices, probs = [], []
    for uci, prob in lp.policy:
        cmove = to_canonical_move(chess.Move.from_uci(uci), board.turn)
        indices.append(me.encode(cmove))
        probs.append(float(prob))

    return {
        "square_tokens": sq.astype(np.int8),
        "state_features": sf.astype(np.float32),
        "legal_indices": np.array(indices, dtype=np.int64),
        "legal_probs": np.array(probs, dtype=np.float32),
        "wdl": np.array(lp.wdl, dtype=np.float32),
        "moves_left": np.float32(lp.moves_left),
    }


def _bytes_feature(b):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[b]))


def _float_list(v):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(v)))


def _int64_list(v):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(v)))


def serialize_example(ex: dict) -> bytes:
    feat = {
        "square_tokens": _bytes_feature(ex["square_tokens"].tobytes()),
        "state_features": _float_list(ex["state_features"]),
        "legal_indices": _int64_list(ex["legal_indices"]),
        "legal_probs": _float_list(ex["legal_probs"]),
        "wdl": _float_list(ex["wdl"]),
        "moves_left": _float_list([float(ex["moves_left"])]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feat)).SerializeToString()


def write_shard(labeled_positions, path: str) -> int:
    """Write an iterable of LabeledPosition to a TFRecord shard. Returns count."""
    n = 0
    with tf.io.TFRecordWriter(path) as w:
        for lp in labeled_positions:
            w.write(serialize_example(encode_example(lp)))
            n += 1
    return n
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk .venv/bin/python -m pytest tests/data/test_preencode.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
rtk git add src/data/preencode.py tests/data/test_preencode.py && rtk git commit -m "feat: pre-encode + TFRecord serialization"
```

---

### Task 4: TFRecord write/read round-trip

**Files:**
- Test: `tests/data/test_shard_roundtrip.py`

- [ ] **Step 1: Write the failing test**

`tests/data/test_shard_roundtrip.py`:
```python
import os
import tensorflow as tf
import chess
from src.data.labeled_position import LabeledPosition
from src.data.preencode import write_shard


def test_write_shard_roundtrip(tmp_path):
    lps = [
        LabeledPosition(chess.STARTING_FEN, [("e2e4", 1.0)], (0.5, 0.4, 0.1), 80.0),
        LabeledPosition(chess.STARTING_FEN, [("d2d4", 0.5), ("c2c4", 0.5)], (0.4, 0.5, 0.1), 79.0),
    ]
    path = str(tmp_path / "shard0.tfrecord")
    n = write_shard(lps, path)
    assert n == 2
    assert os.path.exists(path)
    count = sum(1 for _ in tf.data.TFRecordDataset([path]))
    assert count == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk .venv/bin/python -m pytest tests/data/test_shard_roundtrip.py -v`
Expected: FAIL — collection/import error only if a prior file is missing; otherwise this exercises `write_shard` (already exists) — the test will FAIL first run only if `write_shard` has a bug. If `write_shard` is already correct, this passes; that is acceptable for a verification task — proceed to commit.

- [ ] **Step 3: (No new code expected)** Confirm `write_shard` from Task 3 satisfies the test; if the count assertion fails, fix `write_shard` so each example is written exactly once.

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk .venv/bin/python -m pytest tests/data/test_shard_roundtrip.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add tests/data/test_shard_roundtrip.py && rtk git commit -m "test: TFRecord shard round-trip"
```

---

### Task 5: `tf.data` loader with policy scatter

**Files:**
- Create: `src/data/dataset.py`
- Test: `tests/data/test_dataset.py`

- [ ] **Step 1: Write the failing test**

`tests/data/test_dataset.py`:
```python
import numpy as np
import chess
from src.data.labeled_position import LabeledPosition
from src.data.preencode import write_shard
from src.data.dataset import make_dataset
from src.game.move_encoder import get_move_encoder


def test_loader_batches_and_scatters(tmp_path):
    P = get_move_encoder().policy_size
    lps = [LabeledPosition(chess.STARTING_FEN, [("e2e4", 1.0)], (0.5, 0.4, 0.1), 80.0)] * 4
    path = str(tmp_path / "s.tfrecord")
    write_shard(lps, path)

    ds = make_dataset([path], batch_size=2, policy_size=P, shuffle=False)
    (sq, sf), (pol, wdl, ml) = next(iter(ds))
    assert sq.shape == (2, 64) and sq.dtype.name == "int32"
    assert sf.shape == (2, 18)
    assert pol.shape == (2, P)
    e2e4 = get_move_encoder().encode(chess.Move.from_uci("e2e4"))
    assert abs(float(pol[0, e2e4]) - 1.0) < 1e-5
    assert abs(float(pol[0].numpy().sum()) - 1.0) < 1e-5
    assert wdl.shape == (2, 3) and ml.shape == (2, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk .venv/bin/python -m pytest tests/data/test_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.dataset'`.

- [ ] **Step 3: Write minimal implementation**

`src/data/dataset.py`:
```python
"""tf.data loader: TFRecord shards -> (inputs, targets) batches per contract."""

import tensorflow as tf

_FEATURES = {
    "square_tokens": tf.io.FixedLenFeature([], tf.string),
    "state_features": tf.io.FixedLenFeature([18], tf.float32),
    "legal_indices": tf.io.VarLenFeature(tf.int64),
    "legal_probs": tf.io.VarLenFeature(tf.float32),
    "wdl": tf.io.FixedLenFeature([3], tf.float32),
    "moves_left": tf.io.FixedLenFeature([1], tf.float32),
}


def _make_parse(policy_size: int):
    def _parse(record):
        ex = tf.io.parse_single_example(record, _FEATURES)
        sq = tf.cast(tf.io.decode_raw(ex["square_tokens"], tf.int8), tf.int32)  # [64]
        sq = tf.ensure_shape(sq, [64])
        sf = ex["state_features"]
        idx = tf.sparse.to_dense(ex["legal_indices"])     # [k]
        prob = tf.sparse.to_dense(ex["legal_probs"])      # [k]
        policy = tf.scatter_nd(idx[:, None], prob, [policy_size])  # [P]
        return (sq, sf), (policy, ex["wdl"], ex["moves_left"])
    return _parse


def make_dataset(shard_paths, batch_size: int, policy_size: int,
                 shuffle: bool = True, shuffle_buffer: int = 8192):
    ds = tf.data.TFRecordDataset(shard_paths, num_parallel_reads=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(_make_parse(policy_size), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk .venv/bin/python -m pytest tests/data/test_dataset.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/data/dataset.py tests/data/test_dataset.py && rtk git commit -m "feat: tf.data loader with policy scatter"
```

---

### Task 6: Source readers (Kaggle CSV smoke + ChessBench documented)

**Files:**
- Create: `src/data/chessbench.py`, `tests/data/fixtures/sample_chessdata.csv`
- Test: `tests/data/test_sources.py`

- [ ] **Step 1: Write the failing test + fixture**

`tests/data/fixtures/sample_chessdata.csv`:
```
FEN,Evaluation
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,+20
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1,-15
```

`tests/data/test_sources.py`:
```python
import os
from src.data.chessbench import iter_kaggle_csv
from src.data.labeled_position import LabeledPosition

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "sample_chessdata.csv")


def test_iter_kaggle_csv_yields_value_only_positions():
    lps = list(iter_kaggle_csv(FIX))
    assert len(lps) == 2
    assert all(isinstance(lp, LabeledPosition) for lp in lps)
    # value-only smoke source: policy is uniform over legal moves and sums to 1
    for lp in lps:
        assert abs(sum(p for _, p in lp.policy) - 1.0) < 1e-6
        assert all(0.0 <= w <= 1.0 for w in lp.wdl)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk .venv/bin/python -m pytest tests/data/test_sources.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.chessbench'`.

- [ ] **Step 3: Write minimal implementation**

`src/data/chessbench.py`:
```python
"""Source readers that emit LabeledPosition records.

- iter_kaggle_csv: a value-only SMOKE source (FEN + centipawn eval). Policy is set
  uniform over legal moves purely to exercise the pipeline shape; NOT for real
  training.
- iter_chessbench: the real training source (DeepMind ChessBench action-value
  .bag set: per-legal-move win% + state win% + ply/length). Documented here; the
  full run wires it in. Requires the `searchless_chess` data tooling and is not
  exercised by unit tests (no network).
"""

import csv
import chess

from src.data.labeled_position import LabeledPosition
from src.data.targets import cp_to_winprob, winprob_to_wdl


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


def iter_chessbench(path: str):
    """Real ChessBench action-value source. Implement against the searchless_chess
    .bag readers: for each position emit per-legal-move win% as `policy`
    (via src.data.targets.scores_to_policy on win% logits), `wdl` from the state
    win%, and `moves_left` from the game's ply index and length. Not unit-tested
    (requires the external dataset)."""
    raise NotImplementedError(
        "Wire to DeepMind ChessBench .bag readers for the full run; see docstring."
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk .venv/bin/python -m pytest tests/data/test_sources.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/data/chessbench.py tests/data/fixtures/sample_chessdata.csv tests/data/test_sources.py && rtk git commit -m "feat: Kaggle CSV smoke source + ChessBench reader stub"
```

---

### Task 7: Pre-encode CLI + throughput benchmark

**Files:**
- Create: `scripts/preencode.py`
- Test: `tests/data/test_throughput.py`

- [ ] **Step 1: Write the failing test**

`tests/data/test_throughput.py`:
```python
import time
import pytest
import chess
from src.data.labeled_position import LabeledPosition
from src.data.preencode import write_shard


@pytest.mark.slow
def test_preencode_throughput(tmp_path):
    lps = [LabeledPosition(chess.STARTING_FEN, [("e2e4", 1.0)], (0.5, 0.4, 0.1), 80.0)
           for _ in range(2000)]
    path = str(tmp_path / "bench.tfrecord")
    t0 = time.time()
    n = write_shard(lps, path)
    rate = n / (time.time() - t0)
    print(f"preencode rate: {rate:.0f} positions/sec")
    assert rate > 200  # must comfortably beat GPU consumption
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk .venv/bin/python -m pytest tests/data/test_throughput.py -v -m slow`
Expected: FAIL only if rate < 200/sec (encoder too slow); otherwise PASS. If it fails, profile `encode_example` (the per-move `to_canonical_move`/`encode` loop is the hot path) before proceeding.

- [ ] **Step 3: Write the CLI**

`scripts/preencode.py`:
```python
#!/usr/bin/env python3
"""Pre-encode a source dataset into TFRecord shards.

Usage: python scripts/preencode.py --source kaggle --input data/chessData.csv \
       --out-dir data/shards --shard-size 100000
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.chessbench import iter_kaggle_csv, iter_chessbench
from src.data.preencode import write_shard


def _chunked(it, size):
    chunk = []
    for x in it:
        chunk.append(x)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["kaggle", "chessbench"], required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-size", type=int, default=100000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    src = iter_kaggle_csv(args.input) if args.source == "kaggle" else iter_chessbench(args.input)
    total = 0
    for i, chunk in enumerate(_chunked(src, args.shard_size)):
        path = os.path.join(args.out_dir, f"shard_{i:05d}.tfrecord")
        total += write_shard(chunk, path)
        print(f"wrote {path} ({total} positions)")
    print(f"done: {total} positions")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify the CLI runs on the fixture**

Run: `rtk .venv/bin/python scripts/preencode.py --source kaggle --input tests/data/fixtures/sample_chessdata.csv --out-dir /tmp/ds_smoke --shard-size 1`
Expected: prints two shard writes and `done: 2 positions`.

- [ ] **Step 5: Commit**

```bash
rtk git add scripts/preencode.py tests/data/test_throughput.py && rtk git commit -m "feat: preencode CLI + throughput benchmark"
```

---

## Self-Review (completed by plan author)

- **Spec coverage (§5):** ChessBench action-value source (documented reader) ✓; pre-encode on M1 to packed shards ✓; canonical storage (positions stored already-canonical via `encode_position`/`to_canonical_move`) ✓; `tf.data` loader with prefetch ✓; Kaggle starter for bring-up ✓; throughput gate ✓.
- **Contract consistency:** shard fields (`square_tokens` int8[64], `state_features`[18], `legal_indices`/`legal_probs`, `wdl`[3], `moves_left`) and the loader's `(inputs, targets)` shape with dense `[B,P]` policy match `_interface-contracts.md`. `P` read from `get_move_encoder().policy_size`.
- **Placeholders:** none. `iter_chessbench` is a deliberate documented `NotImplementedError` for the external dataset (not a placeholder for testable code) — clearly out of unit-test scope per the milestone constraint (no network).
- **Naming consistency:** `LabeledPosition`, `encode_example`, `serialize_example`, `write_shard`, `make_dataset`, `iter_kaggle_csv` used consistently across tasks/tests.

## Notes for the implementer
- Tests never touch the network or Stockfish; the real ChessBench wiring (`iter_chessbench`) is exercised only during the actual data build, not CI.
- Value-only Kaggle examples use a uniform policy purely so the pipeline shape is exercised — do **not** train a real model on uniform-policy data; real training uses ChessBench per-move targets.
