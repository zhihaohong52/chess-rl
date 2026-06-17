# Phase 1 Plan 1 — Foundation (Token Encoder + Transformer Network) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Implemented in PyTorch (2026-06-18).** This plan's code blocks are TensorFlow/Keras and are now superseded — the project moved to PyTorch + MPS. See the committed `src/model/{transformer,smolgen,heads}.py` and the spec's update banner for the authoritative implementation.

**Goal:** Build the board-as-tokens encoder and the transformer policy/value/moves-left network (with smolgen), fully unit-tested, with no data/training/MCTS — the foundation every later milestone consumes.

**Architecture:** A 64-square token board representation in canonical (side-to-move) orientation plus an 18-dim global-state vector → a ~10–11M-param transformer (8 layers, d_model 256, 8 heads, smolgen attention bias) → an attention-style from→to policy head over the existing 1924-action move space, a 3-way WDL value head, and a moves-left head.

**Tech Stack:** Python 3.12, TensorFlow/Keras (+ tensorflow-metal on M1), python-chess, NumPy, pytest.

**Contracts:** `docs/superpowers/plans/_interface-contracts.md` (FROZEN — obey exactly).

---

## File Structure

| File | Responsibility |
|---|---|
| `tests/__init__.py`, `tests/conftest.py` | Make `tests/` a package; put repo root on `sys.path` |
| `requirements.txt` (modify) | Add `tensorflow`, `tensorflow-metal`, `pytest` |
| `src/game/orientation.py` (new) | Canonical-orientation helpers (board + move flip) |
| `src/game/token_encoder.py` (new) | Board → `(square_tokens[64], state_features[18])` |
| `config.py` (modify) | Transformer + smolgen + distill hyperparameters |
| `src/model/smolgen.py` (new) | Smolgen layer → per-head `[B,H,64,64]` attention bias |
| `src/model/heads.py` (new) | `policy_index_map`, policy/value/moves-left heads |
| `src/model/transformer.py` (new) | `ChessTransformer` assembling encoder + heads |
| `tests/game/`, `tests/model/` (new) | Unit tests mirroring the above |

All shell commands are prefixed with `rtk` per repo `CLAUDE.md`. Use `.venv/bin/python` and `.venv/bin/python -m pytest`.

---

### Task 0: Environment & test infrastructure

**Files:**
- Create: `tests/__init__.py`, `tests/conftest.py`, `tests/test_smoke.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Write the failing test**

`tests/test_smoke.py`:
```python
def test_imports_and_tf():
    import numpy as np
    import chess
    import tensorflow as tf
    assert tf.__version__ >= "2.16"
    assert chess.square_mirror(0) == 56  # a1 -> a8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk .venv/bin/python -m pytest tests/test_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pytest'` (pytest not yet installed), or `tensorflow` import error.

- [ ] **Step 3: Create test scaffolding and install deps**

`tests/__init__.py`: (empty file)

`tests/conftest.py`:
```python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

Append to `requirements.txt`:
```
tensorflow>=2.16.0
tensorflow-metal>=1.1.0
pytest>=8.0.0
```

Install: `rtk .venv/bin/pip install -r requirements.txt`
(If `tensorflow-metal` fails to resolve on this machine, install `tensorflow` + `pytest` first, then `rtk .venv/bin/pip install tensorflow-metal`; if metal still fails, proceed without it — it only affects local inference speed, tracked in Plan 4.)

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk .venv/bin/python -m pytest tests/test_smoke.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add tests/__init__.py tests/conftest.py tests/test_smoke.py requirements.txt && rtk git commit -m "test: add pytest scaffolding and TF/Metal deps"
```

---

### Task 1: Canonical-orientation helpers

**Files:**
- Create: `src/game/orientation.py`
- Test: `tests/game/__init__.py`, `tests/game/test_orientation.py`

- [ ] **Step 1: Write the failing test**

`tests/game/__init__.py`: (empty)

`tests/game/test_orientation.py`:
```python
import chess
from src.game.orientation import canonical_board, to_canonical_move, from_canonical_move


def test_white_to_move_is_identity():
    b = chess.Board()
    assert canonical_board(b).fen() == b.fen()
    mv = chess.Move.from_uci("e2e4")
    assert to_canonical_move(mv, chess.WHITE) == mv


def test_black_to_move_mirrors_and_side_is_white():
    b = chess.Board()
    b.push_uci("e2e4")  # now black to move
    cb = canonical_board(b)
    assert cb.turn == chess.WHITE  # side-to-move always White on canonical board


def test_move_roundtrip_is_involution():
    b = chess.Board()
    b.push_uci("e2e4")  # black to move
    for mv in b.legal_moves:
        cmv = to_canonical_move(mv, b.turn)
        assert from_canonical_move(cmv, b.turn) == mv
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk .venv/bin/python -m pytest tests/game/test_orientation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.game.orientation'`.

- [ ] **Step 3: Write minimal implementation**

`src/game/orientation.py`:
```python
"""Canonical (side-to-move) orientation helpers."""

import chess


def canonical_board(board: chess.Board) -> chess.Board:
    """Return the board from the side-to-move's perspective.

    python-chess Board.mirror() swaps colors and flips vertically, so the
    side-to-move is always White on the canonical board.
    """
    return board if board.turn == chess.WHITE else board.mirror()


def to_canonical_move(move: chess.Move, turn: bool) -> chess.Move:
    """Map a real move to the canonical frame (mirror squares if Black to move)."""
    if turn == chess.WHITE:
        return move
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )


def from_canonical_move(move: chess.Move, turn: bool) -> chess.Move:
    """Map a canonical-frame move back to the real frame (mirror is its own inverse)."""
    return to_canonical_move(move, turn)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk .venv/bin/python -m pytest tests/game/test_orientation.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
rtk git add src/game/orientation.py tests/game/__init__.py tests/game/test_orientation.py && rtk git commit -m "feat: canonical orientation helpers"
```

---

### Task 2: Token encoder — square tokens

**Files:**
- Create: `src/game/token_encoder.py`
- Test: `tests/game/test_token_encoder.py`

- [ ] **Step 1: Write the failing test**

`tests/game/test_token_encoder.py`:
```python
import chess
import numpy as np
from src.game.token_encoder import encode_square_tokens
from src.game.orientation import canonical_board


def test_startpos_square_tokens():
    toks = encode_square_tokens(canonical_board(chess.Board()))
    assert toks.shape == (64,) and toks.dtype == np.int32
    assert toks[chess.A1] == 4   # White rook (side-to-move) -> id 4
    assert toks[chess.E1] == 6   # White king -> id 6
    assert toks[chess.D8] == 11  # opponent queen -> id 11
    assert toks[chess.E4] == 0   # empty


def test_black_to_move_pieces_become_side_to_move_ids():
    b = chess.Board()
    b.push_uci("e2e4")  # black to move -> canonical mirror
    toks = encode_square_tokens(canonical_board(b))
    # Black's own pieces (now side-to-move) must use ids 1..6 somewhere
    assert toks.max() >= 6 and (toks[(toks >= 1) & (toks <= 6)].size > 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk .venv/bin/python -m pytest tests/game/test_token_encoder.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.game.token_encoder'`.

- [ ] **Step 3: Write minimal implementation**

`src/game/token_encoder.py`:
```python
"""Token encoder: board -> 64 square tokens + 18-dim global-state vector (canonical frame)."""

import chess
import numpy as np

from .orientation import canonical_board

STATE_DIM = 18

# On the canonical board the side-to-move is always White.
_PIECE_TO_ID = {
    (chess.WHITE, chess.PAWN): 1, (chess.WHITE, chess.KNIGHT): 2, (chess.WHITE, chess.BISHOP): 3,
    (chess.WHITE, chess.ROOK): 4, (chess.WHITE, chess.QUEEN): 5, (chess.WHITE, chess.KING): 6,
    (chess.BLACK, chess.PAWN): 7, (chess.BLACK, chess.KNIGHT): 8, (chess.BLACK, chess.BISHOP): 9,
    (chess.BLACK, chess.ROOK): 10, (chess.BLACK, chess.QUEEN): 11, (chess.BLACK, chess.KING): 12,
}


def encode_square_tokens(cboard: chess.Board) -> np.ndarray:
    """Return (64,) int32 piece ids for an already-canonicalized board."""
    toks = np.zeros(64, dtype=np.int32)
    for sq, piece in cboard.piece_map().items():
        toks[sq] = _PIECE_TO_ID[(piece.color, piece.piece_type)]
    return toks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk .venv/bin/python -m pytest tests/game/test_token_encoder.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
rtk git add src/game/token_encoder.py tests/game/test_token_encoder.py && rtk git commit -m "feat: square-token encoding"
```

---

### Task 3: Token encoder — state features + encode_position/encode_batch

**Files:**
- Modify: `src/game/token_encoder.py`
- Test: `tests/game/test_token_encoder.py` (add tests)

- [ ] **Step 1: Write the failing test (append)**

Append to `tests/game/test_token_encoder.py`:
```python
from src.game.token_encoder import encode_state_features, encode_position, encode_batch, STATE_DIM


def test_state_features_fifty_move_and_repetition():
    # halfmove clock 40 set via FEN field 5; canonical frame preserves it
    b = chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 40 80")
    f = encode_state_features(canonical_board(b), repetition_count=2)
    assert f.shape == (STATE_DIM,) and f.dtype == np.float32
    assert abs(f[13] - 0.40) < 1e-6   # halfmove_clock / 100
    assert abs(f[14] - (2 / 3.0)) < 1e-6  # repetition count / 3
    assert f[16] == 1.0  # constant bias


def test_encode_position_and_batch_shapes():
    st, sf = encode_position(chess.Board())
    assert st.shape == (64,) and sf.shape == (STATE_DIM,)
    sts, sfs = encode_batch([chess.Board(), chess.Board()], [0, 0])
    assert sts.shape == (2, 64) and sfs.shape == (2, STATE_DIM)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk .venv/bin/python -m pytest tests/game/test_token_encoder.py -v`
Expected: FAIL — `ImportError: cannot import name 'encode_state_features'`.

- [ ] **Step 3: Write minimal implementation (append to `src/game/token_encoder.py`)**

```python
def encode_state_features(cboard: chess.Board, repetition_count: int = 0) -> np.ndarray:
    """Return (18,) float32 global-state features for an already-canonicalized board."""
    f = np.zeros(STATE_DIM, dtype=np.float32)
    f[0] = float(cboard.has_kingside_castling_rights(chess.WHITE))
    f[1] = float(cboard.has_queenside_castling_rights(chess.WHITE))
    f[2] = float(cboard.has_kingside_castling_rights(chess.BLACK))
    f[3] = float(cboard.has_queenside_castling_rights(chess.BLACK))
    if cboard.ep_square is not None:
        f[4 + chess.square_file(cboard.ep_square)] = 1.0
        f[12] = 1.0
    f[13] = min(cboard.halfmove_clock / 100.0, 1.0)
    f[14] = min(repetition_count / 3.0, 1.0)
    f[15] = min(cboard.fullmove_number / 200.0, 1.0)
    f[16] = 1.0
    f[17] = 0.0
    return f


def encode_position(board: chess.Board, repetition_count: int = 0):
    """Canonicalize and return (square_tokens[64] int32, state_features[18] float32)."""
    cb = canonical_board(board)
    return encode_square_tokens(cb), encode_state_features(cb, repetition_count)


def encode_batch(boards, reps):
    """Return (int32[B,64], float32[B,18]) for a list of boards and repetition counts."""
    sts = np.zeros((len(boards), 64), dtype=np.int32)
    sfs = np.zeros((len(boards), STATE_DIM), dtype=np.float32)
    for i, (b, r) in enumerate(zip(boards, reps)):
        sts[i], sfs[i] = encode_position(b, r)
    return sts, sfs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk .venv/bin/python -m pytest tests/game/test_token_encoder.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
rtk git add src/game/token_encoder.py tests/game/test_token_encoder.py && rtk git commit -m "feat: state features + encode_position/encode_batch"
```

---

### Task 4: Config additions (transformer block)

**Files:**
- Modify: `config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
from config import Config


def test_transformer_config_present():
    assert Config.d_model == 256
    assert Config.n_layers == 8
    assert Config.n_heads == 8
    assert Config.d_ff == 1024
    assert Config.smolgen_compress == 32
    assert Config.smolgen_hidden == 128
    assert Config.smolgen_gen == 128
    assert Config.state_dim == 18
    assert Config.distill_batch_size == 1024
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk .venv/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL — `AttributeError: type object 'Config' has no attribute 'd_model'`.

- [ ] **Step 3: Write minimal implementation (append inside `class Config` in `config.py`)**

```python
    # Transformer architecture (Phase 1 rebuild)
    d_model = 256
    n_layers = 8
    n_heads = 8
    d_ff = 1024
    smolgen_compress = 32
    smolgen_hidden = 128
    smolgen_gen = 128
    state_dim = 18
    transformer_dropout = 0.0

    # Distillation training (Phase 1)
    distill_batch_size = 1024
    distill_lr = 3e-4
    distill_warmup_steps = 2000
    distill_total_steps = 60000
    distill_policy_temperature = 1.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk .venv/bin/python -m pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add config.py tests/test_config.py && rtk git commit -m "feat: transformer + distill config"
```

---

### Task 5: Smolgen layer

**Files:**
- Create: `src/model/smolgen.py`
- Test: `tests/model/__init__.py`, `tests/model/test_smolgen.py`

- [ ] **Step 1: Write the failing test**

`tests/model/__init__.py`: (empty)

`tests/model/test_smolgen.py`:
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from src.model.smolgen import Smolgen


def test_smolgen_output_shape():
    shared_out = layers.Dense(64 * 64, use_bias=False)
    sm = Smolgen(n_heads=8, compress=32, hidden=128, gen=128, shared_out=shared_out)
    x = tf.random.normal((2, 64, 256))
    bias = sm(x)
    assert tuple(bias.shape) == (2, 8, 64, 64)
    assert np.isfinite(bias.numpy()).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk .venv/bin/python -m pytest tests/model/test_smolgen.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.model.smolgen'`.

- [ ] **Step 3: Write minimal implementation**

`src/model/smolgen.py`:
```python
"""Smolgen: learned position-dependent additive attention bias (per head)."""

import tensorflow as tf
from tensorflow.keras import layers


class Smolgen(layers.Layer):
    def __init__(self, n_heads, compress, hidden, gen, shared_out, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.gen = gen
        self.compress = layers.Dense(compress, use_bias=False)
        self.dense1 = layers.Dense(hidden, activation="swish")
        self.ln = layers.LayerNormalization()
        self.dense2 = layers.Dense(n_heads * gen)
        self.shared_out = shared_out  # shared Dense(64*64) instance

    def call(self, x):  # x: [B, 64, d]
        b = tf.shape(x)[0]
        c = self.compress(x)                 # [B, 64, compress]
        c = tf.reshape(c, [b, -1])           # [B, 64*compress]
        h = self.ln(self.dense1(c))          # [B, hidden]
        g = self.dense2(h)                   # [B, n_heads*gen]
        g = tf.reshape(g, [b * self.n_heads, self.gen])
        bias = self.shared_out(g)            # [B*n_heads, 64*64]
        return tf.reshape(bias, [b, self.n_heads, 64, 64])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk .venv/bin/python -m pytest tests/model/test_smolgen.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/model/smolgen.py tests/model/__init__.py tests/model/test_smolgen.py && rtk git commit -m "feat: smolgen attention-bias layer"
```

---

### Task 6: Heads — policy index map + policy/value/moves-left

**Files:**
- Create: `src/model/heads.py`
- Test: `tests/model/test_heads.py`

- [ ] **Step 1: Write the failing test**

`tests/model/test_heads.py`:
```python
import chess
import numpy as np
import tensorflow as tf
from src.model.heads import build_policy_index_map, PolicyHead, ValueHead, MovesLeftHead
from src.game.move_encoder import get_move_encoder


def test_policy_index_map_ranges():
    fi, ti, pr = build_policy_index_map()
    P = get_move_encoder().policy_size
    assert fi.shape == (P,) and ti.shape == (P,) and pr.shape == (P,)
    assert fi.min() >= 0 and fi.max() < 64
    assert ti.min() >= 0 and ti.max() < 64
    assert set(np.unique(pr)).issubset({0, 1, 2, 3})


def test_policy_head_shape_and_finite():
    P = get_move_encoder().policy_size
    head = PolicyHead(d_attn=64)
    sq = tf.random.normal((3, 64, 256))
    logits = head(sq)
    assert tuple(logits.shape) == (3, P)
    assert np.isfinite(logits.numpy()).all()


def test_value_and_movesleft_shapes():
    cls = tf.random.normal((3, 256))
    assert tuple(ValueHead()(cls).shape) == (3, 3)
    ml = MovesLeftHead()(cls)
    assert tuple(ml.shape) == (3, 1)
    assert (ml.numpy() >= 0).all()  # softplus is non-negative
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk .venv/bin/python -m pytest tests/model/test_heads.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.model.heads'`.

- [ ] **Step 3: Write minimal implementation**

`src/model/heads.py`:
```python
"""Policy (attention from->to), WDL value, and moves-left heads."""

import chess
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src.game.move_encoder import get_move_encoder

_PROMO_CLASS = {chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3}


def build_policy_index_map():
    """Return (from_idx[P], to_idx[P], promo_class[P]) from the MoveEncoder."""
    me = get_move_encoder()
    p = me.policy_size
    from_idx = np.zeros(p, dtype=np.int32)
    to_idx = np.zeros(p, dtype=np.int32)
    promo = np.zeros(p, dtype=np.int32)
    for i in range(p):
        mv = me.idx_to_move[i]
        from_idx[i] = mv.from_square
        to_idx[i] = mv.to_square
        promo[i] = _PROMO_CLASS.get(mv.promotion, 0)
    return from_idx, to_idx, promo


class PolicyHead(layers.Layer):
    def __init__(self, d_attn=64, **kwargs):
        super().__init__(**kwargs)
        fi, ti, pr = build_policy_index_map()
        self.flat_ft = tf.constant(fi * 64 + ti, dtype=tf.int32)        # [P]
        up_idx = np.where(pr > 0, fi * 3 + np.maximum(pr - 1, 0), 0)
        self.flat_up = tf.constant(up_idx.astype(np.int32), dtype=tf.int32)  # [P]
        self.up_mask = tf.constant((pr > 0).astype(np.float32), dtype=tf.float32)  # [P]
        self.q = layers.Dense(d_attn, use_bias=False)
        self.k = layers.Dense(d_attn, use_bias=False)
        self.up = layers.Dense(3)  # per-square underpromotion logits
        self.scale = float(d_attn) ** 0.5

    def call(self, sq):  # sq: [B, 64, d]
        b = tf.shape(sq)[0]
        q = self.q(sq)
        k = self.k(sq)
        scores = tf.matmul(q, k, transpose_b=True) / self.scale  # [B, 64, 64]
        base = tf.gather(tf.reshape(scores, [b, 64 * 64]), self.flat_ft, axis=1)  # [B, P]
        up = tf.reshape(self.up(sq), [b, 64 * 3])                                  # [B, 192]
        up_term = tf.gather(up, self.flat_up, axis=1) * self.up_mask               # [B, P]
        return base + up_term


class ValueHead(layers.Layer):
    def __init__(self, hidden=128, **kwargs):
        super().__init__(**kwargs)
        self.d1 = layers.Dense(hidden, activation="relu")
        self.d2 = layers.Dense(3)  # raw WDL logits

    def call(self, cls):
        return self.d2(self.d1(cls))


class MovesLeftHead(layers.Layer):
    def __init__(self, hidden=128, **kwargs):
        super().__init__(**kwargs)
        self.d1 = layers.Dense(hidden, activation="relu")
        self.d2 = layers.Dense(1, activation="softplus")

    def call(self, cls):
        return self.d2(self.d1(cls))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk .venv/bin/python -m pytest tests/model/test_heads.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
rtk git add src/model/heads.py tests/model/test_heads.py && rtk git commit -m "feat: policy/value/moves-left heads"
```

---

### Task 7: Assemble ChessTransformer

**Files:**
- Create: `src/model/transformer.py`
- Test: `tests/model/test_transformer.py`

- [ ] **Step 1: Write the failing test**

`tests/model/test_transformer.py`:
```python
import tempfile
import numpy as np
import tensorflow as tf
from config import Config
from src.model.transformer import ChessTransformer
from src.game.token_encoder import encode_batch
from src.game.move_encoder import get_move_encoder
import chess


def _net():
    return ChessTransformer(Config)


def test_forward_shapes():
    net = _net()
    st, sf = encode_batch([chess.Board(), chess.Board()], [0, 0])
    policy, wdl, ml = net(tf.constant(st), tf.constant(sf))
    P = get_move_encoder().policy_size
    assert tuple(policy.shape) == (2, P)
    assert tuple(wdl.shape) == (2, 3)
    assert tuple(ml.shape) == (2, 1)


def test_param_count_in_budget():
    net = _net()
    st, sf = encode_batch([chess.Board()], [0])
    net(tf.constant(st), tf.constant(sf))  # build
    params = int(sum(np.prod(v.shape) for v in net.trainable_variables))
    assert 8_000_000 < params < 14_000_000, params


def test_save_load_roundtrip():
    net = _net()
    st, sf = encode_batch([chess.Board()], [0])
    p1, _, _ = net(tf.constant(st), tf.constant(sf))
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/w.weights.h5"
        net.save_weights(path)
        net2 = _net()
        net2(tf.constant(st), tf.constant(sf))  # build before load
        net2.load_weights(path)
    p2, _, _ = net2(tf.constant(st), tf.constant(sf))
    assert np.allclose(p1.numpy(), p2.numpy(), atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk .venv/bin/python -m pytest tests/model/test_transformer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.model.transformer'`.

- [ ] **Step 3: Write minimal implementation**

`src/model/transformer.py`:
```python
"""ChessTransformer: token board -> (policy_logits, wdl_logits, moves_left)."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.model.smolgen import Smolgen
from src.model.heads import PolicyHead, ValueHead, MovesLeftHead


class _BiasedMHA(layers.Layer):
    """Multi-head self-attention with an additive per-head bias on the logits."""

    def __init__(self, n_heads, d_model, **kwargs):
        super().__init__(**kwargs)
        self.h = n_heads
        self.d = d_model
        self.dh = d_model // n_heads
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.wo = layers.Dense(d_model)

    def _split(self, t, b, seq):
        return tf.transpose(tf.reshape(t, [b, seq, self.h, self.dh]), [0, 2, 1, 3])

    def call(self, x, bias):  # x: [B,T,d]; bias: [B,h,T,T]
        b = tf.shape(x)[0]
        t = tf.shape(x)[1]
        q = self._split(self.wq(x), b, t)
        k = self._split(self.wk(x), b, t)
        v = self._split(self.wv(x), b, t)
        scores = tf.matmul(q, k, transpose_b=True) / (float(self.dh) ** 0.5) + bias
        a = tf.nn.softmax(scores, axis=-1)
        o = tf.matmul(a, v)  # [B,h,T,dh]
        o = tf.reshape(tf.transpose(o, [0, 2, 1, 3]), [b, t, self.d])
        return self.wo(o)


class _EncoderLayer(layers.Layer):
    def __init__(self, cfg, shared_smolgen_out, **kwargs):
        super().__init__(**kwargs)
        self.ln1 = layers.LayerNormalization()
        self.attn = _BiasedMHA(cfg.n_heads, cfg.d_model)
        self.smolgen = Smolgen(cfg.n_heads, cfg.smolgen_compress,
                               cfg.smolgen_hidden, cfg.smolgen_gen, shared_smolgen_out)
        self.ln2 = layers.LayerNormalization()
        self.ffn = keras.Sequential([
            layers.Dense(cfg.d_ff, activation="gelu"),
            layers.Dense(cfg.d_model),
        ])

    def call(self, x):  # x: [B, 65, d]  (index 0 = CLS, 1..64 = squares)
        h = self.ln1(x)
        bias64 = self.smolgen(h[:, 1:, :])                       # [B,h,64,64]
        bias = tf.pad(bias64, [[0, 0], [0, 0], [1, 0], [1, 0]])  # CLS row/col = 0 -> [B,h,65,65]
        x = x + self.attn(h, bias)
        x = x + self.ffn(self.ln2(x))
        return x


class ChessTransformer(keras.Model):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        d = cfg.d_model
        self.piece_emb = layers.Embedding(13, d)
        self.pos_emb = self.add_weight(name="pos_emb", shape=(1, 64, d),
                                       initializer="random_normal", trainable=True)
        self.cls = self.add_weight(name="cls", shape=(1, 1, d),
                                   initializer="random_normal", trainable=True)
        self.state_mlp = keras.Sequential([
            layers.Dense(d, activation="gelu"),
            layers.Dense(d),
        ])
        shared_smolgen_out = layers.Dense(64 * 64, use_bias=False)
        self.enc = [_EncoderLayer(cfg, shared_smolgen_out) for _ in range(cfg.n_layers)]
        self.final_ln = layers.LayerNormalization()
        self.policy_head = PolicyHead()
        self.value_head = ValueHead()
        self.moves_left_head = MovesLeftHead()

    def call(self, square_tokens, state_features, training=False):
        pe = self.piece_emb(square_tokens) + self.pos_emb       # [B,64,d]
        cond = self.state_mlp(state_features)[:, None, :]       # [B,1,d]
        pe = pe + cond
        b = tf.shape(pe)[0]
        cls = tf.tile(self.cls, [b, 1, 1]) + cond               # [B,1,d]
        x = tf.concat([cls, pe], axis=1)                        # [B,65,d]
        for layer in self.enc:
            x = layer(x)
        x = self.final_ln(x)
        cls_out = x[:, 0, :]
        sq_out = x[:, 1:, :]
        return self.policy_head(sq_out), self.value_head(cls_out), self.moves_left_head(cls_out)

    @tf.function(reduce_retracing=True)
    def predict_batch(self, square_tokens, state_features):
        return self.call(square_tokens, state_features, training=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk .venv/bin/python -m pytest tests/model/test_transformer.py -v`
Expected: PASS (3 passed). If `test_param_count_in_budget` reports a value outside 8–14M, adjust `Config.smolgen_hidden`/`smolgen_gen` (down lowers params) and re-run.

- [ ] **Step 5: Commit**

```bash
rtk git add src/model/transformer.py tests/model/test_transformer.py && rtk git commit -m "feat: assemble ChessTransformer (encoder + smolgen + heads)"
```

---

### Task 8: Full-suite green + foundation summary

**Files:**
- Test: all of `tests/`

- [ ] **Step 1: Run the whole suite**

Run: `rtk .venv/bin/python -m pytest tests/ -v`
Expected: PASS (all tests from Tasks 0–7 green).

- [ ] **Step 2: Sanity-check end-to-end forward from a real board**

Run:
```bash
rtk .venv/bin/python -c "
import chess, tensorflow as tf
from config import Config
from src.model.transformer import ChessTransformer
from src.game.token_encoder import encode_batch
net = ChessTransformer(Config)
st, sf = encode_batch([chess.Board()], [0])
p, w, m = net(tf.constant(st), tf.constant(sf))
print('policy', p.shape, 'wdl', tf.nn.softmax(w).numpy().round(3), 'moves_left', float(m[0,0]))
"
```
Expected: prints policy shape `(1, 1924)`, a 3-vector WDL probability, and a non-negative moves-left scalar.

- [ ] **Step 3: Commit (no-op safety + tag the milestone)**

```bash
rtk git add -A && rtk git commit -m "chore: Phase 1 foundation complete" --allow-empty
```

---

## Self-Review (completed by plan author)

- **Spec coverage:** transformer trunk + smolgen (§4) ✓; spatial from→to policy head over 1924 actions (§4) ✓; WDL value head (§4) ✓; moves-left head (§4) ✓; canonical orientation + fixed encoding incl. 50-move clock & repetition count (§4) ✓; config additions (§7) ✓; env/test setup from the contracts' environment facts ✓. Out of scope here (data, training, MCTS, eval) belong to Plans 2–5.
- **Contract consistency:** `STATE_DIM=18`, piece-id scheme, `(policy_logits[P], wdl_logits[3], moves_left[1])` output tuple, `policy_index_map` semantics, smolgen dims (128/128), and `P = get_move_encoder().policy_size` (never hardcoded) all match `_interface-contracts.md`.
- **Placeholders:** none — every code step has complete code and an exact command.
- **Naming consistency:** `encode_square_tokens`, `encode_state_features`, `encode_position`, `encode_batch`, `build_policy_index_map`, `PolicyHead/ValueHead/MovesLeftHead`, `Smolgen`, `ChessTransformer.call(square_tokens, state_features)` used consistently across tasks and tests.

## Notes for the implementer
- Keras 3 multi-positional-arg `call` is fine for subclassed models invoked directly as `net(st, sf)` (we never use the Functional API here).
- If `tf.gather(..., axis=1)` on a dynamic batch is slow under `@tf.function`, it is still correct; optimize only if Plan 4's latency gate fails.
- The smolgen→`[B,h,64,64]` bias is padded to `[B,h,65,65]` so the `[CLS]` token neither biases nor is biased by smolgen — intentional.
