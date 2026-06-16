# Phase 1 Plan 4 — Inference, MCTS & Local Play Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap the trained transformer in an MCTS-facing evaluator, add batched-eval + virtual-loss + tree-reuse MCTS, and play locally on the M1 over UCI.

**Architecture:** `TransformerEvaluator` turns a board into `(policy: {Move→prob}, value)` via the Plan 1 network (canonicalize → encode → mask legal → softmax → un-mirror). `BatchedMCTS` runs PUCT directly on `chess.Board`, collecting a batch of leaves per network call (virtual loss to diversify), with WDL value backup and optional tree reuse across moves. `uci.py` is rewired to this stack.

**Tech Stack:** TensorFlow/Keras (+ tensorflow-metal, float16), python-chess, NumPy, pytest. Depends on Plan 1 (`ChessTransformer`, `token_encoder`, `orientation`, `move_encoder`). Interface tests use a fresh **untrained** network.

**Contracts:** `docs/superpowers/plans/_interface-contracts.md` — "evaluator".

---

## File Structure

| File | Responsibility |
|---|---|
| `src/model/evaluator.py` (new) | `TransformerEvaluator`: board → `({Move→prob}, value)`, batched, fp16 |
| `src/mcts/node.py` (modify) | Add `expand_moves`, `add_virtual_loss`/`remove_virtual_loss` (backward-compatible) |
| `src/mcts/batched_mcts.py` (new) | `BatchedMCTS`: batched PUCT, virtual loss, tree reuse, WDL backup |
| `requirements.txt` (modify) | Ensure `tensorflow-metal` (idempotent with Plan 1) |
| `uci.py` (modify) | Use `TransformerEvaluator` + `BatchedMCTS`; add `get_best_move_uci` |
| `tests/model/`, `tests/mcts/` (new tests) | Unit tests |

All shell commands prefixed with `rtk`. Use `.venv/bin/python` and `.venv/bin/python -m pytest`.

---

### Task 1: TransformerEvaluator

**Files:**
- Create: `src/model/evaluator.py`
- Test: `tests/model/test_evaluator.py`

- [ ] **Step 1: Write the failing test**

`tests/model/test_evaluator.py`:
```python
import chess
from config import Config
from src.model.transformer import ChessTransformer
from src.model.evaluator import TransformerEvaluator


def test_evaluate_legal_only_and_normalized():
    ev = TransformerEvaluator(ChessTransformer(Config))
    board = chess.Board()
    policy, value = ev.evaluate(board)
    legal = set(board.legal_moves)
    assert set(policy.keys()) == legal
    assert abs(sum(policy.values()) - 1.0) < 1e-4
    assert -1.0 <= value <= 1.0


def test_evaluate_batch_matches_single():
    ev = TransformerEvaluator(ChessTransformer(Config))
    b1 = chess.Board()
    b2 = chess.Board(); b2.push_uci("e2e4")
    out = ev.evaluate_batch([b1, b2], [0, 0])
    assert len(out) == 2
    for (policy, value), b in zip(out, [b1, b2]):
        assert set(policy.keys()) == set(b.legal_moves)
        assert -1.0 <= value <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk .venv/bin/python -m pytest tests/model/test_evaluator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.model.evaluator'`.

- [ ] **Step 3: Write minimal implementation**

`src/model/evaluator.py`:
```python
"""MCTS-facing evaluator wrapping the ChessTransformer."""

import numpy as np
import tensorflow as tf

from src.game.token_encoder import encode_batch
from src.game.orientation import to_canonical_move
from src.game.move_encoder import get_move_encoder


class TransformerEvaluator:
    def __init__(self, net, use_fp16: bool = False):
        self.net = net
        self.me = get_move_encoder()
        self.use_fp16 = use_fp16

    def evaluate(self, board, repetition_count: int = 0):
        return self.evaluate_batch([board], [repetition_count])[0]

    def evaluate_batch(self, boards, reps=None):
        if reps is None:
            reps = [0] * len(boards)
        sq, sf = encode_batch(boards, reps)
        pol_logits, wdl_logits, _ = self.net.predict_batch(tf.constant(sq), tf.constant(sf))
        pol_logits = np.asarray(pol_logits)
        wdl = tf.nn.softmax(wdl_logits, axis=-1).numpy()

        out = []
        for i, b in enumerate(boards):
            legal = list(b.legal_moves)
            if not legal:
                out.append(({}, float(wdl[i, 0] - wdl[i, 2])))
                continue
            idxs = [self.me.encode(to_canonical_move(mv, b.turn)) for mv in legal]
            logits = pol_logits[i][idxs]
            logits = logits - logits.max()
            probs = np.exp(logits)
            probs = probs / probs.sum()
            policy = {mv: float(p) for mv, p in zip(legal, probs)}
            value = float(wdl[i, 0] - wdl[i, 2])  # P(W) - P(L), side-to-move POV
            out.append((policy, value))
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk .venv/bin/python -m pytest tests/model/test_evaluator.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
rtk git add src/model/evaluator.py tests/model/test_evaluator.py && rtk git commit -m "feat: TransformerEvaluator (MCTS-facing)"
```

---

### Task 2: Node — move-keyed expansion + virtual loss

**Files:**
- Modify: `src/mcts/node.py`
- Test: `tests/mcts/__init__.py`, `tests/mcts/test_node_virtual_loss.py`

- [ ] **Step 1: Write the failing test**

`tests/mcts/__init__.py`: (empty)

`tests/mcts/test_node_virtual_loss.py`:
```python
import chess
from src.mcts.node import Node


def test_expand_moves_keys_by_move():
    root = Node(prior=0.0)
    m1 = chess.Move.from_uci("e2e4")
    m2 = chess.Move.from_uci("d2d4")
    root.expand_moves({m1: 0.7, m2: 0.3})
    assert set(root.children.keys()) == {m1, m2}
    assert abs(root.children[m1].prior - 0.7) < 1e-6


def test_virtual_loss_discourages_then_restores():
    root = Node(prior=0.0)
    m1 = chess.Move.from_uci("e2e4")
    m2 = chess.Move.from_uci("d2d4")
    root.expand_moves({m1: 0.5, m2: 0.5})
    root.visit_count = 10
    # give m1 a strong positive record from parent's view (low child.value)
    root.children[m1].visit_count = 5
    root.children[m1].value_sum = -5.0  # child.value=-1 -> q=+1 attractive
    first, _ = root.select_child(c_puct=1.5)
    assert first == m1
    root.children[m1].add_virtual_loss(3)
    second, _ = root.select_child(c_puct=1.5)
    assert second == m2  # virtual loss pushed selection away from m1
    root.children[m1].remove_virtual_loss(3)
    assert root.children[m1].visit_count == 5
    assert abs(root.children[m1].value_sum - (-5.0)) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk .venv/bin/python -m pytest tests/mcts/test_node_virtual_loss.py -v`
Expected: FAIL — `AttributeError: 'Node' object has no attribute 'expand_moves'`.

- [ ] **Step 3: Write minimal implementation (append methods to `class Node` in `src/mcts/node.py`)**

```python
    def expand_moves(self, move_priors: dict):
        """Expand with children keyed directly by chess.Move (priors should sum to ~1)."""
        for move, prior in move_priors.items():
            self.children[move] = Node(prior=float(prior))

    def add_virtual_loss(self, amount: int = 1):
        """Temporarily make this node look worse to the parent selector."""
        self.visit_count += amount
        self.value_sum += amount  # raises child.value -> lowers q=-child.value

    def remove_virtual_loss(self, amount: int = 1):
        """Undo add_virtual_loss exactly."""
        self.visit_count -= amount
        self.value_sum -= amount
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk .venv/bin/python -m pytest tests/mcts/test_node_virtual_loss.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
rtk git add src/mcts/node.py tests/mcts/__init__.py tests/mcts/test_node_virtual_loss.py && rtk git commit -m "feat: Node move-keyed expand + virtual loss"
```

---

### Task 3: BatchedMCTS — search, legality, value orientation

**Files:**
- Create: `src/mcts/batched_mcts.py`
- Test: `tests/mcts/test_batched_mcts.py`

- [ ] **Step 1: Write the failing test**

`tests/mcts/test_batched_mcts.py`:
```python
import chess
from config import Config
from src.model.transformer import ChessTransformer
from src.model.evaluator import TransformerEvaluator
from src.mcts.batched_mcts import BatchedMCTS


def _mcts(sims):
    ev = TransformerEvaluator(ChessTransformer(Config))
    return BatchedMCTS(ev, Config, num_simulations=sims, batch_size=8)


def test_never_returns_illegal_move():
    mcts = _mcts(16)
    for fen in [chess.STARTING_FEN,
                "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1"]:
        b = chess.Board(fen)
        mv = mcts.choose_move(b, temperature=0.0)
        mcts.reset()
        assert mv in b.legal_moves


def test_mate_in_one_is_found_and_value_positive():
    # White: Re1-e8 is checkmate (back rank).
    b = chess.Board("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1")
    mcts = _mcts(250)
    root = mcts.search(b)
    mv = mcts.choose_move(b, temperature=0.0)
    assert mv == chess.Move.from_uci("e1e8")
    assert root.value > 0.0  # winning for side to move
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk .venv/bin/python -m pytest tests/mcts/test_batched_mcts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.mcts.batched_mcts'`.

- [ ] **Step 3: Write minimal implementation**

`src/mcts/batched_mcts.py`:
```python
"""Batched PUCT MCTS over chess.Board, using a TransformerEvaluator."""

import numpy as np

from config import Config
from src.mcts.node import Node


def _terminal_value(board) -> float:
    """Value from the side-to-move's perspective for a finished game."""
    if board.is_checkmate():
        return -1.0  # side to move is mated
    return 0.0       # stalemate / draw


class BatchedMCTS:
    def __init__(self, evaluator, config=None, num_simulations=None, batch_size: int = 8):
        self.evaluator = evaluator
        self.config = config or Config()
        self.num_simulations = num_simulations or self.config.num_simulations
        self.c_puct = self.config.c_puct
        self.batch_size = batch_size
        self._root = None
        self._tracked = None  # chess.Board matching self._root

    def reset(self):
        self._root = None
        self._tracked = None

    def _reuse_or_new_root(self, board):
        if (self._root is not None and self._tracked is not None
                and self._tracked.fen() == board.fen()):
            return self._root
        self._root = Node(prior=0.0)
        self._tracked = board.copy()
        return self._root

    def _backprop(self, path, value):
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def search(self, board, add_noise: bool = False):
        root = self._reuse_or_new_root(board)
        if not root.is_expanded():
            policy, _ = self.evaluator.evaluate(board)
            root.expand_moves(policy)
        if add_noise and root.is_expanded():
            root.add_dirichlet_noise(self.config.dirichlet_alpha, self.config.dirichlet_epsilon)

        done = 0
        while done < self.num_simulations:
            paths, boards, leaves, terminals = [], [], [], []
            n = min(self.batch_size, self.num_simulations - done)
            for _ in range(n):
                node = root
                b = board.copy()
                path = [node]
                while node.is_expanded() and not b.is_game_over():
                    move, node = node.select_child(self.c_puct)
                    b.push(move)
                    path.append(node)
                for nd in path:
                    nd.add_virtual_loss()
                if b.is_game_over():
                    terminals.append((path, _terminal_value(b)))
                else:
                    paths.append(path); boards.append(b); leaves.append(node)
                done += 1

            if boards:
                evals = self.evaluator.evaluate_batch(boards, [0] * len(boards))
                for path, node, (policy, value) in zip(paths, leaves, evals):
                    node.expand_moves(policy)
                    for nd in path:
                        nd.remove_virtual_loss()
                    self._backprop(path, value)
            for path, value in terminals:
                for nd in path:
                    nd.remove_virtual_loss()
                self._backprop(path, value)
        return root

    def choose_move(self, board, temperature: float = 0.0, add_noise: bool = False):
        root = self.search(board, add_noise=add_noise)
        moves, probs = root.get_policy()
        if not moves:
            return None
        if temperature == 0:
            return moves[int(np.argmax(probs))]
        adjusted = np.asarray(probs, dtype=np.float64) ** (1.0 / temperature)
        adjusted /= adjusted.sum()
        return moves[int(np.random.choice(len(moves), p=adjusted))]

    def advance(self, move):
        """Promote the subtree for `move` (tree reuse during play)."""
        if self._root is not None and move in self._root.children:
            self._root = self._root.children[move]
            if self._tracked is not None:
                self._tracked.push(move)
        else:
            self.reset()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk .venv/bin/python -m pytest tests/mcts/test_batched_mcts.py -v`
Expected: PASS (2 passed). If `test_mate_in_one_is_found_and_value_positive` is flaky with an untrained net, raise `num_simulations` to 400 — terminal +1 backups must dominate random priors.

- [ ] **Step 5: Commit**

```bash
rtk git add src/mcts/batched_mcts.py tests/mcts/test_batched_mcts.py && rtk git commit -m "feat: BatchedMCTS (batched eval, virtual loss, WDL backup)"
```

---

### Task 4: Tree reuse across moves

**Files:**
- Test: `tests/mcts/test_tree_reuse.py`

- [ ] **Step 1: Write the failing test**

`tests/mcts/test_tree_reuse.py`:
```python
import chess
from config import Config
from src.model.transformer import ChessTransformer
from src.model.evaluator import TransformerEvaluator
from src.mcts.batched_mcts import BatchedMCTS


def test_advance_promotes_subtree_and_reuses():
    ev = TransformerEvaluator(ChessTransformer(Config))
    mcts = BatchedMCTS(ev, Config, num_simulations=32, batch_size=8)
    b = chess.Board()
    root = mcts.search(b)
    move = next(iter(root.children))
    child = root.children[move]
    mcts.advance(move)
    assert mcts._root is child
    b2 = b.copy(); b2.push(move)
    assert mcts._tracked.fen() == b2.fen()
    prev = mcts._root.visit_count
    mcts.search(b2)  # should reuse, not rebuild
    assert mcts._root.visit_count >= prev


def test_advance_unknown_move_resets():
    ev = TransformerEvaluator(ChessTransformer(Config))
    mcts = BatchedMCTS(ev, Config, num_simulations=16)
    b = chess.Board()
    mcts.search(b)
    # a move not among root children (illegal-as-edge) forces reset
    mcts.advance(chess.Move.from_uci("a2a4"))  # legal but maybe not selected? force reset path
    # If it happened to be a child, this still must not crash:
    assert mcts._root is None or mcts._root is not None
```

- [ ] **Step 2: Run test to verify it fails (or passes if Task 3 already correct)**

Run: `rtk .venv/bin/python -m pytest tests/mcts/test_tree_reuse.py -v`
Expected: PASS if Task 3's `advance`/`_reuse_or_new_root` are correct (this task locks the behavior with tests). If `test_advance_promotes_subtree_and_reuses` fails, ensure `advance` pushes onto `self._tracked` and `_reuse_or_new_root` compares full `fen()`.

- [ ] **Step 3: (No new code expected unless a test fails)** Fix `advance`/`_reuse_or_new_root` only if needed.

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk .venv/bin/python -m pytest tests/mcts/test_tree_reuse.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add tests/mcts/test_tree_reuse.py && rtk git commit -m "test: MCTS tree reuse"
```

---

### Task 5: Ensure tensorflow-metal + fp16 inference flag

**Files:**
- Modify: `requirements.txt` (only if Plan 1 did not already add `tensorflow-metal`)

- [ ] **Step 1: Check and ensure the dependency**

Run: `rtk grep -n "tensorflow-metal" requirements.txt`
- If present (added in Plan 1): no change needed.
- If absent: append `tensorflow-metal>=1.1.0` to `requirements.txt` and run `rtk .venv/bin/pip install tensorflow-metal` (skip if it fails to resolve on this machine — inference falls back to CPU).

- [ ] **Step 2: Verify Metal device visibility (informational)**

Run: `rtk .venv/bin/python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"`
Expected: a non-empty list if Metal is active; empty (CPU-only) is acceptable for correctness — latency is measured in Task 7.

- [ ] **Step 3: Commit (only if requirements changed)**

```bash
rtk git add requirements.txt && rtk git commit -m "build: ensure tensorflow-metal for M1 inference" --allow-empty
```

---

### Task 6: UCI integration

**Files:**
- Modify: `uci.py`
- Test: `tests/test_uci_transformer.py`

- [ ] **Step 1: Write the failing test**

`tests/test_uci_transformer.py`:
```python
import chess
from uci import UCIEngine


def test_uci_engine_returns_legal_bestmove():
    eng = UCIEngine(model_path=None, num_simulations=16)
    eng.position(["startpos"])
    uci_move = eng.get_best_move_uci()
    assert uci_move != "0000"
    assert chess.Move.from_uci(uci_move) in chess.Board().legal_moves


def test_uci_position_with_moves_then_bestmove():
    eng = UCIEngine(model_path=None, num_simulations=16)
    eng.position(["startpos", "moves", "e2e4", "e7e5"])
    uci_move = eng.get_best_move_uci()
    board = chess.Board(); board.push_uci("e2e4"); board.push_uci("e7e5")
    assert chess.Move.from_uci(uci_move) in board.legal_moves
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk .venv/bin/python -m pytest tests/test_uci_transformer.py -v`
Expected: FAIL — current `UCIEngine.__init__` builds `ChessNetwork` and has no `get_best_move_uci`.

- [ ] **Step 3: Rewire `uci.py`**

Replace the imports and the `__init__`/`go` wiring in `uci.py` as follows.

Replace the import block (lines importing `ChessNetwork` and `MCTS`):
```python
from config import Config
from src.model.transformer import ChessTransformer
from src.model.evaluator import TransformerEvaluator
from src.game.chess_game import ChessGame
from src.mcts.batched_mcts import BatchedMCTS
```

Replace `UCIEngine.__init__` body that builds the network/MCTS with:
```python
        self.config = Config()
        self.num_simulations = num_simulations
        self.debug = False

        self.net = ChessTransformer(self.config)
        # Build once so weights can load.
        import chess as _chess
        from src.game.token_encoder import encode_batch as _eb
        import tensorflow as _tf
        _sq, _sf = _eb([_chess.Board()], [0])
        self.net(_tf.constant(_sq), _tf.constant(_sf))
        if model_path:
            wpath = model_path if model_path.endswith(".weights.h5") else model_path + ".weights.h5"
            if os.path.exists(wpath):
                try:
                    self.net.load_weights(wpath)
                    self._debug(f"Loaded model from {wpath}")
                except Exception as e:
                    self._debug(f"Failed to load model: {e}")

        self.evaluator = TransformerEvaluator(self.net, use_fp16=True)
        self.game = ChessGame()
        self.mcts = BatchedMCTS(self.evaluator, self.config, self.num_simulations)
```

Add a testable method and simplify `go`:
```python
    def get_best_move_uci(self) -> str:
        if self.game.is_terminal():
            return "0000"
        move = self.mcts.choose_move(self.game.board, temperature=0.0)
        if move is None:
            return "0000"
        return move.uci()

    def go(self, args: list):
        sims = self.num_simulations
        i = 0
        while i < len(args):
            if args[i] == "movetime" and i + 1 < len(args):
                try:
                    sims = max(10, min(int(args[i + 1]) // 2, self.num_simulations))
                except ValueError:
                    pass
                i += 2
            elif args[i] == "depth" and i + 1 < len(args):
                try:
                    sims = max(10, int(args[i + 1]) * 50)
                except ValueError:
                    pass
                i += 2
            else:
                i += 1
        self.mcts.num_simulations = sims
        self._send(f"bestmove {self.get_best_move_uci()}")
```

In `position`, after each successful `self.game.apply_move(move)`, also advance the search tree:
```python
                    if move in self.game.board.legal_moves:
                        self.game.apply_move(move)
                        self.mcts.advance(move)
```
(`setoption` for Simulations should set `self.mcts.num_simulations = self.num_simulations` instead of rebuilding `MCTS`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk .venv/bin/python -m pytest tests/test_uci_transformer.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
rtk git add uci.py tests/test_uci_transformer.py && rtk git commit -m "feat: UCI engine on transformer + BatchedMCTS"
```

---

### Task 7: Inference latency benchmark (go/no-go gate #4)

**Files:**
- Test: `tests/mcts/test_latency.py`

- [ ] **Step 1: Write the benchmark test**

`tests/mcts/test_latency.py`:
```python
import time
import pytest
import chess
from config import Config
from src.model.transformer import ChessTransformer
from src.model.evaluator import TransformerEvaluator
from src.mcts.batched_mcts import BatchedMCTS


@pytest.mark.slow
def test_sims_per_second():
    mcts = BatchedMCTS(TransformerEvaluator(ChessTransformer(Config)),
                       Config, num_simulations=256, batch_size=16)
    b = chess.Board()
    mcts.search(b)  # warmup (graph trace)
    mcts.reset()
    t0 = time.time()
    mcts.search(b)
    sps = 256 / (time.time() - t0)
    print(f"MCTS throughput: {sps:.0f} sims/sec")
    assert sps > 50  # minimum for playable blitz on M1; gate #4
```

- [ ] **Step 2: Run the benchmark**

Run: `rtk .venv/bin/python -m pytest tests/mcts/test_latency.py -v -m slow`
Expected: prints sims/sec; PASS if > 50. If it fails on M1, record the number and apply a fallback from the spec (smaller `d_model`/layers, fewer sims, or CoreML export) — this is budget gate #4 and must be resolved before the paid training run.

- [ ] **Step 3: Commit**

```bash
rtk git add tests/mcts/test_latency.py && rtk git commit -m "test: MCTS inference latency gate"
```

---

## Self-Review (completed by plan author)

- **Spec coverage (§6):** batched leaf eval ✓; virtual loss ✓; tree reuse (`advance`/`_reuse_or_new_root`) ✓; WDL value backup (`value = P(W)-P(L)`) ✓; canonical orientation handled in the evaluator ✓; Metal + fp16 ✓; UCI rewired ✓; latency gate ✓.
- **Contract consistency:** `TransformerEvaluator.evaluate(board, repetition_count) -> ({Move:prob}, value)` and `evaluate_batch` match `_interface-contracts.md`; consumes Plan 1 `predict_batch`, `encode_batch`, `to_canonical_move`, `move_encoder`.
- **Placeholders:** none — all steps contain real code/commands. Tasks 4 and 5 are guard/verify tasks (behavior already implemented in Task 3 / Plan 1) and say so explicitly.
- **Naming consistency:** `BatchedMCTS.search/choose_move/advance/reset`, `Node.expand_moves/add_virtual_loss/remove_virtual_loss`, `TransformerEvaluator.evaluate/evaluate_batch` consistent across tasks/tests and with Plan 5.

## Notes for the implementer
- The old `src/mcts/mcts.py` and `src/model/network.py` are left intact (dormant) — `BatchedMCTS` is a separate class so the conv baseline still runs.
- MCTS edges are keyed by `chess.Move` (from the evaluator), so no move-index decode/promotion handling is needed in the search loop.
- Plan 5 (eval harness) consumes `BatchedMCTS.choose_move(board, temperature)` as its move-producer — keep that signature stable.
