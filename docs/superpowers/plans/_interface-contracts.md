# Phase 1 Interface Contracts (FROZEN)

> **Update (2026-06-18): the project is now PyTorch (+MPS), not TensorFlow.** The
> conceptual contracts (tokenization, orientation, net I/O shapes, evaluator,
> losses, shard fields) still hold, but the concrete types changed: the net is an
> `nn.Module` (`forward(square_tokens[B,64] long, state_features[B,18] float) ->
> (policy_logits, wdl_logits, moves_left)`, plus `predict_batch` = no-grad eval);
> data shards are `.npz` (not TFRecord) loaded via `make_dataloader(...)` →
> torch `DataLoader`; checkpoints are `.pt` via `state_dict`. See the committed
> code for exact signatures.

These signatures are **fixed**. Every milestone plan implements or consumes them
**exactly** as written. If something here is ambiguous or looks wrong, a plan
author must flag it as a `contract_gap` — **do not silently invent a different
signature.**

Spec: `docs/superpowers/specs/2026-06-17-chess-rl-engine-design.md`.

---

## Constants
- `P = get_move_encoder().policy_size` (currently 1924). **Never hardcode 1924** —
  always read it from the move encoder.
- Transformer (config-driven): `D_MODEL=256`, `N_LAYERS=8`, `N_HEADS=8`,
  `D_FF=1024`.
- `STATE_DIM = 18`.
- `[CLS]` is a **separate learned embedding**, not one of the 13 piece categories.

## Orientation — canonical = side-to-move's perspective
All network I/O (tokens, policy indices, targets) is in the **canonical** frame.
Conversions live in `src/game/orientation.py`:
- `canonical_board(board) -> chess.Board`:
  `board if board.turn == chess.WHITE else board.mirror()`
  (python-chess `Board.mirror()` swaps colors **and** flips vertically, so the
  side-to-move is always White on the canonical board; halfmove clock, fullmove,
  castling, and ep square are preserved/mirrored.)
- `to_canonical_move(move, turn) -> chess.Move`: if `turn == WHITE` return `move`;
  else `chess.Move(chess.square_mirror(move.from_square),
  chess.square_mirror(move.to_square), promotion=move.promotion)`.
- `from_canonical_move(move, turn) -> chess.Move`: **identical** transform (mirror
  is its own inverse).

## `src/game/token_encoder.py`
Everything is computed on the **canonical** board (so side-to-move is White there).
- Piece-id per square: `0=empty`, `1..6` = White P,N,B,R,Q,K (== side-to-move's),
  `7..12` = Black p,n,b,r,q,k (== opponent's).
- `encode_square_tokens(cboard) -> np.ndarray` shape `(64,)` int32. Index = python
  chess square number (`a1=0 … h8=63`); value = piece id above.
- `encode_state_features(cboard, repetition_count: int) -> np.ndarray` shape
  `(18,)` float32:
  - `[0..3]`: castling — `cboard.has_kingside_castling_rights(WHITE)`,
    queenside(WHITE), kingside(BLACK), queenside(BLACK) (own then opponent)
  - `[4..11]`: en-passant file one-hot (8); all zeros if `cboard.ep_square is None`
  - `[12]`: en-passant present (0/1)
  - `[13]`: `min(cboard.halfmove_clock / 100.0, 1.0)`  ← the 50-move signal
  - `[14]`: `min(repetition_count / 3.0, 1.0)`  ← integer count, not binary
  - `[15]`: `min(cboard.fullmove_number / 200.0, 1.0)`
  - `[16]`: `1.0` (constant bias)
  - `[17]`: `0.0` (reserved)
- `encode_position(board, repetition_count=0) -> (square_tokens[64] int32,
  state_features[18] float32)`: canonicalizes internally.
- `encode_batch(boards, reps) -> (np.int32[B,64], np.float32[B,18])`.
- `repetition_count` is supplied by the caller (default 0). In play it is computed
  from history; in training data it defaults to 0 unless the source provides it.

## `src/model/smolgen.py`
- `class Smolgen(keras.layers.Layer)`: from token embeddings `[B,64,D_MODEL]` it
  produces a per-head additive attention bias `[B,N_HEADS,64,64]`.
- Internal dims (config): `smolgen_compress=32`, `smolgen_hidden=128`,
  `smolgen_gen=128`; the final `gen → 64*64` dense is **shared across layers**
  (pass a shared instance in). These dims are tuned to land total params ~10–11M.

## `src/model/heads.py`
- Build once from MoveEncoder a static map `policy_index_map`: for each policy
  index `i`, `(from_sq, to_sq, promo_class)` where `promo_class ∈ {0,1,2,3}`
  (`0` = none/queen, `1/2/3` = underpromo N/B/R), derived from
  `get_move_encoder().idx_to_move`.
- `policy_head(square_emb[B,64,D_MODEL]) -> policy_logits[B,P]`: project squares to
  `from`-query and `to`-key; base logit for move `i` = `q[from_i]·k[to_i]/sqrt(d)`;
  add an underpromotion term from a small head for `promo_class ∈ {1,2,3}`; gather
  into `[B,P]` using `policy_index_map`.
- `value_head(cls[B,D_MODEL]) -> wdl_logits[B,3]` (raw logits).
- `moves_left_head(cls[B,D_MODEL]) -> [B,1]` (softplus, non-negative).

## `src/model/transformer.py`
- `class ChessTransformer(keras.Model)` built from config.
- **call inputs:** `(square_tokens[B,64] int32, state_features[B,18] float32)`.
- **call outputs:** tuple `(policy_logits[B,P], wdl_logits[B,3], moves_left[B,1])`
  — policy & wdl are **raw logits** (no softmax); policy indices match MoveEncoder
  indices on the canonical board.
- Internals: piece embedding (13→D_MODEL) + learned positional embedding
  (64→D_MODEL); a `[CLS]` learned token prepended (seq=65); a state-conditioning
  MLP (`18→D_MODEL`) whose output is added to every token; `N_LAYERS` pre-LN
  encoder layers (MHSA + smolgen bias + FFN, GELU, residual); heads from
  `heads.py` (policy from the 64 square outputs, value/moves-left from `[CLS]`).
- `@tf.function` method `predict_batch(square_tokens, state_features) ->
  (policy_logits, wdl_logits, moves_left)` for fast inference.
- Target total params ~10–13M (assert in a test).

## `src/model/evaluator.py` — the MCTS-facing interface
- `class TransformerEvaluator(net: ChessTransformer)`.
- `evaluate(board, repetition_count=0) -> (policy: dict[chess.Move,float],
  value: float)`: canonicalize → encode → net → mask to legal canonical moves →
  softmax over legal → map canonical moves back to **real** moves for the dict;
  `value = P(W) − P(L)` from the **side-to-move** perspective, ∈ [−1,1].
- `evaluate_batch(boards, reps) -> list[(policy, value)]` (single batched net call).
- Runs with float16 + Metal when available.

## Data shard format (plans 2 & 3) — positions stored already-canonical
TFRecord; one example =
- `square_tokens`: bytes (64 × int8)
- `state_features`: float list (18)
- `legal_indices`: int64 list (variable) — canonical move indices with target prob > 0
- `legal_probs`: float list (variable) — target policy over those indices (sums to 1)
- `wdl`: float list (3)
- `moves_left`: float (scalar)

Loader yields batches: inputs `(square_tokens[B,64] int32, state_features[B,18])`,
targets `(policy_dense[B,P] via scatter, wdl[B,3], moves_left[B,1])`.

## Losses (plan 3)
- policy: soft-target categorical cross-entropy / KL of `policy_dense` against
  `softmax(policy_logits)` (illegal entries are 0 in the dense target).
- value: softmax cross-entropy(`wdl_logits`, `wdl`).
- moves_left: Huber(`moves_left`, target), weight 0.1.
- `total = policy + 1.0·value + 0.1·moves_left`.

## `config.py` additions (keep existing conv params intact)
New transformer block: `d_model, n_layers, n_heads, d_ff, smolgen_compress,
smolgen_hidden, smolgen_gen, state_dim=18, transformer_dropout`, plus distillation
hyperparams (`distill_batch_size, distill_lr, distill_warmup_steps,
distill_total_steps`).

## Environment facts (Plan 1 Task 0 must handle these)
- `.venv` at repo root, **Python 3.12.5**. **TensorFlow is NOT installed**
  (~33 pkgs; `python-chess`, `numpy`, `tqdm` present). **pytest NOT installed.**
  No `tests/` dir, no pytest config.
- Plan 1 Task 0 must: create `tests/` + `tests/__init__.py` + `conftest.py`
  (insert repo root onto `sys.path`); add `pytest` (dev); add `tensorflow` and
  `tensorflow-metal` to `requirements.txt`; install into `.venv`.
- Use `.venv/bin/python -m pytest` and `.venv/bin/pip`. macOS M1: prefer
  `tensorflow` (≥2.16) + `tensorflow-metal`; pin whatever installs cleanly.
- All shell commands in plans are prefixed with `rtk` per repo `CLAUDE.md`.
