# Chess RL Engine — Transformer Rebuild via Supervised Distillation

- **Date:** 2026-06-17
- **Status:** Approved design (pre-implementation)
- **Author:** James (with Claude)
- **Supersedes:** the current conv-net + self-play-first pipeline (kept as dormant baseline)

> **Update (2026-06-18): framework changed to PyTorch.** During implementation we
> hit TensorFlow's broken Apple-Silicon GPU support (`tensorflow-metal` is
> ABI-incompatible with TF 2.21 → CPU-only → MCTS too slow to play). We switched
> the model + training stack to **PyTorch + MPS**, which restored M1 GPU
> acceleration (MCTS **250 sims/sec** on MPS vs 22 on CPU TF). This supersedes the
> "TensorFlow/Keras" choice (D7) and the TF-specific mechanics below: `tf.data`/
> TFRecord → npz shards + torch `Dataset`/`DataLoader`; `@tf.function`/
> `predict_batch` → eager + `torch.no_grad`; `.weights.h5` → `.pt`; mixed
> precision via `torch.autocast`. Architecture, data, MCTS, eval, and all targets
> are unchanged. Implemented on branch `feat/transformer-rebuild`.

---

## 1. Context & Motivation

The repository currently implements an AlphaZero-style chess engine: a small
convolutional policy/value network, PUCT MCTS, and a 3-phase training pipeline
(supervised Stockfish imitation → curriculum vs. Stockfish depths → self-play
RL). The current checkpoint is `checkpoint/model_iter_5.weights.h5` (5 iterations).

A full-codebase audit and a follow-up architecture analysis (with the key
quantitative claims independently fact-checked) established that **the network
architecture is the dominant ceiling on strength**, and that the architecture is
not merely small but *mis-allocated*:

- **Trunk is ~50–80× too small.** Whole net ≈ **582,097 params**; the 4×64
  residual trunk is **329,472** of them. AlphaZero's was ~28–46M (20 blocks ×
  256 filters). After 4 conv blocks the effective receptive field barely spans
  the board, so global concepts can't propagate.
- **The policy head wastes 42.6% of the network.** `Conv2D(2,1×1) →
  Flatten(128) → Dense(1924)`; that `Dense` alone is **248,196 params (42.6% of
  the entire net)** and forces every move through a 128-scalar bottleneck that
  discards the 8×8 grid. AlphaZero/Leela keep the policy *spatial*.
- **Value head is a scalar `tanh`**, conflating "draw" with "unclear".
- **Input-encoding correctness gaps:** no 50-move/halfmove clock plane;
  repetition is binary, not a count; history padding lacks a validity marker —
  the net is partly blind to draw conditions.
- **No modern components:** no squeeze-excitation, no global pooling, no
  attention, no moves-left head.

### Goal & budget decision

The stated aspiration is "better than Stockfish." The analysis made the compute
reality explicit and the following decisions were taken:

- Modern full-strength Stockfish (≈3600+ CCRL) is **not reachable** on a small
  budget; AlphaZero used thousands of TPUs and Lc0 has played >1.5B self-play
  games. Self-play **from scratch** is hopeless at small budgets.
- **Hard budget constraint: ~$10 of GPU rental** (≈ 25–30 GPU-hours on an RTX
  4090 at 2026 spot prices ≈ $0.30–0.40/hr).
- The efficient path at this budget is **supervised distillation** of existing
  free Stockfish-labeled data, **not** self-play.
- **Decision: build the strongest achievable engine for ~$10 as Phase 1,
  architected so it can scale later** ("staged" — option 1 + option 2).
- **Decision: commit to a transformer architecture now** (not an evolved
  conv/SE-ResNet), accepting the larger rewrite, because it is the proven SOTA
  lineage and the intended long-term scaling target.

---

## 2. Goals, Non-Goals & Success Criteria

### Phase 1 goals
- Replace the conv net, plane encoder, and self-play-first training with a
  transformer policy/value net trained by supervised distillation, played via
  MCTS.
- Reach **~2000–2500 Elo** (raw policy ~1900–2200; +search to the top of range)
  for **≤ ~$12** of rented GPU.
- Run the play engine locally on the M1 (Apple Metal) via UCI.

### Non-goals (Phase 1)
- Any self-play RL run (deferred to scaling Phase 3).
- Multi-GPU / distributed training.
- Beating modern full-strength Stockfish.
- Opening books / endgame tablebases.

### Success criteria
1. **Strength:** distilled net + MCTS reaches ~2000–2500 Elo, measured by
   (a) head-to-head vs `model_iter_5` (expect ~100%), (b) matches vs Stockfish
   pinned at fixed skill/depth, (c) tactics-puzzle accuracy.
2. **Budget:** the full rented-GPU run completes for ≤ ~$12.
3. **Local play:** runs on the M1 via UCI at a blitz time control (target ≥ a
   few hundred MCTS sims/move within a few seconds, with Metal + batched eval).

---

## 3. Decisions Log (resolved during design)

| # | Decision | Choice |
|---|----------|--------|
| D1 | Strength vs budget | $10 is the hard limit; reframe to "strongest for ~$10," staged for later scaling |
| D2 | Architecture lineage | **Transformer** now (not SE-ResNet) |
| D3 | Policy distillation target | **Full per-move distribution** (KL) from ChessBench action-values |
| D4 | Smolgen | **Included in Phase 1**, gated by an A/B run before full spend |
| D5 | MCTS upgrades | **All three**: batched eval + tree reuse + virtual loss |
| D6 | Data pre-encoding | **On the M1** (free); upload packed shards to the GPU box |
| D7 | Framework | ~~Stay in TensorFlow/Keras~~ → **PyTorch + MPS** (see 2026-06-18 update banner; TF's M1 GPU support was broken). Framework-agnostic code (MCTS/UCI/move-encoder/eval) reused either way. |
| D8 | Value target | **WDL 3-way** softmax (32-bin win% categorical noted as a refinement) |

---

## 4. Architecture — The Transformer Network

```
Board (FEN) ──canonically flipped to side-to-move's perspective──┐
                                                                 │
  64 squares ── piece-id embedding (13 → 256)  ─┐                │
              + learned positional embed (64 → 256)               │
  global state ─ castling×4, en-passant file,    │  small MLP →   │
    50-move clock, repetition COUNT, fullmove ───┘  conditioning bias added to every token
                                                 ▼
                    [CLS] token  +  64 square tokens     (seq = 65, d_model = 256)
                                                 │
                       ┌─────────────────────────┴────────────────────────┐
                       │  Transformer encoder × 8                          │
                       │  8-head self-attention + FFN(1024), pre-LN, GELU  │
                       │  + smolgen: learned position-dependent 64×64       │
                       │    additive attention bias (param-shared)         │
                       └─────────────────────────┬────────────────────────┘
                                                 │
              ┌──────────────────────────────────┼─────────────────────────────┐
              ▼                                   ▼                              ▼
      64 square outputs                      [CLS] output                  [CLS] output
              │                                   │                              │
   from-query qᵢ / to-key kⱼ               MLP → 3 logits                  MLP → scalar
   logit(i→j) = qᵢ·kⱼ/√d                   softmax → WDL                   = moves-left
   + underpromotion channels                    │                          (small aux loss)
   → gather into 1924 actions             v = P(W) − P(L) ∈ [−1,1]
   → mask illegal → softmax
              │                                   │                              │
          policy π                            value v                       moves_left
              └──────────────── consumed by MCTS (PUCT) ──────────────────────────┘
```

### Input representation (token encoder)
- **64 square tokens.** Each square → learned piece-identity embedding (13
  categories: empty + 6×2 pieces) + learned per-square positional embedding.
- **Canonical orientation.** The board is always flipped to the side-to-move's
  perspective, so the net learns one viewpoint and side-to-move is implicit.
  Moves produced by the policy head are mapped back to the true frame via the
  move encoder.
- **Global-state conditioning.** Castling rights (×4), en-passant file, **50-move
  / halfmove clock**, **repetition count** (not binary), fullmove number →
  small MLP → a vector added as a bias to every token. This closes the
  encoding-correctness gaps.
- **`[CLS]` token** prepended for value/moves-left pooling. Sequence length = 65.

### Trunk
- 8 pre-LN transformer encoder layers; 8 heads; `d_ff = 1024`; GELU; residual.
- **Smolgen** in each layer: compress tokens → small latent → a parameter-shared
  generator emits a per-head `64×64` additive bias on the attention logits (a
  learned, position-dependent spatial prior). Dims kept modest so total params
  stay in the **~10–13M** band the data slice can support.
- Because the sequence is only 65 tokens, attention is near-free; most FLOPs are
  in the FFNs.

### Heads
- **Policy (attention from→to, Lc0-style):** project each square output to a
  "from" query `qᵢ` and "to" key `kⱼ`; `logit(i→j) = qᵢ·kⱼ/√d`, plus dedicated
  channels for underpromotions. Gather into the existing **1924-action** space,
  apply the existing verified legal-move mask, softmax over legal moves. Removes
  the `Dense(128)` bottleneck and restores spatial structure.
- **Value (WDL):** `[CLS]` → MLP → 3 logits → softmax over {Win, Draw, Loss};
  MCTS uses `v = P(W) − P(L)`.
- **Moves-left:** `[CLS]` → MLP → scalar, small auxiliary loss.

### Interface to MCTS
`net(position) → (π over legal moves, scalar v)` (+ optional moves_left) — same
shape the existing PUCT search consumes.

---

## 5. Data & Distillation Training

### Dataset
- **Primary: DeepMind ChessBench action-value set** (`google-deepmind/
  searchless_chess`). 10M Stockfish-16-labeled games → ~15B (state, move, win%)
  points. Use a **bounded slice (~10–50M positions)**.
  - Per position it provides: **state win%** (→ value/WDL target), **per-legal-
    move win%** (→ full soft-policy target), and game ply/length (→ moves-left).
    One dataset feeds all three heads.
- **Starter set (pipeline bring-up): Kaggle `chessData.csv`** (~16M, FEN+cp) —
  tiny, fast, validates the loop before the big download.
- Reference alternatives: Lichess open eval dump (~388M positions, CC0).

### Targets & loss
- **Policy:** `KL( softmax(stockfish_move_winpct / T) ‖ predicted_π )` over legal
  moves (full distribution).
- **Value (WDL):** cross-entropy to a `(W,D,L)` target derived from win% via a
  draw-band mapping. The 32-bin win% categorical (HL-Gauss) is a drop-in
  refinement if 3-way WDL underperforms.
- **Moves-left:** Huber regression to plies-to-end (weight ~0.1).
- `L = KL_policy + 1.0·CE_value + 0.1·movesleft`.

### Training recipe (rented RTX 4090, ~25–30h)
- AdamW; batch 1024–2048; lr ~3e-4 with warmup + cosine decay; weight decay;
  gradient clipping; **mixed precision (bf16)**.
- ~1–2 epochs over the slice (tens of thousands of steps).
- Checkpoint to cloud storage; **spot instance** with resume-on-preemption.

### Pipeline & cost control (budget is won/lost here)
- **Pre-encode offline on the M1** (free): FEN → packed int8 token tensors +
  targets, written as sharded files (~a few–10 GB for the slice). Upload compact
  shards so **paid GPU time is ~100% training, ~0% data prep**.
- `tf.data` from pre-packed shards with prefetch.
- **Staged spend:** (1) full-pipeline smoke test on ~100k positions on the M1;
  (2) short ChessBench-slice + smolgen A/B run (~$1–2); (3) full ~$10 run.

---

## 6. Inference, MCTS & Local M1 Play

### MCTS (reused, with three upgrades — D5)
- **Batched leaf evaluation** (essential for a transformer): one network call
  per batch of leaves. Make `ParallelMCTS` the default play path.
- **Tree reuse:** carry the relevant subtree across moves instead of rebuilding
  the root each move.
- **Virtual loss:** so a single search gathers a batch of *distinct* leaves to
  feed the GPU.
- Backs up `v = P(W) − P(L)` from the WDL head.

### Orientation plumbing
Search flips to the canonical (side-to-move) frame; policy outputs are mapped
back via the move encoder/mask. This is the single most error-prone point — it
gets dedicated tests (Section 8).

### Local M1 play
- **Add `tensorflow-metal`** to `requirements.txt` (currently missing — the M1
  GPU is likely idle today) + **float16** inference, reusing the existing
  `@tf.function` path extended to batched 65-token input.
- Benchmark target: ≥ a few hundred sims/move within a few seconds at blitz.
- **Fallbacks** if Metal attention disappoints: shrink `d_model`/layers, lower
  sim count, or export to **CoreML** (Apple Neural Engine) — optional, not core.
- Existing `uci.py` / `scripts/play.py` reused, swapping in the new evaluator +
  token encoder, with dynamic sim count by time control.

---

## 7. Component Boundaries & File Plan

Four stable interfaces keep scaling a *swap*, not a rewrite:

1. **Encoder:** `position → (token_ids[65], state_features)`
2. **Network:** `batch → (policy_logits[1924], wdl_logits[3], moves_left)` —
   trunk and heads are separate modules.
3. **Evaluator (MCTS-facing):** `evaluate(positions) → (π masked, v)`
4. **Trainer:** consumes `(dataset, net)` — distillation now, self-play later.

| File | Phase-1 status |
|---|---|
| `config.py` | **Modify** — transformer + smolgen dims, data/shard paths, distillation hyperparams |
| `src/game/move_encoder.py` | **Reuse** (1924 actions + legal mask); policy head gathers into it |
| `src/game/chess_game.py` | **Reuse** |
| `src/game/board_encoder.py` | **Dormant** — conv/baseline reference |
| `src/game/token_encoder.py` | **New** — 65-token canonical encoding + global-state conditioning |
| `src/model/network.py` | **Dormant** — old conv net kept as baseline |
| `src/model/transformer.py` (+ `smolgen.py`, `heads.py`) | **New** — trunk, smolgen, policy/WDL/moves-left heads |
| `src/model/evaluator.py` | **New** — MCTS-facing wrapper; Metal/fp16 batched inference |
| `src/mcts/mcts.py`, `node.py` | **Modify** — batched eval default, tree reuse, virtual loss, WDL backup |
| `src/data/` (`chessbench.py`, `preencode.py`, `dataset.py`) | **New** — download/slice, M1 pre-encode to shards, `tf.data` loader |
| `src/training/distill_trainer.py` | **New** — distillation loop |
| `src/training/stockfish_evaluator.py` | **Repurpose** — label-gen → eval opponent |
| `src/eval/` (`arena.py`, `puzzles.py`, `elo.py`) | **New** — vs baseline, vs Stockfish, puzzles, Elo |
| `src/training/{self_play,parallel_self_play,curriculum_trainer,full_pipeline,trainer}.py`, `replay_buffer.py` | **Dormant in Phase 1** — reused later for self-play fine-tuning |
| `uci.py`, `scripts/play.py` | **Reuse** — swap in new evaluator + encoder |
| `scripts/{preencode,distill,eval}.py` | **New** — entry points; `scripts/train*.py` become dormant |

Nothing is deleted; the conv net and self-play stack remain as dormant
baseline/scaling assets.

---

## 8. Evaluation & Testing

### Strength measurement (success criterion #1)
- **Arena vs `model_iter_5`** — 100–200 game match; expect ~100%.
- **Vs Stockfish at pinned strength** — ladder of Skill Level / fixed depth or
  nodes; the ~50% crossover anchors Elo (`stockfish_evaluator.py` as opponent).
- **Elo estimate** — gauntlet vs fixed-Elo opponents → Ordo/BayesElo
  (`src/eval/elo.py`).
- **Tactics puzzles** — Lichess puzzle set, % solved.
- **Move-match** — top-1/top-5 agreement with Stockfish on a held-out set,
  tracked every checkpoint.

### Training-time validation
Held-out split tracking policy KL, value CE, value-sign accuracy, puzzle
accuracy per checkpoint; used to pick the best checkpoint.

### Engineering tests (non-negotiable — chess is an edge-case minefield)
- **Encoder:** canonical-flip correctness; 50-move clock & repetition-count
  populated; en-passant/castling round-trips on known positions.
- **Move-encoder ↔ policy gather:** property test over random games — every
  `python-chess` legal move maps to a unique logit; underpromotions handled;
  mask matches exactly.
- **MCTS:** tree-reuse preserves search invariants; virtual loss doesn't corrupt
  visit counts; **value orientation** correct (winning position scores positive
  for side to move).
- **Integration smoke:** pre-encode 100k → train a few steps → loss drops →
  save/load → play a full legal game over UCI; legality property test.

### Budget-protecting go/no-go gates (all cheap, all before the $10 run)
1. Smoke test passes (legal play, loss decreasing).
2. Data loader ≥ target positions/sec (GPU won't starve).
3. **Smolgen A/B:** smolgen ≥ baseline on validation, else drop it.
4. M1 inference meets the sims/sec target (or a documented fallback chosen).

We do not spend the budget until 1–4 pass.

---

## 9. Staging Path (beyond Phase 1)

Each step is a swap behind the Phase-1 interfaces; the distilled net is the
springboard.

- **Phase 2 — Scale the supervised net** (~$10s–$100s): wider/deeper transformer
  (`d_model` 384–512, 12–16 layers, ~30–100M), more of ChessBench's 15B points,
  more epochs (smolgen already in). Targets the 136M→270M band
  (~2224 internal → 2895 vs-humans as size and data grow together).
- **Phase 3 — Self-play RL fine-tuning** (AlphaZero/Lc0 step): reactivate the
  dormant self-play stack, initialized from the distilled net. Starting RL from
  ~2300 instead of random is the only affordable route past the Stockfish
  teacher's ceiling — multi-GPU and sustained; the gap to modern Stockfish
  remains large.
- **Phase 4 — Frontier** (optional, Lc0-scale): BT-style trunk refinements,
  tablebases, opening book.

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Underfitting at 10–13M on a data slice (< ~2000) | Add data/epochs before params; puzzle-accuracy tracking catches it early |
| Smolgen doesn't pay off on a tight budget | A/B gate before full run; drop if not ≥ baseline |
| M1/Metal attention too slow to play | Benchmark before training; batched eval + fp16; fallbacks (smaller net, fewer sims, CoreML/ANE) |
| Data loading starves the paid GPU | Offline pre-encode to packed shards + throughput gate |
| Value-orientation / canonical-flip bug silently tanks strength | Dedicated MCTS + encoder tests; arena vs baseline |
| Spot preemption mid-run | Checkpoint to cloud + resume |
| ChessBench format/loader complexity | Kaggle starter validates the loop first; bounded slice |
| Scope/budget creep | Staged spend (smoke → A/B → full) behind go/no-go gates |

---

## Appendix A — Verified reference figures

- Current net: **582,097 params** total; trunk **329,472**; policy `Dense`
  **248,196 (42.6%)**. (Trunk count independently confirmed.)
- AlphaZero: ~20 blocks × 256 filters, ~28–46M params (sources vary), `8×8×73`
  policy planes, T=8 history input.
- Lc0: SE-ResNet and transformer "BT" nets (smolgen, WDL, moves-left); BT4 ~191M.
- DeepMind "Grandmaster-Level Chess Without Search" (Ruoss et al., 2024):
  9M / 136M / 270M transformers; **270M ≈ 2895 Lichess blitz with no search**;
  trained by supervised distillation of Stockfish 16 labels.
- NNUE (Stockfish): CPU-efficient incrementally-updatable eval for alpha-beta —
  a different paradigm; not used here.
- GPU spot pricing (2026): RTX 4090 ~$0.30–0.40/hr; A100 ~$0.60–1.07/hr; H100
  ~$1.03–1.55/hr. $10 ≈ ~25–30h on a 4090.
- Datasets: ChessBench ~15B points (from 10M games); Lichess eval dump ~388M
  positions; Kaggle `chessData.csv` ~16M.

## Appendix B — References
- Ruoss et al., "Grandmaster-Level Chess Without Search," 2024 (arXiv 2402.04494);
  `github.com/google-deepmind/searchless_chess`.
- Silver et al., AlphaZero, 2018.
- Leela Chess Zero (Lc0) transformer/BT net documentation (smolgen, WDL,
  moves-left).
- Lichess open database (CC0).
