# SOTA Elo Architecture Recommendations

Codex architecture memo for evolving this project from a strong transformer prototype into a model+search system that can plausibly approach SOTA chess-engine Elo.

## Core Thesis

The current architecture should not be optimized as a pure Stockfish imitation model. Dense Stockfish distillation is useful for bootstrapping, but a SOTA-strength engine needs a closed improvement loop:

```text
policy/value model -> MCTS/search -> stronger targets -> retraining -> promotion by match Elo
```

The model architecture should therefore be designed around search-calibrated policy/value training, not only validation top-1 or cross-entropy against Stockfish labels.

## Current Baseline

The current `ChessTransformer` is approximately:

```text
Input:
  64 canonical square tokens
  18 global state features

Backbone:
  d_model = 256
  n_layers = 8
  n_heads = 8
  d_ff = 1024
  Smolgen-style attention bias

Heads:
  1924-logit policy head
  WDL or HL-Gauss value head
  moves-left auxiliary head

Scale:
  about 10.3M parameters
```

This is a good prototype and distillation baseline. It is not yet a SOTA engine architecture.

## Highest-Leverage Architecture Improvements

### 1. Upgrade the Input Representation

The current board representation is compact but too thin for elite search. Static piece placement plus a few global features misses history and tactical context that matter for value calibration.

Add:

- 2-8 ply compact history or board-delta tokens.
- Repetition bucket features.
- Fifty-move-rule / halfmove-clock bucket embeddings.
- Fullmove or phase embeddings.
- Optional attack/threat features.
- Optional threatened-piece map.

Recommended first version:

```text
current board tokens
+ previous-board delta tokens for 2 plies
+ repetition bucket
+ halfmove bucket
+ phase bucket
```

Do not start with a large history stack. Prove that compact history improves value calibration and MCTS Elo first.

### 2. Replace Flat Policy Emission With Legal-Move Scoring

The current policy emits 1924 logits and filters illegal moves during evaluation. This is workable, but a SOTA policy should score the actual legal move list with move features.

Target shape:

```text
board encoding + legal move feature table -> score per legal move
```

Move features should include:

- Moving piece.
- Captured piece.
- Promotion piece.
- Move type: quiet, capture, check, promotion, castle, en passant.
- Ray direction and distance.
- Knight/pawn special geometry.
- Destination square occupancy.
- Destination square attacked/defended status.
- Global CLS conditioning.

Keep the 1924-index output as a compatibility layer if useful, but make the internal head behave like a candidate legal-move scorer.

### 3. Make Distributional Value the Serious Default

WDL is useful, but SOTA search needs calibrated value, not only correct sign. MCTS is sensitive to miscalibrated leaf values.

Recommended value stack:

```text
distributional value head
expected score projection
WDL projection or auxiliary
moves-left head
optional uncertainty head
```

Use HL-Gauss or another distributional value target as the serious path. Track calibration buckets, not just value sign accuracy.

Avoid using policy logits as action values. Keep these separate:

```text
P(a | s): policy prior
V(s): position value
Q(s, a): post-action expected result
```

If Q is useful, add a separate auxiliary Q head.

### 4. Add Tactical Auxiliary Heads

Tactical failures will cap Elo. Add low-weight auxiliary heads that force the representation to learn tactical structure.

Good candidates:

- Gives-check prediction.
- In-check prediction.
- Legal move count.
- Material balance.
- Captured-piece / capture-value bucket.
- Static exchange evaluation bucket for captures.
- Mate-in-N bucket where labels exist.
- Threatened-piece map.

Suggested loss:

```text
loss =
  policy_weight * policy_ce
+ value_weight * distributional_value_loss
+ moves_left_weight * moves_left_loss
+ q_weight * q_aux_loss
+ tactical_weight * tactical_aux_losses
```

Auxiliary losses should shape representation, not dominate training.

### 5. Treat Smolgen as a Major Ablation Axis

Smolgen-style attention bias is promising because chess has strong square-to-square relational structure. Keep it, but measure its variants.

Ablation ladder:

```text
no-smolgen
shared-smolgen
every-other-layer-smolgen
every-layer-smolgen
geometric-bias + small learned Smolgen
```

Evaluate by MCTS Elo, value calibration, and policy entropy. Do not choose based only on policy top-1.

### 6. Establish a Model Scaling Ladder

The current 10M model is an iteration baseline. A SOTA trajectory needs larger models, but scaling should be gated by measured improvement.

Recommended ladder:

```text
Tiny:   10M params   current fast baseline
Small:  30M params   first serious model
Base:   80M params   serious engine candidate
Large:  150M-250M    SOTA research range
```

Do not jump directly to `Large`. First require:

```text
30M beats 10M under equal MCTS budget
80M beats 30M under equal MCTS budget
```

If scaling does not improve match Elo, the bottleneck is probably data quality, value calibration, search, or target generation rather than raw capacity.

## Search Must Be Part of the Architecture

For SOTA Elo, MCTS is not an evaluation add-on. It is part of the training system.

Required search features:

- Batched GPU inference.
- PUCT with tuned constants.
- Root Dirichlet noise for self-play.
- Temperature schedule.
- Visit-count policy targets.
- Tree reuse.
- Transposition cache or table.
- Repetition/draw-aware terminal handling.
- Resignation calibration.
- Equal-node and equal-time match gates.

The network should train on search outputs:

```text
policy target = MCTS visit distribution
value target = game outcome or search-calibrated value
```

This is the key difference between imitating Stockfish and building an engine that can improve past its teacher.

## Training Curriculum

### Stage 1: Supervised Bootstrap

Train on dense Stockfish policy and value targets.

Goal:

```text
good priors, reasonable value, stable MCTS behavior
```

### Stage 2: Search-Calibrated Evaluation

Use the model inside MCTS and evaluate:

- Policy top-k.
- Policy cross-entropy.
- Value calibration.
- MCTS puzzle solve rate.
- Match Elo against previous checkpoints.

Promotion should be based on match strength, not validation CE.

### Stage 3: Self-Play Target Generation

Generate games using network-guided MCTS.

Train on:

- Visit-count policy distributions.
- Game outcomes.
- Moves-left targets.
- Tactical auxiliaries.

### Stage 4: Checkpoint Promotion Loop

Only promote a checkpoint if it beats the previous best under fixed search conditions.

Target metric:

```text
Elo improves generation over generation without increasing search budget
```

### Stage 5: SOTA Chase

Only after the loop improves reliably:

- Increase model size.
- Increase search budget.
- Tune PUCT.
- Add larger self-play pools.
- Run SPRT-style match testing.
- Compare against fixed Stockfish versions under equal time or node budgets.

## Recommended Next Architecture Variant

The next serious model should be:

```text
Input:
  canonical square tokens
  compact 2-ply history/delta tokens
  global state bucket embeddings
  legal move feature table

Backbone:
  transformer
  12-16 layers
  d_model 384-512
  8-16 heads
  SwiGLU FFN
  stable Pre-LN or RMSNorm
  dropout/residual dropout around 0.05
  Smolgen or geometric+learned attention bias

Heads:
  legal-move scorer policy head
  distributional value head
  moves-left head
  optional Q auxiliary head
  tactical auxiliary heads

Search:
  batched PUCT MCTS
  visit-count target generation
  self-play replay buffer
  checkpoint promotion gates
```

## Practical Priority Order

1. Freeze current `baseline-v1`.
2. Make HL-Gauss/distributional value the serious value baseline.
3. Add compact history and repetition features.
4. Replace or augment the policy head with legal-move feature scoring.
5. Add value calibration metrics.
6. Add head-to-head checkpoint gates at equal MCTS budget.
7. Scale to a 30M model only after the 10M setup is reliable.
8. Close the self-play loop.
9. Scale to 80M+ only after self-play shows monotonic improvement.

## Hard Constraint

Pure Stockfish distillation is unlikely to produce a model that beats Stockfish. It can produce a strong neural engine, but surpassing the teacher requires the system to generate stronger targets than the teacher labels.

The architectural north star should be:

```text
model imitates Stockfish
  -> model guides search
  -> search produces improved targets
  -> model improves beyond previous model
  -> repeat
```

That loop is the credible path toward SOTA Elo.
