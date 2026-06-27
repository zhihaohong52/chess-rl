# Model Architecture Comments for a Stockfish-Beating Target

Codex review from a data analyst / AI researcher perspective. This is not a code hygiene review. It is about whether the current model architecture, objective, data plan, and search integration plausibly move toward an engine that can beat Stockfish.

## Executive Assessment

The current PyTorch transformer is a good prototype, but it is not on a path to beat Stockfish by scaling the present setup alone.

Current model shape:

- 64 canonical square tokens plus a learned CLS token.
- 18 global state features.
- `d_model=256`, `n_layers=8`, `n_heads=8`, `d_ff=1024`.
- Smolgen-style position-dependent attention bias.
- 1924-index move policy head.
- WDL value head or optional HL-Gauss value head.
- Moves-left auxiliary head.
- Around 10.3M parameters for `baseline-v1`.

This is a reasonable Leela-inspired distillation model. It is not yet a superhuman engine architecture. To compete with modern Stockfish, the project needs three things:

1. A stronger representation of chess state, especially history/repetition and tactical move semantics.
2. A training loop that goes beyond imitating Stockfish snapshots.
3. A search stack that is calibrated, batched, and tuned as part of the model, not treated as a downstream consumer.

The right near-term objective is not "beat Stockfish." The right objective is:

```text
build a model+search system that reliably improves with scale and beats its own previous checkpoints under fixed compute
```

If that curve is not established, more parameters and more Stockfish labels will mostly produce a stronger imitation model, not an engine that surpasses the teacher.

## Main Finding

The current architecture is policy-distillation first. A Stockfish-beating system must become policy-value-search co-trained.

Dense Stockfish policy targets are useful bootstrapping data. They should remain the first stage. But a model trained only to reproduce Stockfish move distributions will usually inherit Stockfish's preferences without learning an independent improvement mechanism. AlphaZero-style systems became strong because self-play plus search produced new targets. Leela-style systems became strong because network training and MCTS were coupled over many generations.

The current repo has pieces of this:

- Transformer policy/value network.
- Dense policy distillation.
- MCTS evaluator.
- Gate and ablation scripts.

But those pieces are not yet a closed improvement loop.

## Architecture Comments

### 1. Token State Is Too Thin for Elite Chess

The 64 square tokens plus 18 state features are elegant, but the current representation is intentionally minimal. It does not directly encode enough temporal information for repetition, irreversible move context, or tactical history.

This matters because high-level chess strength is not just static board evaluation. Search needs a value function that understands fortress risk, repetition, move-count dynamics, and historical constraints.

Recommended changes:

- Add compact history tokens or planes for the last 2-8 positions.
- Encode repetition features more richly than a small scalar global feature.
- Encode halfmove clock and fullmove phase with normalized continuous features and bucket embeddings.
- Add attack-map or legal-move-derived auxiliary channels only if measured to improve value calibration.

Do not jump straight to huge history. Start with:

```text
current board tokens
+ previous board delta tokens for 2 plies
+ repetition bucket
+ halfmove bucket
```

A small temporal signal is likely higher leverage than simply widening `d_model`.

### 2. The Policy Head Needs More Move Semantics

The current 1924-index move head is serviceable, but the policy head appears mostly from-square/to-square/promotion structured. For a top engine, the policy head should understand move type and tactical role more explicitly.

Recommended policy-head features:

- Move type: quiet, capture, check, promotion, castle, en passant.
- Moving piece and captured piece.
- Relative geometry: ray direction, ray distance, knight pattern, pawn direction.
- Destination square occupancy and attack status.
- Legality mask injected before any final policy normalization.
- Optional global conditioning from CLS into move scoring.

The policy should score candidate legal moves with a shared move-scoring function, not only emit a flat global vector. A better long-term interface is:

```text
board encoding + legal move list -> score per legal move
```

The flat 1924 logits can remain for speed and compatibility, but internally the head should behave like a move scorer. That makes it easier to add move features, calibrate legal probabilities, and experiment with sampled candidate sets.

### 3. Value Needs to Be Search-Calibrated, Not Just WDL-Accurate

The current WDL and HL-Gauss value heads are useful. The missing piece is search calibration. For MCTS, a slightly miscalibrated value head can be worse than a less accurate but better-calibrated value head.

Recommended value targets:

- WDL from Stockfish or self-play.
- Scalar expected score.
- Distributional value, such as HL-Gauss, as the default serious value target.
- Optional uncertainty head for search risk control.
- Optional Q head for per-move post-action value, but only as an auxiliary head.

Do not reuse policy logits as action values. That conflates two different quantities:

```text
P(a | s): prior move probability
Q(s, a): expected outcome after choosing move a
```

For MCTS, these must remain separate. A model that wants to beat Stockfish needs strong priors and strong leaf values; it should not ask one logit to do both.

### 4. Smolgen Is Worth Keeping, But It Should Be Ablated Carefully

The Smolgen-style attention bias is one of the more promising choices in the current architecture. Chess has strong square-to-square relational structure, and learned attention bias can help the model represent long-range piece interactions.

The risk is parameter and compute allocation. Smolgen at every layer may be less valuable than:

- Shared Smolgen across layers.
- Smolgen every other layer.
- Smaller generated bias dimension.
- Dedicated geometric bias plus smaller learned residual bias.

Recommended ablation order:

```text
baseline-v1
shared-smolgen
every-other-layer-smolgen
geometric-bias-plus-small-smolgen
no-smolgen
```

Evaluate on raw policy match, value calibration, and MCTS strength. Do not decide from policy top-1 alone.

### 5. Model Scale Is Currently Prototype Scale

A 10M parameter transformer can be useful and fast. It is not likely to beat Stockfish with serious search. The goal should be to establish a scaling ladder.

Recommended ladder:

```text
Tiny:   10M params  - fast iteration, current baseline
Small:  30M params  - first serious training runs
Base:   80M params  - serious MCTS candidate
Large: 200M+ params - only after data/search loop works
```

Do not scale to 200M until the 10M and 30M models show monotonic improvement under fixed gates. If 30M does not beat 10M after controlling for data and search, architecture or target quality is the bottleneck.

### 6. Dense Stockfish Policy Is Bootstrapping, Not the Final Objective

Dense Stockfish targets are the right first-stage data. They teach move priors efficiently. But a model trained only on Stockfish labels is bounded by teacher quality and label noise.

Use this curriculum:

1. Supervised dense Stockfish policy + value distillation.
2. MCTS using the network against fixed opponents and previous checkpoints.
3. Generate self-play games with search-improved policies.
4. Train on search visit distributions, WDL outcomes, and value targets.
5. Re-evaluate against frozen baselines.

The key transition is from:

```text
Stockfish says move X is best
```

to:

```text
network-guided search found this stronger distribution, and later outcomes validated it
```

That transition is what creates a chance to surpass the teacher.

### 7. MCTS Must Become a First-Class Training Component

The current evaluator-facing MCTS is useful, but it looks like an evaluation consumer rather than the center of the training algorithm.

For a Stockfish-beating target, MCTS needs:

- Large batched GPU inference.
- Root Dirichlet noise for self-play.
- Temperature schedule.
- Visit-count policy targets.
- Resignation calibration.
- Draw/repetition-aware terminal handling.
- Transposition support or at least careful tree reuse.
- Time/node budget parity in gates.
- Calibration metrics for policy entropy and value error after search.

The network should be trained on search outputs, not just used by search after training.

### 8. Add a Tactical Auxiliary Suite

Modern chess engines live or die tactically. Pure policy imitation may learn tactics implicitly, but auxiliaries can improve sample efficiency.

Recommended auxiliary heads:

- Checkmate-in-N bucket where available.
- Material balance prediction.
- Threat / in-check / gives-check prediction.
- Legal move count prediction.
- Best move category prediction.
- Static exchange evaluation bucket for captures.

Keep auxiliary losses low-weight. They should shape representation, not dominate the policy/value objective.

Suggested loss structure:

```text
loss =
  policy_weight * policy_ce
+ value_weight * distributional_value_loss
+ q_weight * q_aux_loss
+ moves_left_weight * moves_left_loss
+ tactical_aux_weight * tactical_aux_losses
```

The main objective should remain policy + value.

### 9. Current Evaluation Is Too Weak for the Stockfish Goal

Top-1 puzzle match and policy CE are useful diagnostics, but they do not measure engine strength. A model can have better imitation accuracy and still play weaker under search.

Required gates:

- Policy top-1/top-3/top-5 on fixed dense validation sets.
- Value calibration, including expected score calibration buckets.
- MCTS puzzle solving at fixed simulations.
- Head-to-head against previous checkpoint at equal node budget.
- Head-to-head against fixed Stockfish levels at equal time or nodes.
- SPRT-style match testing for serious comparisons.

The main promotion criterion should be match strength, not validation CE.

## Recommended Target Architecture

The next serious architecture should be:

```text
Input:
  canonical board tokens
  compact history/delta tokens
  global state features
  legal move feature table

Backbone:
  transformer, 12-16 layers
  d_model 384-512
  attention heads 8-16
  SwiGLU FFN
  RMSNorm or stable Pre-LN
  dropout/residual dropout around 0.05 for supervised training
  Smolgen or geometric+learned attention bias

Heads:
  legal-move scorer policy head
  distributional value head
  optional Q auxiliary head
  moves-left head
  tactical auxiliary heads

Search:
  batched PUCT MCTS
  visit-count policy targets
  self-play replay buffer
  checkpoint-vs-checkpoint promotion gates
```

This is closer to a Leela/AlphaZero-style system than a pure Stockfish distillation model.

## What I Would Do Next

### Phase 1: Make the Current 10M Model Scientifically Reliable

- Keep `baseline-v1` frozen.
- Train dense policy + HL-Gauss value.
- Add legal-move feature conditioning to the policy head.
- Add compact 2-ply history.
- Add calibration metrics.
- Run fixed MCTS gates.

Success criterion:

```text
new model beats baseline-v1 in head-to-head at equal simulations
```

### Phase 2: Scale to a 30M Model

- Increase `d_model` and depth modestly.
- Keep the same objective and data.
- Do not add new heads in the same run.

Success criterion:

```text
30M model beats 10M model under identical MCTS budget
```

### Phase 3: Close the Self-Play Loop

- Generate MCTS self-play with the best checkpoint.
- Train on visit-count policies and outcomes.
- Promote only if the new checkpoint beats the previous checkpoint.

Success criterion:

```text
Elo improves across generations without increasing search budget
```

### Phase 4: Chase Stockfish

Only after the self-play loop improves reliably:

- Increase model scale.
- Increase MCTS simulations.
- Tune PUCT and temperature.
- Compare against fixed Stockfish versions and node budgets.

At this point "beat Stockfish" becomes a measurable engineering target rather than a slogan.

## Hard Truth

Beating current Stockfish is an extremely high bar. It likely requires large compute, disciplined self-play, strong search engineering, and many failed ablations. A small transformer trained mostly on Stockfish labels can become a good neural chess engine, but it will not automatically exceed Stockfish.

The most important architectural shift is this:

```text
from: model imitates Stockfish
to:   model + search generates stronger targets than the previous model
```

That is the path worth building.
