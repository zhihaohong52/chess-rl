# Chess-RL Model Architecture Findings and Proposed Improvements

Codex review for Claude implementation.

## Executive Summary

The current PyTorch transformer is a better foundation than the legacy
TensorFlow residual CNN, but the latest empirical result changes the priority:
the action-value path should not be treated as the main training direction until
it beats policy-distribution training under equal evaluation.

Claude's recent Stockfish/ChessBench evaluation is the strongest evidence so
far:

| Model | Training data | Raw top-1 | MCTS-100 |
| --- | --- | --- | --- |
| Old policy-distribution | 62k dense, about 29 moves per position | 28.9% on 1k / 27.7% on 300 | 34.0% |
| New action-value | 1.98M pointwise, 1 move per position | 25.7% on 1k / 24.3% on 300 | 32.3% |

This is an important apples-to-apples result: the 30x larger pointwise
action-value dataset did not improve puzzle move selection or MCTS strength.
The practical recommendation is to make dense policy-distribution training the
primary path, keep action-value as an ablation, and focus architecture work on
trainability, policy quality, and evaluator consistency.

## Current Architecture Findings

- The active model is `src/model/transformer.py`, not the legacy
  `src/model/network.py` TensorFlow CNN used by older play/training code.
- The transformer uses 64 canonical square tokens plus a learned CLS token,
  `d_model=256`, 8 layers, 8 attention heads, and a 1024-wide FFN.
- The public forward contract is:
  `forward(square_tokens[B,64], state_features[B,18]) -> (policy_logits, wdl_logits, moves_left)`.
- The policy head is compact: it scores from-square and to-square embeddings,
  with a promotion adjustment, into the existing 1924-move policy index space.
- The value head is WDL softmax, which is preferable to a scalar tanh value
  because it can represent draw probability explicitly.
- The moves-left head is available as auxiliary supervision, but it should stay
  low-weight and diagnostic unless it clearly improves validation strength.
- Smolgen is the dominant encoder cost. In the current config, each encoder
  layer spends roughly 927k parameters on Smolgen, compared with roughly 263k
  on attention and 526k on the FFN.
- `transformer_dropout=0.0`, so the current model has little architectural
  regularization despite being trained from limited dense policy data.

## Data and Objective Findings

- Dense policy targets appear more useful than pointwise action-value targets
  for ChessBench move selection, even with far fewer unique positions.
- The dense policy distribution carries relative move preference information
  within a position. Pointwise action-value training loses much of this
  comparison signal unless each position has many evaluated moves and the
  objective reconstructs a full distribution.
- Reusing policy logits for action-value BCE is semantically awkward: policy
  logits should represent prior preference over legal moves, while action values
  represent expected outcome after choosing a move.
- For MCTS, policy priors and value estimates have different roles. Blending
  them into one move logit can make calibration harder and may explain why more
  pointwise data did not translate into better search results.

## Senior Research Proposal

The recommended research direction is not pure action-value training. The
stronger approach is to use Stockfish multi-move analysis to construct dense
policy distributions, then train the model primarily as a policy-distribution
model.

The core training target should be:

```text
board position -> probability distribution over legal moves
```

The data-generation process should be:

```text
position -> evaluate multiple legal/candidate moves with Stockfish
         -> convert move scores into a soft distribution
         -> train the policy head on that dense distribution
```

Example:

```text
Position X:
e4   -> +0.42
d4   -> +0.38
Nf3  -> +0.31
c4   -> +0.22
a3   -> -0.15

policy_target = softmax(stockfish_scores / temperature)
```

This is expected to outperform one-move-per-position action-value training
because chess move choice is comparative. The model must learn which legal move
is best relative to the other legal moves in the same position. A single
pointwise action-value record does not provide that comparison signal.

Expected near-term performer ranking:

1. Best expected: multi-move Stockfish targets converted into dense
   policy-distribution training, with WDL value supervision.
2. Potentially stronger later: dense policy + WDL + a separate auxiliary Q head.
3. Weakest current path: one-move-per-position action-value training.

Action-value can still be useful, but only if it is redesigned as an auxiliary
objective with sufficient per-position move coverage. It should not replace the
policy-distribution objective until it wins on raw top-1 and MCTS gates.

Recommended multi-head objective:

```text
total_loss =
    policy_ce(policy_logits, dense_stockfish_policy)
  + value_weight * wdl_ce(wdl_logits, stockfish_wdl)
  + q_weight * q_loss(q_logits, per_move_values)
  + moves_left_weight * moves_left_loss
```

Loss priority should be:

```text
policy loss: dominant
WDL value: important
Q/action-value: auxiliary only
moves-left: small auxiliary
```

For candidate move generation, prefer all legal moves when affordable. When not
affordable, include Stockfish MultiPV moves, captures, checks, promotions,
castling moves, and a random sample of quiet legal moves. This gives both the
top of the distribution and enough negative/low-quality alternatives to teach
ranking.

## Proposed Improvements

### 1. Make Policy-Distribution Training the Primary Track

- Keep dense policy-distribution distillation as the default supervised
  objective.
- Treat action-value training as experimental until it beats the dense policy
  baseline on raw top-1 and MCTS-100.
- Prefer generating denser Stockfish targets per position over collecting more
  one-move-per-position records.
- Track results by evaluator type so policy models are evaluated with policy
  evaluators and action-value models are evaluated with action-value evaluators.

### 2. Add a Stable Architecture Preset

- Preserve the existing model as `baseline-v1`.
- Add a trainability-focused `stable-v1` preset with:
  - dropout or residual dropout around attention and FFN, initially 0.05-0.10;
  - optional LayerScale or residual scaling in encoder blocks;
  - unchanged public forward outputs by default.
- Do not scale model depth or width until the stable preset is validated
  against the current policy-distribution baseline.

### 3. Make Smolgen Configurable

- Add a config option such as `smolgen_mode` with values:
  `none`, `every_layer`, `every_other_layer`, and `shared`.
- Benchmark `every_other_layer` and smaller Smolgen hidden/gen dimensions before
  increasing total model size.
- Keep `every_layer` as the baseline-compatible option.
- Report parameter count and validation metrics for each mode.

### 4. Improve the Policy Head Before Adding Raw Capacity

- Keep the 1924-index move interface.
- Enrich the policy head with learned move-type features:
  - move delta or direction;
  - ray distance;
  - knight move flag;
  - castling flag;
  - promotion class.
- Condition policy scoring on CLS/global state as well as from/to square
  embeddings.
- The goal is to make legal chess move semantics easier to learn without
  replacing the rest of the training pipeline.

### 5. Keep Action-Value Separate If It Returns

- If action-value training remains in the repo, add a dedicated Q/action-value
  head instead of overloading `policy_logits`.
- Keep policy logits for priors and Q logits for move outcome estimates.
- In MCTS, make the evaluator choice explicit:
  - policy evaluator: priors from `policy_logits`, root/leaf value from WDL;
  - action-value evaluator: priors still from policy or a calibrated prior head,
    move value from Q only where intentionally configured.
- Do not promote action-value to default unless it beats the policy baseline on
  the same 300-puzzle and 1k-puzzle gates.

### 6. Add Compact Board-History Signal

- The transformer token input currently emphasizes the current canonical board
  plus state features. The older plane encoder represented history more
  directly.
- Add a compact history representation, such as previous-position square tokens
  for the last 2-4 plies or learned per-square history embeddings.
- This may help repetition, en-passant context, tactical continuity, and value
  calibration.

### 7. Add Lightweight Auxiliary Supervision

- Consider low-weight auxiliary heads derived during pre-encoding:
  - material balance;
  - game phase;
  - legal move count bucket;
  - in-check flag;
  - side-to-move expected score bucket.
- These should regularize representation learning, not become product outputs.
- Keep auxiliary losses low-weight and remove any that do not improve policy
  validation or MCTS gates.

## Implementation Notes for Claude

- Preserve the existing default forward contract unless a new optional config
  explicitly requests extra heads.
- Keep changes ablation-friendly. Every architectural change should be
  controlled by config and have a baseline-compatible path.
- Avoid deleting the legacy TensorFlow network unless a separate cleanup task
  requests it; some scripts still reference it.
- Prefer adding concise tests around shape, config behavior, evaluator routing,
  and loss selection over broad rewrites.
- Do not make action-value the default path based on dataset size alone. The
  current evidence says dense policy distribution is stronger.

## Recommended Evaluation Gates

- Raw top-1 on the same 300-puzzle and 1k-puzzle sets used in the latest
  comparison.
- MCTS-100 on the same 300-puzzle set, using each model's correct evaluator.
- Policy metrics: top-1, top-3/top-5, KL or CE, and legal-move calibration.
- Value metrics: WDL CE, value sign accuracy, draw calibration, and terminal
  position sanity checks.
- Throughput metrics: model batch latency and MCTS nodes/sec.
- Ablation table for each proposed architecture change against `baseline-v1`.

## Notes on `scripts/gen_dense_stockfish.py`

Claude's proposed dense Stockfish generation script is aligned with the
recommended research direction. It generates per-position dense policy targets
from Stockfish MultiPV analysis, instead of relying on one-move action-value
records.

Current positive signs:

- The script compiles with `python3 -m py_compile scripts/gen_dense_stockfish.py`.
- It deduplicates FENs before evaluation, avoiding obvious repeated Stockfish
  work.
- It skips terminal/no-legal-move positions.
- It uses one single-threaded Stockfish process per worker, with `Threads=1`
  and `Hash=64`, so `--workers 6` means roughly six CPU cores of Stockfish work.
- It writes normal pre-encoded NPZ shards through `write_shard`, so outputs
  should be compatible with the existing dataloader path.

Do not start the hours-long run without a smoke test. Recommended smoke command:

```bash
rtk test python3 scripts/gen_dense_stockfish.py \
  --fen-bag data/train/action_value-00000-of-02148_data.bag \
  --max-positions 100 \
  --depth 8 \
  --workers 2 \
  --multipv-cap 12 \
  --out-dir data/shards_dense_smoke \
  --shard-size 50
```

After the smoke run, verify that the produced shards load through
`src.data.dataset.make_dataloader` or the normal distillation script before
launching the full generation job.

Recommended first production run:

```bash
python3 scripts/gen_dense_stockfish.py \
  --fen-bag data/train/action_value-00000-of-02148_data.bag \
  --max-positions 50000 \
  --depth 10 \
  --workers 6 \
  --multipv-cap 16 \
  --out-dir data/shards_dense_d10_mpv16 \
  --shard-size 20000 \
  --val-fraction 0.02
```

This is a better first experiment than evaluating all legal moves at depth 12.
All-legal MultiPV can be very expensive in positions with 30-50 legal moves.
The first research objective is to beat the old 62k dense baseline, not to
maximize Stockfish compute per position.

Suggested scaling path:

1. Smoke test: 100 positions, depth 8, `multipv-cap=12`.
2. First production: 50k positions, depth 10, `multipv-cap=16`.
3. Larger production: 100k+ positions if the first production run improves raw
   top-1 and MCTS-100.
4. Higher quality: depth 12 and/or larger `multipv-cap` only after confirming
   the smaller run is useful.

Operational cautions:

- Record exact command, Stockfish version, depth, worker count, MultiPV cap,
  temperature, source bag, output directory, and elapsed time.
- Keep each generated dataset in a separate output directory; do not mix smoke
  and production shards.
- If throughput is low, reduce `multipv-cap` before reducing position count.
- If CPU becomes saturated or the machine is needed interactively, lower
  `--workers`.
- Treat all-legal MultiPV as a later quality experiment, not the default first
  run.

### Optimizing Depth-12 Runs

Depth 12 generation is worth testing, but the default should not be all-legal
MultiPV. Runtime scales roughly with:

```text
positions * candidate moves per position * search depth
```

For depth 12, reduce MultiPV width before reducing position count. Recommended
starting point:

```bash
--depth 12 --multipv-cap 12
```

Use `--multipv-cap 16` only after confirming throughput is acceptable. Avoid
this as a first run:

```bash
--depth 12 --multipv-cap 0
```

All-legal MultiPV can mean 30-50 candidate moves in many positions and should
be treated as a later quality experiment.

Best script-level optimization: add a two-stage candidate strategy.

1. Run a shallow pass at depth 6-8 with wider MultiPV, such as 24-32.
2. Build a candidate set from:
   - top shallow Stockfish moves;
   - all checks;
   - all captures;
   - all promotions;
   - castling moves;
   - a small random sample of quiet legal moves.
3. Run depth 12 only on that candidate set using Stockfish `root_moves`.

This keeps the useful ranking signal while avoiding depth-12 search on many
obviously poor quiet moves.

Keep one Stockfish thread per worker. The current script already configures
`Threads=1`, which is good for multi-process throughput. Set `--workers` near
the number of physical performance cores, leaving 1-2 cores free if the machine
needs to remain interactive.

For depth 12, consider increasing Stockfish hash from 64 MB to 128 MB per
worker, or 256 MB if memory allows:

```text
total_hash_memory = workers * hash_mb
```

Example: 6 workers with 128 MB hash uses about 768 MB for Stockfish hash.

Before any multi-hour depth-12 run, add resume safety:

- write each shard to a temporary file and atomically rename when complete;
- skip already completed shard names on restart;
- write a small manifest with command, worker id, completed shards, and counts.

Recommended first depth-12 command:

```bash
python3 scripts/gen_dense_stockfish.py \
  --fen-bag data/train/action_value-00000-of-02148_data.bag \
  --max-positions 20000 \
  --depth 12 \
  --workers 6 \
  --multipv-cap 12 \
  --out-dir data/shards_dense_d12_mpv12_20k \
  --shard-size 10000 \
  --val-fraction 0.02
```

If this beats the current dense-policy baseline, scale position count before
widening MultiPV. Recommended scaling order:

1. Depth 12, MultiPV 12, 20k positions.
2. Depth 12, MultiPV 12, 50k-100k positions.
3. Depth 12, MultiPV 16, 50k-100k positions.
4. All-legal MultiPV only if earlier runs justify the extra compute.

## Priority Order

1. Lock policy-distribution training/evaluation as the main baseline.
2. Generate multi-move Stockfish targets and convert them into dense policy
   distributions.
3. Train policy + WDL from those dense Stockfish targets.
4. Add stable regularization preset and evaluate against the current policy
   model.
5. Run Smolgen ablations to reduce over-parameterized bias generation.
6. Improve policy head move semantics.
7. Add compact board history.
8. Revisit action-value only with a separate Q head and dense per-position move
   coverage.
