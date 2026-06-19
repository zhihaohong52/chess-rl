# Arena eval: raw policy vs MCTS in actual play (p2-value, 2026-06-19)

Model: `checkpoints/p2_value_300k/best.pt` (preset `p2-value`, HL-Gauss value head,
trained on 300k dense @ T=0.1). Each mover played 20 games vs Stockfish (depth 4)
at two skill levels via `scripts/arena_eval.py`.

| opponent | mover | W/D/L | score | est Elo |
| --- | --- | --- | --- | --- |
| SF skill 1 | raw  | 1/12/7 | 0.350 | ~792 |
| SF skill 1 | mcts | 12/6/2 | **0.750** | ~1091 |
| SF skill 3 | raw  | 4/3/13 | 0.275 | ~932 |
| SF skill 3 | mcts | 11/6/3 | **0.700** | ~1247 |

## Conclusion: the MCTS "regression" was a PUZZLE-METRIC ARTIFACT

In actual play, **MCTS-100 beats the raw policy by +0.40 (skill 1) and +0.425
(skill 3)** — ~+300 Elo, consistent across both opponents, far beyond 20-game
noise (std err ~0.10). The raw policy *loses* even to skill-1 Stockfish (0.350);
MCTS wins comfortably at both levels.

This reverses the conclusion drawn from the puzzle gauge (where MCTS scored *below*
raw, 0.297 vs 0.375). Tactical puzzles reward matching Stockfish's single top move
— exactly what the pattern-matching policy is trained to do — but over a full game
the raw policy accumulates positional/strategic errors that **tree search + the
value head correct**. The earlier "value-head bottleneck breaks MCTS" framing was
based on the wrong metric.

## Implications

1. **MCTS is the engine**, not raw policy. The value head (HL-Gauss, calibrated)
   is doing its job — search is strongly productive in play.
2. **Arena games, not tactical puzzles, are the right strength metric.** Keep
   puzzle raw-top-1 as a fast proxy for *policy* quality, but judge the *engine*
   by arena.
3. The Phase-1/2 worry about MCTS is resolved positively. Carry HL-Gauss forward;
   continue the Phase-2 stack (SwiGLU/dropout/EMA) judging by raw-top-1 (policy)
   AND arena (engine).
4. Caveats: 20 games/skill is noisy; SF skill 1/3 @ depth 4 is a weak ladder, so
   absolute Elo (~1100-1250 with MCTS) is rough. The raw-vs-mcts *gap* is the
   robust result. For the GPU scale-up, use a wider arena ladder + more games.
