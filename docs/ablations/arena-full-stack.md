# Full-stack model: arena + raw gauge (2026-06-19)

Model: `checkpoints/p2_full_300k/best_ema.pt` — preset `p2-value-swiglu-drop`
(HL-Gauss value head + SwiGLU FFN + dropout 0.05) trained on 300k dense @ T=0.1,
7000 steps, with **EMA (decay 0.999)**. The whole Phase-2 feature stack at once.

## Engine comparison (arena, 20 games/skill vs Stockfish depth 4)

| model | raw_top1@1000 | MCTS vs SF-1 | MCTS vs SF-3 | value sign-acc |
| --- | --- | --- | --- | --- |
| baseline-v1 | 0.289 | — | — | 0.831 |
| hf-dense-300k (WDL) | 0.359 | — | — | 0.847 |
| p2-value (HL-Gauss) | 0.375 | 0.750 | 0.700 | 0.873 |
| **p2-full (+SwiGLU+dropout+EMA)** | 0.364 | **0.850** | 0.700 | **0.880** |

Full-stack arena detail: raw 0.350/0.350 vs mcts **0.850 (14W/6D/0L)** / 0.700 at
skill 1 / 3. EMA gave a small raw bump (best.pt 0.360 → best_ema.pt 0.364).

## Verdict: full stack is `phase2-best`

The full stack is the **strongest engine** produced: MCTS beats Stockfish skill 1
at 0.850 (undefeated, +110 est Elo over p2-value) and ties skill 3 at 0.700, with
the best-calibrated value head (sign-acc 0.880). Raw top-1 dipped slightly vs
p2-value (0.364 vs 0.375) — dropout regularization trades a little puzzle-fit — but
**the engine plays with MCTS, and MCTS is the best of any config.** Stacking the
features worked: SwiGLU + EMA improved convergence/generalization, dropout
regularized, and the better value head made search stronger.

`phase2-best` = preset `p2-value-swiglu-drop` + `--ema-decay 0.999`, gauged on
`best_ema.pt`.

## Caveats / next

- 20 games/skill is noisy (std err ~0.10) and SF skill 1/3 @ depth 4 is a weak
  ladder; absolute Elo (~1200) is rough. The full-stack > p2-value gap at skill 1
  (0.85 vs 0.75) is suggestive but within ~1.5 std err — directionally positive,
  consistent with the better value head.
- For the GPU scale-up: use `phase2-best`, a bigger model + more dense shards, and
  a wider arena ladder (skills up to ~10-15) with more games for a firmer Elo.
- Optional: value_loss_weight 0.3-0.5 to recover raw top-1 without losing the
  value/engine gains.
