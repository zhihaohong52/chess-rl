| preset | params | raw_top1@300 | raw_top1@1000 | mcts100_top1@300 | mate_in_1 | top1 | top3 | top5 | policy_ce | legal_mass | wdl_ce | value_sign_acc | draw_cal | batch_latency_ms | positions_per_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p2-value | 10263748 | 0.3867 | 0.3750 | 0.2967 | 0.6667 | 0.3068 | 0.5506 | 0.6659 | 3.2019 | 0.9784 | 2.0057 | 0.8609 | 0.0300 | 142.7209 | 1793.7102 |

> For the hlgauss head: the `wdl_ce` column is the HL-Gauss value cross-entropy
> (64-bucket, larger scale — not comparable to WDL's ~0.44) and `draw_cal` is the
> value MAE `|v̂ − v|` (0.030 = well-calibrated). Row is the HL-Gauss model
> (preset `p2-value`), trained on the 300k dense set, 7000 steps, value_loss_weight=1.0.

## Decision (2026-06-19)

First Phase-2 ablation: HL-Gauss distributional value head vs `hf-dense-300k` (WDL head), same data/recipe.

| gate | baseline-v1 | hf-dense-300k | p2-value (HL-Gauss) |
| --- | --- | --- | --- |
| raw_top1@1000 | 0.289 | 0.359 | **0.375** |
| raw_top1@300 | 0.277 | 0.323 | **0.387** |
| mcts100_top1@300 | 0.340 | 0.310 | 0.297 |
| value sign-acc | 0.831 | 0.847 | 0.861 |
| value MAE | — | — | 0.030 |

**Verdict: KEEP p2-value (raw top-1 improved to 0.375, best of all models), but the
headline goal — flip MCTS ≥ raw — was NOT achieved.**

Key finding: the value head is now *well-calibrated* (MAE 0.030, sign-acc 0.861),
yet shallow MCTS-100 still underperforms raw (0.297 vs 0.375) — worse than even
`hf-dense-300k`'s MCTS. So **the MCTS regression is not primarily a value-calibration
problem.** The most likely explanation is that tactical PUZZLES favor the
pattern-matching policy over positional-value-guided shallow search; a calibrated
positional value cannot supply tactical lookahead that the policy doesn't already
encode, and 100 sims is too shallow to out-search a strong prior. The earlier
c_puct/sims sweep ("more sims → worse") points the same way.

Notable secondary result: HL-Gauss *raised raw top-1* (0.375 > 0.359) even though
in-distribution val_top1 fell (0.315 < 0.342) — the distributional-value auxiliary
task regularized the shared trunk and improved generalization to the (out-of-distribution)
puzzle set. The value_loss_weight=1.0 throttled in-distribution policy (the 64-bucket
CE is larger than WDL CE); a lower weight (~0.3–0.5) may recover val_top1 while keeping
the generalization benefit.

**Implications for the program:**
1. Keep HL-Gauss (raw gain + calibrated value) and carry it forward in the stack.
2. **Re-judge MCTS via arena games, not tactical puzzles** — `src/eval/arena.py` +
   Stockfish opponents — where positional value compounds over a full game. The puzzle
   gauge likely understates MCTS's game value.
3. Optional: value_loss_weight sweep (0.3/0.5) to recover in-distribution policy.
4. Raw policy (0.375) remains the strongest *puzzle* engine at this scale; the GPU
   scale-up (bigger model, deeper effective search) is where MCTS may finally pay off.

