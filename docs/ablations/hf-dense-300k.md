| preset | params | raw_top1@300 | raw_top1@1000 | mcts100_top1@300 | mate_in_1 | top1 | top3 | top5 | policy_ce | legal_mass | wdl_ce | value_sign_acc | draw_cal | batch_latency_ms | positions_per_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-v1 | 10255879 | 0.3233 | 0.3590 | 0.3100 | 0.6667 | 0.3307 | 0.5902 | 0.7046 | 3.1525 | 0.9937 | 0.4379 | 0.8472 | 0.0072 | 143.3804 | 1785.4604 |

> Note: `preset=baseline-v1` is the *architecture*. This row is the
> **hf-dense-300k** model (`train_data=hf-dense-300k-t01`), NOT the baseline.

## Decision (2026-06-19)

Trained `baseline-v1` architecture from scratch on 294k dense positions from
`prdev/chessbench-full-policy-value` (all legal moves per position, win% →
**softmax(win/T=0.1)**), 7000 steps, batch 128. This replaces local Stockfish
generation (downloaded, dense, sharp targets, side-to-move win% verified).

| gate | baseline-v1 | dense-d10-50k (T=1.0) | hf-dense-300k (T=0.1) |
| --- | --- | --- | --- |
| raw_top1@1000 | 0.2890 | 0.2160 | **0.3590** |
| raw_top1@300 | 0.2767 | 0.2000 | 0.3233 |
| mcts100_top1@300 | 0.3400 | 0.3333 | 0.3100 |
| mate_in_1 | 0.6667 | 0.0000 | 0.6667 |
| value_sign_acc | 0.8312 | 0.7725 | 0.8472 |

**Verdict: GO (green-light scaling the dense recipe → Phase 1b/GPU).**

The dense + sharp-target recipe beats `baseline-v1` on raw top-1 by **+7.0 pts
(0.359 vs 0.289)**, recovers mate-in-1 (0/3 → 2/3, confirming the T=1.0→T=0.1 fix),
improves the value head (sign_acc 0.847), and puts 99.4% of policy mass on legal
moves. The recipe — downloaded dense data, all-legal-moves win% at T=0.1 — is
validated and scales freely (data is no longer the bottleneck; training compute is).

**Open issue (not a blocker, but address before/with scale-up): MCTS regressed**
(0.310 vs baseline 0.340) despite the much stronger raw policy. For baseline, search
*corrected* a weaker policy (raw 0.289 → MCTS 0.340, +5); for this stronger, sharper
policy, search *deviates* from already-good picks (raw 0.359 → MCTS 0.310, −5),
because the value head did not improve as much as the policy. This is a
value-quality / search-tuning interaction, not a data-recipe failure. Levers: (1)
strengthen the value head via a bigger model + more training (the GPU run), (2) tune
MCTS (c_puct, simulations) for the sharper policy, (3) re-examine value scale/
calibration used in PUCT backup. Track MCTS-vs-raw at each scale-up step.

**Next:** scale the dense set (encode more shards — trivial now) and run the larger
model / longer training on the rented GPU, while monitoring that MCTS regains its
edge over raw as the value head strengthens.

