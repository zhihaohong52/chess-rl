| preset | params | raw_top1@300 | raw_top1@1000 | mcts100_top1@300 | mate_in_1 | top1 | top3 | top5 | policy_ce | legal_mass | wdl_ce | value_sign_acc | draw_cal | batch_latency_ms | positions_per_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-v1 | 10255879 | 0.2000 | 0.2160 | 0.3333 | 0.0000 | 0.1703 | 0.3677 | 0.5051 | 3.4314 | 0.9577 | 0.5054 | 0.7725 | 0.0734 | 141.8937 | 1804.1675 |

> Note: the `preset` column reads `baseline-v1` because that is the *architecture*
> preset recorded in the sidecar. This row is the **dense-d10-50k** model
> (trained on `data/shards_dense_d10_mpv16`, `train_data=stockfish-d10-mpv16-50k`),
> NOT the baseline checkpoint.

## Decision (2026-06-19)

Trained `baseline-v1` architecture from scratch on the 50k Stockfish dense set
(depth 10, multipv 16, **policy temperature T=1.0**), 3000 steps, batch 128.
Compared to `baseline-v1` (trained on the 62k ChessBench test bag) on the clean
puzzle gates:

| gate | baseline-v1 | dense-d10-50k |
| --- | --- | --- |
| raw_top1@1000 | 0.2890 | 0.2160 |
| raw_top1@300 | 0.2767 | 0.2000 |
| mcts100_top1@300 | 0.3400 | 0.3333 |
| mate_in_1 | 0.6667 | 0.0000 |
| value_sign_acc | 0.8312 | 0.7725 |

**Verdict: NO-GO (do not scale this recipe / do not spend the GPU budget yet).**

The dense-d10-50k model is ~7 pts worse on raw top-1 and solves 0/3 mate-in-1
(baseline 2/3), while MCTS is ~tied. Root cause: the dense **policy targets are
too flat**. Generation used `softmax(win/T)` with **T=1.0** over Stockfish
win-probabilities (range ~0.3–0.7). A forced mate (win≈1.0) then receives only
~10% target probability over ~16 candidate moves; at **T=0.1** (the ChessBench
convention `baseline-v1` inherited) it would receive ~91%. With near-uniform
targets the model gets a weak move-ranking signal and never learns to pick
decisive best moves — exactly matching the 0/3 mate-in-1 and depressed raw top-1.
The value head also trained fine (sign_acc 0.77), consistent with the problem
being target *sharpness*, not data quality.

**Next step (before any scaling):** regenerate a dense set with sharper policy
targets — add/honor a generation `--temperature ≈ 0.1` in
`scripts/gen_dense_stockfish.py` (it currently defaults to
`Config.distill_policy_temperature = 1.0`) — retrain on the same ~50k and
re-gauge. Only if a sharpened-target model matches/beats `baseline-v1` do we
proceed to Phase 1b (generator hardening + large-scale generation + GPU run).
Secondary levers if temperature alone is insufficient: deeper search and more
positions.

