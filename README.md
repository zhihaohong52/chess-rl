# chess-rl

A chess engine built on a 77M-parameter transformer trained by dense policy distillation from Stockfish-16, combined with MCTS at inference and a hybrid move selector (opening book → Syzygy tablebases → MCTS).

**Current strength: ~2000+ Elo** (undefeated vs Stockfish depth-4 ladder).

---

## Architecture

- **Model:** `p3-80m` preset — d_model 512, 16 layers, 16 heads, SwiGLU FFN, HL-Gauss 64-bucket distributional value head, Smolgen attention bias, 1924-logit policy head, moves-left aux head, 18 global state features
- **Training:** Dense policy distillation from Stockfish-16 (ChessBench, 100M positions, 1 epoch, ~51% top-1 move agreement)
- **Inference:** Hybrid mover — Polyglot opening book → Syzygy 3–5 man WDL+DTZ → MCTS (200 sims recommended); optional online 7-man TB via Lichess API

## Quick start

```bash
pip install -r requirements.txt

# Play via UCI (works with Arena, CuteChess, etc.)
./chessrl.sh                          # Linux/Mac
chessrl.bat                           # Windows

# Or directly
python uci.py --model checkpoints/model_final --simulations 200
```

## Training

```bash
# Distill from ChessBench pre-encoded shards (HF: james77mill/chessbench-encoded-npz)
python scripts/distill.py \
    --preset p3-80m \
    --train data/train_shards \
    --steps 100000 \
    --batch 1024 \
    --value-loss-weight 0.3 \
    --stream
```

## Key findings

| Experiment | Result |
|---|---|
| Data lever: 10M model, 100M vs 300k positions | +223 Elo |
| Scale lever: 77M vs 10M, same 100M positions | +101 Elo |
| Search: 50 → 200 sims | +58 Elo |
| Search: 200 → 400 sims | diminishing (~+5) |
| Value loss weight 1.0 → 0.3 | top-1 +3.3%, engine ±noise |
| Opening book + Syzygy 3–5 man bolt-on | system-level gain |

**Data is the primary lever** at this compute budget. The 77M model is under-trained for its size — DeepMind's comparable recipe (270M, full 15B ChessBench corpus) reaches 2895 Lichess blitz without any search.

## Evaluation

```bash
# Elo ladder vs Stockfish skill levels
python scripts/arena_eval.py --simulations 200

# Endgame arena (Syzygy positions)
python scripts/endgame_arena.py
```

## What's next

The pre-encoded HF dataset covers 780 of 1024 source shards (~76%). The clearest path to a major Elo jump is encoding the remaining 244 shards and training on the full corpus, following the data-bound finding above. Self-play / expert iteration is the only path above the Stockfish-16 teacher ceiling.

---

*Branch `feat/transformer-rebuild` — merged to main after Phase 3 full-epoch run.*
