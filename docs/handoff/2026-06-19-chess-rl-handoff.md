# Chess-RL Handoff — 2026-06-19

Branch `feat/transformer-rebuild` (HEAD `bcd98aa`, all pushed). This doc points to
existing artifacts rather than duplicating them — read those for detail.

## TL;DR

A transformer chess engine trained by supervised distillation. Over this session
it went from a weak baseline to an engine that **beats Stockfish skill-1 undefeated
and skill-3** with MCTS. Approach is fully validated; the next lever is **scale on a
rented GPU**. Nothing is half-finished.

## What exists (read these, don't re-derive)

- **Auto-memory (READ FIRST):** `/Users/james/.claude/projects/-Users-james-Documents-GitHub-chess-rl/memory/`
  — `phase-roadmap.md` (full program state + exact `phase2-best` config + next steps),
  `gauge-actionvalue-vs-dense.md`, `chess-rl-env.md` (env gotchas).
- **Specs:** `docs/superpowers/specs/2026-06-19-chess-rl-phase{0,1,2}-*.md`
- **Plans:** `docs/superpowers/plans/2026-06-19-chess-rl-phase{0,1,2}-*.md`
- **Ablation results / decisions:** `docs/ablations/` — `baseline-v1.md`,
  `hf-dense-300k.md`, `p2-value-300k.md`, `arena-p2-value.md`, `arena-full-stack.md`
  (each ends with a Decision section). The arena docs are the important ones.
- **Codex review (informed Phase 2):** `docs/codex_comments/model_architecture_findings.md`
- Git log: `git log --oneline` on the branch (Phase 0/1/2 commits are descriptive).

## What was done

- **Phase 0 — eval/ablation harness.** Preset registry (`src/model/presets.py`),
  checkpoint JSON sidecars + evaluator routing (`src/eval/routing.py`), gate suite
  (`src/eval/gates.py`, `metrics_core.py`, `fixtures.py`), runner `scripts/ablate.py`
  → `docs/ablations/*`. `baseline-v1` locked at **10,255,879 params** (a test guards it).
- **Phase 1 — data pivot.** Action-value (pointwise) underperformed; switched to
  **dense policy distillation**. Discovered a downloadable dense dataset
  `prdev/chessbench-full-policy-value` (HF, MIT; `{fen, moves:{uci:{win_prob,mate}}}`,
  win% side-to-move). Reader `iter_hf_dense` + `scripts/preencode.py --source hf_dense`.
  Result `hf-dense-300k`: raw top-1 **0.359** (baseline 0.289).
- **Phase 2 — recipe/arch (all config-gated, baseline preserved).** HL-Gauss
  distributional value head (`src/model/value_dist.py`, `DistributionalValueHead`),
  SwiGLU FFN, dropout, EMA (`scripts/distill.py --ema-decay`). Presets `p2-value`,
  `p2-value-swiglu`, `p2-value-swiglu-drop`.
- **Key finding (arena):** the MCTS "regression" seen on puzzles was a **metric
  artifact**. `scripts/arena_eval.py` (raw vs MCTS vs Stockfish in games) showed MCTS
  is far stronger in play. **Full stack** (`p2-value-swiglu-drop` + EMA) = strongest
  engine: **MCTS 0.850 vs SF-1 (14W/6D/0L), 0.700 vs SF-3**.

## Current state / where things live (local, gitignored)

- `phase2-best` = preset `p2-value-swiglu-drop` + `--ema-decay 0.999`; gauge `best_ema.pt`.
- Checkpoints (local, `*.pt` gitignored): `checkpoints/baseline-v1/`,
  `checkpoints/distill/` (the original policy model), `checkpoints/hf_dense_300k/`,
  `checkpoints/p2_value_300k/`, `checkpoints/p2_full_300k/` (full stack).
- Data (local, `/data/` gitignored): `data/raw_hf/train-00000.msgpack.zst` (one ~72 MB
  shard ≈ 3.5M dense positions), `data/shards_hf_dense_300k/` (294k train/6k val,
  T=0.1), `data/test/action_value_data.bag` (62k dense), `data/puzzles.csv` (10k).
- Full test suite green: **135 passed, 1 skipped, 2 deselected**.

## Environment gotchas (IMPORTANT)

- Use the off-iCloud venv: `/Users/james/.venvs/chess-rl/bin/python` and `.../bin/pytest`
  (the in-repo `.venv` gets iCloud-evicted and hangs).
- **MPS batch ≤ 128** on this 8 GB M1 (batch 512 silently thrashes for hours).
- `fit()`'s policy-path print isn't flushed → run monitored training with `python -u`.
  (Minor TODO: add `flush=True` so `-u` isn't needed.)
- Prefix shell commands with `rtk` per project CLAUDE.md. `rtk` rewrites `ls` to tree
  format — don't pipe `ls -l` into `awk`; use python `os.path.getsize` to check sizes.
- Stockfish 18 at `/opt/homebrew/bin/stockfish`. Deps added: `zstandard`, `msgpack`.

## Next steps (recommended order)

1. **GPU scale-up (Phase 3)** — the path to real strength, well-justified now.
   ThunderCompute (per-minute, A6000 48 GB ≈ $0.35/hr, ~28 h for the $10 budget).
   Plan: generate/encode more dense shards locally first (free CPU — `iter_hf_dense`,
   one shard holds 3.5M positions; download more via the URL pattern in
   `scripts/download_chessbench.py`/the HF resolve URL), then on the GPU train a
   **bigger model** (raise `d_model`/`n_layers` via a new preset) with `phase2-best`
   recipe + a **wider arena ladder** (skills up to ~10-15, ≥40 games) for firm Elo.
   Honest ceiling: ~1800-2300 Elo is realistic at this budget — **not** full Stockfish
   (~3000+; DeepMind's 270M model used 15B positions for ~2900).
2. **Optional quick tune:** `value_loss_weight` 0.3-0.5 (the 64-bucket HL-Gauss CE is
   larger than WDL CE and throttles policy at weight 1.0) — may recover raw top-1.
3. **Arena is the strength metric, not puzzles.** Keep puzzle raw-top-1 as a fast
   *policy* proxy only.

## Suggested skills for the next session

- `superpowers:brainstorming` then `superpowers:writing-plans` to design/plan the
  Phase 3 GPU scale-up before building (the established workflow in this repo).
- `superpowers:subagent-driven-development` (all subagents as **sonnet**, per the
  user's standing preference) to execute the plan with two-stage review.
- `superpowers:executing-plans` for inline execution of short/run-heavy plans
  (training runs are monitored with the Monitor tool; size step counts to the
  ~1 step/s MPS rate, or far faster on GPU).
- `superpowers:finishing-a-development-branch` when ready to integrate the long-lived
  `feat/transformer-rebuild` branch.

## Watch-outs

- Keep `baseline-v1` bit-identical (the param-count lock test); all new arch is
  config-gated behind presets with baseline-preserving defaults.
- Commit only metrics/code; checkpoints (`*.pt`), shards (`*.npz`), and raw data are
  gitignored — never stage them. `graphify-out/` is also untracked; don't commit it.
- The user prefers structured `AskUserQuestion` for genuine decisions and periodic
  `git push`.
