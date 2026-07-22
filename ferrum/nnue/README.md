# ferrum NNUE — gen-0 reproduction record (M2 Task 1 spike)

**Status:** discovery spike only. $0 spent, no cloud instance created, no
training run. This file is the verified source of truth that later M2 tasks
(2–8) implement against — see `docs/superpowers/plans/2026-07-21-ferrum-m2.md`.
Net binaries and scratch data are gitignored (see `.gitignore` in this
directory); this README is the reproducible record instead, mirroring
`ferrum/bench/anchors/README.md`.

Every fact below is either (a) verified by reading source in this repo or in
a freshly cloned upstream repo at a pinned commit, or (b) explicitly marked
**UNVERIFIED / assumption** where it could not be checked without spending
money or downloading multi-GB assets. Do not treat unmarked prose as
guesswork — it was checked.

---

## Source data

### What's on HF, and where it actually came from

HF dataset `james77mill/chessbench-encoded-npz` (public, CC-BY-4.0): 780
`.npz` shards, flat layout, ~37–39 GB (585 `train_*.npz` × 250k positions +
195 `val_*.npz`; ~100.27M train / ~0.20M val positions total). Per the
project memory note `[[encoded-dataset-hf]]`, these were produced by this
repo's own encoder — **rather than download and inspect a 3 GB shard, this
spike read the exact producing code**, which is authoritative and free:

- `scripts/preencode.py --source hf_dense --input 'data/raw_hf/train-*.msgpack.zst' --out-dir <out> --temperature 0.1`
  — the driver. `hf_dense` routes through `src/data/chessbench.py::iter_hf_dense`.
- `data/raw_hf/train-00000.msgpack.zst` (68.8 MB) sits locally in this repo —
  one raw shard of the public `prdev/chessbench-full-policy-value` dataset:
  zstd+msgpack records `{"fen": str, "moves": {uci: {"win_prob": float in [0,1]}}}`,
  Stockfish-16 win% per legal move (per the design spec and project memory;
  this specific provenance claim about the upstream DeepMind ChessBench paper
  was **not** independently re-verified against external sources in this
  spike — it is carried from `docs/superpowers/specs/2026-07-14-ferrum-nnue-engine-design.md`
  §5 and the `[[phase-roadmap]]`/`[[encoded-dataset-hf]]` memory notes).
- `src/data/chessbench.py::_build_position()` turns one FEN's moves into a
  `LabeledPosition`: `policy = softmax(win/T)` over the given legal moves,
  `wdl = winprob_to_wdl(best move's win%)`, `moves_left = 40.0` (a constant —
  action-value data has no game trajectory, see `_DEFAULT_MOVES_LEFT`).
- `src/data/preencode.py::encode_example()` / `write_shard()` emits the npz
  arrays below. `tests/data/test_preencode.py` asserts this exact schema.

### Verified per-shard schema

| Array | Shape | Dtype | Meaning |
|---|---|---|---|
| `square_tokens` | `[N, 64]` | `int8` | Canonical (side-to-move) board. `0`=empty, `1..6`=mover's own `{P,N,B,R,Q,K}`, `7..12`=opponent's `{P,N,B,R,Q,K}`. If the real side to move is Black, **the whole board is vertically mirrored** (`square ^ 56`) so the array always reads as if the mover were White. Square index = `rank*8+file`; rank 0 = the mover's own back rank post-mirror. Source: `src/game/token_encoder.py::encode_position_fast` (canonical framing defined in `src/game/orientation.py::canonical_board`). |
| `state_features` | `[N, 18]` | `float32` | `[0..3]` = own/opp castling rights (K,Q,k,q in canonical frame), `[4..11]` = en-passant file one-hot (canonical file), `[12]` = ep-present flag, `[13]` = halfmove clock/100 (clamped), `[14]` = repetition count/3 (clamped), `[15]` = fullmove/200 (clamped), `[16]`/`[17]` = constant `1.0`/`0.0` (reserved, unused). **Not needed for gen-0 NNUE** — the design commits to a plain 768-input net with no auxiliary state features. |
| `wdl` | `[N, 3]` | `float32` | **Not raw Stockfish WDL.** It is a *derived* shaping of the scalar win probability `wp` via `src/data/targets.py::winprob_to_wdl(wp, draw_scale=2.0)` — every call site in this repo uses the default `draw_scale=2.0` (never overridden). Side-to-move POV (consistent with `square_tokens`' canonical frame: "mover" = the array's "White"). Sums to exactly 1.0 (verified by `tests/data/test_preencode.py`). |
| `moves_left` | `[N]` | `float32` | Constant `40.0` for every position produced via `hf_dense`/`chessbench` sources. Not a meaningful target; ignore. |
| `legal_indices`, `legal_probs`, `counts` | ragged (CSR) | `int32`/`float32`/`int32` | Policy-head target for the transformer (`counts[i]` legal moves per position, `legal_indices`/`legal_probs` are the concatenated per-position slices). Irrelevant to a value-only NNUE; drop. |

### The critical determination: is this npz usable for gen-0 value-NNUE?

**Yes — and it turns out to be well-suited, not just barely usable**, given two
exact, low-risk derivations (not guesses):

**1. Board reconstruction needs no extra data.** `square_tokens` is *already*
exactly the standard NNUE side-to-move-perspective feature source: own
pieces as ids 1–6, enemy pieces as 7–12, square pre-mirrored for
Black-to-move. That is not "close to" what a perspective net's STM
accumulator wants — it is a literal match, id-for-id, mirror-for-mirror
(verified against `bulletformat`'s own `ChessBoard`/`Chess768`, see "Trainer"
below: same own/enemy nibble convention, same `sq ^ 56` mirror). The
**second** perspective (needed for the `768->Nx2->1` architecture's other
accumulator) does not require the original absolute side-to-move at all — it
is derived purely from `square_tokens` by swapping the own/enemy id ranges
and re-mirroring the square, which is mechanically identical to how
`bullet`'s `Chess768::map_features` derives its own `ntm` feature index from
the same STM-relative record (verified by reading
`crates/bullet_lib/src/game/inputs/chess768.rs` at the pinned commit, below).

**2. The value target is exactly recoverable, not approximate.** `wdl` looks
lossy (a synthetic 3-tuple, not a raw eval) but is not: because every
producing call site fixes `draw_scale=2.0`, the shaping function is
invertible in closed form. Derivation: `winprob_to_wdl` sets
`decisiveness=(2*wp-1)^2`, `d=1-decisiveness`, `w=(1-d)*wp`, `l=(1-d)*(1-wp)`
(and `w+d+l=1` identically, so no renormalization is lost). Solving for `wp`
from `d` alone (robust — does **not** divide by zero at `wp=0.5`, unlike
`wp=w/(w+l)` which is `0/0` exactly at `wp=0.5`):

```python
import math
w, d, l = wdl
sign = 1.0 if w >= l else -1.0
wp = 0.5 + sign * math.sqrt(max(0.0, 1.0 - d)) / 2.0
wp = min(max(wp, 1e-6), 1.0 - 1e-6)        # guard forced-mate extremes
cp = 400.0 * math.log10(wp / (1.0 - wp))    # inverse of src/data/targets.py::cp_to_winprob
```

This recovers the original side-to-move win probability exactly (given the
fixed `draw_scale=2.0`), then the standard logistic inversion gives a
centipawn score equivalent to the design spec's "score = win% inverted
through the sigmoid to centipawns."

**Recommendation:** Task 2's converter reads only `square_tokens` + `wdl`
from each shard (ignoring `state_features`, `moves_left`,
`legal_indices`/`legal_probs`/`counts`), applies the derivation above per
position, and does **not** need to touch the raw `prdev/chessbench-full-policy-value`
msgpack shards directly — the npz is a complete, sufficient, and convenient
source for gen-0. Do the Task 2 Step-2 "local smoke on one shard" (already in
the M2 plan) as the empirical confirmation once an actual HF shard is pulled;
this spike verified the schema from source, not from bytes on disk.

**Filters carried over from the design spec** (§5, "Data pipeline"): drop
in-check positions and positions whose best move is a capture/gives check
(quiet-position training), clamp extreme `cp`, dedupe by Zobrist. These are
Task 2 implementation concerns, not schema facts, so they are only noted
here for continuity.

---

## Trainer (bullet)

**Repo:** <https://github.com/jw1912/bullet>
**Pinned commit:** `cebc78a093d92cbc87e56cfef049184c225270b0` (2026-06-22,
"Speedup `SFBinpackLoader` (#533)"). No tags exist upstream at this point in
history, so the plan pins a raw commit SHA rather than a version tag.
**Companion crate:** `bulletformat = "1.8.0"` (from bullet's own `Cargo.lock`),
independently verified by cloning <https://github.com/jw1912/bulletformat> at
its `v1.8.0` tag, commit `cb6a3602d70b0b6c3abc88fb876810a7bdf0b428`.

Both were cloned **outside** the chess-rl working tree for inspection (not
committed): `bullet` and `bulletformat` under a scratch directory. Verified
locally: `cargo check --release --example simple --features metal` against
this exact `Cargo.lock` **compiles clean** on this Mac (Metal backend, no
CUDA needed for a static check) — so the API surface below is confirmed
current against the pinned commit, not transcribed from possibly-stale docs.

### Training-data binary format ("ChessBoard" / bulletformat)

`docs/3-data.md` calls this "ChessBoard aka bulletformat" — loaded via
`loader::DirectSequentialDataLoader`. Fixed 32-byte little-endian struct
(`crates/chess.rs` in `bulletformat`; size asserted at compile time:
`assert!(size_of::<ChessBoard>() == 32)`):

| Offset | Size | Field | Meaning |
|---|---|---|---|
| 0 | 8 | `occ: u64` | Occupancy bitboard, **side-to-move relative** (vertically mirrored + colors swapped if the real side to move is Black — record is always "as if White to move"). |
| 8 | 16 | `pcs: [u8; 16]` | 4-bit piece code per occupied square, in ascending bit order of `occ` (2 codes/byte). Nibble = `(is_enemy << 3) | piece_type`; `piece_type` 0=P,1=N,2=B,3=R,4=Q,5=K; `is_enemy`=0 for the mover's own pieces, 1 for the opponent's. |
| 24 | 2 | `score: i16` | Side-to-move-relative centipawns. |
| 26 | 1 | `result: u8` | Side-to-move-relative game result: 0=loss, 1=draw, 2=win. |
| 27 | 1 | `ksq: u8` | Mover's king square (post-mirror). |
| 28 | 1 | `opp_ksq: u8` | Opponent's king square, expressed in the *opponent's own* perspective (double-mirrored — see `ChessBoard::from_raw`). Used only by king-bucketed input sets (`ChessBuckets`/HalfKA-style); **unused by plain `Chess768`**, so irrelevant to gen-0. |
| 29 | 3 | `extra: [u8; 3]` | Reserved, zero. |

This is (as noted above) essentially the same convention chess-rl's
`square_tokens` already uses — own pieces low nibble range, opponent pieces
high, same `sq ^ 56` mirror rule — so Task 2's mapping from `square_tokens`
to `occ`/`pcs` is a direct re-encoding, not a re-derivation.

Two viable ways for Task 2 to actually produce the `.bin` (documented here so
Task 2 doesn't have to re-derive them):

- **A (recommended, lower-risk):** emit a plain-text intermediate,
  one line per position, `<FEN>|<score>|<result>`. Verified via
  `crates/utils/src/convert.rs::convert_text` (calls
  `bulletformat::ChessBoard::from_str` then `BulletFormat::write_to_bin`)
  that **only the FEN's piece-placement and side-to-move fields are read** —
  castling/en-passant/clocks are ignored entirely. Since `square_tokens` is
  already the canonical (mover-as-White) frame, a synthetic FEN
  `<placement from square_tokens> w - - 0 1` with the derived `cp`/result as
  the "white-relative" score/result (no further mirroring needed, because the
  frame is already canonical) round-trips correctly. Pack with:
  ```bash
  cargo run --release --package bullet-utils -- convert --from text --input gen0.txt --output gen0.data --threads 8
  ```
- **B (faster, more code to trust):** pack the 32-byte struct directly in
  Python via `struct.pack` per the table above — recommended only with a
  cross-check against (A)'s output on a sample during Task 2's local smoke
  test, given the 100M-row scale makes a silent packing bug expensive to
  discover late.

### Rust config API — concrete example for our architecture

`examples/simple.rs` at the pinned commit (this exact file, only the
constants changed for our target architecture — the API itself is
untouched, confirmed by the `cargo check` above):

```rust
use bullet_lib::{
    game::inputs::Chess768,
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader},
};

const HIDDEN_SIZE: usize = 512;   // design spec §5: "512 hidden per perspective"
const SCALE: i32 = 400;
const QA: i16 = 255;              // community-standard (used verbatim by simple.rs)
const QB: i16 = 64;

fn main() {
    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(Chess768)
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs| {
            let l0 = builder.new_affine("l0", 768, HIDDEN_SIZE);
            let l1 = builder.new_affine("l1", 2 * HIDDEN_SIZE, 1);
            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            l1.forward(stm_hidden.concat(ntm_hidden))
        });

    let schedule = TrainingSchedule {
        net_id: "gen0".to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 40,           // starting point from simple.rs; tune vs our ~100M positions
        },
        // score-only per design spec (see note below on the sign of this value)
        wdl_scheduler: wdl::ConstantWDL { value: 0.0 },
        lr_scheduler: lr::StepLR { start: 0.001, gamma: 0.1, step: 18 },
        save_rate: 10,
    };

    let settings = LocalSettings { threads: 8, test_set: None, output_directory: "checkpoints", batch_queue_size: 64 };
    let data_loader = loader::DirectSequentialDataLoader::new(&["data/gen0.data"]);
    trainer.run(&schedule, &settings, &data_loader);
}
```

**Important sign note (easy to invert by accident):** bullet's own comment
defines `target = wdl * game_result + (1 - wdl) * sigmoid(score / scale)`,
i.e. bullet's `wdl_scheduler` value is the **weight on the game-result
term**. The design spec's "WDL blend λ=1.0 (score-only)" means full weight on
the *score* term — i.e. `1 - wdl_scheduler_value = 1.0`, so
**`wdl_scheduler` must be `ConstantWDL { value: 0.0 }`, not `1.0`**. Recorded
explicitly here because the naming collision (design's λ vs bullet's `wdl`)
is a real inversion trap for Task 3.

Batch size / superbatch count / LR schedule above are copied from the proven
`simple.rs` starting point, **not independently tuned for our ~100M-position
dataset** — Task 3 should treat them as a first-run default, not a verified
optimum.

### Quantisation

`QA=255` (feature-transformer weights/bias, int16), `QB=64` (output weights,
int16), output bias quantised at `QA*QB`. `.round().quantise::<i16>(Q)` =
`round(float * Q)` (contrast: default `.quantise` is truncating, so
`.round()` is deliberately added, per `docs/4-saved-networks.md`).

### Export / checkpoint layout

Per `docs/4-saved-networks.md`, verified against `examples/simple.rs`'s save
format above: a checkpoint directory `<output_directory>/<net_id>-<n>/`
contains `raw.bin` (f32 params, not needed by ferrum), `quantised.bin` (the
actual inference net — **not written if int16 quantisation overflows**,
training otherwise unaffected), and `optimiser_state/`. `quantised.bin` is
the concatenation, in `.save_format` order, of little-endian primitives,
column-major matrices (transposed first if `.transpose()` was specified),
**then the whole file zero-padded up to the next multiple of 64 bytes**.

---

## Net file format

`bullet`'s raw `quantised.bin` has **no self-describing header** — the
consuming engine is expected to hardcode the architecture constants. Since
Task 4's `Nnue::load` needs a "magic/version header, feature-transformer
weights, output weights, biases, scale constants" (per the M2 plan), this
spike defines a thin **ferrum-specific wrapper** that a small export step
(Task 3) prepends to bullet's raw payload — this header is *ferrum's own
invention*, not something bullet provides:

| Offset | Size | Field | Value (gen-0) |
|---|---|---|---|
| 0 | 4 | magic | ASCII `"FeNN"` |
| 4 | 1 | format version | `1` |
| 5 | 1 | reserved | `0` |
| 6 | 2 | `hidden_size: u16` (LE) | `512` |
| 8 | 2 | `qa: i16` (LE) | `255` |
| 10 | 2 | `qb: i16` (LE) | `64` |
| 12 | 2 | `scale: i16` (LE) | `400` |
| 14 | 2 | reserved | `0` |
| 16 | — | payload | bullet's raw quantised arrays, **exact byte count only** (bullet's own 64-byte alignment pad is stripped, not copied — the header's `hidden_size` makes the exact count derivable, so no pad is needed) |

Payload layout (matches `examples/simple.rs`'s commented `Network`/`Accumulator`
structs exactly, generalized to `HIDDEN_SIZE=512`):

| Field | Shape | Bytes (H=512) |
|---|---|---|
| `feature_weights` | `[i16; 768][HIDDEN_SIZE]`, column-major (per-feature contiguous column) | `768 * 512 * 2 = 786,432` |
| `feature_bias` | `[i16; HIDDEN_SIZE]` | `512 * 2 = 1,024` |
| `output_weights` | `[i16; 2*HIDDEN_SIZE]` | `1,024 * 2 = 2,048` |
| `output_bias` | `i16` | `2` |

Payload total = `789,506` bytes; ferrum `gen0.bin` total = `16 + 789,506 =
789,522` bytes (H=512). (Bullet's own `quantised.bin` before stripping the
pad would be `789,568` bytes — `789,506` rounded up to the next multiple of
64 — this is bullet's convention, not ferrum's; the wrapper drops it.)

### Feature indexing convention (Task 4 must match this exactly)

Matches `bulletformat`/`Chess768`'s own convention exactly (verified against
`crates/bullet_lib/src/game/inputs/chess768.rs::map_features` at the pinned
commit), expressed in ferrum's terms (`ferrum/src/types.rs`: `PAWN=0 .. KING=5`,
`pc(color, pt) = color*6+pt`; `ferrum/src/board.rs`: `Board.bb: [Bb; 12]`):

```
for each piece (pt, color) on square sq (ferrum absolute: a1=0 .. h8=63):
    is_enemy  = color != perspective
    view_sq   = if perspective == Black { sq ^ 56 } else { sq }
    feature   = (is_enemy as usize) * 384 + 64 * pt + view_sq as usize   // 0..768
```

Build **two** accumulators per position: `stm_acc` with `perspective =
board.side`, `ntm_acc` with `perspective = board.side.flip()`. Forward pass
(bit-exact port of `simple.rs`'s `Network::evaluate`):

```
screlu(x: i16) -> i32 { let y = i32::from(x).clamp(0, QA); y * y }   // QA=255

output = sum(screlu(stm_acc[i]) * output_weights[i]      for i in 0..H)
       + sum(screlu(ntm_acc[i]) * output_weights[H + i]  for i in 0..H)
output /= QA
output += output_bias
output *= SCALE
output /= QA * QB
```

This is **already side-to-move-relative** by construction (`us` = STM
accumulator is added first in bullet's own `evaluate(us, them)`, matching
`Eval::eval`'s contract in `ferrum/src/eval.rs` — positive = good for
`board.side`) — no extra sign flip needed in `Nnue::eval`, unlike `Hce` which
computes a White-relative score and flips at the end.

---

## Cloud training (ThunderCompute)

**Not launched — read-only queries only** (`get_availability`, `get_pricing`,
`get_specs`, `list_templates`), captured 2026-07-21:

| Spec key | GPU | VRAM | Availability | Price |
|---|---|---|---|---|
| `a6000_x1` | RTX A6000 ×1 | 48 GB | available | **$0.35/h** |

`get_specs` for `a6000_x1`: `vcpuOptions: [6, 8]`, `ramPerVCPUGiB: 8`,
`storageGB: {min: 100, max: 500}`. Recommended: `vcpus=8` (64 GiB RAM),
`disk_size_gb=100` (the minimum — ample for the ~3 GB bullet-format dataset
plus checkpoints). Additional pricing line items from `get_pricing`:
`additional_vcpus: $0.04/h` each (exact included-baseline vCPU count not
surfaced by the read-only tools — treat as a minor addend, well under
$0.10/h at this vCPU range), `disk_gb: $0.0003/GB-h` (→ ~$0.03/h for 100 GB).

Recommended template: `cuda12-9` ("GPU Kernels (CUDA 12.9)" — CUDA 12.9
toolkit + cuDNN + NCCL pre-installed, per `list_templates`); `cuda12-8`
is the documented fallback. Verified from bullet's own `crates/gpu/build.rs`
that the `cuda` feature only needs: `CUDA_PATH` env var set to a directory
containing `include/` and `lib64/`, and dynamic libs `cuda`, `cudart`,
`nvrtc`, `cublas` linkable from there — a standard CUDA toolkit install
satisfies this, no extra packages beyond what the template provides.

Provisioning recipe (**documented, not run** — no instance was created):

```bash
# 1. create (NOT executed in this spike):
#    create_instance(gpu_type="a6000", num_gpus=1, template="cuda12-9",
#                     vcpus=8, disk_size_gb=100)

# 2. on the instance, once running:
curl https://sh.rustup.rs -sSf | sh -s -- -y
source "$HOME/.cargo/env"
nvcc --version                      # confirm toolkit version + locate install prefix
export CUDA_PATH=/usr/local/cuda     # UNVERIFIED on this template — confirm via
                                      # `dirname $(dirname $(command -v nvcc))` on first boot
git clone https://github.com/jw1912/bullet.git
cd bullet && git checkout --detach cebc78a093d92cbc87e56cfef049184c225270b0
cargo build --release --example simple --features cuda   # sanity build before wiring our config

# 3. train (Task 3):
#    scp the Task 2 bulletformat data up, drop in our config (this README's
#    example), `cargo run --release --example gen0 --features cuda`

# 4. retrieve + delete:
#    scp checkpoints/gen0-40/quantised.bin back, prepend the ferrum header
#    (see "Net file format"), then DELETE THE INSTANCE — billing runs only
#    while the instance exists, no idle/auto-stop.
```

**Estimated cost:** design spec (§5/§8) estimates 1–4 GPU-hours per net at
this scale, i.e. `1–4 × $0.35 ≈ $0.35–$1.40` GPU time, **$1–2/run all-in**
with disk + setup overhead — consistent with the existing M2 budget line.
This is an **estimate carried over from the design spec, not a measurement**
— no training has been run in this spike, so treat wall-clock as unverified
until Task 3's first real run.

**Ops rules (unchanged from the design spec, §8):** delete the instance the
moment a session ends (snapshot only if a re-train is imminent); record
actual spend and wall-clock in `ferrum/docs/ledger.md` at Task 3/8; hard
stop + alert at $10/$20 cumulative spend; nothing beyond the ~$20–25
approved pot without asking.

---

## Gate check (per the M2 plan)

- [x] Concrete, verified training-data format (32-byte `ChessBoard`, byte
      offsets above; two viable production paths).
- [x] Concrete `bullet` config (compiled against the pinned commit locally).
- [x] Concrete net-file layout for `Nnue::load` (ferrum header + bullet
      payload, byte-exact, with the feature-indexing formula Task 4 needs).
- [x] Source-data determination made explicitly, with the derivation shown
      (not asserted): the encoded npz is usable, in fact well-suited, for
      gen-0 value-NNUE via the `square_tokens`/`wdl` derivations above.
- No material delta from the design spec's assumptions was found: the
  architecture (768→512×2→1, SCReLU, `bullet`, A6000), the data source, and
  the score-only WDL blend all check out against current upstream `bullet`
  and this repo's actual encoder — the one correction is the λ/`wdl_scheduler`
  sign mapping noted above, which is a documentation nuance, not an
  architecture change.
