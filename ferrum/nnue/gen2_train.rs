// ferrum gen-2 NNUE training config (bullet, pinned cebc78a093d92cbc87e56cfef049184c225270b0).
//
// M6: self-play data, NOT a new architecture. The arch is byte-identical to
// gen-0 (`Chess768`, 768 -> 512x2 -> 1, SCReLU, QA=255/QB=64/SCALE=400), so the
// exported net is a drop-in `EVALFILE` swap for `gen0.bin` with the same FeNN v1
// header and the same 789,522-byte size — ferrum's inference code is untouched.
//
// The ONE experimental change vs gen-0 is `wdl_scheduler`: gen-0 was forced to
// `ConstantWDL { value: 0.0 }` (score-only) because ChessBench has no game
// outcomes. The self-play corpus carries REAL results, so the game-result term
// is informative for the first time. bullet's `wdl` value is the weight on the
// GAME-RESULT term (`target = wdl*result + (1-wdl)*sigmoid(score/scale)`), so
// 0.4 = 40% real outcome / 60% score. Do NOT invert this (see nnue/README.md).
//
// Two variants, selected at run time by `GEN2_VARIANT` (no recompile):
//   v1 (primary)  mix-from-scratch: 63M ChessBench (result byte bucketed from
//                 its own score, see tools/chessbench_to_bullet.py) interleaved
//                 with 14.7M self-play. Random init. Keeps gen-0's breadth.
//   v2 (fallback) fine-tune: self-play only, initialised FROM gen-0's weights.
//                 Sidesteps the mixed-WDL subtlety (spec 6.1) entirely; clean
//                 attribution of the self-play delta.
//
// v2's init comes from `nnue/gen0_to_bullet_weights.py`, which dequantises the
// deployed `gen0.bin` back to f32 in bullet's `weights.bin` wire format — the
// original gen-0 optimiser checkpoint died with its training instance, so this
// reconstruction (accurate to the i16 quantisation step) is the only path back
// to gen-0's weights. Momentum/velocity start at zero, hence the lower LR.
//
// Usage on the training box. NOTE: bullet does NOT auto-discover examples/ —
// the target must be registered in crates/bullet_lib/Cargo.toml or cargo errors
// with "no example target named `gen2`" (verified locally against `cebc78a`):
//   cp ferrum/nnue/gen2_train.rs bullet/examples/gen2.rs
//   cd bullet
//   printf '\n[[example]]\nname = "gen2"\npath = "../../examples/gen2.rs"\n' \
//     >> crates/bullet_lib/Cargo.toml
//   GEN2_VARIANT=v1 GEN2_DATA=data/mix.data \
//     cargo run --release --example gen2 --features cuda
//   GEN2_VARIANT=v2 GEN2_DATA=data/selfplay.data GEN2_INIT=data/gen0_weights.bin \
//     cargo run --release --example gen2 --features cuda
// Then wrap `checkpoints/<net_id>-<n>/quantised.bin` with nnue/build_gen2_bin.py.
use bullet_lib::{
    game::inputs::Chess768,
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::{LocalSettings, TestDataset},
    },
    value::{ValueTrainerBuilder, loader},
};

// FROZEN — must match ferrum's FeNN v1 net and nnue/build_gen2_bin.py.
const HIDDEN_SIZE: usize = 512;
const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;

/// Weight on the game-result term. The M6 experiment; 0.0 reproduces gen-0.
const DEFAULT_WDL: f32 = 0.4;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Variant {
    /// Mix-from-scratch: ChessBench + self-play, random init.
    V1Mix,
    /// Fine-tune from gen-0 weights on self-play only.
    V2FineTune,
}

impl Variant {
    fn net_id(self) -> &'static str {
        match self {
            Variant::V1Mix => "gen2_v1",
            Variant::V2FineTune => "gen2_v2",
        }
    }

    fn default_data(self) -> &'static str {
        match self {
            Variant::V1Mix => "data/mix.data",
            Variant::V2FineTune => "data/selfplay.data",
        }
    }

    /// V1 deliberately reuses gen-0's PROVEN step budget verbatim (40 x 6104
    /// batches of 16384) so the only deltas vs gen-0 are the data and the WDL
    /// blend — not the optimisation schedule.
    ///
    /// V2 is a fine-tune over a thin 14.7M corpus: ~1 epoch per superbatch
    /// (14,694,209 / 16,384 = 897), far fewer superbatches, and a much lower
    /// start LR so gen-0's weights are refined rather than overwritten (its
    /// momentum/velocity are cold, which makes a high LR extra destructive).
    ///
    /// `GEN2_SMOKE=1` shrinks the run to one tiny superbatch: used to dry-run the
    /// whole path (init load -> train -> checkpoint -> FeNN wrap -> engine load)
    /// on a laptop `--features metal` before paying for GPU time.
    fn steps(self, smoke: bool) -> TrainingSteps {
        if smoke {
            return TrainingSteps { batch_size: 16_384, batches_per_superbatch: 8, start_superbatch: 1, end_superbatch: 1 };
        }

        let (batches_per_superbatch, end_superbatch) = match self {
            Variant::V1Mix => (6104, 40),
            Variant::V2FineTune => (897, 12),
        };

        TrainingSteps { batch_size: 16_384, batches_per_superbatch, start_superbatch: 1, end_superbatch }
    }

    fn lr(self) -> lr::StepLR {
        match self {
            Variant::V1Mix => lr::StepLR { start: 0.001, gamma: 0.1, step: 18 },
            Variant::V2FineTune => lr::StepLR { start: 0.0002, gamma: 0.3, step: 5 },
        }
    }

    fn save_rate(self) -> usize {
        match self {
            Variant::V1Mix => 10,
            Variant::V2FineTune => 4,
        }
    }
}

fn env(key: &str) -> Option<String> {
    match std::env::var(key) {
        Ok(value) if !value.is_empty() => Some(value),
        _ => None,
    }
}

fn variant() -> Variant {
    match env("GEN2_VARIANT").unwrap_or_else(|| "v1".to_string()).as_str() {
        "v1" => Variant::V1Mix,
        "v2" => Variant::V2FineTune,
        other => panic!("GEN2_VARIANT must be \"v1\" or \"v2\", got {other:?}"),
    }
}

fn main() {
    let variant = variant();
    let data = env("GEN2_DATA").unwrap_or_else(|| variant.default_data().to_string());
    let val = env("GEN2_VAL");
    let init = env("GEN2_INIT");
    let threads = env("GEN2_THREADS").map_or(8, |t| t.parse().expect("GEN2_THREADS must be an integer"));
    let wdl_weight = env("GEN2_WDL").map_or(DEFAULT_WDL, |w| w.parse().expect("GEN2_WDL must be a float"));
    let smoke = env("GEN2_SMOKE").is_some();

    assert!((0.0..=1.0).contains(&wdl_weight), "GEN2_WDL must be in [0, 1], got {wdl_weight}");
    assert!(
        variant != Variant::V2FineTune || init.is_some(),
        "v2 is a fine-tune: set GEN2_INIT to the gen-0 weights.bin from nnue/gen0_to_bullet_weights.py"
    );

    println!("gen-2 variant : {variant:?} (net_id {})", variant.net_id());
    println!("gen-2 data    : {data}");
    println!("gen-2 val     : {}", val.as_deref().unwrap_or("<none>"));
    println!("gen-2 init    : {}", init.as_deref().unwrap_or("<random>"));
    println!("gen-2 wdl     : {wdl_weight} (weight on the GAME-RESULT term)");
    if smoke {
        println!("gen-2 SMOKE   : 1 superbatch x 8 batches — dry run, NOT a real net");
    }

    // Architecture: identical to gen-0, weight ids/order unchanged so the
    // exported payload stays (l0w, l0b, l1w, l1b) = FeNN v1.
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

    // Weights only (no momentum/velocity) — this is an init, not a resume.
    if let Some(path) = &init {
        trainer.optimiser.load_weights_from_file(path).expect("failed to load GEN2_INIT weights");
        println!("loaded init weights from {path}");
    }

    let schedule = TrainingSchedule {
        net_id: variant.net_id().to_string(),
        eval_scale: SCALE as f32,
        steps: variant.steps(smoke),
        wdl_scheduler: wdl::ConstantWDL { value: wdl_weight },
        lr_scheduler: variant.lr(),
        save_rate: if smoke { 1 } else { variant.save_rate() },
    };

    let settings = LocalSettings {
        threads,
        test_set: val.as_deref().map(TestDataset::at),
        output_directory: "checkpoints",
        batch_queue_size: 64,
    };

    let data_loader = loader::DirectSequentialDataLoader::new(&[data.as_str()]);
    trainer.run(&schedule, &settings, &data_loader);
}
