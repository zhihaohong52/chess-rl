// ferrum gen-1 NNUE training config (bullet, pinned cebc78a093d92cbc87e56cfef049184c225270b0).
//
// Same simple architecture as gen-0 (768*B -> H*2 -> 1, SCReLU) but with a
// king-bucketed, horizontally-mirrored input (`ChessBucketsMirrored`) and a
// larger hidden layer. The bucket layout below is the EXACT 32-entry seed that
// ferrum's `src/nnue.rs` (`KING_BUCKETS`/`NUM_BUCKETS`, Task 2) was byte-verified
// against — DO NOT change one without the other or the trained net's feature
// layout silently mismatches ferrum's loader.
//
// No factoriser / no output buckets on purpose: keeps the exported weight layout
// (l0w, l0b, l1w, l1b, in order) byte-compatible with ferrum's FeNN v2 payload
// (feature_weights[B*768 x H], feature_bias[H], output_weights[2H], output_bias),
// exactly as gen-0 — so `build_gen1_bin.py` is a straight header-prepend like
// `build_gen0_bin.py`.
//
// Recipe is tuned across runs by editing `NET_ID` + the wdl/lr schedulers below
// (bucket count is FIXED at 4 to match the deployed engine). Select the run with
// the lowest validation loss (LocalSettings.test_set = the held-out val shard).
use bullet_lib::{
    game::inputs::{ChessBucketsMirrored, get_num_buckets},
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::{LocalSettings, TestDataset},
    },
    value::{ValueTrainerBuilder, loader},
};

const HIDDEN_SIZE: usize = 1024; // gen-1: 512 -> 1024 per perspective
const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;

// EXACT Task-2 seed (32 entries, a-d files x 8 ranks; bullet folds e-h internally).
// 2x2 quadrant scheme: bucket = 2*(rank>=4) + (file>=2). get_num_buckets => 4.
#[rustfmt::skip]
const BUCKET_LAYOUT: [usize; 32] = [
    0, 0, 1, 1,
    0, 0, 1, 1,
    0, 0, 1, 1,
    0, 0, 1, 1,
    2, 2, 3, 3,
    2, 2, 3, 3,
    2, 2, 3, 3,
    2, 2, 3, 3,
];

fn main() {
    let num_buckets = get_num_buckets(&BUCKET_LAYOUT); // 4
    let num_inputs = 768 * num_buckets; // 3072

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs| {
            let l0 = builder.new_affine("l0", num_inputs, HIDDEN_SIZE);
            let l1 = builder.new_affine("l1", 2 * HIDDEN_SIZE, 1);
            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            l1.forward(stm_hidden.concat(ntm_hidden))
        });

    let schedule = TrainingSchedule {
        net_id: "gen1".to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104, // ~100M positions / 16384 ~= one epoch/superbatch
            start_superbatch: 1,
            end_superbatch: 45,
        },
        // Recipe knob #1 (tune across runs): score-only (ConstantWDL{0.0}) like gen-0,
        // OR a WDL blend that eases in the game-result term, e.g.
        // wdl::LinearWDL { start: 0.2, end: 0.0 }. bullet's value = weight on the
        // GAME-RESULT term, so 0.0 = pure score (design spec lambda=1.0 on score).
        wdl_scheduler: wdl::ConstantWDL { value: 0.0 },
        // Recipe knob #2: LR schedule.
        lr_scheduler: lr::StepLR { start: 0.001, gamma: 0.1, step: 20 },
        save_rate: 15,
    };

    // test_set = held-out val shard -> reported validation loss for model selection.
    let settings = LocalSettings {
        threads: 8,
        test_set: Some(TestDataset::at("data/val.data")),
        output_directory: "checkpoints",
        batch_queue_size: 64,
    };
    let data_loader = loader::DirectSequentialDataLoader::new(&["data/train.data"]);
    trainer.run(&schedule, &settings, &data_loader);
}
