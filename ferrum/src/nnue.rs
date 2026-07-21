use crate::board::{Board, FeatureDelta};
use crate::eval::Eval;
use crate::types::*;

const MAGIC: &[u8; 4] = b"FeNN";
const HEADER_LEN: usize = 16;

/// Gen-0 NNUE evaluation: a 768->Hx2->1 SCReLU network, loaded from ferrum's own
/// ".bin" wrapper format — see `ferrum/nnue/README.md` ("Net file format" /
/// "Feature indexing convention") for the byte-exact spec this module implements
/// against.
///
/// `Eval::eval` does a full refresh (rebuilds both perspective accumulators from
/// scratch) and stays the correctness oracle. The hot search path instead
/// maintains an `Accumulator` incrementally across make/unmake via
/// `fresh_accumulator` (root) + `apply_delta` (O(changed features) per move) —
/// see `Searcher`'s `EvalKind::Nnue` in `search.rs` (M2 Task 5).
pub struct Nnue {
    hidden: usize,
    feature_weights: Box<[i16]>, // 768 * hidden, per-feature contiguous column
    feature_bias: Box<[i16]>,    // hidden
    output_weights: Box<[i16]>,  // 2 * hidden ("us" half then "them" half)
    output_bias: i16,
    qa: i32,
    qb: i32,
    scale: i32,
}

impl Nnue {
    pub fn load(path: &str) -> Result<Nnue, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("reading {path}: {e}"))?;
        Nnue::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Nnue, String> {
        if bytes.len() < HEADER_LEN {
            return Err(format!("file too short for header: {} bytes", bytes.len()));
        }
        if &bytes[0..4] != MAGIC {
            return Err("bad magic (expected \"FeNN\")".into());
        }
        let version = bytes[4];
        if version != 1 {
            return Err(format!("unsupported format version {version}"));
        }
        let hidden = u16::from_le_bytes([bytes[6], bytes[7]]) as usize;
        let qa = i16::from_le_bytes([bytes[8], bytes[9]]) as i32;
        let qb = i16::from_le_bytes([bytes[10], bytes[11]]) as i32;
        let scale = i16::from_le_bytes([bytes[12], bytes[13]]) as i32;

        let payload = &bytes[HEADER_LEN..];
        let expected = 768 * hidden * 2 + hidden * 2 + 2 * hidden * 2 + 2;
        if payload.len() != expected {
            return Err(format!(
                "bad payload length: got {} bytes, expected {expected} for hidden_size={hidden}",
                payload.len()
            ));
        }

        let mut cursor = payload;
        let feature_weights = read_i16s(&mut cursor, 768 * hidden);
        let feature_bias = read_i16s(&mut cursor, hidden);
        let output_weights = read_i16s(&mut cursor, 2 * hidden);
        let output_bias = read_i16s(&mut cursor, 1)[0];

        Ok(Nnue { hidden, feature_weights, feature_bias, output_weights, output_bias, qa, qb, scale })
    }

    /// Builds one perspective's accumulator from scratch by scanning every piece on the
    /// board. Feature indexing matches `ferrum/nnue/README.md`'s convention exactly:
    /// `is_enemy = color != perspective`, `view_sq = sq ^ 56` when `perspective == Black`
    /// (else `sq`), `feature = is_enemy*384 + 64*pt + view_sq`. O(all pieces) — the
    /// full-refresh oracle; `apply_delta` is the O(changed features) fast path.
    fn accumulate(&self, board: &Board, perspective: Color) -> Vec<i32> {
        let mut acc: Vec<i32> = self.feature_bias.iter().map(|&b| b as i32).collect();
        for pt in 0..6 {
            for color in [Color::White, Color::Black] {
                let mut pieces = board.bb[pc(color, pt)];
                while pieces != 0 {
                    let square = pop_lsb(&mut pieces);
                    self.toggle_feature(&mut acc, Self::feature_index(perspective, color, pt, square), true);
                }
            }
        }
        acc
    }

    /// Builds both perspective accumulators (indexed by absolute color, not
    /// side-to-move — see `Accumulator`) from scratch. This is the only place a full
    /// board rescan happens on the hot path: once per search root. Every node inside
    /// the tree instead derives its accumulator from its parent's via `apply_delta`.
    pub fn fresh_accumulator(&self, board: &Board) -> Accumulator {
        Accumulator {
            by_color: [self.accumulate(board, Color::White), self.accumulate(board, Color::Black)],
        }
    }

    fn feature_index(perspective: Color, color: Color, pt: usize, square: u8) -> usize {
        let is_enemy = (color != perspective) as usize;
        let view_sq = if perspective == Color::Black { square ^ 56 } else { square };
        is_enemy * 384 + 64 * pt + view_sq as usize
    }

    fn toggle_feature(&self, acc: &mut [i32], feature: usize, add: bool) {
        let base = feature * self.hidden;
        if add {
            for h in 0..self.hidden { acc[h] += self.feature_weights[base + h] as i32; }
        } else {
            for h in 0..self.hidden { acc[h] -= self.feature_weights[base + h] as i32; }
        }
    }

    /// Updates `acc` in place for one move's `FeatureDelta`, touching only the
    /// changed features (at most 4) instead of rescanning the board — the O(changed
    /// features) incremental counterpart to `fresh_accumulator`'s O(all pieces).
    ///
    /// `forward = true` applies the delta the way `make` changes the board (turn off
    /// every `removed` slot, turn on every `added` slot, for BOTH perspectives — a
    /// piece placement affects both accumulators, just at different feature indices).
    /// `forward = false` reverses it (as `unmake` would: turn off `added`, turn on
    /// `removed`). Either direction, starting from an `acc` that already satisfies the
    /// invariant, must leave `acc` bit-identical to a fresh rebuild at the resulting
    /// position — that is exactly what `incremental_accumulator_matches_full_refresh`
    /// checks at every node of a real search tree.
    pub fn apply_delta(&self, acc: &mut Accumulator, delta: &FeatureDelta, forward: bool) {
        let (off, on) = if forward { (&delta.removed, &delta.added) } else { (&delta.added, &delta.removed) };
        for perspective in [Color::White, Color::Black] {
            let a = &mut acc.by_color[perspective.idx()];
            for &(color, pt, sq) in off.iter().flatten() {
                self.toggle_feature(a, Self::feature_index(perspective, color, pt, sq), false);
            }
            for &(color, pt, sq) in on.iter().flatten() {
                self.toggle_feature(a, Self::feature_index(perspective, color, pt, sq), true);
            }
        }
    }

    /// Runs the forward pass directly against an already-built `Accumulator`,
    /// without touching the board at all — the O(1) read that replaces `eval()`'s
    /// full refresh on the search hot path. Bit-identical to `eval()` given an `acc`
    /// that satisfies the incremental-matches-refresh invariant.
    pub fn eval_accumulator(&self, acc: &Accumulator, side_to_move: Color) -> i32 {
        self.forward(&acc.by_color[side_to_move.idx()], &acc.by_color[side_to_move.flip().idx()])
    }

    fn forward(&self, stm_acc: &[i32], ntm_acc: &[i32]) -> i32 {
        let mut output: i64 = 0;
        for i in 0..self.hidden {
            output += screlu(stm_acc[i], self.qa) * self.output_weights[i] as i64;
        }
        for i in 0..self.hidden {
            output += screlu(ntm_acc[i], self.qa) * self.output_weights[self.hidden + i] as i64;
        }
        output /= self.qa as i64;
        output += self.output_bias as i64;
        output *= self.scale as i64;
        output /= self.qa as i64 * self.qb as i64;
        output as i32
    }
}

/// Two NNUE perspective accumulators, indexed by absolute color — NOT by
/// side-to-move: `by_color[White as usize]` is always the White-perspective
/// accumulator, `by_color[Black as usize]` the Black-perspective one. A piece
/// placement change updates both (at different feature indices, via `apply_delta`);
/// only the forward pass's choice of which one is "us" vs "them" depends on whose
/// turn it is. Consequently a null move — which changes side-to-move but moves no
/// piece — needs no `Accumulator` update at all; `eval_accumulator`'s `side_to_move`
/// argument already handles it.
#[derive(Clone)]
pub struct Accumulator {
    by_color: [Vec<i32>; 2],
}

fn read_i16s(cursor: &mut &[u8], count: usize) -> Box<[i16]> {
    let (chunk, rest) = cursor.split_at(count * 2);
    *cursor = rest;
    chunk.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect()
}

fn screlu(x: i32, qa: i32) -> i64 {
    let y = x.clamp(0, qa) as i64;
    y * y
}

impl Eval for Nnue {
    /// Bit-exact port of `simple.rs`'s `Network::evaluate` (see the README's "Forward
    /// pass" section). Already side-to-move-relative by construction (the STM
    /// accumulator's contribution is summed first, matching bullet's own
    /// `evaluate(us, them)`) — unlike `Hce`, no closing sign flip is needed here.
    fn eval(&self, board: &Board) -> i32 {
        let stm_acc = self.accumulate(board, board.side);
        let ntm_acc = self.accumulate(board, board.side.flip());
        self.forward(&stm_acc, &ntm_acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-rollable little-endian encoder for the `FeNN` wrapper format, mirroring
    /// exactly the byte layout documented in `ferrum/nnue/README.md`.
    #[allow(clippy::too_many_arguments)]
    fn encode_net(
        hidden: usize,
        feature_weights: &[i16],
        feature_bias: &[i16],
        output_weights: &[i16],
        output_bias: i16,
        qa: i16,
        qb: i16,
        scale: i16,
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.push(1); // version
        bytes.push(0); // reserved
        bytes.extend_from_slice(&(hidden as u16).to_le_bytes());
        bytes.extend_from_slice(&qa.to_le_bytes());
        bytes.extend_from_slice(&qb.to_le_bytes());
        bytes.extend_from_slice(&scale.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        for w in feature_weights {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        for w in feature_bias {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        for w in output_weights {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        bytes.extend_from_slice(&output_bias.to_le_bytes());
        bytes
    }

    // Hand-computed fixture: hidden_size=2, all feature weights zero except the two
    // features active on a king-vs-king board (white king e1, black king e8), so the
    // forward pass can be traced by hand end to end.
    //
    // FEN "4k3/8/8/8/8/8/8/4K3 w - - 0 1": white king on e1 (sq 4), black king on e8
    // (sq 60). Feature indices (pt=KING=5):
    //   stm perspective (White):  white king -> is_enemy=0, view_sq=4  -> feature 324
    //                             black king -> is_enemy=1, view_sq=60 -> feature 764
    //   ntm perspective (Black):  white king -> is_enemy=1, view_sq=4^56=60 -> feature 764
    //                             black king -> is_enemy=0, view_sq=60^56=4 -> feature 324
    // Both accumulators sum the same two feature columns (just via different own/enemy
    // roles), so stm_acc == ntm_acc == feature_bias + fw[324] + fw[764] = [31, 12]
    // (bias [1,2], fw[324]=[10,-5], fw[764]=[20,15]).
    //
    // screlu(31, qa=100) = 31*31 = 961; screlu(12, 100) = 12*12 = 144.
    // sum_stm = 961*2 + 144*3 = 1922 + 432 = 2354  (output_weights[0..2] = [2,3])
    // sum_ntm = 961*4 + 144*5 = 3844 + 720 = 4564  (output_weights[2..4] = [4,5])
    // output = 2354 + 4564 = 6918
    // output /= qa(100)      -> 69
    // output += bias(7)      -> 76
    // output *= scale(200)   -> 15200
    // output /= qa*qb(1000)  -> 15
    #[test]
    fn load_and_eval_synthetic_net_matches_hand_computed_forward_pass() {
        let hidden = 2;
        let mut feature_weights = vec![0i16; 768 * hidden];
        feature_weights[324 * hidden] = 10;
        feature_weights[324 * hidden + 1] = -5;
        feature_weights[764 * hidden] = 20;
        feature_weights[764 * hidden + 1] = 15;
        let feature_bias = [1i16, 2];
        let output_weights = [2i16, 3, 4, 5];
        let output_bias = 7i16;
        let bytes = encode_net(hidden, &feature_weights, &feature_bias, &output_weights, output_bias, 100, 10, 200);

        let path = std::env::temp_dir().join(format!("ferrum_nnue_test_{}.bin", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let net = Nnue::load(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(&path).unwrap();

        let board = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        assert_eq!(net.eval(&board), 15);
    }

    #[test]
    fn from_bytes_rejects_bad_magic() {
        let mut bytes = vec![0u8; HEADER_LEN];
        bytes[0..4].copy_from_slice(b"NOPE");
        assert!(Nnue::from_bytes(&bytes).is_err());
    }

    #[test]
    fn from_bytes_rejects_wrong_payload_length() {
        let mut bytes = vec![0u8; HEADER_LEN];
        bytes[0..4].copy_from_slice(MAGIC);
        bytes[4] = 1;
        bytes[6..8].copy_from_slice(&2u16.to_le_bytes()); // hidden_size=2 expects a longer payload
        assert!(Nnue::from_bytes(&bytes).is_err());
    }

    #[test]
    fn load_reports_missing_file() {
        assert!(Nnue::load("/nonexistent/path/gen0.bin").is_err());
    }

    #[test]
    #[ignore = "requires ferrum/nnue/gen0.bin, not yet trained/retrieved (M2 Task 3)"]
    fn loads_real_net() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/nnue/gen0.bin");
        let net = Nnue::load(path).expect("gen0.bin should parse once retrieved");
        let eval = net.eval(&Board::startpos());
        assert!(eval.abs() < 10_000, "startpos eval should be sane, got {eval}");
    }

    /// Depth-limited walk over legal moves (and their unmakes) asserting, at EVERY
    /// node, that an accumulator built purely via `apply_delta` starting from the
    /// position's root is bit-identical — for BOTH perspectives — to a fresh
    /// `accumulate` rebuild of that node's actual board state. This is the invariant
    /// the whole M2 Task 5 incremental design rests on: incremental == full-refresh,
    /// always, for every move type. Also checks that reversing the same delta
    /// (`forward: false`) restores the parent's accumulator exactly — the search
    /// integration itself unmakes via a stack pop rather than a reverse-delta, but
    /// both directions of `apply_delta` are part of the public contract and must
    /// agree.
    fn check_incremental_matches_refresh(net: &Nnue, b: &mut Board, acc: &Accumulator, depth: u32) {
        assert_eq!(
            acc.by_color[Color::White.idx()],
            net.accumulate(b, Color::White),
            "white perspective diverged at {}",
            b.to_fen()
        );
        assert_eq!(
            acc.by_color[Color::Black.idx()],
            net.accumulate(b, Color::Black),
            "black perspective diverged at {}",
            b.to_fen()
        );
        if depth == 0 {
            return;
        }
        let mut moves = Vec::new();
        crate::movegen::generate(b, &mut moves);
        for m in moves {
            let delta = b.feature_delta(m); // must be read before `make` mutates the board
            let undo = b.make(m);
            if b.in_check(b.side.flip()) {
                b.unmake(m, undo);
                continue;
            }

            let mut child = acc.clone();
            net.apply_delta(&mut child, &delta, true);
            check_incremental_matches_refresh(net, b, &child, depth - 1);

            let mut restored = child.clone();
            net.apply_delta(&mut restored, &delta, false);
            assert_eq!(restored.by_color[0], acc.by_color[0], "reverse apply_delta diverged (white) for {}", m.uci());
            assert_eq!(restored.by_color[1], acc.by_color[1], "reverse apply_delta diverged (black) for {}", m.uci());

            b.unmake(m, undo);
        }
    }

    // Shared fixture positions covering every move type `apply_delta` must handle:
    // quiets/captures (all four), both castlings (kiwipete), quiet promotion +
    // capture-promotion (both directions, from the same square), and en passant.
    const INCREMENTAL_TEST_FENS: [&str; 4] = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", // kiwipete
        "n1n5/1P6/8/8/8/8/8/k6K w - - 0 1", // quiet promo + capture-promo
        "8/8/8/2pP4/8/8/8/k6K w - c6 0 1",  // en passant
    ];

    #[test]
    fn incremental_accumulator_matches_full_refresh_synthetic() {
        // A cheap synthetic net (no gen0.bin needed) so this guards the incremental
        // logic even in contexts where the real net isn't present. Every feature
        // weight is distinct and nonzero so a wrong perspective/is_enemy/view_sq
        // index is overwhelmingly likely to produce a visible mismatch rather than
        // accidentally cancel out.
        let hidden = 3;
        let mut feature_weights = vec![0i16; 768 * hidden];
        for f in 0..768 {
            for h in 0..hidden {
                feature_weights[f * hidden + h] = (((f * 7 + h * 3 + 1) % 200) as i16) - 100;
            }
        }
        let feature_bias = vec![5i16; hidden];
        let output_weights = vec![1i16; 2 * hidden];
        let bytes = encode_net(hidden, &feature_weights, &feature_bias, &output_weights, 0, 64, 1, 100);
        let path = std::env::temp_dir().join(format!("ferrum_nnue_inc_test_{}.bin", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let net = Nnue::load(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(&path).unwrap();

        for fen in INCREMENTAL_TEST_FENS {
            let mut b = Board::from_fen(fen).unwrap();
            let acc = net.fresh_accumulator(&b);
            check_incremental_matches_refresh(&net, &mut b, &acc, 3);
        }
    }

    #[test]
    #[ignore = "requires ferrum/nnue/gen0.bin (present locally; not guaranteed in every CI context)"]
    fn incremental_accumulator_matches_full_refresh_real_net() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/nnue/gen0.bin");
        let net = Nnue::load(path).expect("gen0.bin should parse");

        for fen in INCREMENTAL_TEST_FENS {
            let mut b = Board::from_fen(fen).unwrap();
            let acc = net.fresh_accumulator(&b);
            check_incremental_matches_refresh(&net, &mut b, &acc, 3);
        }
    }
}
