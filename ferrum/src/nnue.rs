use crate::board::{Board, FeatureDelta};
use crate::eval::Eval;
use crate::types::*;

const MAGIC: &[u8; 4] = b"FeNN";
const HEADER_LEN: usize = 16;

/// Gen-1 king-bucket count, matching bullet's `ChessBucketsMirrored` input feature set
/// (crates/bullet_lib/src/game/inputs/chess_buckets.rs, bullet commit
/// cebc78a093d92cbc87e56cfef049184c225270b0). This exact value (and `KING_BUCKETS` below)
/// MUST be reused verbatim as the bucket array fed to `ChessBucketsMirrored::new` in the
/// M3 Task 6 bullet training config, or the trained net's feature layout won't match what
/// this file loads.
const NUM_BUCKETS: usize = 4;

/// King-bucket layout: a 2x2 quadrant scheme over each perspective's OWN king square,
/// already expanded to all 64 squares (mirroring bullet's own 32->64 expansion, since our
/// scheme happens not to depend on which side of the mirror line the king's on): bucket =
/// 2*(king_rank >= 4) + (folded_king_file >= 2), where folded_file = min(file, 7-file) is
/// bullet's own e-h-onto-a-d fold. Equivalently, as the 32-entry (files a-d only) seed array
/// bullet's `ChessBucketsMirrored::new` expects (rank-major, must be reused verbatim by Task 6):
///   [0,0,1,1,  0,0,1,1,  0,0,1,1,  0,0,1,1,   // ranks 0-3
///    2,2,3,3,  2,2,3,3,  2,2,3,3,  2,2,3,3]   // ranks 4-7
const KING_BUCKETS: [usize; 64] = [
    0, 0, 1, 1, 1, 1, 0, 0, //
    0, 0, 1, 1, 1, 1, 0, 0, //
    0, 0, 1, 1, 1, 1, 0, 0, //
    0, 0, 1, 1, 1, 1, 0, 0, //
    2, 2, 3, 3, 3, 3, 2, 2, //
    2, 2, 3, 3, 3, 3, 2, 2, //
    2, 2, 3, 3, 3, 3, 2, 2, //
    2, 2, 3, 3, 3, 3, 2, 2, //
];

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
    num_buckets: usize,
    feature_weights: Box<[i16]>, // num_buckets * 768 * hidden, per-feature contiguous column
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
        let hidden = u16::from_le_bytes([bytes[6], bytes[7]]) as usize;
        let qa = i16::from_le_bytes([bytes[8], bytes[9]]) as i32;
        let qb = i16::from_le_bytes([bytes[10], bytes[11]]) as i32;
        let scale = i16::from_le_bytes([bytes[12], bytes[13]]) as i32;
        let num_buckets = match version {
            1 => 1,
            2 => u16::from_le_bytes([bytes[14], bytes[15]]) as usize,
            _ => return Err(format!("unsupported format version {version}")),
        };
        if num_buckets == 0 {
            return Err("num_buckets must be nonzero".into());
        }
        // `forward` divides by `qa` and `qa*qb`, and `screlu` calls `clamp(0, qa)`
        // (which panics if `qa < 0`); a zero/negative divisor from a malformed header
        // must fail here, at the load boundary, rather than panic mid-search.
        if qa <= 0 || qb <= 0 {
            return Err(format!("invalid quantization divisors: qa={qa}, qb={qb} (both must be positive)"));
        }

        let payload = &bytes[HEADER_LEN..];
        let expected = num_buckets * 768 * hidden * 2 + hidden * 2 + 2 * hidden * 2 + 2;
        if payload.len() != expected {
            return Err(format!(
                "bad payload length: got {} bytes, expected {expected} for hidden_size={hidden}, num_buckets={num_buckets}",
                payload.len()
            ));
        }

        let mut cursor = payload;
        let feature_weights = read_i16s(&mut cursor, num_buckets * 768 * hidden);
        let feature_bias = read_i16s(&mut cursor, hidden);
        let output_weights = read_i16s(&mut cursor, 2 * hidden);
        let output_bias = read_i16s(&mut cursor, 1)[0];

        Ok(Nnue { hidden, num_buckets, feature_weights, feature_bias, output_weights, output_bias, qa, qb, scale })
    }

    /// Builds one perspective's accumulator from scratch by scanning every piece on the
    /// board. Feature indexing matches `ferrum/nnue/README.md`'s convention exactly:
    /// `is_enemy = color != perspective`, `view_sq = sq ^ 56` when `perspective == Black`
    /// (else `sq`), `feature = is_enemy*384 + 64*pt + view_sq`. O(all pieces) — the
    /// full-refresh oracle; `apply_delta` is the O(changed features) fast path.
    fn accumulate(&self, board: &Board, perspective: Color) -> Vec<i32> {
        let mut acc: Vec<i32> = self.feature_bias.iter().map(|&b| b as i32).collect();
        let (bucket, mirror) = if self.num_buckets == 1 {
            (0, false)
        } else {
            Nnue::king_context(board.king_sq(perspective), perspective)
        };
        for pt in 0..6 {
            for color in [Color::White, Color::Black] {
                let mut pieces = board.bb[pc(color, pt)];
                while pieces != 0 {
                    let sq = pop_lsb(&mut pieces);
                    self.toggle_feature(&mut acc, Nnue::feature_index_bucketed(perspective, bucket, mirror, color, pt, sq), true);
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

    /// Returns `(king_bucket, mirror)` for `perspective`'s own king at `king_sq` (absolute
    /// square), matching bullet's `ChessBucketsMirrored::map_features` exactly: the
    /// perspective's vertical flip is applied FIRST (the same `view_sq` transform
    /// `feature_index` uses), and only THEN does the (already-flipped) square index into
    /// `KING_BUCKETS` and decide the horizontal mirror (folded file > 3, i.e. king on the
    /// e-h files post-flip => mirror). Byte-verified against bullet's own mapper output in
    /// `bucketed_feature_index_matches_bullet_reference`.
    fn king_context(king_sq: u8, perspective: Color) -> (usize, bool) {
        let view_sq = if perspective == Color::Black { king_sq ^ 56 } else { king_sq };
        (KING_BUCKETS[view_sq as usize], view_sq % 8 > 3)
    }

    /// Bucketed generalisation of `feature_index`: same perspective-flip and
    /// `is_enemy*384 + 64*pt` structure, plus (when `mirror`) a file fold (`^7`) applied
    /// after the perspective flip, plus the bucket's `768`-wide offset. For
    /// `bucket == 0, mirror == false` this is byte-identical to `feature_index` (see
    /// `bucketed_zero_bucket_no_mirror_matches_gen0_formula`) — the invariant M3 Task 3
    /// relies on to keep the v1 (unbucketed) path unchanged.
    fn feature_index_bucketed(
        perspective: Color,
        bucket: usize,
        mirror: bool,
        color: Color,
        pt: usize,
        square: u8,
    ) -> usize {
        let is_enemy = (color != perspective) as usize;
        let mut view_sq = if perspective == Color::Black { square ^ 56 } else { square };
        if mirror {
            view_sq ^= 7;
        }
        bucket * 768 + is_enemy * 384 + 64 * pt + view_sq as usize
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

    /// Bucketed, board-aware incremental update. `board` is the position AFTER the
    /// move (for `forward`; BEFORE it for the reverse direction) — the state whose
    /// accumulator we're producing. `apply_delta`'s plain `feature_index` path is only
    /// correct when neither perspective's king context (bucket/mirror) changes; a king
    /// move is the only thing that can change a king context, so whichever
    /// perspective's OWN king moved gets a full refresh straight from `board`, while
    /// the other perspective still updates incrementally with its (unchanged) king
    /// context. The non-moving perspective's king didn't move, so its king context is
    /// unchanged by this move and equals what the parent accumulator used — indexing
    /// the ≤4 changed pieces with it is correct. Refreshing the moving perspective on
    /// ANY king move (even one that stays within the same bucket) is always correct
    /// and simpler than diffing buckets; king moves are rare enough that the O(all
    /// pieces) refresh cost doesn't matter. For v1 nets (`num_buckets == 1`, no
    /// buckets/mirroring at all) this delegates straight to `apply_delta`.
    pub fn apply_delta_bucketed(&self, acc: &mut Accumulator, delta: &FeatureDelta, board: &Board, forward: bool) {
        if self.num_buckets == 1 {
            return self.apply_delta(acc, delta, forward);
        }
        let king_moved = delta
            .removed
            .iter()
            .flatten()
            .chain(delta.added.iter().flatten())
            .find(|(_, pt, _)| *pt == KING)
            .map(|&(c, _, _)| c);
        for perspective in [Color::White, Color::Black] {
            if king_moved == Some(perspective) {
                acc.by_color[perspective.idx()] = self.accumulate(board, perspective);
            } else {
                let (bucket, mirror) = Nnue::king_context(board.king_sq(perspective), perspective);
                let (off, on) = if forward { (&delta.removed, &delta.added) } else { (&delta.added, &delta.removed) };
                let a = &mut acc.by_color[perspective.idx()];
                for &(color, pt, sq) in off.iter().flatten() {
                    self.toggle_feature(a, Nnue::feature_index_bucketed(perspective, bucket, mirror, color, pt, sq), false);
                }
                for &(color, pt, sq) in on.iter().flatten() {
                    self.toggle_feature(a, Nnue::feature_index_bucketed(perspective, bucket, mirror, color, pt, sq), true);
                }
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

impl Accumulator {
    /// Copies `src`'s contents into `self`, reusing `self`'s existing heap buffers
    /// (no allocation as long as the hidden sizes match, which they always do) — the
    /// pool-reuse primitive `EvalKind::push_delta`/`reset_accumulator` need to advance
    /// the search stack without a clone-then-free per node.
    pub fn copy_from(&mut self, src: &Accumulator) {
        for c in 0..2 { self.by_color[c].copy_from_slice(&src.by_color[c]); }
    }
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
        version: u8,
        num_buckets: u16,
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
        bytes.push(version);
        bytes.push(0); // reserved
        bytes.extend_from_slice(&(hidden as u16).to_le_bytes());
        bytes.extend_from_slice(&qa.to_le_bytes());
        bytes.extend_from_slice(&qb.to_le_bytes());
        bytes.extend_from_slice(&scale.to_le_bytes());
        bytes.extend_from_slice(&num_buckets.to_le_bytes());
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
        let bytes = encode_net(1, 1, hidden, &feature_weights, &feature_bias, &output_weights, output_bias, 100, 10, 200);

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
    fn from_bytes_rejects_nonpositive_quantization() {
        // A well-sized header with qa/qb <= 0 must be rejected at load, not panic in
        // `forward`/`screlu` during the first eval.
        let hidden = 2;
        let fw = vec![0i16; 768 * hidden];
        let fb = vec![0i16; hidden];
        let ow = vec![0i16; 2 * hidden];
        for (qa, qb) in [(0i16, 64i16), (255, 0), (-1, 64), (255, -1)] {
            let bytes = encode_net(1, 1, hidden, &fw, &fb, &ow, 0, qa, qb, 400);
            assert!(Nnue::from_bytes(&bytes).is_err(), "qa={qa} qb={qb} must be rejected");
        }
    }

    #[test]
    fn from_bytes_parses_v2_bucketed_header() {
        let hidden = 2;
        let num_buckets = 4usize;
        let fw = vec![0i16; num_buckets * 768 * hidden];
        let fb = vec![0i16; hidden];
        let ow = vec![0i16; 2 * hidden];
        let bytes = encode_net(2, num_buckets as u16, hidden, &fw, &fb, &ow, 0, 255, 64, 400);
        let net = Nnue::from_bytes(&bytes).expect("v2 net parses");
        assert_eq!(net.num_buckets, num_buckets);
        assert_eq!(net.hidden, hidden);
        assert_eq!(net.feature_weights.len(), num_buckets * 768 * hidden);
    }

    // Reference table generated by running bullet's ACTUAL `ChessBucketsMirrored` mapper
    // (crates/bullet_lib/src/game/inputs/chess_buckets.rs) at the pinned commit
    // cebc78a093d92cbc87e56cfef049184c225270b0, against the SAME `KING_BUCKETS`/32-entry
    // seed array chosen above (see the scratch harness that built each case's FEN, parsed
    // it via `bulletformat::ChessBoard::from_str`, and read off the `stm`-side feature
    // index `ChessBucketsMirrored::map_features` emits for the piece of interest -- i.e.
    // these 32 expected indices are bullet's own program output, not a ferrum port).
    // Tuple shape: ((perspective, own_king_sq, piece_color, piece_type, piece_sq), expected).
    // Cases span: king on a-d and e-h files (both mirror states), both perspectives, own
    // and enemy test pieces, several bucket regions, and the king itself as the test piece.
    type BucketCase = ((Color, u8, Color, usize, u8), usize);

    #[rustfmt::skip]
    const CASES: &[BucketCase] = &[
        ((Color::White, 0, Color::White, 0, 8), 8),
        ((Color::White, 0, Color::Black, 0, 48), 432),
        ((Color::White, 3, Color::White, 1, 21), 853),
        ((Color::White, 3, Color::Black, 4, 59), 1467),
        ((Color::White, 27, Color::White, 2, 20), 916),
        ((Color::White, 24, Color::White, 3, 31), 223),
        ((Color::White, 59, Color::White, 4, 3), 2563),
        ((Color::White, 56, Color::Black, 5, 63), 2303),
        ((Color::White, 4, Color::White, 0, 12), 779),
        ((Color::White, 7, Color::Black, 0, 55), 432),
        ((Color::White, 60, Color::White, 1, 45), 2410),
        ((Color::White, 63, Color::Black, 4, 7), 2176),
        ((Color::White, 36, Color::White, 3, 39), 2528),
        ((Color::White, 39, Color::White, 2, 32), 1703),
        ((Color::Black, 0, Color::Black, 0, 8), 1584),
        ((Color::Black, 3, Color::Black, 1, 21), 2413),
        ((Color::Black, 24, Color::White, 4, 31), 2215),
        ((Color::Black, 59, Color::Black, 3, 3), 1019),
        ((Color::Black, 56, Color::White, 5, 63), 711),
        ((Color::Black, 27, Color::Black, 2, 20), 2476),
        ((Color::Black, 4, Color::Black, 0, 12), 2355),
        ((Color::Black, 7, Color::White, 0, 55), 1928),
        ((Color::Black, 60, Color::Black, 1, 45), 850),
        ((Color::Black, 63, Color::White, 4, 7), 696),
        ((Color::Black, 36, Color::Black, 3, 39), 984),
        ((Color::Black, 39, Color::Black, 2, 32), 159),
        ((Color::White, 0, Color::White, 5, 0), 320),
        ((Color::White, 4, Color::White, 5, 4), 1091),
        ((Color::Black, 59, Color::Black, 5, 59), 1091),
        ((Color::Black, 60, Color::Black, 5, 60), 1091),
        ((Color::White, 30, Color::White, 5, 30), 345),
        ((Color::Black, 33, Color::Black, 5, 33), 345),
    ];

    #[test]
    fn bucketed_feature_index_matches_bullet_reference() {
        for &((perspective, king_sq, color, pt, square), expected) in CASES {
            let (bucket, mirror) = Nnue::king_context(king_sq, perspective);
            let got = Nnue::feature_index_bucketed(perspective, bucket, mirror, color, pt, square);
            assert_eq!(
                got, expected,
                "perspective={perspective:?} king_sq={king_sq} color={color:?} pt={pt} square={square} bucket={bucket} mirror={mirror}"
            );
        }
    }

    #[test]
    fn bucketed_zero_bucket_no_mirror_matches_gen0_formula() {
        // Constraint M3 Task 3 relies on: at bucket 0 with no mirror, the bucketed formula
        // must reduce byte-for-byte to gen-0's plain `feature_index`, for every
        // (perspective, color, pt, square) combination -- not just the cases above.
        for perspective in [Color::White, Color::Black] {
            for color in [Color::White, Color::Black] {
                for pt in 0..6 {
                    for square in 0..64u8 {
                        assert_eq!(
                            Nnue::feature_index_bucketed(perspective, 0, false, color, pt, square),
                            Nnue::feature_index(perspective, color, pt, square)
                        );
                    }
                }
            }
        }
    }

    /// A synthetic v2 bucketed net (distinct nonzero weights) so bucketed-path tests
    /// don't need a trained net. Mirrors `synthetic`/`incremental_..._synthetic` style.
    fn synthetic_bucketed_net(num_buckets: usize, hidden: usize) -> Nnue {
        let feat_rows = num_buckets * 768;
        let mut fw = vec![0i16; feat_rows * hidden];
        for f in 0..feat_rows {
            for h in 0..hidden {
                fw[f * hidden + h] = (((f * 7 + h * 3 + 1) % 200) as i16) - 100;
            }
        }
        let fb = vec![5i16; hidden];
        let ow = vec![1i16; 2 * hidden];
        let bytes = encode_net(2, num_buckets as u16, hidden, &fw, &fb, &ow, 0, 255, 64, 400);
        Nnue::from_bytes(&bytes).unwrap()
    }

    /// Color-swap + vertical rank-flip of a FEN's board, flip side-to-move. Only valid
    /// for fixtures with no castling rights / no ep square (use '-' for both) so the
    /// transform is a clean board mirror.
    fn mirror_fen(fen: &str) -> String {
        let mut parts = fen.split_whitespace();
        let board = parts.next().unwrap();
        let side = parts.next().unwrap();
        let swap_case = |row: &&str| -> String {
            row.chars()
                .map(|c| {
                    if c.is_ascii_uppercase() {
                        c.to_ascii_lowercase()
                    } else if c.is_ascii_lowercase() {
                        c.to_ascii_uppercase()
                    } else {
                        c
                    }
                })
                .collect()
        };
        let rows: Vec<&str> = board.split('/').collect();
        let mirrored: Vec<String> = rows.iter().rev().map(swap_case).collect();
        let new_side = if side == "w" { "b" } else { "w" };
        format!("{} {} - - 0 1", mirrored.join("/"), new_side)
    }

    #[test]
    fn bucketed_eval_is_perspective_symmetric() {
        let net = synthetic_bucketed_net(NUM_BUCKETS, 8);
        for fen in [
            "r3k2r/pp1q1ppp/2n2n2/3pp3/3PP3/2N2N2/PP1Q1PPP/R3K2R w - - 0 1",
            "8/2k2p2/3p4/1P6/2P5/3K4/5P2/8 w - - 0 1",
            "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",
        ] {
            let b = Board::from_fen(fen).unwrap();
            let m = Board::from_fen(&mirror_fen(fen)).unwrap();
            assert_eq!(net.eval(&b), net.eval(&m), "asymmetry at {fen} vs {}", mirror_fen(fen));
        }
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
            let king_moved = delta.removed.iter().flatten().chain(delta.added.iter().flatten()).any(|&(_, pt, _)| pt == KING);
            let undo = b.make(m);
            if b.in_check(b.side.flip()) {
                b.unmake(m, undo);
                continue;
            }

            let mut child = acc.clone();
            net.apply_delta_bucketed(&mut child, &delta, b, true);
            check_incremental_matches_refresh(net, b, &child, depth - 1);

            // A king move's refresh can't be reversed from the delta alone (the
            // pre-move king context is gone once `b` has moved past it), and the
            // real search never reverse-deltas anyway (it pops the accumulator
            // pool instead) -- so only check reverse-restoration when no king
            // moved, where `b`'s (post-move) king context for both perspectives
            // still equals what the parent accumulator used.
            if !king_moved {
                let mut restored = child.clone();
                net.apply_delta_bucketed(&mut restored, &delta, b, false);
                assert_eq!(restored.by_color[0], acc.by_color[0], "reverse apply_delta diverged (white) for {}", m.uci());
                assert_eq!(restored.by_color[1], acc.by_color[1], "reverse apply_delta diverged (black) for {}", m.uci());
            }

            b.unmake(m, undo);
        }
    }

    // Shared fixture positions covering every move type `apply_delta_bucketed` must
    // handle: quiets/captures (all four), both castlings (kiwipete), quiet promotion +
    // capture-promotion (both directions, from the same square), en passant, and —
    // the case gen-0's plain incremental path couldn't handle — king moves (including
    // castling) that cross a king bucket and/or the mirror line.
    const INCREMENTAL_TEST_FENS: [&str; 7] = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", // kiwipete: castling both sides
        "n1n5/1P6/8/8/8/8/8/k6K w - - 0 1", // promo + capture-promo
        "8/8/8/2pP4/8/8/8/k6K w - c6 0 1",  // en passant
        "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", // white O-O/O-O-O: king e1->g1/c1 (bucket + mirror change)
        "8/8/8/3k4/8/4K3/8/8 w - - 0 1",    // both kings roam across buckets/mirror
        "4k3/8/8/8/8/8/8/3K4 w - - 0 1",    // kings sit ON the d/e mirror boundary, far apart: a
                                            // single lateral step (d1-e1, e8-d8) crosses the mirror
    ];

    #[test]
    fn incremental_accumulator_matches_full_refresh_synthetic() {
        // A cheap synthetic BUCKETED net (no gen0.bin needed) so this guards the
        // incremental logic -- including the king-move bucket/mirror refresh path --
        // even in contexts where the real net isn't present. Every feature weight is
        // distinct and nonzero so a wrong perspective/is_enemy/view_sq/bucket/mirror
        // index is overwhelmingly likely to produce a visible mismatch rather than
        // accidentally cancel out.
        let net = synthetic_bucketed_net(NUM_BUCKETS, 3);

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

    #[test]
    #[ignore = "requires ferrum/nnue/gen1.bin (the trained king-bucketed net, present locally)"]
    fn incremental_accumulator_matches_full_refresh_real_gen1_net() {
        // Same invariant as above but against the REAL bucketed gen-1 net: exercises the
        // king-move accumulator refresh (Task 4) with the actual trained weights over
        // fixtures that cross buckets/mirror and castle.
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/nnue/gen1.bin");
        let net = Nnue::load(path).expect("gen1.bin should parse");
        assert!(net.num_buckets > 1, "gen1.bin must be a bucketed v2 net");
        for fen in INCREMENTAL_TEST_FENS {
            let mut b = Board::from_fen(fen).unwrap();
            let acc = net.fresh_accumulator(&b);
            check_incremental_matches_refresh(&net, &mut b, &acc, 3);
        }
    }
}
