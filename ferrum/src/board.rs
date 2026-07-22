use crate::attacks::*;
use crate::types::{
    bb, file_of, lsb, pc, pop_lsb, sq, sq_name, parse_sq, Bb, Color, BISHOP, KING, KNIGHT, PAWN,
    QUEEN, ROOK,
};
use crate::zobrist::z;

/// Which castling rights survive a move touching this square.
/// Rights bits: WK=1 WQ=2 BK=4 BQ=8. a1=WQ rook, h1=WK rook, e1=W king, mirrored for black.
const CASTLE_MASK: [u8; 64] = {
    let mut m = [15u8; 64];
    m[0] = 15 - 2;   // a1
    m[7] = 15 - 1;   // h1
    m[4] = 15 - 3;   // e1
    m[56] = 15 - 8;  // a8
    m[63] = 15 - 4;  // h8
    m[60] = 15 - 12; // e8
    m
};

pub struct Undo {
    captured: Option<usize>,
    castling: u8,
    ep: Option<u8>,
    halfmove: u16,
    hash: u64,
}

pub struct NullUndo {
    ep: Option<u8>,
    hash: u64,
    halfmove: u16,
}

/// Which NNUE feature slots a move turns off/on, expressed piece-agnostically as
/// `(color, piece_type, square)` triples — the exact information an incremental
/// accumulator update needs (M2 Task 5), computed without any NNUE-specific
/// knowledge here (`nnue.rs` maps each triple to a feature index per perspective).
/// Every move type needs at most 2 removed + 2 added slots: quiet/double-push/
/// quiet-promo use 1+1, any capture (incl. en passant, promo-capture) uses 2+1,
/// and castling — the only case touching two own pieces — uses 2+2 (king, rook).
///
/// Computed from the position *before* `make` is applied; call it first.
#[derive(Clone, Copy, Debug)]
pub struct FeatureDelta {
    pub removed: [Option<(Color, usize, u8)>; 2],
    pub added: [Option<(Color, usize, u8)>; 2],
}

#[derive(Clone)]
pub struct Board {
    pub bb: [Bb; 12],
    pub occ: [Bb; 2],
    pub side: Color,
    pub castling: u8, // WK=1 WQ=2 BK=4 BQ=8
    pub ep: Option<u8>,
    pub halfmove: u16,
    pub fullmove: u16,
    pub hash: u64,
}

impl Board {
    pub fn all(&self) -> Bb { self.occ[0] | self.occ[1] }

    pub fn piece_on(&self, s: u8) -> Option<usize> {
        (0..12).find(|&p| self.bb[p] & bb(s) != 0)
    }

    pub fn king_sq(&self, c: Color) -> u8 { lsb(self.bb[pc(c, KING)]) }

    /// Is square `s` attacked by color `c`?
    pub fn attacked(&self, s: u8, c: Color) -> bool {
        // pawn_attacks(c.flip(), s) = squares from which a c-pawn attacks s
        if pawn_attacks(c.flip(), s) & self.bb[pc(c, PAWN)] != 0 { return true; }
        if knight_attacks(s) & self.bb[pc(c, KNIGHT)] != 0 { return true; }
        if king_attacks(s) & self.bb[pc(c, KING)] != 0 { return true; }
        let occ = self.all();
        if bishop_attacks(s, occ) & (self.bb[pc(c, BISHOP)] | self.bb[pc(c, QUEEN)]) != 0 { return true; }
        if rook_attacks(s, occ) & (self.bb[pc(c, ROOK)] | self.bb[pc(c, QUEEN)]) != 0 { return true; }
        false
    }

    pub fn in_check(&self, c: Color) -> bool { self.attacked(self.king_sq(c), c.flip()) }

    pub(crate) fn compute_hash(&self) -> u64 {
        let zb = z();
        let mut h = 0u64;
        for p in 0..12 {
            let mut b = self.bb[p];
            while b != 0 { h ^= zb.piece[p][pop_lsb(&mut b) as usize]; }
        }
        if self.side == Color::Black { h ^= zb.side; }
        h ^= zb.castling[self.castling as usize];
        if let Some(e) = self.ep { h ^= zb.ep_file[file_of(e) as usize]; }
        h
    }

    pub fn from_fen(fen: &str) -> Result<Board, String> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 4 { return Err("bad fen".into()); }
        let mut b = Board {
            bb: [0; 12], occ: [0; 2], side: Color::White, castling: 0,
            ep: None, halfmove: 0, fullmove: 1, hash: 0,
        };
        let (mut f, mut r) = (0u8, 7u8);
        for ch in parts[0].chars() {
            match ch {
                '/' => {
                    if r == 0 { return Err("too many ranks".into()); }
                    f = 0; r -= 1;
                }
                '1'..='8' => {
                    f += ch as u8 - b'0';
                    if f > 8 { return Err("rank overflow".into()); }
                }
                _ => {
                    if f >= 8 { return Err("rank overflow".into()); }
                    let c = if ch.is_uppercase() { Color::White } else { Color::Black };
                    let pt = match ch.to_ascii_lowercase() {
                        'p' => PAWN, 'n' => KNIGHT, 'b' => BISHOP,
                        'r' => ROOK, 'q' => QUEEN, 'k' => KING,
                        _ => return Err(format!("bad piece {ch}")),
                    };
                    let s = sq(f, r);
                    b.bb[pc(c, pt)] |= bb(s);
                    b.occ[c.idx()] |= bb(s);
                    f += 1;
                }
            }
        }
        if b.bb[pc(Color::White, KING)].count_ones() != 1
            || b.bb[pc(Color::Black, KING)].count_ones() != 1
        {
            return Err("invalid king count".into());
        }
        // Pawns on ranks 1/8 would panic movegen in debug (square arithmetic
        // off the board) or silently corrupt the Move encoding in release.
        const BACK_RANKS: Bb = 0xFF00_0000_0000_00FF;
        if (b.bb[pc(Color::White, PAWN)] | b.bb[pc(Color::Black, PAWN)]) & BACK_RANKS != 0 {
            return Err("pawn on back rank".into());
        }
        b.side = if parts[1] == "w" { Color::White } else { Color::Black };
        for ch in parts[2].chars() {
            b.castling |= match ch { 'K' => 1, 'Q' => 2, 'k' => 4, 'q' => 8, _ => 0 };
        }
        if parts[3] != "-" { b.ep = parse_sq(parts[3]); }
        if parts.len() > 4 { b.halfmove = parts[4].parse().unwrap_or(0); }
        if parts.len() > 5 { b.fullmove = parts[5].parse().unwrap_or(1); }
        b.hash = b.compute_hash();
        Ok(b)
    }

    pub fn startpos() -> Board {
        Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap()
    }

    pub fn to_fen(&self) -> String {
        let mut s = String::new();
        for r in (0..8).rev() {
            let mut empty: u8 = 0;
            for f in 0..8 {
                match self.piece_on(sq(f, r)) {
                    None => empty += 1,
                    Some(p) => {
                        if empty > 0 { s.push((b'0' + empty) as char); empty = 0; }
                        s.push(b"PNBRQKpnbrqk"[p] as char);
                    }
                }
            }
            if empty > 0 { s.push((b'0' + empty) as char); }
            if r > 0 { s.push('/'); }
        }
        s.push(' ');
        s.push_str(if self.side == Color::White { "w" } else { "b" });
        s.push(' ');
        if self.castling == 0 { s.push('-'); } else {
            for (bit, ch) in [(1, 'K'), (2, 'Q'), (4, 'k'), (8, 'q')] {
                if self.castling & bit != 0 { s.push(ch); }
            }
        }
        s.push(' ');
        match self.ep { Some(e) => s.push_str(&sq_name(e)), None => s.push('-') }
        s.push_str(&format!(" {} {}", self.halfmove, self.fullmove));
        s
    }

    fn move_piece(&mut self, p: usize, from: u8, to: u8) {
        let zb = z();
        self.bb[p] ^= bb(from) | bb(to);
        self.occ[p / 6] ^= bb(from) | bb(to);
        self.hash ^= zb.piece[p][from as usize] ^ zb.piece[p][to as usize];
    }
    fn toggle_piece(&mut self, p: usize, s: u8) {
        self.bb[p] ^= bb(s);
        self.occ[p / 6] ^= bb(s);
        self.hash ^= z().piece[p][s as usize];
    }

    pub fn make(&mut self, m: crate::moves::Move) -> Undo {
        use crate::moves::*;
        let zb = z();
        let us = self.side;
        let mut undo = Undo {
            captured: None, castling: self.castling, ep: self.ep,
            halfmove: self.halfmove, hash: self.hash,
        };
        let (from, to, flags) = (m.from(), m.to(), m.flags());
        let piece = self.piece_on(from).expect("no piece on from-square");

        // clear old ep from hash; new ep set below if DPP
        if let Some(e) = self.ep { self.hash ^= zb.ep_file[file_of(e) as usize]; }
        self.ep = None;
        self.halfmove += 1;
        if piece % 6 == PAWN { self.halfmove = 0; }

        if m.is_capture() {
            let cap_sq = if flags == FLAG_EP {
                if us == Color::White { to - 8 } else { to + 8 }
            } else { to };
            let cap = self.piece_on(cap_sq).expect("no captured piece");
            self.toggle_piece(cap, cap_sq);
            undo.captured = Some(cap);
            self.halfmove = 0;
        }

        self.move_piece(piece, from, to);

        if m.is_promo() {
            self.toggle_piece(piece, to);                       // remove pawn
            self.toggle_piece(pc(us, m.promo_pt()), to);        // drop promo piece
        } else if flags == FLAG_OO {
            let (rf, rt) = if us == Color::White { (7, 5) } else { (63, 61) };
            self.move_piece(pc(us, ROOK), rf, rt);
        } else if flags == FLAG_OOO {
            let (rf, rt) = if us == Color::White { (0, 3) } else { (56, 59) };
            self.move_piece(pc(us, ROOK), rf, rt);
        } else if flags == FLAG_DPP {
            let ep = if us == Color::White { from + 8 } else { from - 8 };
            self.ep = Some(ep);
            self.hash ^= zb.ep_file[file_of(ep) as usize];
        }

        self.hash ^= zb.castling[self.castling as usize];
        self.castling &= CASTLE_MASK[from as usize] & CASTLE_MASK[to as usize];
        self.hash ^= zb.castling[self.castling as usize];

        if us == Color::Black { self.fullmove += 1; }
        self.side = us.flip();
        self.hash ^= zb.side;
        undo
    }

    pub fn unmake(&mut self, m: crate::moves::Move, u: Undo) {
        use crate::moves::*;
        let us = self.side.flip(); // the side that made the move
        let (from, to, flags) = (m.from(), m.to(), m.flags());

        if m.is_promo() {
            self.toggle_piece(pc(us, m.promo_pt()), to);
            self.toggle_piece(pc(us, PAWN), to);
        }
        let piece = self.piece_on(to).expect("no piece on to-square");
        self.move_piece(piece, to, from);

        if flags == FLAG_OO {
            let (rf, rt) = if us == Color::White { (7, 5) } else { (63, 61) };
            self.move_piece(pc(us, ROOK), rt, rf);
        } else if flags == FLAG_OOO {
            let (rf, rt) = if us == Color::White { (0, 3) } else { (56, 59) };
            self.move_piece(pc(us, ROOK), rt, rf);
        }
        if let Some(cap) = u.captured {
            let cap_sq = if flags == FLAG_EP {
                if us == Color::White { to - 8 } else { to + 8 }
            } else { to };
            self.toggle_piece(cap, cap_sq);
        }
        if us == Color::Black { self.fullmove -= 1; }
        self.side = us;
        self.castling = u.castling;
        self.ep = u.ep;
        self.halfmove = u.halfmove;
        self.hash = u.hash; // full restore — no incremental unmake hashing needed
    }

    pub fn has_non_pawn_material(&self, c: Color) -> bool {
        (self.bb[pc(c, KNIGHT)] | self.bb[pc(c, BISHOP)] | self.bb[pc(c, ROOK)] | self.bb[pc(c, QUEEN)]) != 0
    }

    pub fn make_null(&mut self) -> NullUndo {
        let zb = z();
        let u = NullUndo { ep: self.ep, hash: self.hash, halfmove: self.halfmove };
        if let Some(e) = self.ep {
            self.hash ^= zb.ep_file[file_of(e) as usize];
        }
        self.ep = None;
        self.halfmove += 1;
        self.side = self.side.flip();
        self.hash ^= zb.side;
        u
    }

    pub fn unmake_null(&mut self, u: NullUndo) {
        self.side = self.side.flip();
        self.ep = u.ep;
        self.hash = u.hash;
        self.halfmove = u.halfmove;
    }

    /// See `FeatureDelta`. Must be called on `self` *before* `self.make(m)` — it
    /// reads `piece_on(from)`/`piece_on(to)` against the pre-move position, mirroring
    /// exactly the same capture-square/promo-piece/castling-rook logic `make` itself
    /// uses (kept independent of `make` so this stays a pure, non-mutating read).
    pub fn feature_delta(&self, m: crate::moves::Move) -> FeatureDelta {
        use crate::moves::*;
        let us = self.side;
        let (from, to, flags) = (m.from(), m.to(), m.flags());
        let piece_pt = self.piece_on(from).expect("no piece on from-square") % 6;

        let mut removed = [Some((us, piece_pt, from)), None];
        let mut added = [Some((us, if m.is_promo() { m.promo_pt() } else { piece_pt }, to)), None];

        if m.is_capture() {
            let cap_sq = if flags == FLAG_EP {
                if us == Color::White { to - 8 } else { to + 8 }
            } else { to };
            let cap_pt = self.piece_on(cap_sq).expect("no captured piece") % 6;
            removed[1] = Some((us.flip(), cap_pt, cap_sq));
        } else if flags == FLAG_OO {
            let (rf, rt) = if us == Color::White { (7, 5) } else { (63, 61) };
            removed[1] = Some((us, ROOK, rf));
            added[1] = Some((us, ROOK, rt));
        } else if flags == FLAG_OOO {
            let (rf, rt) = if us == Color::White { (0, 3) } else { (56, 59) };
            removed[1] = Some((us, ROOK, rf));
            added[1] = Some((us, ROOK, rt));
        }

        FeatureDelta { removed, added }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    pub const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    #[test]
    fn fen_roundtrip() {
        let b = Board::from_fen(STARTPOS).unwrap();
        assert_eq!(b.to_fen(), STARTPOS);
        assert_eq!(b.side, Color::White);
        assert_eq!(b.castling, 15);
        assert_eq!(b.bb[pc(Color::White, PAWN)].count_ones(), 8);
        assert_eq!(b.piece_on(4), Some(pc(Color::White, KING)));
        let kiwi = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        assert_eq!(Board::from_fen(kiwi).unwrap().to_fen(), kiwi);
    }
    #[test]
    fn attacked_squares() {
        let b = Board::from_fen(STARTPOS).unwrap();
        assert!(b.attacked(20, Color::White));   // e3 attacked by white pawns
        assert!(!b.attacked(36, Color::White));  // e5 not attacked by white
        assert!(!b.in_check(Color::White));
        // Re1 vs Ke8 down the open e-file: Black is in check, White is not.
        let checked = Board::from_fen("4k3/8/8/8/8/8/8/4RK2 b - - 0 1").unwrap();
        assert!(checked.in_check(Color::Black));
        assert!(!checked.in_check(Color::White));
    }
    #[test]
    fn hash_differs() {
        let a = Board::from_fen(STARTPOS).unwrap();
        let c = Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").unwrap();
        assert_ne!(a.hash, c.hash);
        // Castling rights alone must change the hash.
        let no_castle = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1").unwrap();
        assert_ne!(a.hash, no_castle.hash);
        // En-passant square alone must change the hash (black just played d7-d5).
        let d5 = "rnbqkbnr/ppp1pppp/8/3p4/8/8/PPPPPPPP/RNBQKBNR w KQkq";
        let with_ep = Board::from_fen(&format!("{d5} d6 0 2")).unwrap();
        let without_ep = Board::from_fen(&format!("{d5} - 0 2")).unwrap();
        assert_ne!(with_ep.hash, without_ep.hash);
    }
    #[test]
    fn fen_rejects_malformed() {
        // 9 ranks: must not underflow the rank counter.
        assert!(Board::from_fen("8/8/8/8/8/8/8/8/8 w - - 0 1").is_err());
        // 9 files in a rank: must not shift off the board.
        assert!(Board::from_fen("rnbqkbnrr/8/8/8/8/8/8/8 w - - 0 1").is_err());
        // No kings: king_sq/in_check would misbehave downstream.
        assert!(Board::from_fen("8/8/8/8/8/8/8/8 w - - 0 1").is_err());
        // Missing side/castling/ep fields.
        assert!(Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR").is_err());
    }
    #[test]
    fn fen_rejects_back_rank_pawns() {
        // White pawn on e8 (kings a1/h8, non-adjacent).
        assert!(Board::from_fen("4P2k/8/8/8/8/8/8/K7 w - - 0 1").is_err());
        // Black pawn on e1 (kings a1/h8, non-adjacent).
        assert!(Board::from_fen("7k/8/8/8/8/8/8/K3p3 w - - 0 1").is_err());
        // Control: same material one rank inward parses fine.
        assert!(Board::from_fen("7k/4P3/8/8/8/8/4p3/K7 w - - 0 1").is_ok());
    }
    #[test]
    fn null_move_round_trips() {
        let mut b = Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").unwrap();
        let (h, side, ep) = (b.hash, b.side, b.ep);
        let u = b.make_null();
        assert_eq!(b.side, side.flip());
        assert_ne!(b.hash, h);
        b.unmake_null(u);
        assert_eq!(b.hash, h);
        assert_eq!(b.side, side);
        assert_eq!(b.ep, ep);
    }
    #[test]
    fn null_move_round_trips_with_ep() {
        // After 1.e4, the ep square is set (e3); this exercises the ep-file
        // XOR path in make_null/unmake_null that the Kiwipete FEN (no ep
        // square) does not touch.
        let mut b = Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1").unwrap();
        assert_eq!(b.ep, Some(sq(4, 2))); // e3
        let (h, side, ep) = (b.hash, b.side, b.ep);
        let u = b.make_null();
        assert_eq!(b.side, side.flip());
        assert_ne!(b.hash, h);
        b.unmake_null(u);
        assert_eq!(b.hash, h);
        assert_eq!(b.side, side);
        assert_eq!(b.ep, ep);
    }
    #[test]
    fn non_pawn_material_guard() {
        let kp = Board::from_fen("k7/8/8/8/8/8/4P3/K7 w - - 0 1").unwrap();
        assert!(!kp.has_non_pawn_material(Color::White));
        let kr = Board::from_fen("k7/8/8/8/8/8/8/K3R3 w - - 0 1").unwrap();
        assert!(kr.has_non_pawn_material(Color::White));
    }

    /// Shape-level coverage for `feature_delta` (NNUE Task 5's incremental-update
    /// input) independent of the `nnue` module: finds `uci` among the position's
    /// legal moves and asserts the exact `removed`/`added` triples it must produce.
    type FeatureSlots = Vec<(Color, usize, u8)>;
    fn delta_of(fen: &str, uci: &str) -> (FeatureSlots, FeatureSlots) {
        let b = Board::from_fen(fen).unwrap();
        let mut ms = Vec::new();
        crate::movegen::generate(&b, &mut ms);
        let m = ms.into_iter().find(|m| m.uci() == uci).expect("move not legal");
        let d = b.feature_delta(m);
        (d.removed.into_iter().flatten().collect(), d.added.into_iter().flatten().collect())
    }

    #[test]
    fn feature_delta_quiet_move() {
        let (removed, added) = delta_of(STARTPOS, "e2e4");
        assert_eq!(removed, vec![(Color::White, PAWN, sq(4, 1))]);
        assert_eq!(added, vec![(Color::White, PAWN, sq(4, 3))]);
    }

    #[test]
    fn feature_delta_capture() {
        // Kiwipete: Bxb4 (e2b4... use an actual capture in the position) — Nxd5 is
        // simplest: white knight c3 takes the black pawn... actually use the
        // pre-verified capture e5xd7? Simplest reliable capture: Nc3xb1? Not present.
        // Use e5xf7+ (knight takes f7 pawn) which the kiwipete perft suite recognizes.
        let (removed, added) = delta_of(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "e5f7",
        );
        assert_eq!(removed.len(), 2);
        assert!(removed.contains(&(Color::White, KNIGHT, sq(4, 4))));
        assert!(removed.contains(&(Color::Black, PAWN, sq(5, 6))));
        assert_eq!(added, vec![(Color::White, KNIGHT, sq(5, 6))]);
    }

    #[test]
    fn feature_delta_en_passant_removes_the_captured_pawns_own_square() {
        // "dxc6 e.p.": the captured black pawn sits on c5, not on the c6 target square.
        let (removed, added) = delta_of("8/8/8/2pP4/8/8/8/k6K w - c6 0 1", "d5c6");
        assert_eq!(removed.len(), 2);
        assert!(removed.contains(&(Color::White, PAWN, sq(3, 4))));
        assert!(removed.contains(&(Color::Black, PAWN, sq(2, 4)))); // c5, not c6
        assert_eq!(added, vec![(Color::White, PAWN, sq(2, 5))]); // c6
    }

    #[test]
    fn feature_delta_quiet_promotion() {
        let (removed, added) = delta_of("n1n5/1P6/8/8/8/8/8/k6K w - - 0 1", "b7b8q");
        assert_eq!(removed, vec![(Color::White, PAWN, sq(1, 6))]);
        assert_eq!(added, vec![(Color::White, QUEEN, sq(1, 7))]);
    }

    #[test]
    fn feature_delta_capture_promotion() {
        let (removed, added) = delta_of("n1n5/1P6/8/8/8/8/8/k6K w - - 0 1", "b7a8q");
        assert_eq!(removed.len(), 2);
        assert!(removed.contains(&(Color::White, PAWN, sq(1, 6))));
        assert!(removed.contains(&(Color::Black, KNIGHT, sq(0, 7))));
        assert_eq!(added, vec![(Color::White, QUEEN, sq(0, 7))]);
    }

    #[test]
    fn feature_delta_castling_moves_both_king_and_rook() {
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let (removed, added) = delta_of(fen, "e1g1"); // white O-O
        assert_eq!(removed.len(), 2);
        assert!(removed.contains(&(Color::White, KING, sq(4, 0))));
        assert!(removed.contains(&(Color::White, ROOK, sq(7, 0))));
        assert_eq!(added.len(), 2);
        assert!(added.contains(&(Color::White, KING, sq(6, 0))));
        assert!(added.contains(&(Color::White, ROOK, sq(5, 0))));

        // Same position, black to move, for black's O-O-O.
        let fen_black = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1";
        let (removed, added) = delta_of(fen_black, "e8c8"); // black O-O-O
        assert_eq!(removed.len(), 2);
        assert!(removed.contains(&(Color::Black, KING, sq(4, 7))));
        assert!(removed.contains(&(Color::Black, ROOK, sq(0, 7))));
        assert_eq!(added.len(), 2);
        assert!(added.contains(&(Color::Black, KING, sq(2, 7))));
        assert!(added.contains(&(Color::Black, ROOK, sq(3, 7))));
    }
}
