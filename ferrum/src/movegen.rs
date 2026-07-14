use crate::attacks::*;
use crate::board::Board;
use crate::moves::*;
use crate::types::*;

pub fn generate(b: &Board, list: &mut Vec<Move>) {
    let us = b.side;
    let them = us.flip();
    let own = b.occ[us.idx()];
    let opp = b.occ[them.idx()];
    let all = own | opp;

    // --- pawns ---
    let pawns = b.bb[pc(us, PAWN)];
    let (push, promo_rank): (i8, u8) = if us == Color::White { (8, 7) } else { (-8, 0) };
    let mut p = pawns;
    while p != 0 {
        let from = pop_lsb(&mut p);
        let to1 = (from as i8 + push) as u8;
        if all & bb(to1) == 0 {
            if rank_of(to1) == promo_rank {
                for n in 0..4 { list.push(Move::new(from, to1, FLAG_PROMO + n)); }
            } else {
                list.push(Move::new(from, to1, FLAG_QUIET));
                let start_rank = if us == Color::White { 1 } else { 6 };
                if rank_of(from) == start_rank {
                    let to2 = (from as i8 + 2 * push) as u8;
                    if all & bb(to2) == 0 {
                        list.push(Move::new(from, to2, FLAG_DPP));
                    }
                }
            }
        }
        let mut caps = pawn_attacks(us, from) & opp;
        while caps != 0 {
            let to = pop_lsb(&mut caps);
            if rank_of(to) == promo_rank {
                for n in 0..4 { list.push(Move::new(from, to, FLAG_PROMO + FLAG_CAP + n)); }
            } else {
                list.push(Move::new(from, to, FLAG_CAP));
            }
        }
        if let Some(ep) = b.ep {
            if pawn_attacks(us, from) & bb(ep) != 0 {
                list.push(Move::new(from, ep, FLAG_EP));
            }
        }
    }

    // --- knights, king ---
    for (pt, att) in [
        (KNIGHT, knight_attacks as fn(u8) -> Bb),
        (KING, king_attacks as fn(u8) -> Bb),
    ] {
        let mut pieces = b.bb[pc(us, pt)];
        while pieces != 0 {
            let from = pop_lsb(&mut pieces);
            push_targets(list, from, att(from) & !own, opp);
        }
    }
    // --- sliders ---
    for pt in [BISHOP, ROOK, QUEEN] {
        let mut pieces = b.bb[pc(us, pt)];
        while pieces != 0 {
            let from = pop_lsb(&mut pieces);
            let a = match pt {
                BISHOP => bishop_attacks(from, all),
                ROOK => rook_attacks(from, all),
                _ => queen_attacks(from, all),
            };
            push_targets(list, from, a & !own, opp);
        }
    }

    // --- castling (fully legal when emitted) ---
    let (ks_bit, qs_bit, king_from) = if us == Color::White { (1u8, 2u8, 4u8) } else { (4u8, 8u8, 60u8) };
    if b.bb[pc(us, KING)] & bb(king_from) != 0 && !b.in_check(us) {
        // king side: f/g empty, f/g not attacked (e already checked via !in_check)
        if b.castling & ks_bit != 0
            && all & (bb(king_from + 1) | bb(king_from + 2)) == 0
            && !b.attacked(king_from + 1, them) && !b.attacked(king_from + 2, them)
        {
            list.push(Move::new(king_from, king_from + 2, FLAG_OO));
        }
        // queen side: b/c/d empty, c/d not attacked
        if b.castling & qs_bit != 0
            && all & (bb(king_from - 1) | bb(king_from - 2) | bb(king_from - 3)) == 0
            && !b.attacked(king_from - 1, them) && !b.attacked(king_from - 2, them)
        {
            list.push(Move::new(king_from, king_from - 2, FLAG_OOO));
        }
    }
}

fn push_targets(list: &mut Vec<Move>, from: u8, mut targets: Bb, opp: Bb) {
    while targets != 0 {
        let to = pop_lsb(&mut targets);
        let fl = if opp & bb(to) != 0 { FLAG_CAP } else { FLAG_QUIET };
        list.push(Move::new(from, to, fl));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    fn count_legal(fen: &str) -> usize {
        let mut b = Board::from_fen(fen).unwrap();
        let mut ms = Vec::new();
        generate(&b, &mut ms);
        ms.into_iter().filter(|&m| {
            let u = b.make(m);
            let ok = !b.in_check(b.side.flip());
            b.unmake(m, u);
            ok
        }).count()
    }

    #[test]
    fn startpos_20_moves() {
        assert_eq!(count_legal("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"), 20);
    }
    #[test]
    fn kiwipete_48_moves() {
        assert_eq!(count_legal("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"), 48);
    }
    #[test]
    fn promotions_and_ep() {
        // white pawn on b7 promotes (4) + capture-promos onto a8/c8 knights (8) + white king a1... careful: kings must exist.
        assert_eq!(count_legal("n1n5/1P6/8/8/8/8/8/k6K w - - 0 1"), 4 + 8 + 3); // +3 king moves (h1: g1,g2,h2)
        // en passant is generated and legal
        assert_eq!(count_legal("8/8/8/2pP4/8/8/8/k6K w - c6 0 1"), 2 + 3);      // d6 push, dxc6 ep + 3 king moves
    }
    #[test]
    fn make_unmake_restores_everything() {
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let mut b = Board::from_fen(fen).unwrap();
        let (h0, f0) = (b.hash, b.to_fen());
        let mut ms = Vec::new();
        generate(&b, &mut ms);
        for m in ms {
            let u = b.make(m);
            b.unmake(m, u);
            assert_eq!(b.hash, h0, "hash broken by {}", m.uci());
            assert_eq!(b.to_fen(), f0, "fen broken by {}", m.uci());
        }
    }
}
