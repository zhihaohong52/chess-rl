use crate::board::Board;
use crate::movegen::generate;

pub fn perft(b: &mut Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }

    let mut moves = Vec::with_capacity(64);
    generate(b, &mut moves);
    let mut nodes = 0;
    for m in moves {
        let undo = b.make(m);
        if !b.in_check(b.side.flip()) {
            nodes += if depth == 1 { 1 } else { perft(b, depth - 1) };
        }
        b.unmake(m, undo);
    }
    nodes
}

/// Print a per-root-move breakdown for debugging a perft mismatch.
pub fn perft_divide(b: &mut Board, depth: u32) {
    let mut moves = Vec::with_capacity(64);
    generate(b, &mut moves);
    let mut total = 0;
    for m in moves {
        let undo = b.make(m);
        if !b.in_check(b.side.flip()) {
            let nodes = if depth <= 1 { 1 } else { perft(b, depth - 1) };
            println!("{}: {}", m.uci(), nodes);
            total += nodes;
        }
        b.unmake(m, undo);
    }
    println!("total: {total}");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(fen: &str, depth: u32) -> u64 {
        perft(&mut Board::from_fen(fen).unwrap(), depth)
    }

    const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    const KIWI: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    const POS3: &str = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
    const POS4: &str = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
    const POS5: &str = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";
    const POS6: &str = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10";

    #[test]
    fn perft_shallow() {
        assert_eq!(p(START, 1), 20);
        assert_eq!(p(START, 2), 400);
        assert_eq!(p(START, 3), 8_902);
        assert_eq!(p(START, 4), 197_281);
        assert_eq!(p(KIWI, 1), 48);
        assert_eq!(p(KIWI, 2), 2_039);
        assert_eq!(p(KIWI, 3), 97_862);
        assert_eq!(p(POS3, 3), 2_812);
        assert_eq!(p(POS3, 5), 674_624);
        assert_eq!(p(POS4, 3), 9_467);
        assert_eq!(p(POS5, 3), 62_379);
        assert_eq!(p(POS6, 3), 89_890);
    }

    #[test]
    #[ignore]
    fn perft_deep() {
        assert_eq!(p(START, 5), 4_865_609);
        assert_eq!(p(START, 6), 119_060_324);
        assert_eq!(p(KIWI, 4), 4_085_603);
        assert_eq!(p(POS3, 6), 11_030_083);
        assert_eq!(p(POS4, 4), 422_333);
        assert_eq!(p(POS5, 4), 2_103_487);
        assert_eq!(p(POS6, 4), 3_894_594);
    }
}
