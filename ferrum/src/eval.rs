use crate::board::Board;
use crate::types::*;

pub trait Eval { fn eval(&self, board: &Board) -> i32; }

pub const MATERIAL: [i32; 6] = [100, 320, 330, 500, 900, 0];
pub struct Hce;

fn pst(piece: usize, square: u8, color: Color) -> i32 {
    let s = if color == Color::White { square ^ 56 } else { square };
    let file = (s & 7) as i32;
    let rank = (s >> 3) as i32;
    let center = 6 - (file - 3).abs() - (rank - 3).abs();
    match piece {
        PAWN => rank * 5 + if file == 3 || file == 4 { 4 } else { 0 },
        KNIGHT => center * 7 - if file == 0 || file == 7 { 12 } else { 0 },
        BISHOP => center * 3,
        ROOK => rank * 2,
        QUEEN => center * 2,
        KING => -center * 3,
        _ => 0,
    }
}

impl Eval for Hce {
    fn eval(&self, board: &Board) -> i32 {
        let mut score = 0;
        for piece in 0..6 {
            for color in [Color::White, Color::Black] {
                let mut pieces = board.bb[pc(color, piece)];
                while pieces != 0 {
                    let square = pop_lsb(&mut pieces);
                    let value = MATERIAL[piece] + pst(piece, square, color);
                    if color == Color::White { score += value; } else { score -= value; }
                }
            }
        }
        if board.side == Color::White { score } else { -score }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn startpos_symmetric() { assert_eq!(Hce.eval(&Board::startpos()), 0); }
    #[test]
    fn material_dominates() {
        let white = Board::from_fen("k7/8/8/8/8/8/8/KQ6 w - - 0 1").unwrap();
        let black = Board::from_fen("k7/8/8/8/8/8/8/KQ6 b - - 0 1").unwrap();
        assert!(Hce.eval(&white) > 800);
        assert!(Hce.eval(&black) < -800);
    }
    #[test]
    fn pst_prefers_center_knight() {
        let center = Board::from_fen("k7/8/8/8/4N3/8/8/K7 w - - 0 1").unwrap();
        let corner = Board::from_fen("k7/8/8/8/8/8/8/K6N w - - 0 1").unwrap();
        assert!(Hce.eval(&center) > Hce.eval(&corner));
    }
}
