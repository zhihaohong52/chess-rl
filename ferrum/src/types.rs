pub type Bb = u64;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Color { White, Black }

impl Color {
    pub fn flip(self) -> Color {
        match self { Color::White => Color::Black, Color::Black => Color::White }
    }
    pub fn idx(self) -> usize { self as usize }
}

pub const PAWN: usize = 0;
pub const KNIGHT: usize = 1;
pub const BISHOP: usize = 2;
pub const ROOK: usize = 3;
pub const QUEEN: usize = 4;
pub const KING: usize = 5;

/// Index into 12-element piece-bitboard arrays.
pub fn pc(c: Color, pt: usize) -> usize { c.idx() * 6 + pt }

pub fn bb(sq: u8) -> Bb { 1u64 << sq }
/// Precondition: `b != 0` — on an empty bitboard this returns the bogus index 64; callers must guard.
pub fn lsb(b: Bb) -> u8 { b.trailing_zeros() as u8 }
/// Precondition: `b != 0` — on an empty bitboard this panics in debug (underflow) and returns the bogus index 255 in release; callers must guard.
pub fn msb(b: Bb) -> u8 { 63 - b.leading_zeros() as u8 }
/// Precondition: `*b != 0` — on an empty bitboard this panics in debug (underflow) and returns the bogus index 64 in release; callers must guard.
pub fn pop_lsb(b: &mut Bb) -> u8 { let s = lsb(*b); *b &= *b - 1; s }
pub fn file_of(sq: u8) -> u8 { sq & 7 }
pub fn rank_of(sq: u8) -> u8 { sq >> 3 }
pub fn sq(file: u8, rank: u8) -> u8 { rank * 8 + file }

pub fn sq_name(s: u8) -> String {
    format!("{}{}", (b'a' + file_of(s)) as char, rank_of(s) + 1)
}

pub fn parse_sq(s: &str) -> Option<u8> {
    let b = s.as_bytes();
    if b.len() != 2 { return None; }
    let f = b[0].wrapping_sub(b'a');
    let r = b[1].wrapping_sub(b'1');
    if f > 7 || r > 7 { return None; }
    Some(sq(f, r))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn square_math() {
        assert_eq!(sq(0, 0), 0);            // a1
        assert_eq!(sq(7, 7), 63);           // h8
        assert_eq!(sq(4, 0), 4);            // e1
        assert_eq!(file_of(12), 4);         // e2 -> file e
        assert_eq!(rank_of(12), 1);         // e2 -> rank 2
        assert_eq!(sq_name(36), "e5");
        assert_eq!(parse_sq("e5"), Some(36));
        assert_eq!(parse_sq("i9"), None);
        assert_eq!(parse_sq(""), None);
        assert_eq!(parse_sq("e"), None);
        assert_eq!(parse_sq("e55"), None);
        assert_eq!(parse_sq("a9"), None);
        assert_eq!(parse_sq("i1"), None);
    }
    #[test]
    fn bitboard_ops() {
        let mut b: Bb = bb(0) | bb(63);
        assert_eq!(lsb(b), 0);
        assert_eq!(pop_lsb(&mut b), 0);
        assert_eq!(pop_lsb(&mut b), 63);
        assert_eq!(b, 0);
        assert_eq!(msb(bb(0) | bb(63)), 63);
        assert_eq!(msb(bb(5)), 5);
    }
    #[test]
    fn piece_index() {
        assert_eq!(pc(Color::White, PAWN), 0);
        assert_eq!(pc(Color::Black, PAWN), 6);
        assert_eq!(pc(Color::Black, KING), 11);
        assert_eq!(Color::White.flip(), Color::Black);
    }
}
