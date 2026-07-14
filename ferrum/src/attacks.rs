use std::sync::OnceLock;
use crate::types::*;

struct Tables {
    knight: [Bb; 64],
    king: [Bb; 64],
    pawn: [[Bb; 64]; 2], // [color][sq]
    rays: [[Bb; 64]; 8], // [dir][sq]
}

static TABLES: OnceLock<Tables> = OnceLock::new();

// (df, dr) per direction index; positive = attacked squares have higher index
const DIRS: [(i8, i8); 8] = [
    (0, 1), (0, -1), (1, 0), (-1, 0),   // N S E W
    (1, 1), (-1, 1), (1, -1), (-1, -1), // NE NW SE SW
];
const POSITIVE: [bool; 8] = [true, false, true, false, true, true, false, false];

fn on_board(f: i8, r: i8) -> bool { (0..8).contains(&f) && (0..8).contains(&r) }

fn build() -> Tables {
    let mut t = Tables {
        knight: [0; 64], king: [0; 64], pawn: [[0; 64]; 2], rays: [[0; 64]; 8],
    };
    for s in 0u8..64 {
        let (f, r) = (file_of(s) as i8, rank_of(s) as i8);
        for (df, dr) in [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)] {
            if on_board(f + df, r + dr) { t.knight[s as usize] |= bb(sq((f + df) as u8, (r + dr) as u8)); }
        }
        for (df, dr) in DIRS {
            if on_board(f + df, r + dr) { t.king[s as usize] |= bb(sq((f + df) as u8, (r + dr) as u8)); }
        }
        for (c, dr) in [(0usize, 1i8), (1usize, -1i8)] {
            for df in [-1i8, 1] {
                if on_board(f + df, r + dr) { t.pawn[c][s as usize] |= bb(sq((f + df) as u8, (r + dr) as u8)); }
            }
        }
        for (d, &(df, dr)) in DIRS.iter().enumerate() {
            let (mut cf, mut cr) = (f + df, r + dr);
            while on_board(cf, cr) {
                t.rays[d][s as usize] |= bb(sq(cf as u8, cr as u8));
                cf += df; cr += dr;
            }
        }
    }
    t
}

fn tables() -> &'static Tables { TABLES.get_or_init(build) }

pub fn knight_attacks(s: u8) -> Bb { tables().knight[s as usize] }
pub fn king_attacks(s: u8) -> Bb { tables().king[s as usize] }
pub fn pawn_attacks(c: Color, s: u8) -> Bb { tables().pawn[c.idx()][s as usize] }

/// Attacks along one ray direction from `s`, cut at the first blocker in `occ`.
/// `occ` must be the union of ALL pieces (both colors). The returned set INCLUDES
/// the nearest blocker's square whether friend or enemy.
fn ray_attack(t: &Tables, dir: usize, s: u8, occ: Bb) -> Bb {
    let ray = t.rays[dir][s as usize];
    let blockers = ray & occ;
    if blockers == 0 { return ray; }
    let blocker = if POSITIVE[dir] { lsb(blockers) } else { msb(blockers) };
    ray ^ t.rays[dir][blocker as usize]
}

/// Rook attacks from `s`. `occ` must be the union of ALL pieces (both colors);
/// the result INCLUDES the nearest blocker's square whether friend or enemy,
/// so callers must mask out own pieces for move generation.
pub fn rook_attacks(s: u8, occ: Bb) -> Bb {
    let t = tables();
    ray_attack(t, 0, s, occ) | ray_attack(t, 1, s, occ) | ray_attack(t, 2, s, occ) | ray_attack(t, 3, s, occ)
}
/// Bishop attacks from `s`. `occ` must be the union of ALL pieces (both colors);
/// the result INCLUDES the nearest blocker's square whether friend or enemy,
/// so callers must mask out own pieces for move generation.
pub fn bishop_attacks(s: u8, occ: Bb) -> Bb {
    let t = tables();
    ray_attack(t, 4, s, occ) | ray_attack(t, 5, s, occ) | ray_attack(t, 6, s, occ) | ray_attack(t, 7, s, occ)
}
/// Queen attacks from `s`. Same `occ` contract as `rook_attacks`/`bishop_attacks`:
/// `occ` is the union of ALL pieces; the result includes the nearest blockers.
pub fn queen_attacks(s: u8, occ: Bb) -> Bb { rook_attacks(s, occ) | bishop_attacks(s, occ) }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn leapers() {
        assert_eq!(knight_attacks(28).count_ones(), 8);          // e4: 8 targets
        assert_eq!(knight_attacks(0), bb(10) | bb(17));          // a1: b3, c2
        assert_eq!(king_attacks(0).count_ones(), 3);             // a1: a2,b1,b2
        assert_eq!(pawn_attacks(Color::White, 12), bb(19) | bb(21)); // e2 -> d3,f3
        assert_eq!(pawn_attacks(Color::Black, 51), bb(42) | bb(44)); // d7 -> c6,e6
    }
    #[test]
    fn sliders() {
        assert_eq!(rook_attacks(0, 0).count_ones(), 14);         // a1, empty board
        // rook a1 with blocker on a3: attacks a2,a3 + b1..h1 = 9 squares
        assert_eq!(rook_attacks(0, bb(16)).count_ones(), 9);
        assert_eq!(bishop_attacks(27, 0).count_ones(), 13);      // d4, empty board
        // bishop d4, blocker f6: NE ray stops at f6
        let a = bishop_attacks(27, bb(45));
        assert!(a & bb(45) != 0 && a & bb(54) == 0);             // hits f6, not g7
        assert_eq!(queen_attacks(27, 0).count_ones(), 27);       // d4, empty board
        // Negative-direction (msb) branch coverage:
        // rook h8, blocker h6: south ray cut (h7,h6) + west ray full (g8..a8) = 9
        assert_eq!(rook_attacks(63, bb(47)).count_ones(), 9);
        // bishop d4, blocker b2: SW ray hits b2, not a1
        let a = bishop_attacks(27, bb(9));
        assert!(a & bb(9) != 0 && a & bb(0) == 0);
        // rook e4 blocked e2(south) & e6(north): 4 vertical + 7 horizontal = 11
        assert_eq!(rook_attacks(28, bb(12) | bb(44)).count_ones(), 11);
    }
}
