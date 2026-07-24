//! Self-play data generation (M6). Plays fixed-node games with the v0.4.0 engine,
//! filters to quiet positions, and streams White-relative `FEN | score | wdl` text
//! for bullet training. Reuses search/board/eval unchanged.

use crate::board::Board;
use crate::moves::Move;
use crate::movegen::generate;
use crate::search::{Limits, Searcher};
use crate::types::Color;

/// Deterministic, seedable xorshift64. Used ONLY for opening-move selection, so
/// datagen is reproducible per seed and decorrelated across worker seeds.
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        // xorshift cannot start from 0; substitute a fixed nonzero constant.
        Self { state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed } }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform-ish integer in `0..n` (modulo bias is negligible for the small `n`
    /// used in opening selection). Panics if `n == 0`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

pub struct SelfplayConfig {
    pub seed: u64,
    pub games: u64,
    pub nodes: u64,
    pub out: String,
    pub shard_size: usize,
    pub random_plies: usize,
    pub mb: usize,
    pub net: Option<String>,
}

#[derive(Clone, Copy, PartialEq)]
pub enum Outcome { BlackWin = 0, Draw = 1, WhiteWin = 2 }

/// One searched position, labeled White-relative. `white_score` is centipawns
/// from White's POV; the boolean flags feed the Task 4 quiet filter.
pub struct Cand {
    pub fen: String,
    pub white_score: i32,
    pub in_check: bool,
    pub best_is_capture: bool,
    pub best_gives_check: bool,
    pub is_mate_score: bool,
    pub hash: u64,
}

pub struct Game {
    pub cands: Vec<Cand>,
    pub outcome: Outcome,
}

const MATE_BOUND: i32 = 29_000; // mirrors search.rs

/// Legal moves (movegen is pseudo-legal; filter by make/in_check like uci.rs).
pub fn legal_moves(board: &Board) -> Vec<Move> {
    let mut pseudo = Vec::new();
    generate(board, &mut pseudo);
    let mut copy = board.clone();
    pseudo
        .into_iter()
        .filter(|&mv| {
            let undo = copy.make(mv);
            let legal = !copy.in_check(copy.side.flip());
            copy.unmake(mv, undo);
            legal
        })
        .collect()
}

/// True if `mv` gives check to the side that will be to move after it.
fn gives_check(board: &Board, mv: Move) -> bool {
    let mut copy = board.clone();
    let undo = copy.make(mv);
    let check = copy.in_check(copy.side); // side is now the opponent-to-move
    copy.unmake(mv, undo);
    check
}

pub fn play_game(searcher: &mut Searcher, rng: &mut XorShift64, cfg: &SelfplayConfig) -> Game {
    let mut board = Board::startpos();
    let mut hashes: Vec<u64> = vec![board.hash];

    // Opening diversification: `random_plies` uniform-random legal moves, unrecorded.
    for _ in 0..cfg.random_plies {
        let legal = legal_moves(&board);
        if legal.is_empty() { break; }
        let mv = legal[rng.below(legal.len())];
        board.make(mv);
        hashes.push(board.hash);
    }

    let mut cands = Vec::new();
    let limits = Limits { nodes: Some(cfg.nodes), ..Default::default() };

    // Adjudication counters (White-relative).
    let mut win_streak = 0i32;  // consecutive plies |white_score| >= 1000, same winning side
    let mut win_side_white = false;
    let mut draw_streak = 0i32;  // consecutive plies |white_score| <= 10 after move 40

    let outcome = loop {
        let legal = legal_moves(&board);
        if legal.is_empty() {
            break if board.in_check(board.side) {
                if board.side == Color::White { Outcome::BlackWin } else { Outcome::WhiteWin }
            } else { Outcome::Draw };
        }
        if board.halfmove >= 100 { break Outcome::Draw; }
        // Threefold: this position already occurred >= 2 times before now.
        if hashes.iter().filter(|&&h| h == board.hash).count() >= 3 { break Outcome::Draw; }

        let (best, stm_score) = searcher.think_scored(&mut board, &limits, &hashes);
        if best == Move::NONE { break Outcome::Draw; } // defensive: no move from search
        let white_score = if board.side == Color::White { stm_score } else { -stm_score };

        cands.push(Cand {
            fen: board.to_fen(),
            white_score,
            in_check: board.in_check(board.side),
            best_is_capture: best.is_capture(),
            best_gives_check: gives_check(&board, best),
            is_mate_score: stm_score.abs() >= MATE_BOUND,
            hash: board.hash,
        });

        // Adjudication bookkeeping.
        if white_score.abs() >= 1000 {
            let side = white_score > 0;
            if win_streak > 0 && side == win_side_white { win_streak += 1; }
            else { win_streak = 1; win_side_white = side; }
        } else { win_streak = 0; }
        if board.fullmove >= 40 && white_score.abs() <= 10 { draw_streak += 1; } else { draw_streak = 0; }

        board.make(best);
        hashes.push(board.hash);

        if win_streak >= 4 {
            break if win_side_white { Outcome::WhiteWin } else { Outcome::BlackWin };
        }
        if draw_streak >= 8 { break Outcome::Draw; }
    };

    Game { cands, outcome }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::Searcher;

    #[test]
    fn play_game_terminates_and_is_deterministic() {
        // Low node budget keeps the test fast; HCE searcher (no net needed for logic test).
        let cfg = SelfplayConfig {
            seed: 42, games: 1, nodes: 800, out: String::new(),
            shard_size: 1_000_000, random_plies: 4, mb: 8, net: None,
        };
        let mut s1 = Searcher::new(cfg.mb);
        let mut r1 = XorShift64::new(cfg.seed);
        let g1 = play_game(&mut s1, &mut r1, &cfg);

        let mut s2 = Searcher::new(cfg.mb);
        let mut r2 = XorShift64::new(cfg.seed);
        let g2 = play_game(&mut s2, &mut r2, &cfg);

        assert!(!g1.cands.is_empty(), "a game must produce candidate positions");
        assert!(matches!(g1.outcome, Outcome::WhiteWin | Outcome::BlackWin | Outcome::Draw));
        // Same seed + same engine ⇒ identical game.
        assert_eq!(g1.cands.len(), g2.cands.len(), "same seed must reproduce the game");
        assert_eq!(g1.outcome as u8, g2.outcome as u8);
    }

    #[test]
    fn prng_is_deterministic_per_seed_and_varies_across_seeds() {
        let mut a = XorShift64::new(12345);
        let seq_a: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        // Same seed reproduces the exact sequence.
        let mut a2 = XorShift64::new(12345);
        let seq_a2: Vec<u64> = (0..8).map(|_| a2.next_u64()).collect();
        assert_eq!(seq_a, seq_a2, "same seed must reproduce the sequence");
        // Different seed diverges.
        let mut b = XorShift64::new(67890);
        let seq_b: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
        assert_ne!(seq_a, seq_b, "different seeds must diverge");
        // No zero-lock, values vary.
        assert!(seq_a.iter().all(|&x| x != 0));
        assert!(seq_a[0] != seq_a[1]);
    }

    #[test]
    fn below_stays_in_range() {
        let mut r = XorShift64::new(1);
        for _ in 0..1000 {
            assert!(r.below(20) < 20);
            assert!(r.below(1) == 0);
        }
    }
}
