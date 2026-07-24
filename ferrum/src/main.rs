#![allow(dead_code)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::possible_missing_else)]

mod types;
mod attacks;
mod bench;
mod zobrist;
mod board;
mod eval;
mod nnue;
mod moves;
mod movegen;
mod perft;
mod search;
mod selfplay;
mod tt;
mod uci;

use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("perft") => {
            let depth = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
            let fen = args.get(3).cloned().unwrap_or_else(|| {
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".into()
            });
            let mut board = board::Board::from_fen(&fen).expect("bad FEN");
            let start = std::time::Instant::now();
            let nodes = perft::perft(&mut board, depth);
            let elapsed = start.elapsed().as_secs_f64();
            println!("perft({depth}) = {nodes}  ({:.2} Mnps)", nodes as f64 / elapsed / 1e6);
        }
        Some("divide") => {
            let depth = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
            let fen = args.get(3).cloned().unwrap_or_else(|| {
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".into()
            });
            let mut board = board::Board::from_fen(&fen).expect("bad FEN");
            perft::perft_divide(&mut board, depth);
        }
        Some("bench") => bench::bench(),
        Some("selfplay") => selfplay::main_selfplay(&args),
        _ => uci::run(),
    }
}
