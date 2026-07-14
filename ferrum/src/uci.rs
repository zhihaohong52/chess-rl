use crate::board::Board;
use crate::movegen::generate;
use crate::moves::Move;
use crate::search::{Limits, Searcher};
use std::io::BufRead;

pub fn run() {
    let stdin = std::io::stdin();
    let mut board = Board::startpos();
    let mut history = Vec::new();
    let mut searcher = Searcher::new(64);

    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let tokens: Vec<_> = line.split_whitespace().collect();
        match tokens.first().copied() {
            Some("uci") => println!("id name ferrum 0.1.0\nid author James\noption name Hash type spin default 64 min 1 max 4096\nuciok"),
            Some("isready") => println!("readyok"),
            Some("setoption") if tokens.get(2) == Some(&"Hash") => {
                if let Some(Ok(mb)) = tokens.get(4).map(|value| value.parse()) {
                    searcher = Searcher::new(mb);
                }
            }
            Some("ucinewgame") => {
                searcher.tt.clear();
                board = Board::startpos();
                history.clear();
            }
            Some("position") => set_position(&tokens, &mut board, &mut history),
            Some("go") => {
                let limits = parse_limits(&tokens);
                let best = searcher.think(&mut board, &limits, &history);
                println!("bestmove {}", if best == Move::NONE { "0000".into() } else { best.uci() });
            }
            Some("quit") => break,
            _ => {}
        }
    }
}

fn set_position(tokens: &[&str], board: &mut Board, history: &mut Vec<u64>) {
    let mut index = 1;
    if tokens.get(index) == Some(&"startpos") {
        *board = Board::startpos();
        index += 1;
    } else if tokens.get(index) == Some(&"fen") {
        let end = tokens[index + 1..].iter().position(|&token| token == "moves")
            .map(|offset| index + 1 + offset).unwrap_or(tokens.len());
        if let Ok(parsed) = Board::from_fen(&tokens[index + 1..end].join(" ")) { *board = parsed; }
        index = end;
    }
    history.clear();
    history.push(board.hash);
    if tokens.get(index) == Some(&"moves") {
        for uci in &tokens[index + 1..] {
            if let Some(mv) = find_move(board, uci) {
                board.make(mv);
                history.push(board.hash);
            }
        }
    }
}

fn parse_limits(tokens: &[&str]) -> Limits {
    let mut limits = Limits::default();
    let mut index = 1;
    while index < tokens.len() {
        let value = tokens.get(index + 1).and_then(|value| value.parse::<u64>().ok());
        match tokens[index] {
            "depth" => limits.depth = value.map(|value| value as u32),
            "movetime" => limits.movetime = value,
            "wtime" => limits.wtime = value,
            "btime" => limits.btime = value,
            "winc" => limits.winc = value,
            "binc" => limits.binc = value,
            _ => { index += 1; continue; }
        }
        index += 2;
    }
    if limits.depth.is_none() && limits.movetime.is_none() && limits.wtime.is_none() && limits.btime.is_none() {
        limits.depth = Some(6);
    }
    limits
}

fn find_move(board: &Board, uci: &str) -> Option<Move> {
    let mut moves = Vec::new();
    generate(board, &mut moves);
    let mut copy = board.clone();
    moves.into_iter().find(|mv| {
        if mv.uci() != uci { return false; }
        let undo = copy.make(*mv);
        let legal = !copy.in_check(copy.side.flip());
        copy.unmake(*mv, undo);
        legal
    })
}
