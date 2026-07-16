use crate::{board::Board, eval::{Eval, Hce, MATERIAL}, movegen::generate, moves::*, tt::*, types::*};
use std::time::{Duration, Instant};
pub const MATE: i32 = 30_000;
const MATE_BOUND: i32 = MATE - 1_000;
#[derive(Default)] pub struct Limits { pub depth: Option<u32>, pub movetime: Option<u64>, pub wtime: Option<u64>, pub btime: Option<u64>, pub winc: Option<u64>, pub binc: Option<u64> }
pub struct Searcher { pub tt: Tt, nodes: u64, deadline: Option<Instant>, stopped: bool, history: Vec<u64>, eval: Hce }
impl Searcher {
    pub fn new(mb: usize) -> Self { Self { tt:Tt::new(mb), nodes:0, deadline:None, stopped:false, history:Vec::new(), eval:Hce } }
    pub fn node_count(&self) -> u64 { self.nodes }
    pub fn think(
        &mut self,
        board: &mut Board,
        limits: &Limits,
        history: &[u64],
    ) -> Move {
        self.nodes = 0;
        self.stopped = false;
        self.history = history.to_vec();
        self.deadline = deadline(board, limits);

        let mut best = Move::NONE;
        let mut prev_score = 0i32;
        for depth in 1..=limits.depth.unwrap_or(64) {
            let score = self.aspiration(board, depth, prev_score);
            if self.stopped {
                break;
            }
            prev_score = score;
            if let Some(e) = self.tt.probe(board.hash) {
                best = e.mv;
            }
            println!(
                "info depth {depth} score {} nodes {} pv {}",
                score_text(score),
                self.nodes,
                if best == Move::NONE {
                    "(none)".into()
                } else {
                    best.uci()
                }
            );
            if best == Move::NONE || score.abs() > MATE_BOUND {
                break;
            }
        }
        best
    }

    fn aspiration(&mut self, b: &mut Board, depth: u32, prev: i32) -> i32 {
        if depth < 4 || prev.abs() > MATE_BOUND {
            return self.negamax(b, depth as i32, -MATE, MATE, 0);
        }

        let mut delta = 25;
        let mut alpha = prev.saturating_sub(delta).max(-MATE);
        let mut beta = prev.saturating_add(delta).min(MATE);
        loop {
            let score = self.negamax(b, depth as i32, alpha, beta, 0);
            if self.stopped || (score > alpha && score < beta) {
                return score;
            }
            if score <= alpha {
                if alpha == -MATE {
                    return score;
                }
                alpha = alpha.saturating_sub(delta).max(-MATE);
            } else {
                debug_assert!(score >= beta);
                if beta == MATE {
                    return score;
                }
                beta = beta.saturating_add(delta).min(MATE);
            }
            delta = delta.saturating_add(delta / 2).min(MATE);
        }
    }

    fn timed_out(&mut self) -> bool { if self.nodes & 2047 == 0 { if self.deadline.is_some_and(|d| Instant::now() >= d) { self.stopped=true; } } self.stopped }
    fn repeated(&self, hash:u64)->bool { self.history.iter().rev().skip(1).step_by(2).any(|&h|h==hash) }

    fn negamax(
        &mut self,
        b: &mut Board,
        depth: i32,
        mut alpha: i32,
        beta: i32,
        ply: i32,
    ) -> i32 {
        self.nodes += 1;
        if self.timed_out() {
            return 0;
        }
        if ply > 0 && (b.halfmove >= 100 || self.repeated(b.hash)) {
            return 0;
        }
        if depth <= 0 {
            return self.qsearch(b, alpha, beta);
        }

        let alpha0 = alpha;
        let tt_entry = self.tt.probe(b.hash);
        let tt_move = tt_entry.map(|e| e.mv).unwrap_or(Move::NONE);
        if let Some(e) = tt_entry {
            if ply > 0 && e.depth as i32 >= depth {
                let score = from_tt(e.score, ply);
                if e.bound == BOUND_EXACT
                    || e.bound == BOUND_LOWER && score >= beta
                    || e.bound == BOUND_UPPER && score <= alpha
                {
                    return score;
                }
            }
        }

        let mut moves = Vec::with_capacity(64);
        generate(b, &mut moves);
        order(b, &mut moves, tt_move);

        let mut legal = 0;
        let mut best = -MATE - 1;
        let mut best_move = Move::NONE;
        for m in moves {
            let undo = b.make(m);
            if b.in_check(b.side.flip()) {
                b.unmake(m, undo);
                continue;
            }
            legal += 1;
            self.history.push(b.hash);
            let score = -self.negamax(b, depth - 1, -beta, -alpha, ply + 1);
            self.history.pop();
            b.unmake(m, undo);
            if self.stopped {
                return 0;
            }
            if score > best {
                best = score;
                best_move = m;
            }
            if score > alpha {
                alpha = score;
                if alpha >= beta {
                    break;
                }
            }
        }

        if legal == 0 {
            return if b.in_check(b.side) { -MATE + ply } else { 0 };
        }
        let bound = if best <= alpha0 {
            BOUND_UPPER
        } else if best >= beta {
            BOUND_LOWER
        } else {
            BOUND_EXACT
        };
        self.tt.store(b.hash, best_move, to_tt(best, ply), depth as i8, bound);
        best
    }
    fn qsearch(&mut self,b:&mut Board,mut alpha:i32,beta:i32)->i32 { self.nodes+=1;if self.timed_out(){return 0} let stand=self.eval.eval(b);if stand>=beta{return stand} if stand>alpha{alpha=stand} let mut moves=Vec::new();generate(b,&mut moves);moves.retain(|m|m.is_capture()||m.is_promo());order(b,&mut moves,Move::NONE);for m in moves{let u=b.make(m);if b.in_check(b.side.flip()){b.unmake(m,u);continue}let score=-self.qsearch(b,-beta,-alpha);b.unmake(m,u);if score>alpha{alpha=score;if alpha>=beta{break}}}alpha }
}
fn order(b:&Board,moves:&mut [Move],tt:Move){moves.sort_by_key(|m|if *m==tt{-1_000_000}else if m.is_capture(){let v=b.piece_on(m.to()).map(|p|MATERIAL[p%6]).unwrap_or(100);let a=b.piece_on(m.from()).map(|p|MATERIAL[p%6]).unwrap_or(0);-(10_000+v*10-a)}else if m.is_promo(){-9_000}else{0})}
fn to_tt(s:i32,ply:i32)->i32{if s>MATE_BOUND{s+ply}else if s < -MATE_BOUND{s-ply}else{s}} fn from_tt(s:i32,ply:i32)->i32{if s>MATE_BOUND{s-ply}else if s < -MATE_BOUND{s+ply}else{s}}
fn score_text(s:i32)->String{if s.abs()>MATE_BOUND{format!("mate {}",if s>0{(MATE-s+1)/2}else{-((MATE+s+1)/2)})}else{format!("cp {s}")}}
fn deadline(b:&Board,l:&Limits)->Option<Instant>{if let Some(ms)=l.movetime{return Some(Instant::now()+Duration::from_millis(ms.saturating_sub(20)))}let(time,inc)=if b.side==Color::White{(l.wtime,l.winc)}else{(l.btime,l.binc)};let time=time?;Some(Instant::now()+Duration::from_millis((time/25+inc.unwrap_or(0)/2).max(10).min(time.saturating_sub(50).max(10))))}
#[cfg(test)] mod tests { use super::*; fn best(f:&str,d:u32)->String{let mut b=Board::from_fen(f).unwrap();Searcher::new(4).think(&mut b,&Limits{depth:Some(d),..Default::default()},&[]).uci()} #[test]fn finds_mate_in_1(){assert_eq!(best("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1",4),"e1e8")} #[test]fn takes_free_queen(){assert_eq!(best("k7/8/8/3q4/8/2N5/8/K7 w - - 0 1",4),"c3d5")} #[test]fn stalemate_is_none(){let mut b=Board::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap();assert_eq!(Searcher::new(1).think(&mut b,&Limits{depth:Some(3),..Default::default()},&[]),Move::NONE)}
    #[test]
    fn aspiration_still_finds_winning_capture() {
        assert_eq!(
            best("k7/8/8/3q4/8/2N5/8/K7 w - - 0 1", 6),
            "c3d5"
        );
    }

    #[test]
    fn aspiration_returns_at_a_mate_bound() {
        let mut b = Board::from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1").unwrap();
        let mut s = Searcher::new(1);
        assert_eq!(s.aspiration(&mut b, 4, 0), -MATE);
    }
}
