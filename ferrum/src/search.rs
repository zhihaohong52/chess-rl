use crate::{board::Board, eval::{Eval, Hce, MATERIAL}, movegen::generate, moves::*, tt::*, types::*};
use std::time::{Duration, Instant};
pub const MATE: i32 = 30_000;
const MATE_BOUND: i32 = MATE - 1_000;
const MAX_PLY: usize = 128;

#[cfg(test)]
std::thread_local! {
    static ASPIRATION_RETRIES: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn reset_aspiration_retries() {
    ASPIRATION_RETRIES.with(|retries| retries.set(0));
}

#[cfg(test)]
fn aspiration_retries() -> u32 {
    ASPIRATION_RETRIES.with(std::cell::Cell::get)
}
#[derive(Default)] pub struct Limits { pub depth: Option<u32>, pub movetime: Option<u64>, pub wtime: Option<u64>, pub btime: Option<u64>, pub winc: Option<u64>, pub binc: Option<u64> }
pub struct Searcher { pub tt: Tt, nodes: u64, deadline: Option<Instant>, stopped: bool, history: Vec<u64>, killers: [[Move; 2]; MAX_PLY], eval: Hce }
impl Searcher {
    pub fn new(mb: usize) -> Self { Self { tt:Tt::new(mb), nodes:0, deadline:None, stopped:false, history:Vec::new(), killers: [[Move::NONE; 2]; MAX_PLY], eval:Hce } }
    pub fn node_count(&self) -> u64 { self.nodes }
    pub fn think(
        &mut self,
        board: &mut Board,
        limits: &Limits,
        history: &[u64],
    ) -> Move {
        self.nodes = 0;
        self.killers = [[Move::NONE; 2]; MAX_PLY];
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
                #[cfg(test)]
                ASPIRATION_RETRIES.with(|retries| {
                    retries.set(retries.get().saturating_add(1));
                });
                alpha = alpha.saturating_sub(delta).max(-MATE);
            } else {
                debug_assert!(score >= beta);
                if beta == MATE {
                    return score;
                }
                #[cfg(test)]
                ASPIRATION_RETRIES.with(|retries| {
                    retries.set(retries.get().saturating_add(1));
                });
                beta = beta.saturating_add(delta).min(MATE);
            }
            delta = delta.saturating_add(delta / 2).min(MATE);
        }
    }

    fn timed_out(&mut self) -> bool { if self.nodes & 2047 == 0 { if self.deadline.is_some_and(|d| Instant::now() >= d) { self.stopped=true; } } self.stopped }
    fn repeated(&self, hash:u64)->bool { self.history.iter().rev().skip(1).step_by(2).any(|&h|h==hash) }

    fn store_killer(&mut self, ply: i32, m: Move) {
        let p = (ply as usize).min(MAX_PLY - 1);
        if self.killers[p][0] != m {
            self.killers[p][1] = self.killers[p][0];
            self.killers[p][0] = m;
        }
    }

    fn order_moves(&self, b: &Board, moves: &mut [Move], tt: Move, ply: i32) {
        let p = (ply as usize).min(MAX_PLY - 1);
        moves.sort_by_key(|m| {
            if *m == tt {
                -2_000_000
            } else if m.is_capture() {
                let victim = b.piece_on(m.to()).map(|piece| MATERIAL[piece % 6]).unwrap_or(100);
                let attacker = b.piece_on(m.from()).map(|piece| MATERIAL[piece % 6]).unwrap_or(0);
                -(1_000_000 + victim * 10 - attacker)
            } else if m.is_promo() {
                -900_000
            } else if *m == self.killers[p][0] || *m == self.killers[p][1] {
                -800_000
            } else {
                0
            }
        });
    }

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
        if ply >= MAX_PLY as i32 {
            return self.qsearch(b, alpha, beta);
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

        let in_check = b.in_check(b.side);
        let static_eval = if in_check { 0 } else { self.eval.eval(b) };
        // Reverse futility pruning (static null-move): at shallow depth, if the static
        // eval beats beta by a depth-scaled margin, assume the node holds and prune.
        // Fail-soft: returns the static eval.
        if depth <= 6 && !in_check && beta.abs() < MATE_BOUND && static_eval - 80 * depth >= beta {
            return static_eval;
        }

        // Null-move pruning: skip a turn and see if the position is still >= beta.
        // Guarded against check and likely-zugzwang (side must have non-pawn material).
        // Eval-gated: only attempted when static_eval already looks >= beta.
        // Fail-hard: returns beta (not the null score) on cutoff.
        if depth >= 3 && ply > 0 && beta.abs() < MATE_BOUND
            && !in_check && static_eval >= beta && b.has_non_pawn_material(b.side)
        {
            let r = 2 + depth / 4;
            let u = b.make_null();
            self.history.push(b.hash);
            let s = -self.negamax(b, depth - 1 - r, -beta, -beta + 1, ply + 1);
            self.history.pop();
            b.unmake_null(u);
            if self.stopped {
                return 0;
            }
            if s >= beta {
                return beta;
            }
        }

        let mut moves = Vec::with_capacity(64);
        generate(b, &mut moves);
        self.order_moves(b, &mut moves, tt_move, ply);

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
            let gives_check = b.in_check(b.side);
            let quiet = !m.is_capture() && !m.is_promo();
            // late move pruning: skip late quiets at shallow depth once we have a real best
            if legal > 1 && quiet && !in_check && !gives_check && best > -MATE_BOUND
                && depth <= 3 && legal > 4 + depth * depth
            {
                b.unmake(m, undo);
                legal -= 1;                 // this move was not actually searched
                continue;
            }
            // futility pruning: skip late quiets whose static eval can't reach alpha
            if legal > 1 && quiet && !in_check && !gives_check && best > -MATE_BOUND
                && depth <= 4 && static_eval + 100 * depth <= alpha
            {
                b.unmake(m, undo);
                legal -= 1;
                continue;
            }
            self.history.push(b.hash);
            // Late move reductions (LMR): late quiet, non-checking moves at depth >= 3
            // (when not already in check) get a shallower search first — reduced by
            // 1 ply, or 2 once legal > 6. This is full-window, not PVS/null-window:
            // both the reduced probe and the full-depth re-search use the same
            // [-beta, -alpha] window, and the re-search only runs at full depth
            // when the reduced score beats alpha.
            let reduce = if depth >= 3 && legal > 3 && quiet && !in_check && !gives_check {
                1 + (legal > 6) as i32
            } else {
                0
            };
            let score = if reduce > 0 {
                let reduced = -self.negamax(b, depth - 1 - reduce, -beta, -alpha, ply + 1);
                if reduced > alpha {
                    -self.negamax(b, depth - 1, -beta, -alpha, ply + 1)
                } else {
                    reduced
                }
            } else {
                -self.negamax(b, depth - 1, -beta, -alpha, ply + 1)
            };
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
                    if !m.is_capture() && !m.is_promo() {
                        self.store_killer(ply, m);
                    }
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
#[cfg(test)]
mod tests {
    use super::*;

    const WINNING_CAPTURE_FEN: &str = "k7/8/8/3q4/8/2N5/8/K7 w - - 0 1";
    const NEGATIVE_MATE_FEN: &str = "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1";
    const RA8_MATE_FEN: &str = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1";

    fn best(f: &str, d: u32) -> String {
        let mut b = Board::from_fen(f).unwrap();
        Searcher::new(4)
            .think(&mut b, &Limits { depth: Some(d), ..Default::default() }, &[])
            .uci()
    }

    fn full_window_score(fen: &str, depth: u32) -> i32 {
        let mut board = Board::from_fen(fen).unwrap();
        Searcher::new(1).negamax(&mut board, depth as i32, -MATE, MATE, 0)
    }

    fn aspiration_score(fen: &str, depth: u32, prev: i32) -> i32 {
        let mut board = Board::from_fen(fen).unwrap();
        Searcher::new(1).aspiration(&mut board, depth, prev)
    }

    #[test]
    fn finds_mate_in_1() {
        assert_eq!(best("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1", 4), "e1e8");
    }

    #[test]
    fn takes_free_queen() {
        assert_eq!(best(WINNING_CAPTURE_FEN, 4), "c3d5");
    }

    #[test]
    fn stalemate_is_none() {
        let mut b = Board::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap();
        assert_eq!(
            Searcher::new(1).think(&mut b, &Limits { depth: Some(3), ..Default::default() }, &[]),
            Move::NONE
        );
    }

    #[test]
    fn aspiration_retries_on_ordinary_fail_high() {
        let expected = full_window_score(WINNING_CAPTURE_FEN, 6);
        assert!((26..=MATE_BOUND).contains(&expected));

        reset_aspiration_retries();
        let actual = aspiration_score(WINNING_CAPTURE_FEN, 6, 0);

        assert_eq!(actual, expected);
        assert!(aspiration_retries() > 0);
    }

    #[test]
    fn aspiration_retries_on_ordinary_fail_low() {
        let expected = full_window_score(WINNING_CAPTURE_FEN, 6);
        assert!((26..=975).contains(&expected));

        reset_aspiration_retries();
        let actual = aspiration_score(WINNING_CAPTURE_FEN, 6, 1_000);

        assert_eq!(actual, expected);
        assert!(aspiration_retries() > 0);
    }

    #[test]
    fn aspiration_returns_at_a_mate_bound() {
        let expected = full_window_score(NEGATIVE_MATE_FEN, 4);
        assert_eq!(expected, -MATE);

        reset_aspiration_retries();
        let actual = aspiration_score(NEGATIVE_MATE_FEN, 4, 0);

        assert_eq!(actual, expected);
        assert!(aspiration_retries() > 0);
    }

    #[test]
    fn aspiration_retries_to_positive_mate_bound() {
        let expected = full_window_score(RA8_MATE_FEN, 4);
        assert_eq!(expected, MATE - 1);

        reset_aspiration_retries();
        let actual = aspiration_score(RA8_MATE_FEN, 4, MATE_BOUND);

        assert_eq!(actual, expected);
        assert_eq!(actual, MATE - 1);
        assert!(aspiration_retries() > 0);
    }

    #[test]
    fn killer_stored_and_deduped() {
        let mut s = Searcher::new(1);
        let m = Move::new(12, 28, 0);
        s.store_killer(3, m);
        assert_eq!(s.killers[3][0], m);
        s.store_killer(3, m);
        assert_eq!(s.killers[3][1], Move::NONE);
        let m2 = Move::new(11, 27, 0);
        s.store_killer(3, m2);
        assert_eq!(s.killers[3][0], m2);
        assert_eq!(s.killers[3][1], m);
    }

    #[test]
    fn rfp_keeps_tactics() {
        assert_eq!(best("k7/8/8/3q4/8/2N5/8/K7 w - - 0 1", 7), "c3d5");
        assert_eq!(best("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1", 5), "e1e8");
    }

    #[test]
    fn lmr_still_finds_deep_tactic() {
        // Winning knight fork must survive reductions + re-search.
        assert_eq!(best("k7/8/8/3q4/8/2N5/8/K7 w - - 0 1", 8), "c3d5");
    }

    #[test]
    fn lmp_keeps_tactics() {
        assert_eq!(best("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1", 6), "e1e8");
    }

    #[test]
    fn futility_keeps_tactics() {
        assert_eq!(best("k7/8/8/3q4/8/2N5/8/K7 w - - 0 1", 6), "c3d5");
    }

    #[test]
    fn move_ordering_has_exact_killer_precedence() {
        let b = Board::from_fen("7k/P7/8/1p6/8/8/1R6/K7 w - - 0 1").unwrap();
        let tt = Move::new(9, 8, FLAG_QUIET);
        let capture = Move::new(9, 33, FLAG_CAP);
        let quiet_promotion = Move::new(48, 56, FLAG_PROMO + 3);
        let killer = Move::new(9, 10, FLAG_QUIET);
        let quiet = Move::new(9, 11, FLAG_QUIET);
        let mut moves = vec![quiet, killer, quiet_promotion, capture, tt];
        let mut s = Searcher::new(1);
        s.killers[0][0] = killer;

        s.order_moves(&b, &mut moves, tt, 0);

        assert_eq!(moves, vec![tt, capture, quiet_promotion, killer, quiet]);
    }
}
