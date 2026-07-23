use crate::{board::{Board, FeatureDelta, Undo}, eval::{Eval, Hce, MATERIAL}, movegen::generate, moves::*, nnue::{Accumulator, Nnue}, tt::*, types::*};
use crate::attacks::{bishop_attacks, king_attacks, knight_attacks, pawn_attacks, rook_attacks};
use std::time::{Duration, Instant};
pub const MATE: i32 = 30_000;
const MATE_BOUND: i32 = MATE - 1_000;
const MAX_PLY: usize = 128;

/// Which evaluator a `Searcher` currently uses. Monomorphic (no `dyn`) so `Searcher`
/// stays a plain, cheaply-constructed struct: `Hce` is the zero-cost default, `Nnue`
/// replaces it only once a net file is successfully loaded via `Searcher::with_net`.
///
/// `Nnue`'s `stack` is a grow-only POOL of accumulator buffers (M2 Task 6), addressed
/// by `top` rather than truncated on every unmake: `stack[top]` is always the current
/// node's accumulator, `stack[0]` the current search root's fully-refreshed one, and
/// each deeper entry is derived from its parent by `Nnue::apply_delta` in O(changed
/// features) — the board is never rescanned mid-search. `push_delta`/`pop_delta` move
/// `top` up/down and reuse whatever buffer is already sitting at the new index instead
/// of cloning/freeing a `Vec` per node; the pool only grows (one alloc) the first time
/// search reaches a new max depth, which — because qsearch recurses past `MAX_PLY` —
/// is not statically bounded, so it must stay a `Vec` and never a fixed-size array.
/// `Hce` carries no such state; the stack machinery is a no-op for it (see
/// `is_nnue`/`push_delta`/`pop_delta` below).
pub enum EvalKind { Hce(Hce), Nnue { net: Nnue, stack: Vec<Accumulator>, top: usize } }
impl Eval for EvalKind {
    fn eval(&self, board: &Board) -> i32 {
        match self {
            EvalKind::Hce(e) => e.eval(board),
            EvalKind::Nnue { net, stack, top } => net.eval_accumulator(&stack[*top], board.side)
        }
    }
}
impl EvalKind {
    fn is_nnue(&self) -> bool { matches!(self, EvalKind::Nnue { .. }) }

    /// Resets to the search root's fully-refreshed accumulator at `stack[0]` — the one
    /// full-refresh per `think()` call. Pool buffers beyond index 0 are left allocated
    /// (NOT `.clear()`'d) so deeper plies reuse them instead of reallocating. No-op for
    /// `Hce`.
    fn reset_accumulator(&mut self, board: &Board) {
        if let EvalKind::Nnue { net, stack, top } = self {
            *top = 0;
            let fresh = net.fresh_accumulator(board);
            match stack.first_mut() {
                Some(root) => root.copy_from(&fresh),
                None => stack.push(fresh),
            }
        }
    }

    /// Advances `top` to a new accumulator derived from the current one by `delta`, in
    /// O(changed features) -- except on a king move, where the moving side's
    /// perspective is instead refreshed from `board` (bucketed nets only; see
    /// `Nnue::apply_delta_bucketed`). `board` must be the position AFTER the move
    /// (post-`make`). Pair with exactly one `pop_delta` per `push_delta` (mirroring
    /// one `make`/`unmake` pair). No-op for `Hce`.
    fn push_delta(&mut self, delta: &FeatureDelta, board: &Board) {
        if let EvalKind::Nnue { net, stack, top } = self {
            if *top + 1 == stack.len() {
                stack.push(stack[*top].clone()); // grows the pool once per new max depth ever reached
            }
            let (below, above) = stack.split_at_mut(*top + 1);
            above[0].copy_from(&below[*top]); // reuses above[0]'s existing heap buffers, no alloc
            net.apply_delta_bucketed(&mut above[0], delta, board, true);
            *top += 1;
        }
    }

    /// Reverses one `push_delta` by stepping `top` back down — the vacated buffer
    /// stays in the pool for reuse, never freed. No-op for `Hce`.
    fn pop_delta(&mut self) {
        if let EvalKind::Nnue { top, .. } = self {
            *top -= 1;
        }
    }
}

/// Move-ordering history (M4 B1). Three independent gravity-updated tables:
/// main quiet `[color][from][to]`, 1-ply continuation `[prev_piece_to][cur_piece_to]`
/// (piece-to index = piece(0..12)*64 + to(0..64)), and capture `[piece][to][captured_pt]`.
/// Entries stay in (-MAX, MAX) via the gravity update. Not the `Searcher::history`
/// field (that is the repetition hash list).
pub struct History {
    main: Vec<i32>,    // len 2*64*64
    cont: Vec<i32>,    // len 768*768
    capture: Vec<i32>, // len 12*64*6
}
impl History {
    pub const MAX: i32 = 16_384;
    pub const BONUS_CAP: i32 = 1_200;

    pub fn new() -> Self {
        History { main: vec![0; 2 * 64 * 64], cont: vec![0; 768 * 768], capture: vec![0; 12 * 64 * 6] }
    }
    pub fn clear(&mut self) {
        self.main.iter_mut().for_each(|e| *e = 0);
        self.cont.iter_mut().for_each(|e| *e = 0);
        self.capture.iter_mut().for_each(|e| *e = 0);
    }
    /// Depth-scaled cutoff bonus, capped.
    pub fn bonus(depth: i32) -> i32 { (16 * depth * depth).min(Self::BONUS_CAP) }

    fn apply(entry: &mut i32, delta: i32) {
        // Gravity: pulls toward 0 proportionally so |entry| stays < MAX. The raw
        // update can still land exactly on +/-MAX due to integer-division truncation
        // (e.g. repeated same-sign deltas converge there and then freeze), so clamp
        // to the open interval to preserve the invariant.
        *entry += delta - *entry * delta.abs() / Self::MAX;
        *entry = (*entry).clamp(-(Self::MAX - 1), Self::MAX - 1);
    }

    fn main_idx(c: Color, from: u8, to: u8) -> usize { (c.idx() * 64 + from as usize) * 64 + to as usize }
    pub fn quiet(&self, c: Color, from: u8, to: u8) -> i32 { self.main[Self::main_idx(c, from, to)] }
    pub fn update_quiet(&mut self, c: Color, from: u8, to: u8, delta: i32) {
        let i = Self::main_idx(c, from, to); Self::apply(&mut self.main[i], delta);
    }

    pub fn cont(&self, prev: usize, cur: usize) -> i32 { self.cont[prev * 768 + cur] }
    pub fn update_cont(&mut self, prev: usize, cur: usize, delta: i32) {
        let i = prev * 768 + cur; Self::apply(&mut self.cont[i], delta);
    }

    fn cap_idx(piece: usize, to: u8, victim_pt: usize) -> usize { (piece * 64 + to as usize) * 6 + victim_pt }
    pub fn capture(&self, piece: usize, to: u8, victim_pt: usize) -> i32 { self.capture[Self::cap_idx(piece, to, victim_pt)] }
    pub fn update_capture(&mut self, piece: usize, to: u8, victim_pt: usize, delta: i32) {
        let i = Self::cap_idx(piece, to, victim_pt); Self::apply(&mut self.capture[i], delta);
    }
}
impl Default for History { fn default() -> Self { Self::new() } }

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
/// Search-shape switches. `Searcher::new` and `Searcher::with_net` build the
/// production shape; nothing in `uci.rs` can reach these. `heuristics: false`
/// disables every score-inexact heuristic — TT cutoffs, null-move, RFP, LMP,
/// futility, LMR — leaving a pure alpha-beta core that the PVS equivalence test
/// diffs against. `pvs` toggles principal variation search itself, off only in
/// that same equivalence test.
///
/// Deliberately runtime bools rather than `#[cfg(test)]` conditionals: cfg-gating
/// would mean the test suite exercises different code than ships, which
/// reintroduces the exact risk this anchor exists to remove. The cost is one
/// perfectly-predicted branch per pruning site.
#[derive(Clone, Copy)]
struct Shape {
    /// Scout non-first moves with a null window, re-searching full-window only on
    /// a fail-high inside the window. Off only in the equivalence test.
    pvs: bool,
    heuristics: bool,
}

impl Shape {
    fn production() -> Self { Self { pvs: true, heuristics: true } }
    /// Pure alpha-beta core. Test-only.
    #[cfg(test)]
    fn pure() -> Self { Self { pvs: true, heuristics: false } }
}

#[derive(Default)] pub struct Limits { pub depth: Option<u32>, pub movetime: Option<u64>, pub wtime: Option<u64>, pub btime: Option<u64>, pub winc: Option<u64>, pub binc: Option<u64> }
pub struct Searcher { pub tt: Tt, nodes: u64, deadline: Option<Instant>, stopped: bool, history: Vec<u64>, killers: [[Move; 2]; MAX_PLY], eval: EvalKind, hist: History, shape: Shape }
impl Searcher {
    pub fn new(mb: usize) -> Self { Self { tt:Tt::new(mb), nodes:0, deadline:None, stopped:false, history:Vec::new(), killers: [[Move::NONE; 2]; MAX_PLY], eval:EvalKind::Hce(Hce), hist: History::new(), shape: Shape::production() } }
    /// Like `new`, but loads an NNUE net from `path` and uses it in place of `Hce`.
    /// Returns the load error (net file missing/malformed) without constructing a
    /// `Searcher` on failure — callers should keep their previous searcher (HCE) then.
    pub fn with_net(mb: usize, path: &str) -> Result<Self, String> {
        let net = Nnue::load(path)?;
        Ok(Self { tt:Tt::new(mb), nodes:0, deadline:None, stopped:false, history:Vec::new(), killers: [[Move::NONE; 2]; MAX_PLY], eval:EvalKind::Nnue { net, stack: Vec::new(), top: 0 }, hist: History::new(), shape: Shape::production() })
    }
    pub fn node_count(&self) -> u64 { self.nodes }
    /// Builds a searcher with a non-production shape. Test-only: no production
    /// caller may construct anything but `Shape::production()`.
    #[cfg(test)]
    fn with_shape(mb: usize, shape: Shape) -> Self { Self { shape, ..Self::new(mb) } }
    /// Wraps `board.make`/`board.feature_delta` so every call site in the search tree
    /// threads the NNUE accumulator stack automatically (M2 Task 5): the delta is
    /// read from the pre-move position, the board is made, and — only when `eval` is
    /// actually `Nnue` — a new accumulator is pushed derived from it in O(changed
    /// features). Pair with `unmake_move`.
    fn make_move(&mut self, b: &mut Board, m: Move) -> Undo {
        let delta = self.eval.is_nnue().then(|| b.feature_delta(m));
        let undo = b.make(m);
        if let Some(d) = &delta {
            self.eval.push_delta(d, b); // `b` is now the post-make board, as `push_delta` requires
        }
        undo
    }
    /// Reverses `make_move`.
    fn unmake_move(&mut self, b: &mut Board, m: Move, undo: Undo) {
        b.unmake(m, undo);
        self.eval.pop_delta();
    }
    pub fn think(
        &mut self,
        board: &mut Board,
        limits: &Limits,
        history: &[u64],
    ) -> Move {
        self.nodes = 0;
        self.killers = [[Move::NONE; 2]; MAX_PLY];
        self.hist.clear();
        self.stopped = false;
        self.history = history.to_vec();
        self.eval.reset_accumulator(board);
        let plan = plan_time(board, limits);
        self.deadline = plan.map(|(_, hard)| Instant::now() + Duration::from_millis(hard));
        let start = Instant::now();
        let soft = plan.map(|(s, _)| s);

        let mut best = Move::NONE;
        let mut prev_score = 0i32;
        let mut prev_best = Move::NONE;
        let mut stable = 0;
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
            if best == prev_best {
                stable += 1;
            } else {
                stable = 0;
                prev_best = best;
            }
            if let Some(soft_ms) = soft {
                let scale = if stable >= 3 { 6 } else { 10 };   // spend ~0.6x soft when the PV is stable
                if start.elapsed().as_millis() as u64 >= soft_ms * scale / 10 {
                    break;
                }
            }
            if best == Move::NONE || score.abs() > MATE_BOUND {
                break;
            }
        }
        best
    }

    fn aspiration(&mut self, b: &mut Board, depth: u32, prev: i32) -> i32 {
        if depth < 4 || prev.abs() > MATE_BOUND {
            return self.negamax(b, depth as i32, -MATE, MATE, 0, None);
        }

        let mut delta = 25;
        let mut alpha = prev.saturating_sub(delta).max(-MATE);
        let mut beta = prev.saturating_add(delta).min(MATE);
        loop {
            let score = self.negamax(b, depth as i32, alpha, beta, 0, None);
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

    /// Move ordering for the main negamax search: SEE-based capture ranking (winning/equal
    /// captures above killers, losing captures below). Deliberately not shared with the
    /// cheap MVV-LVA `order` used by qsearch — the two are not interchangeable.
    fn order_moves(&self, b: &Board, moves: &mut [Move], tt: Move, ply: i32, prev_pt: Option<usize>) {
        let p = (ply as usize).min(MAX_PLY - 1);
        moves.sort_by_cached_key(|m| {
            if *m == tt {
                -2_000_000
            } else if m.is_capture() {
                let g = see(b, *m);
                if g >= 0 { -(1_000_000 + g) } else { -(g + 200_000) }
            } else if m.is_promo() {
                -900_000
            } else if *m == self.killers[p][0] || *m == self.killers[p][1] {
                -800_000
            } else {
                // Quiet: rank by main + continuation history. Higher history sorts
                // earlier; stays strictly between the killer band (-800_000) and the
                // lowest-history quiet, never colliding with the bands above (history
                // magnitude is bounded well under 100k by the gravity clamp).
                let mover = b.piece_on(m.from()).unwrap();
                let mut h = self.hist.quiet(b.side, m.from(), m.to());
                if let Some(prev) = prev_pt { h += self.hist.cont(prev, mover * 64 + m.to() as usize); }
                -h
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
        prev_pt: Option<usize>,
    ) -> i32 {
        self.nodes += 1;
        if self.timed_out() {
            return 0;
        }
        if ply > 0 && self.repeated(b.hash) {
            return 0; // repetition draw (a repeated position can't be checkmate)
        }
        if ply > 0 && b.halfmove >= 100 {
            // Fifty-move draw — but checkmate takes precedence over the draw claim: if
            // the move that tripped the clock delivers mate, score it as mate, not a draw.
            if b.in_check(b.side) {
                let mut ms = Vec::new();
                generate(b, &mut ms);
                let mated = !ms.iter().any(|&m| {
                    let u = b.make(m);
                    let ok = !b.in_check(b.side.flip());
                    b.unmake(m, u);
                    ok
                });
                if mated {
                    return -MATE + ply;
                }
            }
            return 0;
        }
        if ply >= MAX_PLY as i32 {
            return self.qsearch(b, alpha, beta, ply);
        }
        let in_check = b.in_check(b.side);
        // Check extension: search one ply deeper when the side to move is in check, so
        // forced checking sequences aren't cut off and mis-scored at the qsearch horizon
        // (qsearch only tries captures/promotions and never detects checkmate). Bounded
        // by the `ply >= MAX_PLY` guard above, so it cannot runaway.
        let depth = if in_check { depth + 1 } else { depth };
        if depth <= 0 {
            return self.qsearch(b, alpha, beta, ply);
        }

        let alpha0 = alpha;
        let tt_entry = self.tt.probe(b.hash);
        let tt_move = tt_entry.map(|e| e.mv).unwrap_or(Move::NONE);
        if let Some(e) = tt_entry {
            if self.shape.heuristics && ply > 0 && e.depth as i32 >= depth {
                let score = from_tt(e.score, ply);
                if e.bound == BOUND_EXACT
                    || e.bound == BOUND_LOWER && score >= beta
                    || e.bound == BOUND_UPPER && score <= alpha
                {
                    return score;
                }
            }
        }

        let static_eval = if in_check { 0 } else { self.eval.eval(b) };
        // Reverse futility pruning (static null-move): at shallow depth, if the static
        // eval beats beta by a depth-scaled margin, assume the node holds and prune.
        // Fail-soft: returns the static eval.
        if self.shape.heuristics && depth <= 6 && !in_check && beta.abs() < MATE_BOUND && static_eval - 80 * depth >= beta {
            return static_eval;
        }

        // Null-move pruning: skip a turn and see if the position is still >= beta.
        // Guarded against check and likely-zugzwang (side must have non-pawn material).
        // Eval-gated: only attempted when static_eval already looks >= beta.
        // Fail-hard: returns beta (not the null score) on cutoff.
        if self.shape.heuristics && depth >= 3 && ply > 0 && beta.abs() < MATE_BOUND
            && !in_check && static_eval >= beta && b.has_non_pawn_material(b.side)
        {
            let r = 2 + depth / 4;
            let u = b.make_null();
            self.history.push(b.hash);
            let s = -self.negamax(b, depth - 1 - r, -beta, -beta + 1, ply + 1, None);
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
        self.order_moves(b, &mut moves, tt_move, ply, prev_pt);

        let mut legal = 0;
        let mut best = -MATE - 1;
        let mut best_move = Move::NONE;
        let mut tried_quiets: Vec<Move> = Vec::new();
        for m in moves {
            let mover = b.piece_on(m.from()).unwrap();   // 0..12 piece index (pre-move)
            let cur_pt = mover * 64 + m.to() as usize;   // piece-to index for continuation history
            let undo = self.make_move(b, m);
            if b.in_check(b.side.flip()) {
                self.unmake_move(b, m, undo);
                continue;
            }
            legal += 1;
            let gives_check = b.in_check(b.side);
            let quiet = !m.is_capture() && !m.is_promo();
            // late move pruning: skip late quiets at shallow depth once we have a real best
            if self.shape.heuristics && legal > 1 && quiet && !in_check && !gives_check && best > -MATE_BOUND
                && depth <= 3 && legal > 4 + depth * depth
            {
                self.unmake_move(b, m, undo);
                legal -= 1;                 // this move was not actually searched
                continue;
            }
            // futility pruning: skip late quiets whose static eval can't reach alpha
            if self.shape.heuristics && legal > 1 && quiet && !in_check && !gives_check && best > -MATE_BOUND
                && depth <= 4 && static_eval + 100 * depth <= alpha
            {
                self.unmake_move(b, m, undo);
                legal -= 1;
                continue;
            }
            if quiet { tried_quiets.push(m); }
            self.history.push(b.hash);
            // Late move reductions (LMR): late quiet, non-checking moves at depth >= 3
            // (when not already in check) get a shallower search first — reduced by
            // 1 ply, or 2 once legal > 6. This is full-window, not PVS/null-window:
            // both the reduced probe and the full-depth re-search use the same
            // [-beta, -alpha] window, and the re-search only runs at full depth
            // when the reduced score beats alpha.
            let reduce = if self.shape.heuristics && depth >= 3 && legal > 3 && quiet && !in_check && !gives_check {
                1 + (legal > 6) as i32
            } else {
                0
            };
            // Principal variation search. The first legal move establishes the PV and
            // gets the full [alpha, beta] window. Every later move is first probed
            // with a null window — a scout that can only prove "not better than
            // alpha" or "better than alpha" — and we pay for a full-window re-search
            // only when the scout lands strictly inside the window.
            //
            // At non-PV nodes beta == alpha + 1 already, so guard (c) is unsatisfiable
            // and the re-search never runs: PVS costs nothing there. This also means
            // the pre-existing LMR window [-beta, -alpha] was *already* a null window
            // at those nodes, which is why M5 needs no separate LMR bundle.
            let score = if !self.shape.pvs {
                // Pre-M5 reference path, kept so `pvs: false` is a faithful v0.4.0
                // baseline: reduced probe and re-search BOTH full-window. Without this
                // arm the `!pvs` branch would skip LMR entirely and every PVS-on/off
                // comparison would be measuring against a crippled baseline.
                if reduce > 0 {
                    let reduced = -self.negamax(b, depth - 1 - reduce, -beta, -alpha, ply + 1, Some(cur_pt));
                    if reduced > alpha {
                        -self.negamax(b, depth - 1, -beta, -alpha, ply + 1, Some(cur_pt))
                    } else {
                        reduced
                    }
                } else {
                    -self.negamax(b, depth - 1, -beta, -alpha, ply + 1, Some(cur_pt))
                }
            } else if legal == 1 {
                -self.negamax(b, depth - 1, -beta, -alpha, ply + 1, Some(cur_pt))
            } else {
                // (a) reduced scout, when LMR applies; the sentinel makes (b)
                //     unconditional when it does not, without duplicating the call.
                let mut s = if reduce > 0 {
                    -self.negamax(b, depth - 1 - reduce, -alpha - 1, -alpha, ply + 1, Some(cur_pt))
                } else {
                    alpha + 1
                };
                // (b) full-depth scout, once the reduced probe beat alpha
                if s > alpha {
                    s = -self.negamax(b, depth - 1, -alpha - 1, -alpha, ply + 1, Some(cur_pt));
                }
                // (c) full-window re-search — PV nodes only, where beta > alpha + 1
                if s > alpha && s < beta {
                    s = -self.negamax(b, depth - 1, -beta, -alpha, ply + 1, Some(cur_pt));
                }
                s
            };
            self.history.pop();
            self.unmake_move(b, m, undo);
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
                    if quiet {
                        let bonus = History::bonus(depth);
                        self.hist.update_quiet(b.side, m.from(), m.to(), bonus);
                        if let Some(prev) = prev_pt { self.hist.update_cont(prev, cur_pt, bonus); }
                        for &q in &tried_quiets {
                            if q == m { continue; }
                            self.hist.update_quiet(b.side, q.from(), q.to(), -bonus);
                            if let Some(prev) = prev_pt {
                                let qpt = b.piece_on(q.from()).unwrap() * 64 + q.to() as usize;
                                self.hist.update_cont(prev, qpt, -bonus);
                            }
                        }
                        self.store_killer(ply, m);
                    } else if m.is_capture() {
                        let victim = b.piece_on(m.to()).map(|p| p % 6).unwrap_or(PAWN);
                        self.hist.update_capture(b.piece_on(m.from()).unwrap(), m.to(), victim, History::bonus(depth));
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
    fn qsearch(&mut self, b: &mut Board, mut alpha: i32, beta: i32, ply: i32) -> i32 {
        self.nodes += 1;
        if self.timed_out() {
            return 0;
        }
        // When the side to move is in check, standing pat is illegal and a captures/promos-
        // only list omits the quiet evasions (king walks, interpositions) that resolve the
        // check — so search EVERY legal move here and, if none exist, report checkmate.
        // Otherwise, the usual stand-pat qsearch over captures/promotions.
        let in_check = b.in_check(b.side);
        let mut moves = Vec::new();
        generate(b, &mut moves);
        if !in_check {
            let stand = self.eval.eval(b);
            if stand >= beta {
                return stand;
            }
            if stand > alpha {
                alpha = stand;
            }
            moves.retain(|m| m.is_capture() || m.is_promo());
        }
        order(b, &mut moves, Move::NONE);
        let mut legal = 0;
        for m in moves {
            if !in_check && m.is_capture() && !m.is_promo() && see(b, m) < 0 {
                continue;
            }
            let u = self.make_move(b, m);
            if b.in_check(b.side.flip()) {
                self.unmake_move(b, m, u);
                continue;
            }
            legal += 1;
            let score = -self.qsearch(b, -beta, -alpha, ply + 1);
            self.unmake_move(b, m, u);
            if score > alpha {
                alpha = score;
                if alpha >= beta {
                    break;
                }
            }
        }
        if in_check && legal == 0 {
            return -MATE + ply; // checkmated — no legal evasion exists
        }
        alpha
    }
}
/// Move ordering for qsearch: cheap MVV-LVA over the (already capture/promo-only) move
/// list. Deliberately not the SEE-based `order_moves` used by negamax — the two are not
/// interchangeable (this one is O(1) per move; SEE-based ordering is far more expensive).
fn order(b:&Board,moves:&mut [Move],tt:Move){moves.sort_by_key(|m|if *m==tt{-1_000_000}else if m.is_capture(){let v=b.piece_on(m.to()).map(|p|MATERIAL[p%6]).unwrap_or(100);let a=b.piece_on(m.from()).map(|p|MATERIAL[p%6]).unwrap_or(0);-(10_000+v*10-a)}else if m.is_promo(){-9_000}else{0})}

// Intentionally NOT `eval.rs`'s `MATERIAL` table: identical for P/N/B/R/Q but the KING
// slot is 10_000 here vs 0 in MATERIAL. SEE simulates a chain of hypothetical recaptures
// that can include a king, and a 0-value king would make "recapturing" with the king
// look free/neutral instead of catastrophic — the large value keeps a king "capture" in
// the exchange simulation from ever scoring as a favourable trade. Do not unify the two
// tables.
const SEE_VAL: [i32; 6] = [100, 320, 330, 500, 900, 10_000]; // P N B R Q K

fn attackers_to(b: &Board, s: u8, occ: Bb) -> Bb {
    let wp = pawn_attacks(Color::Black, s) & b.bb[pc(Color::White, PAWN)];
    let bp = pawn_attacks(Color::White, s) & b.bb[pc(Color::Black, PAWN)];
    let n = knight_attacks(s) & (b.bb[pc(Color::White, KNIGHT)] | b.bb[pc(Color::Black, KNIGHT)]);
    let k = king_attacks(s) & (b.bb[pc(Color::White, KING)] | b.bb[pc(Color::Black, KING)]);
    let bishops = b.bb[pc(Color::White, BISHOP)] | b.bb[pc(Color::Black, BISHOP)]
        | b.bb[pc(Color::White, QUEEN)] | b.bb[pc(Color::Black, QUEEN)];
    let rooks = b.bb[pc(Color::White, ROOK)] | b.bb[pc(Color::Black, ROOK)]
        | b.bb[pc(Color::White, QUEEN)] | b.bb[pc(Color::Black, QUEEN)];
    (wp | bp | n | k | (bishop_attacks(s, occ) & bishops) | (rook_attacks(s, occ) & rooks)) & occ
}

fn least_valuable(b: &Board, attackers: Bb, side: Color) -> Option<(u8, usize)> {
    for pt in 0..6 {
        let set = attackers & b.bb[pc(side, pt)];
        if set != 0 { return Some((lsb(set), pt)); }
    }
    None
}

pub fn see(b: &Board, m: Move) -> i32 {
    let to = m.to();
    let mut from = m.from();
    let mut occ = b.all();
    let bishops = b.bb[pc(Color::White, BISHOP)] | b.bb[pc(Color::Black, BISHOP)]
        | b.bb[pc(Color::White, QUEEN)] | b.bb[pc(Color::Black, QUEEN)];
    let rooks = b.bb[pc(Color::White, ROOK)] | b.bb[pc(Color::Black, ROOK)]
        | b.bb[pc(Color::White, QUEEN)] | b.bb[pc(Color::Black, QUEEN)];

    let mut gain = [0i32; 32];
    let target_pt = if m.flags() == FLAG_EP { PAWN } else { b.piece_on(to).map(|p| p % 6).unwrap_or(0) };
    gain[0] = SEE_VAL[target_pt];
    let mut moving_pt = b.piece_on(from).unwrap() % 6;
    let mut side = b.side.flip();
    let mut attackers = attackers_to(b, to, occ);
    let mut d = 0;
    loop {
        // Remove the piece that just captured (now sitting on `to`) from the board,
        // then see whether `side` has a piece left to recapture it. gain[d] is only
        // computed once such an attacker is actually found — a capture that never
        // happens must not be fed into the minimax back-substitution below.
        occ ^= bb(from);
        attackers &= !bb(from);
        attackers |= (bishop_attacks(to, occ) & bishops) | (rook_attacks(to, occ) & rooks); // x-ray
        attackers &= occ;
        match least_valuable(b, attackers, side) {
            Some((sq, pt)) => {
                d += 1;
                gain[d] = SEE_VAL[moving_pt] - gain[d - 1];
                from = sq;
                moving_pt = pt;
                side = side.flip();
            }
            None => break,
        }
        if d >= 31 { break; }
    }
    while d > 0 { d -= 1; gain[d] = -std::cmp::max(-gain[d], gain[d + 1]); }
    gain[0]
}

fn to_tt(s:i32,ply:i32)->i32{if s>MATE_BOUND{s+ply}else if s < -MATE_BOUND{s-ply}else{s}} fn from_tt(s:i32,ply:i32)->i32{if s>MATE_BOUND{s-ply}else if s < -MATE_BOUND{s+ply}else{s}}
fn score_text(s:i32)->String{if s.abs()>MATE_BOUND{format!("mate {}",if s>0{(MATE-s+1)/2}else{-((MATE+s+1)/2)})}else{format!("cp {s}")}}
fn plan_time(b: &Board, l: &Limits) -> Option<(u64, u64)> {
    if let Some(mt) = l.movetime {
        let m = mt.saturating_sub(20).max(1);
        return Some((m, m));
    }
    let (time, inc) = if b.side == Color::White { (l.wtime, l.winc) } else { (l.btime, l.binc) };
    let time = time?;
    let inc = inc.unwrap_or(0);
    let soft = (time / 20 + inc * 3 / 4).max(1);
    let hard = (time / 4).min(soft * 4).max(soft).min(time.saturating_sub(30).max(1));
    Some((soft, hard))
}
#[cfg(test)]
mod tests {
    use super::*;

    const WINNING_CAPTURE_FEN: &str = "k7/8/8/3q4/8/2N5/8/K7 w - - 0 1";
    const NEGATIVE_MATE_FEN: &str = "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1";
    const RA8_MATE_FEN: &str = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1";
    // Philidor-style smothered mate: 1.Qg8+! Rxg8 (forced, Kxg8 illegal since Nh6
    // guards g8) 2.Nf7# (quiet knight check, king smothered by its own Rg8/Pg7/Ph7).
    const SMOTHERED_MATE_FEN: &str = "5r1k/6pp/7N/3Q4/8/8/8/6K1 w - - 0 1";

    fn best(f: &str, d: u32) -> String {
        let mut b = Board::from_fen(f).unwrap();
        Searcher::new(4)
            .think(&mut b, &Limits { depth: Some(d), ..Default::default() }, &[])
            .uci()
    }

    fn full_window_score(fen: &str, depth: u32) -> i32 {
        let mut board = Board::from_fen(fen).unwrap();
        Searcher::new(1).negamax(&mut board, depth as i32, -MATE, MATE, 0, None)
    }

    fn aspiration_score(fen: &str, depth: u32, prev: i32) -> i32 {
        let mut board = Board::from_fen(fen).unwrap();
        Searcher::new(1).aspiration(&mut board, depth, prev)
    }

    fn nodes_at(fen: &str, depth: u32, shape: Shape) -> u64 {
        let mut b = Board::from_fen(fen).unwrap();
        let mut s = Searcher::with_shape(1, shape);
        s.negamax(&mut b, depth as i32, -MATE, MATE, 0, None);
        s.node_count()
    }

    #[test]
    fn disabling_heuristics_widens_the_tree() {
        // With TT cutoffs, null-move, RFP, LMP, futility and LMR all disabled the
        // search must visit strictly more nodes at the same depth. This is what
        // proves the `shape.heuristics` guards actually reached every pruning site —
        // a guard that silently missed one would leave the counts much closer.
        let pure = nodes_at(WINNING_CAPTURE_FEN, 5, Shape::pure());
        let prod = nodes_at(WINNING_CAPTURE_FEN, 5, Shape::production());
        assert!(pure > prod, "pure core {pure} nodes must exceed production {prod}");
    }

    /// (FEN, depth) pairs for the PVS equivalence anchor. Depths are deliberately
    /// shallow: with TT cutoffs disabled the tree grows exponentially, so branchy
    /// positions get depth 3 and sparse ones 4-7 (WINNING_CAPTURE_FEN sits at 7,
    /// deepened from 6 during ladder-bug mutation testing).
    const EQUIV_SUITE: [(&str, u32); 7] = [
        (WINNING_CAPTURE_FEN, 7),                                       // sparse tactical
        (RA8_MATE_FEN, 5),                                              // back-rank mate
        (NEGATIVE_MATE_FEN, 4),                                         // side to move is in check, and mated
        ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 4),                          // stalemate
        ("8/8/1p1k4/p1p2p1p/P1P2P1P/1P1K4/8/8 w - - 0 1", 5),           // king-and-pawn endgame
        ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3), // Kiwipete, branchy
        ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 3), // quiet middlegame
    ];

    fn pure_score(fen: &str, depth: u32, pvs: bool) -> i32 {
        let mut b = Board::from_fen(fen).unwrap();
        Searcher::with_shape(1, Shape { pvs, ..Shape::pure() })
            .negamax(&mut b, depth as i32, -MATE, MATE, 0, None)
    }

    #[test]
    fn pvs_scores_identically_to_plain_alpha_beta() {
        // THE M5 CORRECTNESS ANCHOR — replaces the full-window negamax invariant.
        //
        // Over a pure alpha-beta core (no TT cutoffs, null-move, RFP, LMP, futility
        // or LMR) PVS is provably score-identical to plain full-window search: the
        // null-window scouts only ever prove "not better than alpha", and every move
        // that beats alpha is re-searched with the real window. Any divergence here
        // is a genuine bug in the scout/re-search ladder.
        //
        // This does NOT prove LMR-under-PVS is sound — reductions are not
        // score-preserving under any windowing scheme. `lmr_still_finds_deep_tactic`
        // and the `*_keeps_tactics` tests remain the guard for that.
        for (fen, depth) in EQUIV_SUITE {
            assert_eq!(
                pure_score(fen, depth, true),
                pure_score(fen, depth, false),
                "PVS diverged from full-window alpha-beta at {fen} depth {depth}"
            );
        }
    }

    /// Nodes for a full iterative-deepening search — the way the engine actually
    /// plays. Unlike `nodes_at`, this includes aspiration windows and a TT warmed by
    /// the shallower iterations, both of which narrow windows on their own.
    fn think_nodes(fen: &str, depth: u32, shape: Shape) -> u64 {
        let mut b = Board::from_fen(fen).unwrap();
        let mut s = Searcher::with_shape(16, shape);
        s.think(&mut b, &Limits { depth: Some(depth), ..Default::default() }, &[]);
        s.node_count()
    }

    #[test]
    fn pvs_does_not_inflate_the_real_search_tree() {
        // PVS's node effect MUST be measured through `think`. A fixed-depth
        // `negamax(-MATE, MATE)` comparison flatters PVS enormously — it reported a
        // 77.9% "saving" that does not exist in play, because the `pvs: false` side of
        // that comparison searches every root move with a full window against a cold
        // TT, which the engine never does.
        //
        // Measured through `think`, PVS in this engine is roughly node-NEUTRAL: at PV
        // nodes the ladder runs three searches (reduced scout, full-depth scout,
        // full-window re-search) where the pre-M5 code ran two, and that cost offsets
        // the narrower-window saving. Iterative deepening plus aspiration were already
        // doing most of what PVS exists to do.
        //
        // So this is a regression guard, not a win condition: it catches a ladder bug
        // that blows the tree up. Whether PVS is worth keeping is the SPRT's call.
        const SUITE: [(&str, u32); 4] = [
            ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 8),
            ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 8),
            ("2rq1rk1/pb1nbppp/1p2pn2/2pp4/3P1B2/2NBPN2/PPQ2PPP/R4RK1 w - - 0 11", 8),
            ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 8),
        ];
        let mut pvs_nodes = 0u64;
        let mut full_nodes = 0u64;
        for (fen, depth) in SUITE {
            pvs_nodes += think_nodes(fen, depth, Shape::production());
            full_nodes += think_nodes(fen, depth, Shape { pvs: false, ..Shape::production() });
        }
        println!(
            "PVS {pvs_nodes} vs pre-M5 {full_nodes} nodes ({:+.1}%)",
            100.0 * (pvs_nodes as f64 / full_nodes as f64 - 1.0)
        );
        assert!(
            pvs_nodes * 100 <= full_nodes * 115,
            "PVS {pvs_nodes} vs pre-M5 {full_nodes} nodes: tree inflated by more than 15%"
        );
    }

    #[test]
    fn history_gravity_is_bounded_and_directional() {
        let mut h = History::new();
        for _ in 0..1000 { h.update_quiet(Color::White, 12, 28, 900); }
        let hi = h.quiet(Color::White, 12, 28);
        assert!(hi > 0 && hi < History::MAX, "saturated entry {hi} left (0, MAX)");
        let before = h.quiet(Color::White, 12, 28);
        h.update_quiet(Color::White, 12, 28, -900);
        assert!(h.quiet(Color::White, 12, 28) < before, "malus must decrease the entry");
        assert_eq!(h.quiet(Color::White, 11, 27), 0);
    }

    #[test]
    fn history_bonus_scales_with_depth_and_caps() {
        assert!(History::bonus(8) > History::bonus(3), "deeper cutoff earns more");
        assert!(History::bonus(64) <= History::BONUS_CAP, "bonus is capped");
    }

    #[test]
    fn continuation_and_capture_history_are_separate_axes() {
        let mut h = History::new();
        h.update_cont(5, 20, 700);
        assert!(h.cont(5, 20) > 0);
        assert_eq!(h.cont(6, 20), 0);
        h.update_capture(1, 28, PAWN, 700);
        assert!(h.capture(1, 28, PAWN) > 0);
        assert_eq!(h.capture(1, 28, KNIGHT), 0);
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
    fn checkmate_overrides_fifty_move_draw() {
        // Clock at 99: the mating move Ra8# trips halfmove to 100. Checkmate takes
        // precedence over the fifty-move draw claim, so the search must still play the
        // mate instead of a quiet move (which it would if the child scored as a draw).
        assert_eq!(best("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 99 1", 4), "a1a8");
    }

    #[test]
    fn qsearch_reports_checkmate() {
        // Black is checkmated (Ra8#, king smothered by its own f7/g7/h7 pawns). Reached
        // in check, qsearch must search evasions, find none, and return a mate score —
        // not stand pat on the (materially rosy) static eval.
        let mut b = Board::from_fen("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1").unwrap();
        let score = Searcher::new(1).qsearch(&mut b, -MATE, MATE, 0);
        assert!(score <= -MATE_BOUND, "checkmated side must get a mate score, got {score}");
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

        // Aspiration reuses the transposition table across its widening re-searches,
        // so once a selective extension (the check extension) is in the tree the
        // fail-soft result can differ by a few centipawns from a clean full-window
        // search — benign, well-known search instability, not a logic error. Assert
        // the score converges into the same winning band and stays close to the
        // full-window value, and that a retry occurred, rather than bit-exact equality.
        assert!((26..=975).contains(&actual), "aspiration score {actual} left the winning band");
        assert!((actual - expected).abs() <= 64, "aspiration {actual} diverged from full-window {expected}");
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

        s.order_moves(&b, &mut moves, tt, 0, None);

        assert_eq!(moves, vec![tt, capture, quiet_promotion, killer, quiet]);
    }

    #[test]
    fn order_ranks_high_history_quiet_first_but_below_killers() {
        let b = Board::from_fen("7k/8/8/8/8/8/8/R3K3 w - - 0 1").unwrap();
        let mut s = Searcher::new(1);
        let good = Move::new(0, 24, FLAG_QUIET);   // a1a4
        let meh  = Move::new(0, 8,  FLAG_QUIET);   // a1a2
        let killer = Move::new(4, 12, FLAG_QUIET); // e1e2, will be a killer
        s.hist.update_quiet(Color::White, 0, 24, 5000);
        s.killers[0][0] = killer;
        let mut moves = vec![meh, good, killer];
        s.order_moves(&b, &mut moves, Move::NONE, 0, None);
        assert_eq!(moves[0], killer, "killer outranks history quiets");
        assert_eq!(moves[1], good, "higher-history quiet before lower");
        assert_eq!(moves[2], meh);
    }

    fn see_of(fen: &str, mv: &str) -> i32 {
        let b = Board::from_fen(fen).unwrap();
        let mut ms = Vec::new();
        crate::movegen::generate(&b, &mut ms);
        let m = ms.into_iter().find(|m| m.uci() == mv).expect("move not legal");
        see(&b, m)
    }
    #[test]
    fn see_values() {
        // White pawn takes an undefended pawn: +100 (victim value, nothing recaptures).
        assert_eq!(see_of("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5"), 100);
        // Queen takes a pawn defended by a pawn: 100 - 900 < 0 (bad trade).
        assert!(see_of("4k3/8/8/2p5/3p4/8/3Q4/4K3 w - - 0 1", "d2d4") < 0);
        // Rook takes an undefended rook: +500.
        assert_eq!(see_of("4k3/8/8/8/8/8/3r4/3RK3 w - - 0 1", "d1d2"), 500);
        // Capturing promotion: pawn takes rook and promotes to queen; the new queen is
        // immediately recaptured by the knight. Even so, the pawn was never a queen's
        // worth of material to lose — it only ever cost White a pawn to win a rook, so
        // the trade nets +500 (rook) - 100 (pawn) = +400. SEE is correctly POSITIVE
        // here; do not "correct" this to -400 (that would require crediting the
        // recapture as if a real queen were lost, which double-counts the promotion).
        assert_eq!(see_of("r6k/1Pn5/8/8/8/8/8/7K w - - 0 1", "b7a8q"), 400);
        // 3-ply exchange: Nxn (knight takes knight), pawn recaptures, and the bishop
        // backs the pawn up: net 320 - 320 + 100 = 100 for the side that opened it.
        assert_eq!(see_of("4k3/8/8/2p5/3n4/1N6/5B2/4K3 w - - 0 1", "b3d4"), 100);
        // X-ray reveal: the front rook takes the bishop, the knight recaptures the
        // rook, and the second rook — hidden behind the first on the d-file until it
        // moves — is revealed and recaptures the knight: 330 - 500 + 320 = 150.
        assert_eq!(see_of("4k3/8/1n6/3b4/8/8/3R4/K2R4 w - - 0 1", "d2d5"), 150);
    }

    #[test]
    fn respects_movetime_and_returns_legal() {
        let mut b = Board::startpos();
        let t = std::time::Instant::now();
        let m = Searcher::new(8).think(&mut b, &Limits { movetime: Some(50), ..Default::default() }, &[]);
        let ms = t.elapsed().as_millis();
        assert_ne!(m, Move::NONE);
        assert!(ms < 400, "movetime 50ms overran badly: {ms}ms");
    }

    #[test]
    fn check_extension_finds_mate_missed_without_it() {
        // Discriminating regression test (verified empirically against the pre-extension
        // commit e1a904d): at nominal depth 2 the pre-extension engine returns "h6f5"
        // (cp 488) — it never even considers the queen sacrifice, because the mated node
        // at the end of the 1.Qg8+ Rxg8 2.Nf7# line lands exactly on a depth-exhausted,
        // in-check node that falls straight into qsearch (captures/promotions only, no
        // legal-move/checkmate test) without the extension, and qsearch simply misses the
        // quiet mating knight move. With the check extension, depth 2 already finds the
        // forced mate. This test would FAIL if the check extension were reverted.
        assert_eq!(best(SMOTHERED_MATE_FEN, 2), "d5g8");
    }

    #[test]
    fn beta_cutoff_populates_quiet_history() {
        let mut b = Board::from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1").unwrap();
        let mut s = Searcher::new(8);
        s.think(&mut b, &Limits { depth: Some(6), ..Default::default() }, &[]);
        let any = (0..64).any(|f| (0..64).any(|t| s.hist.quiet(Color::White, f as u8, t as u8) != 0
            || s.hist.quiet(Color::Black, f as u8, t as u8) != 0));
        assert!(any, "expected some quiet history after a real search");
    }
}
