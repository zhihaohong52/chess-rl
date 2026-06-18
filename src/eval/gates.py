"""Run-level gate suite: puzzles, MCTS, dense policy/value, throughput.

Objective-agnostic gates (raw top-1, throughput) take a net; objective-aware
gates (MCTS) take a routed evaluator. run_gates assembles a flat metrics dict.
"""

import time

import chess
import numpy as np
import torch

from config import Config
from src.game.token_encoder import encode_position
from src.game.orientation import to_canonical_move
from src.game.move_encoder import get_move_encoder
from src.eval import metrics_core as mc


def raw_top1(net, device, puzzles) -> float:
    """Argmax legal-move accuracy (no search). Objective-agnostic."""
    me = get_move_encoder()
    net.eval().to(device)
    if not puzzles:
        return 0.0
    correct = 0
    for pz in puzzles:
        board = chess.Board(pz.fen)
        legal = list(board.legal_moves)
        if not legal:
            continue
        sq, sf = encode_position(board, 0)
        sq_t = torch.tensor(sq[None], dtype=torch.long, device=device)
        sf_t = torch.tensor(sf[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            pol, _, _ = net.predict_batch(sq_t, sf_t)
        pol = pol[0].cpu().numpy()
        idxs = [me.encode(to_canonical_move(mv, board.turn)) for mv in legal]
        best = legal[int(np.argmax(pol[idxs]))]
        if best == pz.solution_moves[0]:
            correct += 1
    return correct / len(puzzles)


def mcts_top1(evaluator, puzzles, simulations: int) -> float:
    """MCTS-N argmax accuracy using the routed evaluator."""
    from src.mcts.batched_mcts import BatchedMCTS
    if not puzzles or simulations <= 0:
        return 0.0
    mcts = BatchedMCTS(evaluator, Config, num_simulations=simulations)
    correct = 0
    for pz in puzzles:
        mcts.reset()
        mv = mcts.choose_move(chess.Board(pz.fen), temperature=0.0)
        if mv == pz.solution_moves[0]:
            correct += 1
    return correct / len(puzzles)


def mate_in_one_acc(net, device, positions) -> float:
    """Fraction of mate-in-1 positions where argmax legal move is the mate."""
    me = get_move_encoder()
    net.eval().to(device)
    if not positions:
        return 0.0
    correct = 0
    for fen, uci in positions:
        board = chess.Board(fen)
        legal = list(board.legal_moves)
        sq, sf = encode_position(board, 0)
        sq_t = torch.tensor(sq[None], dtype=torch.long, device=device)
        sf_t = torch.tensor(sf[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            pol, _, _ = net.predict_batch(sq_t, sf_t)
        pol = pol[0].cpu().numpy()
        idxs = [me.encode(to_canonical_move(mv, board.turn)) for mv in legal]
        best = legal[int(np.argmax(pol[idxs]))]
        if best == chess.Move.from_uci(uci):
            correct += 1
    return correct / len(positions)


def policy_value_metrics(net, device, val_loader, max_batches: int = 50) -> dict:
    """top1/3/5, policy CE, legal mass, WDL CE, value sign acc, draw cal."""
    net.eval().to(device)
    acc = {k: 0.0 for k in ("top1", "top3", "top5", "policy_ce", "legal_mass",
                            "wdl_ce", "value_sign_acc", "draw_cal")}
    n = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            if i >= max_batches:
                break
            (sq, sf), (pol_t, wdl_t, _ml) = inputs, targets
            sq = sq.to(device); sf = sf.to(device)
            pol_t = pol_t.to(device); wdl_t = wdl_t.to(device)
            pol, wdl, _ = net(sq, sf)
            acc["top1"] += mc.policy_topk_match(pol, pol_t, 1)
            acc["top3"] += mc.policy_topk_match(pol, pol_t, 3)
            acc["top5"] += mc.policy_topk_match(pol, pol_t, 5)
            acc["policy_ce"] += mc.policy_cross_entropy(pol, pol_t)
            acc["legal_mass"] += mc.legal_mass(pol, pol_t)
            acc["wdl_ce"] += mc.wdl_cross_entropy(wdl, wdl_t)
            acc["value_sign_acc"] += mc.value_sign_acc(wdl, wdl_t)
            acc["draw_cal"] += mc.draw_calibration(wdl, wdl_t)
            n += 1
    n = max(n, 1)
    return {k: v / n for k, v in acc.items()}


def throughput(net, device, batch: int = 256, iters: int = 5) -> dict:
    """Forward latency (ms/batch) and positions/sec at the given batch size."""
    net.eval().to(device)
    sq = torch.zeros(batch, 64, dtype=torch.long, device=device)
    sf = torch.zeros(batch, 18, dtype=torch.float32, device=device)
    with torch.no_grad():
        net.predict_batch(sq, sf)  # warmup
        t0 = time.perf_counter()
        for _ in range(iters):
            net.predict_batch(sq, sf)
        dt = (time.perf_counter() - t0) / iters
    return {"batch_latency_ms": dt * 1000.0, "positions_per_sec": batch / dt}


def run_gates(net, evaluator, device, puzzle_counts=(300, 1000),
              mcts_puzzles=300, mcts_sims=100, dense_val=True,
              throughput_batch=256, throughput_iters=5) -> dict:
    """Assemble the full gate dict. Set dense_val=False / mcts_puzzles=0 to skip
    the heavy parts in fast smoke runs."""
    from src.eval.fixtures import (load_gate_puzzles, mate_in_one_positions,
                                   ensure_dense_val)
    out = {}
    maxc = max(puzzle_counts) if puzzle_counts else 0
    all_pz = load_gate_puzzles(maxc) if maxc else []
    for c in puzzle_counts:
        out[f"raw_top1@{c}"] = raw_top1(net, device, all_pz[:c])
    if mcts_puzzles:
        out[f"mcts{mcts_sims}_top1@{mcts_puzzles}"] = mcts_top1(
            evaluator, all_pz[:mcts_puzzles], mcts_sims)
    out["mate_in_1"] = mate_in_one_acc(net, device, mate_in_one_positions())
    if dense_val:
        from src.data.dataset import make_dataloader
        path = ensure_dense_val()
        loader = make_dataloader([path], batch_size=256,
                                 policy_size=get_move_encoder().policy_size,
                                 shuffle=False)
        out.update(policy_value_metrics(net, device, loader))
    out.update(throughput(net, device, batch=throughput_batch,
                          iters=throughput_iters))
    return out
