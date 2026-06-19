"""MCTS-facing evaluator wrapping ChessTransformer (PyTorch)."""

import numpy as np
import torch
import torch.nn.functional as F

from src.game.token_encoder import encode_batch
from src.game.orientation import to_canonical_move
from src.game.move_encoder import get_move_encoder


class TransformerEvaluator:
    def __init__(self, net, use_fp16: bool = False, device=None, objective: str = "policy"):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.net = net.to(device)
        self.net.eval()
        self.me = get_move_encoder()
        self.use_fp16 = use_fp16
        if objective not in ("policy", "action_value"):
            raise ValueError(f"objective must be 'policy' or 'action_value', got {objective}")
        self.objective = objective

    def evaluate(self, board, repetition_count: int = 0):
        return self.evaluate_batch([board], [repetition_count])[0]

    def evaluate_batch(self, boards, reps=None):
        if reps is None:
            reps = [0] * len(boards)
        sq, sf = encode_batch(boards, reps)
        sq_t = torch.tensor(sq, dtype=torch.long, device=self.device)
        sf_t = torch.tensor(sf, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            pol_logits, wdl_logits, _ = self.net.predict_batch(sq_t, sf_t)

        pol_logits = pol_logits.cpu().numpy()
        wdl = F.softmax(wdl_logits.cpu(), dim=-1).numpy()

        out = []
        for i, b in enumerate(boards):
            legal = list(b.legal_moves)
            if not legal:
                if getattr(self.net, "value_head_type", "wdl") == "hlgauss":
                    from src.model.value_dist import bucket_centers
                    probs_v = wdl[i]                        # already softmax over K buckets
                    centers = bucket_centers(len(probs_v)).numpy()
                    vhat = float((probs_v * centers).sum())
                    _no_legal_value = 2.0 * vhat - 1.0
                else:
                    _no_legal_value = float(wdl[i, 0] - wdl[i, 2])
                out.append(({}, _no_legal_value))
                continue
            idxs = [self.me.encode(to_canonical_move(mv, b.turn)) for mv in legal]
            logits = pol_logits[i][idxs]
            if self.objective == "action_value":
                # sigmoid(logit) = predicted Q(s,a) = win%, side-to-move POV.
                q = 1.0 / (1.0 + np.exp(-logits))
                probs = q / q.sum() if q.sum() > 0 else np.full(len(q), 1.0 / len(q))
                policy = {mv: float(p) for mv, p in zip(legal, probs)}
                value = float(2.0 * q.max() - 1.0)  # V(s) ~ best move's win%, in [-1,1]
            else:
                logits = logits - logits.max()
                probs = np.exp(logits)
                probs = probs / probs.sum()
                policy = {mv: float(p) for mv, p in zip(legal, probs)}
                if getattr(self.net, "value_head_type", "wdl") == "hlgauss":
                    from src.model.value_dist import bucket_centers
                    probs_v = wdl[i]                        # already softmax over K buckets
                    centers = bucket_centers(len(probs_v)).numpy()
                    vhat = float((probs_v * centers).sum())
                    value = 2.0 * vhat - 1.0
                else:
                    value = float(wdl[i, 0] - wdl[i, 2])  # P(W) - P(L)
            out.append((policy, value))
        return out
