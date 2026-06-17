"""Policy (attention from->to), WDL value, and moves-left heads. PyTorch."""

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.game.move_encoder import get_move_encoder

_PROMO_CLASS = {chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3}


def build_policy_index_map():
    """Return (from_idx[P], to_idx[P], promo_class[P]) int64 arrays from the MoveEncoder."""
    me = get_move_encoder()
    p = me.policy_size
    from_idx = np.zeros(p, dtype=np.int64)
    to_idx = np.zeros(p, dtype=np.int64)
    promo = np.zeros(p, dtype=np.int64)
    for i in range(p):
        mv = me.idx_to_move[i]
        from_idx[i] = mv.from_square
        to_idx[i] = mv.to_square
        promo[i] = _PROMO_CLASS.get(mv.promotion, 0)
    return from_idx, to_idx, promo


class PolicyHead(nn.Module):
    def __init__(self, d_model, d_attn=64):
        super().__init__()
        fi, ti, pr = build_policy_index_map()
        self.register_buffer("flat_ft", torch.tensor(fi * 64 + ti, dtype=torch.long))
        up_idx = np.where(pr > 0, fi * 3 + np.maximum(pr - 1, 0), 0).astype(np.int64)
        self.register_buffer("flat_up", torch.tensor(up_idx, dtype=torch.long))
        self.register_buffer("up_mask", torch.tensor((pr > 0).astype(np.float32)))
        self.q = nn.Linear(d_model, d_attn, bias=False)
        self.k = nn.Linear(d_model, d_attn, bias=False)
        self.up = nn.Linear(d_model, 3)
        self.scale = float(d_attn) ** 0.5

    def forward(self, sq):  # sq: [B, 64, d]
        b = sq.shape[0]
        q = self.q(sq)
        k = self.k(sq)
        scores = torch.matmul(q, k.transpose(1, 2)) / self.scale  # [B, 64, 64]
        base = scores.reshape(b, 64 * 64)[:, self.flat_ft]        # [B, P]
        up = self.up(sq).reshape(b, 64 * 3)                       # [B, 192]
        up_term = up[:, self.flat_up] * self.up_mask              # [B, P]
        return base + up_term


class ValueHead(nn.Module):
    def __init__(self, d_model, hidden=128):
        super().__init__()
        self.d1 = nn.Linear(d_model, hidden)
        self.d2 = nn.Linear(hidden, 3)

    def forward(self, cls):
        return self.d2(F.relu(self.d1(cls)))


class MovesLeftHead(nn.Module):
    def __init__(self, d_model, hidden=128):
        super().__init__()
        self.d1 = nn.Linear(d_model, hidden)
        self.d2 = nn.Linear(hidden, 1)

    def forward(self, cls):
        return F.softplus(self.d2(F.relu(self.d1(cls))))
