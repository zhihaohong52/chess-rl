"""ChessTransformer (PyTorch): token board -> (policy_logits, wdl_logits, moves_left)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.smolgen import Smolgen
from src.model.heads import PolicyHead, ValueHead, MovesLeftHead, DistributionalValueHead


class BiasedMHA(nn.Module):
    """Multi-head self-attention with an additive per-head bias on the logits."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h = n_heads
        self.d = d_model
        self.dh = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, x, bias):  # x: [B,T,d]; bias: [B,h,T,T]
        b, t, _ = x.shape

        def split(y):
            return y.reshape(b, t, self.h, self.dh).transpose(1, 2)  # [B,h,T,dh]

        q, k, v = split(self.wq(x)), split(self.wk(x)), split(self.wv(x))
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.dh ** 0.5) + bias
        a = torch.softmax(scores, dim=-1)
        o = torch.matmul(a, v).transpose(1, 2).reshape(b, t, self.d)
        return self.wo(o)


class EncoderLayer(nn.Module):
    def __init__(self, cfg, shared_smolgen_out):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = BiasedMHA(cfg.d_model, cfg.n_heads)
        self.smolgen = Smolgen(cfg.d_model, cfg.n_heads, cfg.smolgen_compress,
                               cfg.smolgen_hidden, cfg.smolgen_gen, shared_smolgen_out)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d_model)
        )

    def forward(self, x):  # x: [B, 65, d] (index 0 = CLS, 1..64 = squares)
        h = self.ln1(x)
        bias64 = self.smolgen(h[:, 1:, :])             # [B,h,64,64]
        bias = F.pad(bias64, (1, 0, 1, 0))             # CLS row/col = 0 -> [B,h,65,65]
        x = x + self.attn(h, bias)
        x = x + self.ffn(self.ln2(x))
        return x


class ChessTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.piece_emb = nn.Embedding(13, d)
        self.pos_emb = nn.Parameter(torch.randn(1, 64, d) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.state_mlp = nn.Sequential(
            nn.Linear(cfg.state_dim, d), nn.GELU(), nn.Linear(d, d)
        )
        shared_smolgen_out = nn.Linear(cfg.smolgen_gen, 64 * 64, bias=False)
        self.enc = nn.ModuleList(
            [EncoderLayer(cfg, shared_smolgen_out) for _ in range(cfg.n_layers)]
        )
        self.final_ln = nn.LayerNorm(d)
        self.policy_head = PolicyHead(d)
        self.value_head_type = getattr(cfg, "value_head_type", "wdl")
        self.value_buckets = getattr(cfg, "value_buckets", 64)
        if self.value_head_type == "hlgauss":
            self.value_head = DistributionalValueHead(d, self.value_buckets)
        else:
            self.value_head = ValueHead(d)
        self.moves_left_head = MovesLeftHead(d)

    def forward(self, square_tokens, state_features):
        """square_tokens: [B,64] long; state_features: [B,18] float.

        Returns (policy_logits[B,P], value_out[B,3]|[B,K], moves_left[B,1]).
        """
        pe = self.piece_emb(square_tokens) + self.pos_emb       # [B,64,d]
        cond = self.state_mlp(state_features).unsqueeze(1)      # [B,1,d]
        pe = pe + cond
        b = pe.shape[0]
        cls = self.cls.expand(b, -1, -1) + cond                 # [B,1,d]
        x = torch.cat([cls, pe], dim=1)                         # [B,65,d]
        for layer in self.enc:
            x = layer(x)
        x = self.final_ln(x)
        cls_out = x[:, 0, :]
        sq_out = x[:, 1:, :]
        return self.policy_head(sq_out), self.value_head(cls_out), self.moves_left_head(cls_out)

    @torch.no_grad()
    def predict_batch(self, square_tokens, state_features):
        """Eval-mode forward for inference (no grad)."""
        self.eval()
        return self.forward(square_tokens, state_features)
