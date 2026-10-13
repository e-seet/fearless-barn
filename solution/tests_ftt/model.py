"""FT-Transformer (Gorishniy et al. 2021) -- attention over FEATURES, not neighbours.

Why this architecture and not another one: every architecture this repo has
closed varies how the model consumes the *movement stream* -- the GRU state
encoder (`tests_seqrnn`, residual corr 0.980 vs the MLP), bidirectional
queue-attention (`pipeline_queueattn_antigravity`), the matched/unmatched
router, TabR-style retrieval. `tests_seqrnn/RESULT.md` states the conclusion
outright: richer per-flight neighbour/sequence representation is foreclosed.

Column-attention is a different inductive bias and the one tabular family with
no entry in this build. Each feature becomes a token and the blocks learn
feature-pair interactions directly, rather than through the MLP's single
concatenated vector. The only claim being tested is whether that produces
DECORRELATED errors -- see `decorrelation.py` for the pre-registered bar.

Deliberately the same data pipeline, optimizer, early stopping and output format
as `tests_lirf_seeds/train_seeds.py` (the fast mlp_antigravity loop), so any
delta is attributable to the architecture and the saved predictions drop
straight into the existing screen.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """Every feature -> one d_token vector. Numericals get a per-feature weight
    vector and bias (x_i * W_i + b_i); categoricals get an embedding table."""

    def __init__(self, cat_cardinalities: "dict[str, int]", num_features: int,
                 d_token: int = 64):
        super().__init__()
        self.cat_cols = list(cat_cardinalities.keys())
        self.num_features = num_features
        self.d_token = d_token

        self.cat_emb = nn.ModuleDict()
        for col, card in cat_cardinalities.items():
            # card + 1 for unseen categories, matching TabularMLP's convention
            self.cat_emb[col] = nn.Embedding(card + 1, d_token)

        if num_features > 0:
            self.num_weight = nn.Parameter(torch.empty(num_features, d_token))
            self.num_bias = nn.Parameter(torch.empty(num_features, d_token))
            nn.init.normal_(self.num_weight, std=d_token ** -0.5)
            nn.init.normal_(self.num_bias, std=d_token ** -0.5)

        self.cls = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls, std=d_token ** -0.5)

    def forward(self, x_cat: "dict[str, torch.Tensor]", x_num: torch.Tensor) -> torch.Tensor:
        toks = []
        for col in self.cat_cols:
            toks.append(self.cat_emb[col](x_cat[col]).unsqueeze(1))   # (B, 1, d)
        if self.num_features > 0:
            # (B, F, 1) * (F, d) -> (B, F, d)
            toks.append(x_num.unsqueeze(-1) * self.num_weight + self.num_bias)
        x = torch.cat(toks, dim=1)
        cls = self.cls.expand(x.shape[0], -1, -1)
        return torch.cat([cls, x], dim=1)                             # (B, 1+F, d)


class _Block(nn.Module):
    """Pre-norm MHSA + FFN, the FT-Transformer arrangement."""

    def __init__(self, d_token: int, n_heads: int, ffn_factor: float,
                 attn_dropout: float, ffn_dropout: float, residual_dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(d_token, n_heads, dropout=attn_dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(d_token)
        d_hidden = int(d_token * ffn_factor)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_hidden),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_hidden, d_token),
        )
        self.res_drop = nn.Dropout(residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.res_drop(a)
        x = x + self.res_drop(self.ffn(self.norm2(x)))
        return x


class FTTransformer(nn.Module):
    def __init__(self, cat_cardinalities: "dict[str, int]", num_features: int,
                 d_token: int = 64, n_blocks: int = 3, n_heads: int = 8,
                 ffn_factor: float = 4 / 3, attn_dropout: float = 0.2,
                 ffn_dropout: float = 0.1, residual_dropout: float = 0.0,
                 prior_mean: float = 990.0):
        super().__init__()
        if d_token % n_heads:
            raise ValueError(f"d_token {d_token} must be divisible by n_heads {n_heads}")
        self.tokenizer = FeatureTokenizer(cat_cardinalities, num_features, d_token)
        self.blocks = nn.ModuleList([
            _Block(d_token, n_heads, ffn_factor, attn_dropout, ffn_dropout,
                   residual_dropout)
            for _ in range(n_blocks)
        ])
        self.head_norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)
        # Same trick as TabularMLP: start at the training mean so the first
        # epochs are spent learning structure, not the intercept.
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, prior_mean)

    def forward(self, x_cat: "dict[str, torch.Tensor]", x_num: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(x_cat, x_num)
        for blk in self.blocks:
            x = blk(x)
        # FT-Transformer reads out from the [CLS] token only.
        return self.head(nn.functional.gelu(self.head_norm(x[:, 0]))).squeeze(-1)
