"""TabularMLP + attention pooling over the departure-queue neighbour set.

Identical to `pipeline_mlp_antigravity.model.TabularMLP` except the input to
the backbone also includes a learned context vector summarising the K
temporally-nearest other departures (see `neighbors.py`), via a single-head
masked attention-pooling layer -- not a fixed aggregate (mean/count), so the
model can weigh e.g. "the one same-runway neighbour 4 minutes ago" very
differently from "eight neighbours spread across 90 minutes."
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _QueueAttention(nn.Module):
    def __init__(self, n_features: int = 3, d_model: int = 16, d_ctx: int = 16):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.score = nn.Linear(d_model, 1)
        self.out = nn.Linear(d_model, d_ctx)

    def forward(self, neighbor_feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """neighbor_feats: (B, K, n_features), mask: (B, K) in {0, 1}."""
        h = torch.tanh(self.proj(neighbor_feats))          # (B, K, d_model)
        scores = self.score(h).squeeze(-1)                  # (B, K)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        # A fully-masked row (no neighbours in window) has all scores == -1e9,
        # so softmax degenerates to a uniform distribution over non-existent
        # neighbours -- zero it explicitly rather than let attention "invent"
        # a queue for flights that had none.
        has_any = (mask.sum(dim=1, keepdim=True) > 0).float()
        weights = weights * mask * has_any
        ctx = (weights.unsqueeze(-1) * h).sum(dim=1)         # (B, d_model)
        return self.out(ctx) * has_any                       # (B, d_ctx)


class TabularMLPQueue(nn.Module):
    def __init__(self, cat_cardinalities: dict[str, int], num_features: int,
                 hidden_dims: tuple[int, ...] = (256, 128, 64),
                 dropout: float = 0.1, prior_mean: float = 990.0,
                 queue_n_features: int = 3, queue_d_model: int = 16,
                 queue_d_ctx: int = 16):
        super().__init__()
        self.cat_cols = list(cat_cardinalities.keys())
        self.embeddings = nn.ModuleDict()
        total_emb_dim = 0

        for col, card in cat_cardinalities.items():
            emb_dim = min(48, max(4, int(card ** 0.5)))
            self.embeddings[col] = nn.Embedding(card + 1, emb_dim)
            total_emb_dim += emb_dim

        self.num_bn = nn.BatchNorm1d(num_features) if num_features > 0 else nn.Identity()
        self.queue_attn = _QueueAttention(queue_n_features, queue_d_model, queue_d_ctx)

        in_dim = total_emb_dim + num_features + queue_d_ctx
        prev_dim = in_dim
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.backbone = nn.Sequential(*layers)

        self.head = nn.Linear(prev_dim, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, prior_mean)

    def forward(self, x_cat: dict[str, torch.Tensor], x_num: torch.Tensor,
               neighbor_feats: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
        emb_outs = [self.embeddings[col](x_cat[col]) for col in self.cat_cols]

        x_parts = []
        if emb_outs:
            x_parts.append(torch.cat(emb_outs, dim=1))
        if x_num.shape[1] > 0:
            x_parts.append(self.num_bn(x_num))
        x_parts.append(self.queue_attn(neighbor_feats, neighbor_mask))

        x = torch.cat(x_parts, dim=1)
        feat = self.backbone(x)
        return self.head(feat).squeeze(-1)
