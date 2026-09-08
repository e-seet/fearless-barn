"""PyTorch Tabular MLP with Entity Embeddings for categorical features.

Provides inductive bias diversity against gradient boosted trees:
- Entity embeddings for high-cardinality geometry (stand_runway, adep_stand, etc.)
- Batch normalization and dropout for continuous tabular features
- Residual feed-forward blocks with GELU activations
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TabularMLP(nn.Module):
    def __init__(self, cat_cardinalities: dict[str, int], num_features: int,
                 hidden_dims: tuple[int, ...] = (256, 128, 64),
                 dropout: float = 0.1, prior_mean: float = 990.0):
        super().__init__()
        self.cat_cols = list(cat_cardinalities.keys())
        self.embeddings = nn.ModuleDict()
        total_emb_dim = 0

        for col, card in cat_cardinalities.items():
            emb_dim = min(48, max(4, int(card ** 0.5)))
            # card + 1 to account for unknown/unseen categories
            self.embeddings[col] = nn.Embedding(card + 1, emb_dim)
            total_emb_dim += emb_dim

        self.num_bn = nn.BatchNorm1d(num_features) if num_features > 0 else nn.Identity()

        in_dim = total_emb_dim + num_features
        layers = []
        prev_dim = in_dim

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

    def forward(self, x_cat: dict[str, torch.Tensor], x_num: torch.Tensor) -> torch.Tensor:
        emb_outs = []
        for col in self.cat_cols:
            emb_outs.append(self.embeddings[col](x_cat[col]))

        x_parts = []
        if emb_outs:
            x_parts.append(torch.cat(emb_outs, dim=1))
        if x_num.shape[1] > 0:
            x_parts.append(self.num_bn(x_num))

        x = torch.cat(x_parts, dim=1)
        feat = self.backbone(x)
        out = self.head(feat).squeeze(-1)
        return out
