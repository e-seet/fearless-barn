"""PyTorch Tabular MLP with Entity Embeddings for categorical features.

Provides inductive bias diversity against gradient boosted trees:
- Entity embeddings for high-cardinality geometry (stand_runway, adep_stand, etc.)
- Batch normalization and dropout for continuous tabular features
- Feed-forward blocks with GELU activations, plain by default; pass
  `residual=True` for projected-skip-connection blocks instead (see
  `_ResidualBlock`) -- this flag exists specifically to A/B test that shape.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _ResidualBlock(nn.Module):
    """Linear->BN->GELU->Dropout with a (projected, if needed) skip connection.

    The docstring above has claimed "residual feed-forward blocks" since this
    file was written, but the backbone was a plain `nn.Sequential` with no
    skip connections at all -- this is the actual implementation, gated by
    `residual=True` so the untested default path is unchanged.
    """
    def __init__(self, prev_dim: int, h_dim: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(prev_dim, h_dim)
        self.bn = nn.BatchNorm1d(h_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.shortcut = nn.Identity() if prev_dim == h_dim else nn.Linear(prev_dim, h_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop(self.act(self.bn(self.linear(x))))
        return out + self.shortcut(x)


class TabularMLP(nn.Module):
    def __init__(self, cat_cardinalities: dict[str, int], num_features: int,
                 hidden_dims: tuple[int, ...] = (256, 128, 64),
                 dropout: float = 0.1, prior_mean: float = 990.0,
                 residual: bool = False):
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
        prev_dim = in_dim

        if residual:
            blocks = []
            for h_dim in hidden_dims:
                blocks.append(_ResidualBlock(prev_dim, h_dim, dropout))
                prev_dim = h_dim
            self.backbone = nn.ModuleList(blocks)
        else:
            layers = []
            for h_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(nn.BatchNorm1d(h_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
                prev_dim = h_dim
            self.backbone = nn.Sequential(*layers)

        self.residual = residual
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
        if self.residual:
            feat = x
            for block in self.backbone:
                feat = block(feat)
        else:
            feat = self.backbone(x)
        out = self.head(feat).squeeze(-1)
        return out
