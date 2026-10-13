"""Backbone variants for tests_backbone (see PREREGISTERED.md).

`BackboneMLP` reproduces `pipeline_mlp_antigravity.model.TabularMLP` exactly when
`ple_edges is None and k == 1` (same categorical embedding sizes, numeric
BatchNorm, Linear-BN-GELU-Dropout blocks, zero-initialised head at the prior
mean), and adds two independent switches:

* `ple_edges`: (F, T+1) quantile bin edges per numeric feature. The numeric
  path becomes piecewise-linear encoding, then a per-feature Linear(T -> d_ple)
  and ReLU, flattened (Gorishniy et al. 2022, "PLE" + linear + ReLU).
* `k > 1`: TabM-mini (Gorishniy et al. 2024). The input row is replicated k
  times and multiplied by a per-member adapter (random +-1 init). The backbone
  is shared, and there are k heads. forward() returns (B, k) in training;
  `predict_mean` averages the members.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PLE(nn.Module):
    def __init__(self, edges: torch.Tensor, d_out: int = 16):
        super().__init__()
        lo, hi = edges[:, :-1], edges[:, 1:]
        width = hi - lo
        self.register_buffer("lo", lo)
        self.register_buffer("width", width.clamp(min=1e-6))
        self.register_buffer("valid", (width > 1e-9).float())   # padded bins -> 0
        F, T = lo.shape
        self.weight = nn.Parameter(torch.empty(F, T, d_out))
        self.bias = nn.Parameter(torch.zeros(F, d_out))
        nn.init.normal_(self.weight, std=T ** -0.5)
        self.out_dim = F * d_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x (B, F)
        enc = ((x.unsqueeze(-1) - self.lo) / self.width).clamp(0.0, 1.0) * self.valid
        h = torch.einsum("bft,ftd->bfd", enc, self.weight) + self.bias
        return torch.relu(h).flatten(1)


class BackboneMLP(nn.Module):
    def __init__(self, cat_cardinalities: "dict[str, int]", num_features: int,
                 hidden_dims=(256, 128, 64), dropout: float = 0.1, prior_mean: float = 990.0,
                 ple_edges: "torch.Tensor | None" = None, d_ple: int = 16, k: int = 1):
        super().__init__()
        self.cat_cols = list(cat_cardinalities)
        self.embeddings = nn.ModuleDict()
        emb_dim_total = 0
        for col, card in cat_cardinalities.items():
            d = min(48, max(4, int(card ** 0.5)))
            self.embeddings[col] = nn.Embedding(card + 1, d)
            emb_dim_total += d
        if ple_edges is not None:
            self.num = PLE(ple_edges, d_ple)
            num_dim = self.num.out_dim
        else:
            self.num = nn.BatchNorm1d(num_features)
            num_dim = num_features
        in_dim = emb_dim_total + num_dim
        self.k = k
        if k > 1:
            self.adapter = nn.Parameter(torch.randint(0, 2, (k, in_dim)).float() * 2 - 1)
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head_w = nn.Parameter(torch.zeros(k, prev))
        self.head_b = nn.Parameter(torch.full((k,), float(prior_mean)))

    def forward(self, x_cat, x_num) -> torch.Tensor:
        parts = [torch.cat([self.embeddings[c](x_cat[c]) for c in self.cat_cols], dim=1),
                 self.num(x_num)]
        x = torch.cat(parts, dim=1)                                   # (B, D)
        B = x.shape[0]
        if self.k > 1:
            x = (x.unsqueeze(1) * self.adapter).reshape(B * self.k, -1)
        h = self.backbone(x).reshape(B, self.k, -1)                   # (B, k, H)
        return (h * self.head_w).sum(-1) + self.head_b                # (B, k)

    @torch.no_grad()
    def predict_mean(self, x_cat, x_num) -> torch.Tensor:
        return self.forward(x_cat, x_num).mean(1)
