"""GRU sequence encoder + entity-embedding tabular MLP, joint model.

The tabular half (embeddings, numeric batchnorm, feed-forward backbone) is
IDENTICAL in shape to `pipeline_mlp_antigravity.model.TabularMLP` so any
gain/loss versus that arm is attributable to the added sequence branch, not
a capacity or preprocessing difference. The GRU processes the backward-only
movement sequence from `sequence_features.py` (oldest to most recent) and
its final hidden state is concatenated in as one more input block.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SeqTabularModel(nn.Module):
    def __init__(self, cat_cardinalities: dict, num_features: int,
                 seq_feature_dim: int = 4, gru_hidden: int = 32,
                 hidden_dims: tuple = (256, 128, 64), dropout: float = 0.1,
                 prior_mean: float = 990.0):
        super().__init__()
        self.cat_cols = list(cat_cardinalities.keys())
        self.embeddings = nn.ModuleDict()
        total_emb_dim = 0
        for col, card in cat_cardinalities.items():
            emb_dim = min(48, max(4, int(card ** 0.5)))
            self.embeddings[col] = nn.Embedding(card + 1, emb_dim)
            total_emb_dim += emb_dim

        self.num_bn = nn.BatchNorm1d(num_features) if num_features > 0 else nn.Identity()

        # mask is appended as an extra per-step channel so the GRU can learn
        # to discount padded (no-history) steps rather than reading a zero
        # vector as "a real simultaneous event with zero gap".
        self.gru = nn.GRU(input_size=seq_feature_dim + 1, hidden_size=gru_hidden,
                          num_layers=1, batch_first=True)
        self.gru_hidden = gru_hidden

        in_dim = total_emb_dim + num_features + gru_hidden
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

    def forward(self, x_cat: dict, x_num: torch.Tensor,
                x_seq: torch.Tensor, x_seq_mask: torch.Tensor) -> torch.Tensor:
        emb_outs = [self.embeddings[col](x_cat[col]) for col in self.cat_cols]

        seq_in = torch.cat([x_seq, x_seq_mask.unsqueeze(-1)], dim=-1)
        _, h_n = self.gru(seq_in)          # h_n: (1, batch, gru_hidden)
        h_final = h_n[-1]                   # (batch, gru_hidden)

        x_parts = []
        if emb_outs:
            x_parts.append(torch.cat(emb_outs, dim=1))
        if x_num.shape[1] > 0:
            x_parts.append(self.num_bn(x_num))
        x_parts.append(h_final)

        x = torch.cat(x_parts, dim=1)
        feat = self.backbone(x)
        return self.head(feat).squeeze(-1)
