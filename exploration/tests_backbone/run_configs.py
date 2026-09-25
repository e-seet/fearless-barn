#!/usr/bin/env python3
"""Train backbone variants (PREREGISTERED.md) on forward or seasonal_full.

Same data pipeline, optimizer and early stopping as tests_hp_diversity/run_configs.py;
only the model changes. For k > 1 (TabM) the training loss is the mean of the
per-member MSEs and predictions are the member mean.

usage: python tests_backbone/run_configs.py base ple tabm tabm_ple [--split seasonal_full]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import config
from common import dataset as D
from common import features as F
from common.bootstrap import seed_everything
from pipeline_mlp_antigravity.train import encode_data
from tests_backbone.models import BackboneMLP

FS, POL, PATIENCE, BATCH, N_BINS = "with_aobt", "keep_all", 6, 2048, 16
CLIP = (config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
STOP_FLAG = HERE / "STOP"
CONFIGS = {"base": dict(ple=False, k=1), "ple": dict(ple=True, k=1),
           "tabm": dict(ple=False, k=32), "tabm_ple": dict(ple=True, k=32)}


def quantile_edges(num: torch.Tensor, n_bins: int) -> torch.Tensor:
    """(F, n_bins+1) strictly-increasing-where-possible quantile edges; bins that
    collapse (repeated quantiles) get zero width and are masked inside PLE."""
    x = num.float().cpu().numpy()
    rows = []
    for j in range(x.shape[1]):
        q = np.quantile(x[:, j], np.linspace(0, 1, n_bins + 1))
        u = np.unique(q)
        pad = np.full(n_bins + 1 - len(u), u[-1])
        rows.append(np.concatenate([u, pad]))
    return torch.tensor(np.stack(rows), dtype=torch.float32)


@torch.no_grad()
def predict(model, cat, num, bs=8192):
    model.eval()
    out = [model.predict_mean({c: cat[c][i:i + bs] for c in cat}, num[i:i + bs]).float().cpu().numpy()
           for i in range(0, len(num), bs)]
    return np.clip(np.concatenate(out), *CLIP)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", choices=list(CONFIGS))
    ap.add_argument("--seeds", default="42,43,44,45,46,47")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--split", default="forward", choices=["forward", "seasonal_full"])
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(",")]
    SPLIT, pre = a.split, ("" if a.split == "forward" else a.split + "_")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr, ytr, _ = D.load_part(SPLIT, "train", FS, filter_target=True, outlier_policy=POL)
    Xva, yva, mva = D.load_part(SPLIT, "val", FS, filter_target=False)
    Xte, yte, mte = D.load_part(SPLIT, "test", FS, filter_target=False)
    Xva, Xte = D.align_columns(Xtr, Xva), D.align_columns(Xtr, Xte)
    yva, yte = np.asarray(yva, float), np.asarray(yte, float)
    cat_cols = [c for c in F.CATEGORICAL if c in Xtr.columns]
    num_cols = [c for c in Xtr.columns if c not in cat_cols and pd.api.types.is_numeric_dtype(Xtr[c])]
    cat_maps, card = {}, {}
    for col in cat_cols:
        cats = sorted(Xtr[col].dropna().astype("object").unique())
        cat_maps[col] = {v: i for i, v in enumerate(cats)}
        card[col] = len(cats)
    med = Xtr[num_cols].astype("float64").median()
    sc = StandardScaler().fit(Xtr[num_cols].astype("float64").fillna(med).to_numpy())
    todev = lambda cn: ({c: t.to(dev) for c, t in cn[0].items()}, cn[1].to(dev))
    tr = todev(encode_data(Xtr, cat_cols, num_cols, cat_maps, sc, med))
    va = todev(encode_data(Xva, cat_cols, num_cols, cat_maps, sc, med))
    te = todev(encode_data(Xte, cat_cols, num_cols, cat_maps, sc, med))
    edges = quantile_edges(tr[1], N_BINS).to(dev)
    y_t = torch.tensor(np.asarray(ytr, dtype="float32"), device=dev)
    n = len(y_t)
    meta_path = HERE / f"meta{'' if SPLIT == 'forward' else '_' + SPLIT}.npz"
    if not meta_path.exists():
        np.savez(meta_path, yva=yva, yte=yte, apva=mva["ADEP_mvt"].astype(str).to_numpy(),
                 apte=mte["ADEP_mvt"].astype(str).to_numpy())

    for cname in a.configs:
        cfg = CONFIGS[cname]
        for seed in seeds:
            out = HERE / f"preds_{pre}{cname}_seed{seed}.npz"
            if out.exists():
                print(f"{cname} seed {seed} exists, skip", flush=True)
                continue
            if STOP_FLAG.exists():
                print("STOP flag present -- exiting", flush=True)
                return 0
            t0 = time.time()
            seed_everything(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = BackboneMLP(card, len(num_cols), prior_mean=float(np.mean(ytr)),
                                ple_edges=edges if cfg["ple"] else None, k=cfg["k"]).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            best, best_ep, bad, bpv, bpt, hist = 1e18, 0, 0, None, None, []
            for ep in range(1, a.epochs + 1):
                model.train()
                perm = torch.randperm(n, device=dev)
                for i in range(0, n, BATCH):
                    b = perm[i:i + BATCH]
                    opt.zero_grad()
                    out_k = model({c: tr[0][c][b] for c in tr[0]}, tr[1][b])      # (B, k)
                    loss = ((out_k - y_t[b].unsqueeze(1)) ** 2).mean()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    opt.step()
                pv = predict(model, *va)
                r = float(np.sqrt(np.mean((yva - pv) ** 2)))
                hist.append(r)
                if r < best:
                    best, best_ep, bad, bpv, bpt = r, ep, 0, pv, predict(model, *te)
                else:
                    bad += 1
                    if bad >= PATIENCE:
                        break
            hit_cap = ep >= a.epochs and bad < PATIENCE
            np.savez(out, pv=bpv, pt=bpt, best_epoch=best_ep, val=best, epochs_run=ep,
                     hit_cap=hit_cap, history=np.asarray(hist), config=cname)
            print(f"{cname} seed {seed}: best_ep {best_ep} val {best:.2f} ran {ep} "
                  f"[{time.time()-t0:.0f}s]{'  *** HIT CAP ***' if hit_cap else ''}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
