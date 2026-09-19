#!/usr/bin/env python3
"""Experiment: snapshot / weight averaging inside one MLP training run, as a
variance-reduction lever for `mlp_antigravity`.

Motivation (tests_local_vs_board/RESULT.md): the board pays much more for MLP
variance reduction than local val shows (1->3 seeds: local val -0.9, board
-18). Seed-averaging has plateaued at 12-18 seeds, so this asks whether the
epoch-to-epoch noise INSIDE a run (val RMSE swings +-10 between adjacent
epochs at constant LR 1e-3) can be averaged away cheaply, and whether that
still helps once seeds are averaged too.

Same model, data and hyperparameters as pipeline_mlp_antigravity/train.py
(TabularMLP (256,128,64), dropout 0.1, AdamW lr 1e-3 wd 1e-4, batch 2048,
MSE, grad-clip 5, forward/with_aobt/keep_all) EXCEPT: (1) fixed EPOCHS (default 60; the 25-epoch first attempt did NOT converge, see analysis_e25_NOT_CONVERGED.txt), no
early stopping, so every snapshot exists; (2) batching is done by direct GPU
tensor indexing instead of a per-row Dataset (mathematically the same
shuffled-minibatch SGD, ~10x faster). Val AND test predictions are stored
every epoch.

Variants, all scored from ONE run per seed (paired within run -- seed noise
cancels in the contrasts):
  best   argmin-val epoch over all epochs (stand-in for early stopping; biased
         IN FAVOUR of the baseline on val because it is selected on val)
  avg13  ('window') mean prediction over epochs SNAP_LO..EPOCHS (41..60)
  avg21  ('last5') mean prediction over the last 5 epochs
  swa13  ('swa') weights averaged over the window, BN statistics recomputed on a
         300k-row training subsample, then predict

Pre-registered bar (before running): a variant counts only if, on TEST, it
beats `best` in all 3 seeds with mean paired gain >= 3s single-model, AND the
3-seed-ensemble version beats the ensemble of `best` by >= 2s. Val is reported
but not decisive (baseline is val-selected).
"""
from __future__ import annotations

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
from pipeline_mlp_antigravity.model import TabularMLP
from pipeline_mlp_antigravity.train import encode_data

SPLIT, FEATURESET, POLICY = "forward", "with_aobt", "keep_all"
import os
EPOCHS = int(os.environ.get("SNAP_EPOCHS", 60))
BATCH, LR, WD = 2048, 1e-3, 1e-4
SNAP_LO = int(os.environ.get("SNAP_LO", 41))   # first epoch of the averaging window
CLIP_LO, CLIP_HI = config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC


def clip(p):
    return np.clip(p, CLIP_LO, CLIP_HI)


def to_dev(cat, num, dev):
    return {c: t.to(dev) for c, t in cat.items()}, num.to(dev)


@torch.no_grad()
def predict(model, cat, num, dev, bs=8192):
    model.eval()
    n = len(num)
    out = []
    for i in range(0, n, bs):
        xc = {c: cat[c][i:i + bs] for c in cat}
        out.append(model(xc, num[i:i + bs]).float().cpu().numpy())
    return clip(np.concatenate(out))


def main(seed: int) -> None:
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    Xtr, ytr, _ = D.load_part(SPLIT, "train", FEATURESET, filter_target=True, outlier_policy=POLICY)
    Xva, yva, _ = D.load_part(SPLIT, "val", FEATURESET, filter_target=False)
    Xte, yte, _ = D.load_part(SPLIT, "test", FEATURESET, filter_target=False)
    Xva = D.align_columns(Xtr, Xva)
    Xte = D.align_columns(Xtr, Xte)
    yva = np.asarray(yva, dtype="float64")
    yte = np.asarray(yte, dtype="float64")
    cat_cols = [c for c in F.CATEGORICAL if c in Xtr.columns]
    num_cols = [c for c in Xtr.columns if c not in cat_cols and pd.api.types.is_numeric_dtype(Xtr[c])]

    cat_maps, card = {}, {}
    for col in cat_cols:
        cats = sorted(Xtr[col].dropna().astype("object").unique())
        cat_maps[col] = {v: i for i, v in enumerate(cats)}
        card[col] = len(cats)
    medians = Xtr[num_cols].astype("float64").median()
    scaler = StandardScaler().fit(Xtr[num_cols].astype("float64").fillna(medians).to_numpy())

    tr = to_dev(*encode_data(Xtr, cat_cols, num_cols, cat_maps, scaler, medians), dev)
    va = to_dev(*encode_data(Xva, cat_cols, num_cols, cat_maps, scaler, medians), dev)
    te = to_dev(*encode_data(Xte, cat_cols, num_cols, cat_maps, scaler, medians), dev)
    y_t = torch.tensor(np.asarray(ytr, dtype="float32"), device=dev)
    n = len(y_t)
    print(f"[seed {seed}] loaded n_train={n}  [{time.time()-t0:.0f}s]", flush=True)

    model = TabularMLP(card, num_features=len(num_cols), hidden_dims=(256, 128, 64),
                       dropout=0.1, prior_mean=float(np.mean(ytr))).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    crit = nn.MSELoss()

    pv, pt, snaps = [], [], {}
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        sse = 0.0
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            xc = {c: tr[0][c][b] for c in tr[0]}
            opt.zero_grad()
            pred = model(xc, tr[1][b])
            loss = crit(pred, y_t[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            sse += ((pred.detach() - y_t[b]) ** 2).sum().item()
        p_va, p_te = predict(model, *va, dev), predict(model, *te, dev)
        pv.append(p_va); pt.append(p_te)
        if ep >= SNAP_LO:
            snaps[ep] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        r = float(np.sqrt(np.mean((yva - p_va) ** 2)))
        print(f"[seed {seed}] ep {ep:2d} train {np.sqrt(sse/n):7.2f} val {r:7.2f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    # ---- SWA: average weights over the window, recompute BN on a train subsample ----
    avg = {}
    for k in snaps[SNAP_LO]:
        if snaps[SNAP_LO][k].dtype.is_floating_point:
            avg[k] = torch.stack([snaps[e][k] for e in snaps]).mean(0)
        else:
            avg[k] = snaps[EPOCHS][k]
    model.load_state_dict(avg)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.reset_running_stats()
            m.momentum = None
    g = torch.Generator(device=dev); g.manual_seed(seed)
    sub = torch.randperm(n, device=dev, generator=g)[:300_000]
    model.train()
    with torch.no_grad():
        for i in range(0, len(sub), BATCH):
            b = sub[i:i + BATCH]
            model({c: tr[0][c][b] for c in tr[0]}, tr[1][b])
    swa_va, swa_te = predict(model, *va, dev), predict(model, *te, dev)

    np.savez(HERE / f"preds_e{EPOCHS}_seed{seed}.npz", pv=np.array(pv), pt=np.array(pt),
             swa_va=swa_va, swa_te=swa_te, yva=np.asarray(yva, dtype="float64"),
             yte=np.asarray(yte, dtype="float64"))
    print(f"[seed {seed}] saved  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    for s in (int(a) for a in sys.argv[1:]) or (42,):
        main(s)
