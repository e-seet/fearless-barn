#!/usr/bin/env python3
"""Season-matched sample weighting for the MLP (PREREGISTERED.md).

Production mlp_antigravity recipe on the fast GPU loop; the only change is a
per-row weight exp(-d/TAU) in the MSE, d = circular month distance to the target.

usage: python tests_season_weight/run.py --split seasonal_full --targets 1,7
       python tests_season_weight/run.py --split forward --targets 11,12
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
from pipeline_mlp_antigravity.model import TabularMLP
from pipeline_mlp_antigravity.train import encode_data

FS, POL, BATCH, PATIENCE, TAU = "with_aobt", "keep_all", 2048, 6, 3.0
CLIP = (config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
STOP_FLAG = HERE / "STOP"


def month_of(meta) -> np.ndarray:
    return pd.to_datetime(meta["month"].astype(str)).dt.month.to_numpy()


def circ(a, b):
    d = np.abs(a - b) % 12
    return np.minimum(d, 12 - d)


@torch.no_grad()
def predict(model, cat, num, bs=8192):
    model.eval()
    out = [model({c: cat[c][i:i + bs] for c in cat}, num[i:i + bs]).float().cpu().numpy()
           for i in range(0, len(num), bs)]
    return np.clip(np.concatenate(out), *CLIP)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["forward", "seasonal_full"])
    ap.add_argument("--targets", required=True, help="comma-separated target months, e.g. 1,7")
    ap.add_argument("--seeds", default="42,43,44,45,46,47")
    ap.add_argument("--epochs", type=int, default=50)
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(",")]
    targets = [int(t) for t in a.targets.split(",")]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr, ytr, mtr = D.load_part(a.split, "train", FS, filter_target=True, outlier_policy=POL)
    Xva, yva, mva = D.load_part(a.split, "val", FS, filter_target=False)
    Xte, yte, mte = D.load_part(a.split, "test", FS, filter_target=False)
    Xva, Xte = D.align_columns(Xtr, Xva), D.align_columns(Xtr, Xte)
    yva, yte = np.asarray(yva, float), np.asarray(yte, float)
    assert len(Xtr) == len(ytr) == len(mtr)
    m_tr = month_of(mtr)
    print(f"{a.split}: train months {sorted(set(m_tr))}, n={len(Xtr)}", flush=True)
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
    y_t = torch.tensor(np.asarray(ytr, dtype="float32"), device=dev)
    n = len(y_t)
    meta_path = HERE / f"meta_{a.split}.npz"
    if not meta_path.exists():
        np.savez(meta_path, yva=yva, yte=yte, apte=mte["ADEP_mvt"].astype(str).to_numpy(),
                 mte=month_of(mte))

    modes = [("base", None)] + [(f"t{t:02d}", t) for t in targets]
    for mode, t in modes:
        w_np = np.ones(n) if t is None else np.exp(-circ(m_tr, t) / TAU)
        w_t = torch.tensor(w_np.astype("float32"), device=dev)
        if t is not None:
            print(f"{mode}: effective sample {w_np.sum()**2 / (w_np**2).sum() / n:.1%} of n", flush=True)
        for seed in seeds:
            out = HERE / f"preds_{a.split}_{mode}_seed{seed}.npz"
            if out.exists():
                print(f"{out.name} exists, skip", flush=True)
                continue
            if STOP_FLAG.exists():
                print("STOP flag present -- exiting", flush=True)
                return 0
            t0 = time.time()
            seed_everything(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = TabularMLP(card, num_features=len(num_cols), hidden_dims=(256, 128, 64),
                               dropout=0.1, prior_mean=float(np.average(ytr, weights=w_np))).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            best, best_ep, bad, bpv, bpt, hist = 1e18, 0, 0, None, None, []
            for ep in range(1, a.epochs + 1):
                model.train()
                perm = torch.randperm(n, device=dev)
                for i in range(0, n, BATCH):
                    b = perm[i:i + BATCH]
                    opt.zero_grad()
                    r2 = (model({c: tr[0][c][b] for c in tr[0]}, tr[1][b]) - y_t[b]) ** 2
                    loss = (r2 * w_t[b]).sum() / w_t[b].sum()
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
                     hit_cap=hit_cap, history=np.asarray(hist), target_month=-1 if t is None else t, tau=TAU)
            print(f"{mode} seed {seed}: best_ep {best_ep} val {best:.2f} ran {ep} [{time.time()-t0:.0f}s]"
                  f"{'  *** HIT CAP ***' if hit_cap else ''}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
