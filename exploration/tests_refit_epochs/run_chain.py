#!/usr/bin/env python3
"""Refit-analog epoch curves for the mlp_antigravity recipe. See PREREGISTERED.md.

Per seed, three stages, each trained for a FIXED number of epochs with the
predictions saved at EVERY epoch (float32):

  A  train Jan-Jun  (forward_inner train)        -> scored Jul-Aug
  B  train Jan-Aug  (forward train)              -> scored Sep-Oct, Nov-Dec
  C  train Jan-Oct  (forward train + val)        -> scored Nov-Dec

The recipe is tests_lirf_seeds/train_seeds.py (the production mlp_antigravity
recipe on GPU tensors) with no early stopping. The patience-6 choice is derived
afterwards from the saved curve, which is identical to what a patience run
would have seen, because training does not depend on when you stop it.

usage: python tests_refit_epochs/run_chain.py 42 43 ... [--epochs 50]
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

FS, POL, BATCH = "with_aobt", "keep_all", 2048
CLIP = (config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
STOP_FLAG = HERE / "STOP"


@torch.no_grad()
def predict(model, cat, num, bs=8192):
    model.eval()
    out = []
    for i in range(0, len(num), bs):
        out.append(model({c: cat[c][i:i + bs] for c in cat}, num[i:i + bs]).float().cpu().numpy())
    return np.clip(np.concatenate(out), *CLIP)


def load_seasonal():
    """PREREGISTERED_seasonal.md: one stage S on seasonal_full, scored on Dec (val)
    and Jan+Jul (test) every epoch."""
    Xtr, ytr, _ = D.load_part("seasonal_full", "train", FS, filter_target=True, outlier_policy=POL)
    evals = {}
    for name, part in (("dec", "val"), ("janjul", "test")):
        X, y, m = D.load_part("seasonal_full", part, FS, filter_target=False)
        evals[name] = (X, np.asarray(y, float), m["ADEP_mvt"].astype(str).to_numpy(), X.index.to_numpy())
    return {"S": (Xtr, np.asarray(ytr, float), ["dec", "janjul"])}, evals


def load_stages():
    """{stage: (Xtr, ytr, {eval_name: X})} plus {eval_name: (y, airports)}."""
    fi_tr, fi_ytr, _ = D.load_part("forward_inner", "train", FS, filter_target=True, outlier_policy=POL)
    fw_tr, fw_ytr, _ = D.load_part("forward", "train", FS, filter_target=True, outlier_policy=POL)
    # Stage C's extra training months get the TRAINING outlier policy, exactly as
    # `full` applies it to all 12 months.
    fw_va_f, fw_yva_f, _ = D.load_part("forward", "val", FS, filter_target=True, outlier_policy=POL)
    c_tr = pd.concat([fw_tr, D.align_columns(fw_tr, fw_va_f)])
    c_ytr = pd.concat([pd.Series(np.asarray(fw_ytr, float)), pd.Series(np.asarray(fw_yva_f, float))]).to_numpy()

    evals = {}
    for name, (split, part) in {"julaug": ("forward_inner", "val"),
                                "sepoct": ("forward", "val"),
                                "novdec": ("forward", "test")}.items():
        X, y, m = D.load_part(split, part, FS, filter_target=False)
        evals[name] = (X, np.asarray(y, float), m["ADEP_mvt"].astype(str).to_numpy(), X.index.to_numpy())
    stages = {
        "A": (fi_tr, np.asarray(fi_ytr, float), ["julaug"]),
        "B": (fw_tr, np.asarray(fw_ytr, float), ["sepoct", "novdec"]),
        "C": (c_tr, c_ytr, ["novdec"]),
    }
    return stages, evals


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds", type=int, nargs="+")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seasonal", action="store_true",
                    help="PREREGISTERED_seasonal.md surface instead of stages A/B/C")
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stages, evals = load_seasonal() if a.seasonal else load_stages()
    pre = "seasonal_" if a.seasonal else ""
    # eval metadata, once
    meta_out = HERE / f"meta{'_seasonal' if a.seasonal else ''}.npz"
    if not meta_out.exists():
        np.savez(meta_out, **{f"y_{k}": v[1] for k, v in evals.items()},
                 **{f"ap_{k}": v[2] for k, v in evals.items()},
                 **{f"ids_{k}": v[3] for k, v in evals.items()},
                 **{f"n_train_{s}": len(v[0]) for s, v in stages.items()})
    for s, (Xtr, ytr, ev) in stages.items():
        assert len(Xtr) == len(ytr), (s, len(Xtr), len(ytr))
        print(f"stage {s}: train n={len(Xtr)} -> {ev}", flush=True)

    for seed in a.seeds:
        out = HERE / f"curves_{pre}seed{seed}.npz"
        if out.exists():
            print(f"seed {seed} exists, skip", flush=True)
            continue
        res = {}
        t0 = time.time()
        for s, (Xtr, ytr, ev) in stages.items():
            if STOP_FLAG.exists():
                print("STOP flag present -- exiting, seed unsaved", flush=True)
                return 0
            cat_cols = [c for c in F.CATEGORICAL if c in Xtr.columns]
            num_cols = [c for c in Xtr.columns
                        if c not in cat_cols and pd.api.types.is_numeric_dtype(Xtr[c])]
            cat_maps, card = {}, {}
            for col in cat_cols:
                cats = sorted(Xtr[col].dropna().astype("object").unique())
                cat_maps[col] = {v: i for i, v in enumerate(cats)}
                card[col] = len(cats)
            med = Xtr[num_cols].astype("float64").median()
            sc = StandardScaler().fit(Xtr[num_cols].astype("float64").fillna(med).to_numpy())
            todev = lambda cn: ({c: t.to(dev) for c, t in cn[0].items()}, cn[1].to(dev))
            tr = todev(encode_data(Xtr, cat_cols, num_cols, cat_maps, sc, med))
            ev_t = {e: todev(encode_data(D.align_columns(Xtr, evals[e][0]), cat_cols, num_cols,
                                         cat_maps, sc, med)) for e in ev}
            y_t = torch.tensor(ytr.astype("float32"), device=dev)
            n = len(y_t)

            seed_everything(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = TabularMLP(card, num_features=len(num_cols), hidden_dims=(256, 128, 64),
                               dropout=0.1, prior_mean=float(np.mean(ytr))).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            crit = nn.MSELoss()
            curves = {e: np.zeros((a.epochs, len(evals[e][1])), dtype="float32") for e in ev}
            for ep in range(1, a.epochs + 1):
                model.train()
                perm = torch.randperm(n, device=dev)
                for i in range(0, n, BATCH):
                    b = perm[i:i + BATCH]
                    opt.zero_grad()
                    loss = crit(model({c: tr[0][c][b] for c in tr[0]}, tr[1][b]), y_t[b])
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    opt.step()
                msg = []
                for e in ev:
                    p = predict(model, *ev_t[e])
                    curves[e][ep - 1] = p
                    y = evals[e][1]
                    msg.append(f"{e} {np.sqrt(np.mean((y - p) ** 2)):.2f}")
                if ep % 10 == 0 or ep == 1:
                    print(f"  seed {seed} stage {s} ep {ep:2d}/{a.epochs} " + " | ".join(msg)
                          + f" [{time.time()-t0:.0f}s]", flush=True)
            for e in ev:
                res[f"{s}_{e}"] = curves[e]
            del tr, ev_t, model, opt
            torch.cuda.empty_cache()
        np.savez(out, **res, epochs=a.epochs)
        print(f"seed {seed}: done [{time.time()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
