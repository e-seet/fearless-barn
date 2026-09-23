#!/usr/bin/env python3
"""Paired comparison after the rec-3 retrain: truncated (cap 25) vs converged (cap 45).

`alpha_cap_vs_converged.py` grouped seeds by their own `hit_max_epochs` flag.
After the retrain every seed is converged, so that grouping collapses -- and it
was the WRONG comparison anyway: it contrasted *different* seeds (unpaired),
where between-seed variance of ~6 s swamps the effect. See the 2026-09-23
correction in ../CONVERGENCE_AUDIT.md sec.1.

This does the paired version. For each of the 12 retrained seeds it loads BOTH
models -- the truncated one preserved in TRUNCATED25/ and the converged one now
in artifacts/runs/ -- predicts both on forward_inner val, and compares:

  1. per-seed val RMSE            (paired: does the cap cost per-member accuracy?)
  2. the MLP ENSEMBLE val RMSE    (does it survive averaging? the open question)
  3. the solved per-airport alphas (does it move the shipped weights?)

Claim (1) is already answered for the first five seeds (mean -3.64 s). Claims
(2) and (3) are what decides whether any of this reaches a submission.

No training. usage: python tests_mlp_convergence/alpha_paired_old_vs_new.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import config
from common import dataset as D
from common.metrics import rmse
from common.predict import predict_arm
from pipeline_mlp_antigravity.model import TabularMLP
from pipeline_mlp_antigravity.train import TabularDataset, encode_data
from pipeline_per_airport.model import AIRPORT_COL
from pipeline_airport_operator_multimodel.train import solve_alpha_per_airport

FS, POL, WS = "with_aobt", "keep_all", "forward_inner"
RETRAINED = [42, 43, 44, 48, 49, 50, 51, 53, 54, 55, 57, 58]
ALREADY_OK = [45, 46, 47, 52, 56, 59]      # never truncated; identical in both sets
BACKUP = HERE / "TRUNCATED25"


def predict_from_dir(run_dir: Path, X):
    """Load an mlp_antigravity run from an ARBITRARY directory and predict.

    predict_arm() derives its path from (arm, protocol, seed), so it cannot
    reach the backed-up truncated runs. This mirrors the loader in
    pipeline_mlp_antigravity/evaluate.py.
    """
    with open(run_dir / "preprocessor.pkl", "rb") as fh:
        prep = pickle.load(fh)
    cat_t, num_t = encode_data(X, prep["cat_cols"], prep["num_cols"],
                               prep["cat_maps"], prep["scaler"], prep["medians"])
    loader = DataLoader(TabularDataset(cat_t, num_t), batch_size=4096,
                        shuffle=False, num_workers=0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularMLP(prep["cat_cardinalities"], num_features=len(prep["num_cols"]),
                       hidden_dims=tuple(prep.get("hidden_dims", (256, 128, 64))),
                       dropout=prep.get("dropout", 0.1),
                       prior_mean=prep.get("prior_mean", 990.0),
                       residual=prep.get("residual", False)).to(dev)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=dev))
    model.eval()
    out = []
    with torch.no_grad():
        for x_cat, x_num in loader:
            x_cat = {c: x_cat[c].to(dev) for c in x_cat}
            out.append(model(x_cat, x_num.to(dev)).cpu().numpy())
    return np.clip(np.concatenate(out), config.PREDICTION_CLIP_MIN_SEC,
                   config.PREDICTION_CLIP_MAX_SEC)


def main() -> int:
    Xtr, _, _ = D.load_part(WS, "train", FS, filter_target=True, outlier_policy=POL)
    X, y, meta = D.load_part(WS, "val", FS, filter_target=False)
    X = D.align_columns(Xtr, X)
    y = np.asarray(y, float)
    apx = meta[AIRPORT_COL].astype("object").to_numpy()
    tree = predict_arm("airport_operator_blend", WS, "val", FS, 42, POL,
                       X=X, meta=meta, Xtr=Xtr)
    print(f"forward_inner val n={len(y)} | tree {rmse(y, tree):.2f}\n")

    old_p, new_p = {}, {}
    print(f"{'seed':>5} {'old val':>9} {'new val':>9} {'delta':>8}  {'old ep':>7} {'new ep':>7}")
    deltas = []
    for s in RETRAINED:
        run = ROOT / f"artifacts/runs/mlp_antigravity__{WS}-{FS}-{POL}__seed{s}"
        bak = BACKUP / f"seed{s}"
        nj = json.loads((run / "model.json").read_text())
        c = nj.get("convergence", {})
        if c.get("hit_max_epochs") or not c.get("early_stop_fired"):
            raise SystemExit(f"seed {s} has not been retrained to convergence yet "
                             f"(best {c.get('best_epoch')} / cap {c.get('max_epochs')}). "
                             f"Wait for the retrain to finish.")
        oj = json.loads((bak / "model.json").read_text())
        old_p[s] = predict_from_dir(bak, X)
        new_p[s] = predict_from_dir(run, X)
        ro, rn = rmse(y, old_p[s]), rmse(y, new_p[s])
        deltas.append(rn - ro)
        print(f"{s:>5} {ro:9.2f} {rn:9.2f} {rn-ro:+8.2f}  "
              f"{oj.get('best_epoch'):>7} {nj.get('best_epoch'):>7}")
    print(f"\n  paired mean delta: {np.mean(deltas):+.2f} s   "
          f"({sum(1 for d in deltas if d < -0.005)}/{len(deltas)} seeds improved, "
          f"{sum(1 for d in deltas if abs(d) <= 0.005)} unchanged)")

    # Seeds never truncated -- identical in both ensembles, included in both.
    for s in ALREADY_OK:
        run = ROOT / f"artifacts/runs/mlp_antigravity__{WS}-{FS}-{POL}__seed{s}"
        p = predict_from_dir(run, X)
        old_p[s] = new_p[s] = p

    seeds18 = sorted(old_p)
    ens_old = np.mean([old_p[s] for s in seeds18], axis=0)
    ens_new = np.mean([new_p[s] for s in seeds18], axis=0)
    print(f"\n=== THE QUESTION: does it survive averaging? (18-seed ensemble) ===")
    print(f"  MLP ensemble val RMSE  old {rmse(y, ens_old):.3f}  ->  new {rmse(y, ens_new):.3f}"
          f"   ({rmse(y, ens_new)-rmse(y, ens_old):+.3f} s)")

    w_old = solve_alpha_per_airport(y, tree, ens_old, apx)
    w_new = solve_alpha_per_airport(y, tree, ens_new, apx)
    shipped = json.loads((ROOT / "artifacts/runs/airport_operator_multimodel__full-"
                          "with_aobt-keep_all__seed42__tag=msmlp18fi/model.json"
                          ).read_text())["weights_per_airport"]
    print(f"\n=== alphas (weight on the TREE side) ===")
    print(f"{'airport':<8} {'shipped v25':>12} {'old ens':>9} {'new ens':>9} {'shift':>8}")
    shifts = []
    for a in sorted(w_old):
        sh = w_new[a] - w_old[a]
        shifts.append(sh)
        print(f"{a:<8} {shipped.get(a, float('nan')):12.4f} {w_old[a]:9.4f} "
              f"{w_new[a]:9.4f} {sh:+8.4f}")
    print(f"\n  mean |shift| {np.mean(np.abs(shifts)):.4f}   mean shift {np.mean(shifts):+.4f}")

    bl = lambda w, e: np.array([w[a] for a in apx]) * tree + \
        (1 - np.array([w[a] for a in apx])) * e
    print(f"\n  blend val RMSE  old {rmse(y, bl(w_old, ens_old)):.3f}  ->  "
          f"new {rmse(y, bl(w_new, ens_new)):.3f}"
          f"   ({rmse(y, bl(w_new, ens_new))-rmse(y, bl(w_old, ens_old)):+.3f} s)")
    print("\nNOTE: this is the weight-solve surface, so the blend number is "
          "in-sample for the alphas. The ensemble RMSE above it is not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
