#!/usr/bin/env python3
"""Honest three-member per-airport solve: FTT + tree parent + MLP.

The descriptive w-sweep in `vs_shipped_blend.py` was read on the same surfaces
it reported, so its w could not be used. This solves the weights where this
build always solves them -- on `forward_inner` val (Jul-Aug, a holdout carved
from TRAIN, never the evaluation surface) -- and then applies them, frozen, to
forward val (Sep-Oct) and forward test (Nov-Dec).

Members, mirroring `pipeline_airport_operator_catboost_multimodel` exactly with
FTT in the slot CatBoost occupies there:

  p_a = FT-Transformer, seed-averaged
  p_b = airport_operator_blend   (the tree parent)
  p_c = mlp_antigravity, seed-averaged

and reusing that arm's OWN simplex solver rather than a new one -- it is the
repo's documented fix for plain NNLS pushing a single weight past 1.0 at LIRF
on a small outlier-heavy sample.

Baseline is the SHIPPED two-member blend `airport_operator_multimodel` (tag
msmlp), not an MLP-only ensemble: `tests_matched_unmatched_router` is the
precedent for a candidate that beat a weak comparator and lost to this one.

No training. usage: python tests_ftt/solve_three_member.py [--l2 0.0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import config
from common import dataset as D
from common.predict import predict_arm
from pipeline_airport_operator_catboost_multimodel.train import (
    solve_weights_per_airport_simplex)

FS, POL = "with_aobt", "keep_all"
SEEDS = [42, 43, 44]
SHIPPED_ARM, SHIPPED_TAG, SEED = "airport_operator_multimodel", "msmlp", 42
rm = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))


def ftt_mean(split_sfx: str, seeds=SEEDS):
    """Seed-averaged FTT predictions, with the convergence gate applied."""
    ps = []
    for s in seeds:
        z = np.load(HERE / f"preds_seed{s}{split_sfx}.npz", allow_pickle=True)
        if bool(z["hit_cap"]):
            raise SystemExit(
                f"seed {s}{split_sfx} HIT THE EPOCH CAP (best {int(z['best_epoch'])} "
                f"of {int(z['epochs_run'])}). Truncated, not converged -- raise "
                f"--epochs and re-run before solving weights on it "
                f"(../CONVERGENCE_AUDIT.md sec.1).")
        ps.append(z["pv"])
    return np.mean(ps, axis=0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--l2", type=float, default=0.0,
                    help="shrinkage toward equal weights in the simplex solve")
    a = ap.parse_args(argv)

    # ---- 1. solve on forward_inner val -------------------------------------
    WS = "forward_inner"
    Xw_tr, _, _ = D.load_part(WS, "train", FS, filter_target=True, outlier_policy=POL)
    Xw, yw, meta_w = D.load_part(WS, "val", FS, filter_target=False)
    Xw = D.align_columns(Xw_tr, Xw)
    yw = np.asarray(yw, float)
    ap_w = meta_w["ADEP_mvt"].astype(str).to_numpy()

    p_tree_w = predict_arm("airport_operator_blend", WS, "val", FS, SEED, POL,
                           X=Xw, meta=meta_w, Xtr=Xw_tr)
    p_mlp_w = np.mean([predict_arm("mlp_antigravity", WS, "val", FS, s, POL,
                                   X=Xw, meta=meta_w, Xtr=Xw_tr) for s in SEEDS], axis=0)
    p_ftt_w = ftt_mean("_forward_inner_lr1e-3")
    assert len(p_ftt_w) == len(yw), f"FTT {len(p_ftt_w)} vs forward_inner val {len(yw)}"

    print(f"forward_inner val n={len(yw)}  "
          f"tree {rm(yw, p_tree_w):.2f} | mlp {rm(yw, p_mlp_w):.2f} | ftt {rm(yw, p_ftt_w):.2f}")

    W = solve_weights_per_airport_simplex(yw, p_ftt_w, p_tree_w, p_mlp_w, ap_w,
                                          l2_shrinkage=a.l2)
    print("\nsolved weights per airport [FTT, tree, MLP] (simplex, w>=0, sum=1):")
    for k in sorted(W):
        w = W[k]
        print(f"  {k}: FTT {w[0]:.3f}  tree {w[1]:.3f}  MLP {w[2]:.3f}")

    # ---- 2. apply FROZEN to forward val and test ---------------------------
    SP = "forward"
    Xtr, _, _ = D.load_part(SP, "train", FS, filter_target=True, outlier_policy=POL)
    out = {"l2_shrinkage": a.l2, "weights_per_airport": W, "seeds": SEEDS}
    for part, pk in (("val", "pv"), ("test", "pt")):
        X, y, meta = D.load_part(SP, part, FS, filter_target=False)
        X = D.align_columns(Xtr, X)
        y = np.asarray(y, float)
        apx = meta["ADEP_mvt"].astype(str).to_numpy()

        p_tree = predict_arm("airport_operator_blend", SP, part, FS, SEED, POL,
                             X=X, meta=meta, Xtr=Xtr)
        p_mlp = np.mean([predict_arm("mlp_antigravity", SP, part, FS, s, POL,
                                     X=X, meta=meta, Xtr=Xtr) for s in SEEDS], axis=0)
        p_ftt = np.mean([np.load(HERE / f"preds_seed{s}_lr1e-3_e132.npz",
                                 allow_pickle=True)[pk] for s in SEEDS], axis=0)
        shipped = predict_arm(SHIPPED_ARM, SP, part, FS, SEED, POL,
                              X=X, meta=meta, Xtr=Xtr, tag=SHIPPED_TAG)

        three = np.zeros(len(y))
        for k in np.unique(apx):
            m = apx == k
            w = W.get(str(k), [1 / 3, 1 / 3, 1 / 3])
            three[m] = w[0] * p_ftt[m] + w[1] * p_tree[m] + w[2] * p_mlp[m]
        three = np.clip(three, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)

        L = apx == "LIRF"
        ib = y <= 7200
        print(f"\n===== forward {part.upper()} (n={len(y)}) =====")
        row = lambda nm, p: print(
            f"  {nm:22s} pooled {rm(y,p):7.2f} | in-band {rm(y[ib],p[ib]):7.2f} | "
            f"LIRF {rm(y[L],p[L]):8.2f} | non-LIRF {rm(y[~L],p[~L]):7.2f}")
        row("shipped 2-member", shipped)
        row("3-member (FTT added)", three)
        d = rm(y, three) - rm(y, shipped)
        dn = rm(y[~L], three[~L]) - rm(y[~L], shipped[~L])
        di = rm(y[ib], three[ib]) - rm(y[ib], shipped[ib])
        print(f"  delta vs shipped: pooled {d:+.2f} | in-band {di:+.2f} | non-LIRF {dn:+.2f}"
              f"   (negative = better)")
        out[part] = {"shipped": rm(y, shipped), "three_member": rm(y, three),
                     "delta_pooled": d, "delta_in_band": di, "delta_non_lirf": dn,
                     "shipped_lirf": rm(y[L], shipped[L]), "three_lirf": rm(y[L], three[L])}

    (HERE / "solve_three_member.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'solve_three_member.json'}")
    print("\nNOTE: weights solved on forward_inner (never on val/test). The board has "
          "reversed local wins of this size three times (congest, idband, candidate 9).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
