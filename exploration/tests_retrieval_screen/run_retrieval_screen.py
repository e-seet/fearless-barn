#!/usr/bin/env python3
"""Screen (no training): does a retrieval-style neighbour mean carry information
the shipped blend does not already have?

Gate before building any TabR-style retrieval-augmented model. Retrieval's
claimed advantage is that it degrades to a neighbour average where entity
embeddings have no signal (median 2 occurrences/level, TAXI_OUT_DRIVERS.md
sec.2u). So the retrieved-neighbour mean is computed with hierarchical
back-off (finest cell with >= K=32 training rows wins) and compared against
the SHIPPED BLEND's residual, not against y directly:

    d = retrieval_mean - blend_pred        resid = y - blend_pred

If retrieval adds information the blend lacks, d correlates with resid.
Permutation control: shuffle d within airport (keeps the airport-level
structure the blend already has, destroys row-level alignment), 200 draws.
Ceiling: OLS resid ~ a + b*d, fit on val and applied to test (cross-surface,
honest) plus in-sample on each, as pooled-RMSE improvement.

Pre-registered bar (before running): out-of-sample (val-fit -> test) pooled
RMSE gain must clear ~2s AND rho must exceed the permutation 99th percentile,
otherwise the full retrieval build is not worth its epochs. (~5s is the
board-visibility floor; 2s is the lower screening bar because sec.3 of the
local-vs-board join shows the board amplifies local differences ~2x.)
"""
from __future__ import annotations

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

SPLIT, FEATURESET, POLICY = "forward", "with_aobt", "keep_all"
SEED, TAG = 42, "msmlp"
K = 32
LEVELS = [
    ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "hour", "AIRCRAFT_TYPE_mvt"],
    ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "hour"],
    ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt"],
    ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"],
    ["ADEP_mvt", "RUNWAY_mvt", "hour"],
    ["ADEP_mvt", "hour"],
]


def key(X, cols):
    s = X[cols[0]].astype(str)
    for c in cols[1:]:
        s = s + "|" + X[c].astype(str)
    return s


def retrieve(Xtr, ytr, Xq):
    """Hierarchical back-off cell mean; returns (mean, level_used)."""
    out = np.full(len(Xq), np.nan)
    lvl = np.full(len(Xq), -1, dtype="int64")
    for li, cols in enumerate(LEVELS):
        ktr = key(Xtr, cols)
        g = pd.DataFrame({"k": ktr.to_numpy(), "y": ytr}).groupby("k")["y"].agg(["mean", "count"])
        g = g[g["count"] >= K]
        kq = key(Xq, cols)
        m = kq.map(g["mean"])
        fill = np.isnan(out) & m.notna().to_numpy()
        out[fill] = m.to_numpy()[fill]
        lvl[fill] = li
    return out, lvl


def rmse(e):
    return float(np.sqrt(np.mean(np.asarray(e) ** 2)))


def main() -> int:
    Xtr, ytr, _ = D.load_part(SPLIT, "train", FEATURESET, filter_target=True, outlier_policy=POLICY)
    ytr = np.asarray(ytr, dtype="float64")
    res = {}
    for part in ("val", "test"):
        X, y, meta = D.load_part(SPLIT, part, FEATURESET, filter_target=False)
        X = D.align_columns(Xtr, X)
        y = np.asarray(y, dtype="float64")
        p = predict_arm("airport_operator_multimodel", SPLIT, part, FEATURESET, SEED, POLICY,
                        X=X, meta=meta, tag=TAG)
        r, lvl = retrieve(Xtr, ytr, X)
        ok = np.isfinite(r)
        ap = meta["ADEP_mvt"].astype(str).to_numpy()
        print(f"[{part}] n={len(y)} retrieval coverage={ok.mean():.4%}  "
              f"level dist={np.bincount(lvl[ok], minlength=len(LEVELS)) / ok.sum()}")
        res[part] = dict(y=y, p=p, r=r, ok=ok, ap=ap)

    def stats(part, do_perm=True):
        d_ = res[part]
        y, p, r, ok, ap = d_["y"], d_["p"], d_["r"], d_["ok"], d_["ap"]
        resid = (y - p)[ok]
        d = (r - p)[ok]
        rho = float(np.corrcoef(d, resid)[0, 1])
        line = f"[{part}] blend RMSE(covered)={rmse(resid):.3f}  rho(d,resid)={rho:+.4f}"
        if do_perm:
            rng = np.random.default_rng(0)
            apo = ap[ok]
            rhos = []
            idx_by = {a: np.flatnonzero(apo == a) for a in np.unique(apo)}
            for _ in range(200):
                dp = d.copy()
                for a, ix in idx_by.items():
                    dp[ix] = d[rng.permutation(ix)]
                rhos.append(np.corrcoef(dp, resid)[0, 1])
            rhos = np.array(rhos)
            line += f"  perm mean={rhos.mean():+.4f} p99(|rho|)={np.percentile(np.abs(rhos), 99):.4f}"
        print(line)
        return d, resid

    dv, rv = stats("val")
    dt, rt = stats("test")

    def ols(d, resid):
        A = np.vstack([np.ones_like(d), d]).T
        coef, *_ = np.linalg.lstsq(A, resid, rcond=None)
        return coef

    cv = ols(dv, rv)
    ct = ols(dt, rt)
    print(f"\nOLS resid ~ a + b*d   val-fit: a={cv[0]:+.2f} b={cv[1]:+.4f}   test-fit: a={ct[0]:+.2f} b={ct[1]:+.4f}")

    def gain(d, resid, coef):
        new = resid - (coef[0] + coef[1] * d)
        return rmse(resid), rmse(new)

    # pooled RMSE over ALL rows: uncovered rows keep their blend residual
    for part, d, resid, coef, tag in (("val", dv, rv, cv, "in-sample"),
                                       ("test", dt, rt, ct, "in-sample"),
                                       ("test", dt, rt, cv, "OUT-OF-SAMPLE (val-fit -> test)"),
                                       ("val", dv, rv, ct, "OUT-OF-SAMPLE (test-fit -> val)")):
        dd = res[part]
        ok = dd["ok"]
        full = dd["y"] - dd["p"]
        new_full = full.copy()
        new_full[ok] = resid - (coef[0] + coef[1] * d)
        print(f"  {part:4s} {tag:34s} pooled RMSE {rmse(full):8.3f} -> {rmse(new_full):8.3f}  "
              f"(gain {rmse(full) - rmse(new_full):+.3f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
