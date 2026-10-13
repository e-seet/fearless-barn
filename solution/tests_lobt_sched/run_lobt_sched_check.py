#!/usr/bin/env python3
"""Diagnostic (no training, no feature build): does the explicit pairwise
flight-plan clock difference `lobt_sched_sec = LOBT - SCHED` carry residual
signal the current best model doesn't already have?

Motivation: reviewing a competitor's (merry-quicksand, rank 7/160 at 257.55)
public experiment log (reports/MATCHED_MODEL.md in saurabhdxt-labs/
prc-challenge) found their single biggest feature-family gain (+8.17s,
established on one fold) came from handing a GBT explicit PAIRWISE
differences between flight-plan clocks, rather than letting the tree infer
them from the individual clock columns -- their stated mechanism: "trees
cannot form the difference of two inputs; handing the differences over
directly is what moved the number." Cross-checking `common/features.py`
found this project already computes nearly all of their differences
explicitly (`delta_aobt_sec`, `aobt_minus_eobt_sec`, `aobt_minus_lobt_sec`,
`aobt_minus_sched_sec`, `eobt_minus_iobt_sec`, `lobt_minus_iobt_sec`) --
except one: `lobt_sched_sec = LOBT - SCHED`. We have both halves
(`delta_sched_sec = MVT-SCHED`, `delta_lobt_sec = MVT-LOBT`) as separate
features, whose difference recovers LOBT-SCHED exactly, but never as its
own explicit column. Their own E1b (four more clock differences added to
an already-rich set) found +0.26s, "not established, family exhausted" --
so the prior here is low, but it's a specific, concretely falsifiable claim
and cheap to check before dismissing.

Method: same convention as sec.2p/sec.2s/sec.2z/tests_wtc_sequencing --
compute the candidate column, correlate it against the current best single
arm's (mlp_antigravity, seed42) residual on forward/val AND forward/test
(both, since a single surface isn't decisive per this project's own
discipline), plus a permutation control, then an in-sample OLS ceiling on
pooled RMSE (optimistic upper bound). No features.py edit, no fingerprint
bump, no retrain.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import config
from common import dataset as D
from common.predict import predict_arm

SPLIT, FEATURESET, OUTLIER_POLICY = "forward", "with_aobt", "keep_all"
BASELINE_ARM, BASELINE_SEED = "mlp_antigravity", 42


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def check(part: str) -> None:
    print("\n" + "=" * 70)
    print("[lobt-sched] part = {}".format(part))
    X, y, meta = D.load_part(SPLIT, part, FEATURESET, filter_target=False)
    y = np.asarray(y, dtype="float64")
    p = predict_arm(BASELINE_ARM, SPLIT, part, FEATURESET, BASELINE_SEED,
                     OUTLIER_POLICY, X=X, meta=meta)
    resid = y - p

    assert "delta_sched_sec" in X.columns and "delta_lobt_sec" in X.columns
    lobt_sched_sec = (X["delta_sched_sec"] - X["delta_lobt_sec"]).to_numpy()
    mask = np.isfinite(lobt_sched_sec) & np.isfinite(resid)
    print("  rows: {}  usable (both finite): {} ({:.2%})".format(
        len(mask), mask.sum(), mask.mean()))

    r = resid[mask]
    feat = lobt_sched_sec[mask]
    rho = corr(feat, r)
    print("[result] rho(lobt_sched_sec, resid) = {:+.4f}".format(rho))
    print("[result] naive R^2 (rho^2) = {:.5f}".format(rho ** 2))

    # permutation control: shuffle the feature relative to the residual
    rng = np.random.default_rng(42)
    feat_shuf = rng.permutation(feat)
    print("[permutation control] rho(shuffled, resid) = {:+.4f}".format(
        corr(feat_shuf, r)))

    # in-sample OLS ceiling on pooled RMSE (optimistic upper bound, same
    # convention as sec.2p/sec.2h/tests_wtc_sequencing)
    from sklearn.linear_model import LinearRegression
    ols = LinearRegression().fit(feat.reshape(-1, 1), r)
    correction_masked = ols.predict(feat.reshape(-1, 1))
    correction = np.zeros(len(resid))
    correction[mask] = correction_masked
    old_rmse = float(np.sqrt(np.mean(resid ** 2)))
    new_rmse = float(np.sqrt(np.mean((resid - correction) ** 2)))
    print("[ceiling] pooled RMSE: baseline (mlp) = {:.3f}  "
          "with in-sample OLS correction = {:.3f}  ceiling = {:+.3f}s".format(
              old_rmse, new_rmse, new_rmse - old_rmse))

    # per-airport breakdown
    if "ADEP_mvt" in meta.columns:
        airport = meta["ADEP_mvt"].astype(str).to_numpy()
        print("\nper-airport (n, rho, ceiling_s):")
        for a in sorted(set(airport)):
            m2 = mask & (airport == a)
            if m2.sum() < 50:
                continue
            r2 = resid[m2]
            f2 = lobt_sched_sec[m2]
            rho2 = corr(f2, r2)
            ols2 = LinearRegression().fit(f2.reshape(-1, 1), r2)
            corr2 = ols2.predict(f2.reshape(-1, 1))
            old2 = float(np.sqrt(np.mean(r2 ** 2)))
            new2 = float(np.sqrt(np.mean((r2 - corr2) ** 2)))
            print("  {:6s} n={:6d}  rho={:+.4f}  ceiling={:+.3f}s".format(
                a, int(m2.sum()), rho2, new2 - old2))


def main() -> int:
    for part in ("val", "test"):
        check(part)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
