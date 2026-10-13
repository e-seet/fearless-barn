#!/usr/bin/env python3
"""Does the destination/runway pattern found in LIRF's AOBT-unmatched
population (run_lirf_flightpath_export.py) actually correlate with the
current model's residual -- i.e. can it differentiate WHICH unmatched rows
are worst, not just confirm membership in a population already known to be
bad (sec.2q)?

Runway precision check first: main runway 25 carries 88% of unmatched rows
at the SAME rate as overall (ratio 0.98 -- proportional, not concentrated).
The two reverse-configuration ends of the minor runway, 16L/34R, show a
real but small-sample elevated rate (9.4x / 5.6x, n=14/21) -- everything
else is close to proportional. So "singling out the runway" is mostly NO
(the corrupted rows use the same main runway as everyone else) with one
minor, small-n exception.

Method: restrict to LIRF's has_issue (no flight record) DEP rows in
forward/val+test, join the destination overrepresentation ratio and a
reverse-runway flag, correlate against mlp_antigravity's residual on THESE
rows specifically (does destination/runway predict how bad an
already-flagged row's error is), plus a permutation control.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
sys.path.insert(0, str(ROOT))

import config
from common import io as cio
from common import dataset as D
from common.predict import predict_arm

SPLIT, FEATURESET, OUTLIER_POLICY = "forward", "with_aobt", "keep_all"
BASELINE_ARM, BASELINE_SEED = "mlp_antigravity", 42
REVERSE_RUNWAYS = {"16L", "34R"}


def main() -> int:
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt",
            "AIRCRAFT_OPERATOR_flt"]
    mov = cio.load_training(columns=cols)
    dep = mov[(mov["PHASE_mvt"].eq("DEP")) & (mov["ADEP_mvt"].eq("LIRF"))].copy()
    dep["has_issue"] = dep["AIRCRAFT_OPERATOR_flt"].isna()
    issue_full = dep[dep["has_issue"]]

    # destination overrepresentation ratio (share-of-issues / share-of-total),
    # computed on the FULL YEAR pool -- same method as run_lirf_flightpath_export.py
    dest = pd.DataFrame({
        "n_issue": issue_full["ADES_mvt"].value_counts(),
        "n_total": dep["ADES_mvt"].value_counts(),
    }).fillna(0)
    dest["overrep_ratio"] = ((dest["n_issue"] / dest["n_issue"].sum()) /
                             (dest["n_total"] / dest["n_total"].sum()))
    dest_map = dest["overrep_ratio"].to_dict()

    dep_key = dep.set_index("MVT_ID_mvt")
    dep_key["ades_overrep_ratio"] = dep_key["ADES_mvt"].map(dest_map)
    dep_key["is_reverse_runway"] = dep_key["RUNWAY_mvt"].isin(REVERSE_RUNWAYS).astype("float64")

    print("[atfm] loading {} baseline (seed{}) on {}/val+test...".format(
        BASELINE_ARM, BASELINE_SEED, SPLIT))
    rows = []
    for part in ["val", "test"]:
        X, y, meta = D.load_part(SPLIT, part, FEATURESET, filter_target=False)
        y = np.asarray(y, dtype="float64")
        p = predict_arm(BASELINE_ARM, SPLIT, part, FEATURESET, BASELINE_SEED,
                        OUTLIER_POLICY, X=X, meta=meta)
        resid = y - p
        airport = meta["ADEP_mvt"].astype(str).to_numpy() if "ADEP_mvt" in meta.columns else None
        ids = meta.index.to_numpy()
        m = airport == "LIRF"
        for i, mvt_id in enumerate(ids[m]):
            if mvt_id in dep_key.index and dep_key.loc[mvt_id, "has_issue"]:
                rows.append({
                    "mvt_id": mvt_id, "part": part,
                    "resid": resid[m][i], "abs_resid": abs(resid[m][i]),
                    "ades_overrep_ratio": dep_key.loc[mvt_id, "ades_overrep_ratio"],
                    "is_reverse_runway": dep_key.loc[mvt_id, "is_reverse_runway"],
                })

    df = pd.DataFrame(rows)
    print("\nLIRF has_issue rows found in forward val+test: {}".format(len(df)))
    if len(df) < 20:
        print("Too few rows for a meaningful correlation -- stopping.")
        return 0

    def corr(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else float("nan")

    r = df["resid"].to_numpy()
    ar = df["abs_resid"].to_numpy()
    dest_ratio = df["ades_overrep_ratio"].fillna(1.0).to_numpy()
    rev = df["is_reverse_runway"].to_numpy()

    print("\n[result] rho(ades_overrep_ratio, resid)     = {:+.4f}".format(corr(dest_ratio, r)))
    print("[result] rho(ades_overrep_ratio, abs(resid)) = {:+.4f}".format(corr(dest_ratio, ar)))
    print("[result] rho(is_reverse_runway, resid)       = {:+.4f}  (n_reverse={})".format(
        corr(rev, r), int(rev.sum())))
    print("[result] rho(is_reverse_runway, abs(resid))  = {:+.4f}".format(corr(rev, ar)))

    rng = np.random.default_rng(42)
    dest_shuf = rng.permutation(dest_ratio)
    rev_shuf = rng.permutation(rev)
    print("\n[permutation control] rho(shuffled dest_ratio, resid) = {:+.4f}".format(corr(dest_shuf, r)))
    print("[permutation control] rho(shuffled reverse_rwy, resid) = {:+.4f}".format(corr(rev_shuf, r)))

    print("\nmean |resid| when is_reverse_runway=1 (n={}): {:.1f}s".format(
        int(rev.sum()), ar[rev == 1].mean() if rev.sum() else float("nan")))
    print("mean |resid| when is_reverse_runway=0 (n={}): {:.1f}s".format(
        int((rev == 0).sum()), ar[rev == 0].mean()))

    # in-sample OLS ceiling restricted to THIS subpopulation's own pooled RMSE
    from sklearn.linear_model import LinearRegression
    Xo = np.column_stack([dest_ratio, rev])
    ols = LinearRegression().fit(Xo, r)
    corrected = r - ols.predict(Xo)
    old_rmse = float(np.sqrt(np.mean(r ** 2)))
    new_rmse = float(np.sqrt(np.mean(corrected ** 2)))
    print("\n[ceiling, WITHIN the has_issue subpopulation only] RMSE baseline={:.1f} "
          "corrected={:.1f}  ceiling={:+.1f}s (n={})".format(old_rmse, new_rmse, new_rmse - old_rmse, len(df)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
