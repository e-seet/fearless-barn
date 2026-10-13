#!/usr/bin/env python3
"""Diagnostic (no training, no feature build): does a NEIGHBOUR's wake-
turbulence category (WTC), relative to the current flight's own, correlate
with the current best model's residual?

Motivation (a participant's speculative post, conversation.md sec.5): ATC
must sequence departures for wake-turbulence separation minima, so a light
aircraft immediately following a heavy one might see a longer taxi/hold than
the reverse ordering. The aircraft's OWN WTC (`WK_TBL_CAT_flt`) is already a
native feature every arm sees; what's untested is a NEIGHBOUR's WTC relative
to this flight's. `pipeline_queueattn_antigravity` (TAXI_OUT_DRIVERS.md
sec.2k) attended over each departure's 8 nearest neighbours but never fed in
neighbour aircraft type/WTC -- that architecture closed flat on the features
it did use, so before touching it again (expensive: multi-seed GPU retrain),
this checks for ANY signal at all with a cheap correlation ceiling, matching
this project's established discipline (sec.2h, sec.2o, sec.2p, sec.2z all
did this before deciding whether to build).

Method: for every DEP row, find the immediately PRECEDING departure at the
same (airport, runway) -- the aircraft ATC most directly had to sequence
behind on that runway -- and read its WTC. Ordinal-encode WTC
(L=0 < M=1 < H=2 < J=3, ICAO wake-strength ordering) and build:
  - prev_heavier: 1 if the preceding aircraft's WTC ordinal > this flight's
    own (the wake-separation-relevant direction: light behind heavy)
  - wtc_diff: own_ord - prev_ord (signed, more general than the flag)
Both correlated against mlp_antigravity's (forward split, seed42) val
residual -- the SAME baseline the closed queue-attention architecture
extends -- plus a permutation control (shuffle neighbour assignment) to
confirm any observed correlation isn't a construction artifact.

Whole-period computation before split-subsetting (matching build_congestion's
established rule against train/serve skew at split boundaries). Forward-
looking-in-time-only (a departure's PRECEDING same-runway neighbour, not a
following one) since sequencing/wake-separation is a backward-looking
physical constraint by definition here, unlike the general "any nearby
movement" congestion features.
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

WTC_ORDER = {"L": 0, "M": 1, "H": 2, "J": 3}
SPLIT, FEATURESET, OUTLIER_POLICY = "forward", "with_aobt", "keep_all"
BASELINE_ARM, BASELINE_SEED = "mlp_antigravity", 42


def main() -> int:
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "RUNWAY_mvt",
            "MVT_TIME_UTC_mvt", "WK_TBL_CAT_flt"]
    print("[wtc-sequencing] loading whole-period movement data...")
    mov = cio.load_training(columns=cols)
    dep = mov[mov["PHASE_mvt"].eq("DEP")].copy()
    dep["MVT_TIME_UTC_mvt"] = pd.to_datetime(dep["MVT_TIME_UTC_mvt"], utc=True).dt.tz_localize(None)
    dep["wtc_ord"] = dep["WK_TBL_CAT_flt"].map(WTC_ORDER)
    dep = dep.sort_values("MVT_TIME_UTC_mvt")
    print("  DEP rows: {}  (WTC known: {:.2%})".format(len(dep), dep["wtc_ord"].notna().mean()))

    # ---- per (airport, runway) sorted timeline: find each row's immediately
    # preceding same-runway departure ----
    dep = dep.dropna(subset=["RUNWAY_mvt"])
    prev_wtc = np.full(len(dep), np.nan)
    prev_id = np.full(len(dep), np.nan)
    times = dep["MVT_TIME_UTC_mvt"].to_numpy()
    wtc = dep["wtc_ord"].to_numpy()
    ids = dep["MVT_ID_mvt"].to_numpy()
    key = dep["ADEP_mvt"].astype(str) + "|" + dep["RUNWAY_mvt"].astype(str)
    dep = dep.assign(_key=key.to_numpy())
    for k, idx in dep.groupby("_key").indices.items():
        idx = np.asarray(idx)  # already time-sorted (dep is globally sorted)
        prev_wtc[idx[1:]] = wtc[idx[:-1]]
        prev_id[idx[1:]] = ids[idx[:-1]]

    dep = dep.assign(prev_wtc_ord=prev_wtc, prev_mvt_id=prev_id)
    have_pair = dep["wtc_ord"].notna() & dep["prev_wtc_ord"].notna()
    print("  rows with a same-runway preceding departure AND both WTCs known: "
          "{} ({:.2%} of DEP)".format(have_pair.sum(), have_pair.mean()))

    dep["wtc_diff"] = dep["wtc_ord"] - dep["prev_wtc_ord"]
    dep["prev_heavier"] = (dep["prev_wtc_ord"] > dep["wtc_ord"]).astype("float64")
    dep.loc[~have_pair, ["wtc_diff", "prev_heavier"]] = np.nan

    # ---- baseline residual on forward/val ----
    print("[wtc-sequencing] loading {} baseline (seed{}) on {}/val...".format(
        BASELINE_ARM, BASELINE_SEED, SPLIT))
    Xva, yva, meta_va = D.load_part(SPLIT, "val", FEATURESET, filter_target=False)
    yva = np.asarray(yva, dtype="float64")
    p_va = predict_arm(BASELINE_ARM, SPLIT, "val", FEATURESET, BASELINE_SEED,
                       OUTLIER_POLICY, X=Xva, meta=meta_va)
    resid = yva - p_va

    val_ids = meta_va[config.ID].to_numpy() if config.ID in meta_va.columns else meta_va.index.to_numpy()
    dep_idx = dep.set_index("MVT_ID_mvt")
    aligned = dep_idx.reindex(val_ids)
    mask = aligned["prev_heavier"].notna().to_numpy()
    print("  val rows joined: {}  (usable, has_pair): {} ({:.2%})".format(
        len(val_ids), mask.sum(), mask.mean()))

    r = resid[mask]
    diff = aligned["wtc_diff"].to_numpy()[mask]
    flag = aligned["prev_heavier"].to_numpy()[mask]

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    rho_diff = corr(diff, r)
    rho_flag = corr(flag, r)
    mean_resid_prev_heavier = r[flag == 1].mean()
    mean_resid_other = r[flag == 0].mean()
    print("\n[result] rho(wtc_diff, resid)   = {:+.4f}".format(rho_diff))
    print("[result] rho(prev_heavier, resid) = {:+.4f}".format(rho_flag))
    print("[result] mean resid | prev_heavier=1 (n={}): {:+.2f}s".format(
        int((flag == 1).sum()), mean_resid_prev_heavier))
    print("[result] mean resid | prev_heavier=0 (n={}): {:+.2f}s".format(
        int((flag == 0).sum()), mean_resid_other))
    print("[result] naive OLS ceiling (r^2 of prev_heavier alone): {:.5f}".format(rho_flag ** 2))

    # ---- permutation control: shuffle which neighbour's WTC each row sees ----
    rng = np.random.default_rng(42)
    flag_shuf = rng.permutation(flag)
    diff_shuf = rng.permutation(diff)
    print("\n[permutation control] rho(shuffled wtc_diff, resid)   = {:+.4f}".format(corr(diff_shuf, r)))
    print("[permutation control] rho(shuffled prev_heavier, resid) = {:+.4f}".format(corr(flag_shuf, r)))

    # ---- in-sample OLS ceiling on POOLED RMSE (optimistic upper bound,
    # same convention as sec.2p/sec.2h): fit resid ~ prev_heavier by OLS,
    # use the fitted correction, recompute pooled RMSE on the FULL val set
    # (unmatched rows get zero correction) ----
    from sklearn.linear_model import LinearRegression
    ols = LinearRegression().fit(flag.reshape(-1, 1), r)
    correction_masked = ols.predict(flag.reshape(-1, 1))
    correction = np.zeros(len(resid))
    correction[mask] = correction_masked
    old_rmse = float(np.sqrt(np.mean(resid ** 2)))
    new_rmse = float(np.sqrt(np.mean((resid - correction) ** 2)))
    print("\n[ceiling] pooled val RMSE: baseline (mlp) = {:.3f}  "
          "with in-sample OLS correction = {:.3f}  ceiling = {:+.3f}s".format(
              old_rmse, new_rmse, new_rmse - old_rmse))

    # ---- per-airport breakdown ----
    airport = meta_va["ADEP_mvt"].astype(str).to_numpy() if "ADEP_mvt" in meta_va.columns else None
    if airport is not None:
        print("\nper-airport (n usable, rho(prev_heavier,resid), mean resid | prev_heavier=1, "
              "mean resid | prev_heavier=0, frac prev_heavier=1):")
        for a in sorted(pd.unique(airport)):
            m2 = mask & (airport == a)
            if m2.sum() < 50:
                continue
            r2 = yva[m2] - p_va[m2]
            flag2 = aligned["prev_heavier"].to_numpy()[m2]
            if flag2.sum() < 5 or (len(flag2) - flag2.sum()) < 5:
                continue
            rho2 = corr(flag2, r2)
            print("  {:6s} n={:6d}  rho={:+.4f}  resid|1={:+7.2f}  resid|0={:+7.2f}  frac1={:.2%}".format(
                a, int(m2.sum()), rho2, r2[flag2 == 1].mean(), r2[flag2 == 0].mean(), flag2.mean()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
