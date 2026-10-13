#!/usr/bin/env python3
"""Diagnostic (no training, no feature build): does EUROCONTROL's own daily
Airport Arrival ATFM Delay dataset correlate with the current model's
taxi-out residual?

Background: this project's own schema audit (TAXI_OUT_DRIVERS.md sec.2b)
found NO CTOT/slot/regulation field in the competition data at all, and
explicitly flagged that testing an ATFM-regulation hypothesis would need an
external source -- exactly what this is. Per EUROCONTROL's own definition
(ansperformance.eu/capacity/atfm_apt/): "Aircraft expected to arrive during
a period of congestion are given ATFM delay AT THEIR DEPARTURE AIRPORT,
under the authority of the Network Manager" -- i.e. this is a genuine,
network-wide, exogenous ground-hold mechanism attributed to the DEPARTURE
airport/day, not a re-statement of local traffic density the existing
congestion features might already capture.

Data source: EUROCONTROL Aviation Intelligence Portal (ansperformance.eu),
dataset APT-DLY, bulk download
https://www.eurocontrol.int/performance/data/download/xls/Airport_Arrival_ATFM_Delay.xlsx
(the "DATA" sheet: YEAR/MONTH/FLT_DATE/APT_ICAO granularity, daily, going
back to 2019). Verified complete coverage for all 10 target airports across
2025 (365/365 days each) AND the real scored period Jan+Jul 2026 (no gaps).
Licensing: NOT yet organizer-confirmed for this challenge (conversation.md
sec.1, open question) -- this run is a diagnostic only, not a shipped
feature; do not build into features.py without resolving that first.

Method: join DLY_APT_ARR_1 (minutes of ATFM delay attributed to this
airport/day) and a derived pct_regulated (FLT_ARR_1_DLY_15 / FLT_ARR_1, the
share of that day's flights delayed >15min) to each DEP row in forward/val
by (ADEP_mvt, calendar date). Correlate against mlp_antigravity's (seed42)
val residual, plus a permutation control (shuffle the date-level join).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import config
from common import io as cio
from common import dataset as D
from common.predict import predict_arm

SCRATCH = Path(os.environ.get("ATFM_SCRATCH",
    r"C:\Users\Admin\AppData\Local\Temp\claude\C--Users-Admin-Desktop-eddie-fearless-barn"
    r"\68f2e258-1f8e-48ed-8028-54fecced0b45\scratchpad"))
APT_DLY_PARQUET = SCRATCH / "apt_dly.parquet"

SPLIT, FEATURESET, OUTLIER_POLICY = "forward", "with_aobt", "keep_all"
BASELINE_ARM, BASELINE_SEED = "mlp_antigravity", 42
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]


def main() -> int:
    print("[atfm] loading EUROCONTROL APT-DLY (cached parquet)...")
    atfm = pd.read_parquet(APT_DLY_PARQUET)
    atfm = atfm[atfm["APT_ICAO"].isin(AIRPORTS)].copy()
    atfm["FLT_DATE"] = pd.to_datetime(atfm["FLT_DATE"]).dt.date
    atfm["pct_regulated"] = atfm["FLT_ARR_1_DLY_15"] / atfm["FLT_ARR_1"].replace(0, np.nan)
    atfm["DLY_APT_ARR_1"] = atfm["DLY_APT_ARR_1"].fillna(0.0)
    atfm_key = atfm.set_index(["APT_ICAO", "FLT_DATE"])[["DLY_APT_ARR_1", "pct_regulated", "FLT_ARR_1"]]
    print("  atfm rows (10 airports): {}  date range: {} to {}".format(
        len(atfm), atfm["FLT_DATE"].min(), atfm["FLT_DATE"].max()))

    print("[atfm] loading DEP row dates/airports (whole training pool)...")
    mov = cio.load_training(columns=[config.ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
    dep = mov[mov["PHASE_mvt"].eq("DEP")].copy()
    dep["date"] = pd.to_datetime(dep["MVT_TIME_UTC_mvt"], utc=True).dt.tz_localize(None).dt.date
    dep_key = dep.set_index([config.ID])[["ADEP_mvt", "date"]]

    print("[atfm] loading {} baseline (seed{}) on {}/val...".format(
        BASELINE_ARM, BASELINE_SEED, SPLIT))
    Xva, yva, meta_va = D.load_part(SPLIT, "val", FEATURESET, filter_target=False)
    yva = np.asarray(yva, dtype="float64")
    p_va = predict_arm(BASELINE_ARM, SPLIT, "val", FEATURESET, BASELINE_SEED,
                       OUTLIER_POLICY, X=Xva, meta=meta_va)
    resid = yva - p_va
    val_ids = meta_va.index.to_numpy()

    joined = dep_key.reindex(val_ids)
    joined = joined.join(atfm_key, on=["ADEP_mvt", "date"])
    mask = joined["DLY_APT_ARR_1"].notna().to_numpy()
    print("  val rows joined: {}  usable (has ATFM record that day): {} ({:.2%})".format(
        len(val_ids), mask.sum(), mask.mean()))

    r = resid[mask]
    dly = joined["DLY_APT_ARR_1"].to_numpy()[mask]
    pct = joined["pct_regulated"].fillna(0.0).to_numpy()[mask]

    def corr(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[ok], b[ok])[0, 1])

    rho_dly = corr(dly, r)
    rho_pct = corr(pct, r)
    print("\n[result] rho(DLY_APT_ARR_1 that day, resid) = {:+.4f}".format(rho_dly))
    print("[result] rho(pct_regulated that day, resid)   = {:+.4f}".format(rho_pct))

    # OLS ceiling, both predictors jointly, on the full val pool (unmatched -> 0 correction)
    from sklearn.linear_model import LinearRegression
    X_ols = np.column_stack([dly, pct])
    ols = LinearRegression().fit(X_ols, r)
    correction = np.zeros(len(resid))
    correction[mask] = ols.predict(X_ols)
    old_rmse = float(np.sqrt(np.mean(resid ** 2)))
    new_rmse = float(np.sqrt(np.mean((resid - correction) ** 2)))
    print("\n[ceiling] pooled val RMSE: baseline = {:.3f}  with in-sample OLS correction = {:.3f}  "
          "ceiling = {:+.3f}s".format(old_rmse, new_rmse, new_rmse - old_rmse))

    # permutation control: shuffle which date's ATFM record each row sees
    rng = np.random.default_rng(42)
    dly_shuf = rng.permutation(dly)
    pct_shuf = rng.permutation(pct)
    print("\n[permutation control] rho(shuffled DLY, resid) = {:+.4f}".format(corr(dly_shuf, r)))
    print("[permutation control] rho(shuffled pct, resid)  = {:+.4f}".format(corr(pct_shuf, r)))

    # per-airport breakdown
    airport = joined["ADEP_mvt"].to_numpy()
    print("\nper-airport (n usable, rho(DLY,resid), rho(pct,resid), mean DLY_APT_ARR_1/day):")
    for a in AIRPORTS:
        m2 = mask & (airport == a)
        if m2.sum() < 50:
            continue
        r2 = resid[m2]
        dly2 = joined["DLY_APT_ARR_1"].to_numpy()[m2]
        pct2 = joined["pct_regulated"].fillna(0.0).to_numpy()[m2]
        print("  {:6s} n={:6d}  rho_dly={:+.4f}  rho_pct={:+.4f}  mean_dly={:8.1f}".format(
            a, int(m2.sum()), corr(dly2, r2), corr(pct2, r2), dly2.mean()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
