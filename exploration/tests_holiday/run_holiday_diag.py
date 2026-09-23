#!/usr/bin/env python3
"""Diagnostic (no retrain, no features.py edit): do national public holidays
(and the travel-surge days immediately around them) carry residual signal?

Uses the open-source `holidays` PyPI package (no account, no API, pure
computation from public calendar rules) for each airport's own country:
EDDF/EDDM->DE, EGLL->GB, EHAM->NL, LEBL/LEMD->ES, LFPG->FR, LIRF->IT,
LSZH->CH, LTFM->TR. Holiday date is matched against the departure's UTC
calendar date (not localized to each country's own timezone -- a
deliberate simplification; the mismatch is at most a few hours near
midnight local time and doesn't change the diagnostic's purpose).

Method: same "ceiling before training" pattern as SS2h/SS2j/SS2p/SS2r/SS2w/
SS2z/wind-runway -- in-sample joint OLS of
[is_holiday, is_day_before_holiday, is_day_after_holiday] against a quick
single-seed `per_airport` baseline's residual, per airport, on `forward`
test (secondary) and `seasonal` test (pre-registered decisive, bar 5s).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import config
from common import io as cio
from common import dataset as D
from common.bootstrap import seed_everything
from common.features import CATEGORICAL
from pipeline_per_airport.model import PerAirportLGBM

SEED = 42
FEATURESET, POLICY = "with_aobt", "keep_all"
BAR_SEC = 5.0

COUNTRY_OF = {
    "EDDF": "DE", "EDDM": "DE", "EGLL": "GB", "EHAM": "NL",
    "LEBL": "ES", "LEMD": "ES", "LFPG": "FR", "LIRF": "IT",
    "LSZH": "CH", "LTFM": "TR",
}

PARAMS = {
    "objective": "l2", "metric": "rmse",
    "learning_rate": 0.05, "num_leaves": 255, "min_data_in_leaf": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "max_cat_threshold": 64, "cat_smooth": 20.0,
    "verbosity": -1, "num_threads": 0,
    "seed": SEED, "bagging_seed": SEED, "feature_fraction_seed": SEED,
    "device_type": "cpu",
}
NUM_BOOST_ROUND, PATIENCE = 5000, 200


def build_holiday_calendars():
    cals = {}
    for ap, country in COUNTRY_OF.items():
        cals[ap] = holidays.country_holidays(country, years=range(2024, 2027))
    return cals


def holiday_flags(dates: "pd.Series", airports: "pd.Series", cals: dict):
    is_hol = np.zeros(len(dates), dtype="int8")
    is_before = np.zeros(len(dates), dtype="int8")
    is_after = np.zeros(len(dates), dtype="int8")
    d = dates.dt.date.to_numpy()
    ap = airports.to_numpy()
    one_day = pd.Timedelta(days=1)
    for a in pd.unique(ap):
        if a not in cals:
            continue
        cal = cals[a]
        sel = ap == a
        dd = pd.to_datetime(d[sel])
        is_hol[np.flatnonzero(sel)] = np.array([x in cal for x in dd.date])
        is_before[np.flatnonzero(sel)] = np.array([(x + one_day) in cal for x in dd])
        is_after[np.flatnonzero(sel)] = np.array([(x - one_day) in cal for x in dd])
    return is_hol, is_before, is_after


def ols_ceiling(resid: "np.ndarray", regressors: "np.ndarray"):
    ok = np.isfinite(regressors).all(axis=1) & np.isfinite(resid)
    n_tot = len(resid)
    if ok.sum() < 20:
        return float("nan"), 0, n_tot
    Xr = np.column_stack([np.ones(ok.sum()), regressors[ok]])
    yr = resid[ok]
    coef, *_ = np.linalg.lstsq(Xr, yr, rcond=None)
    pred = Xr @ coef
    rmse_before = np.sqrt(np.mean(yr ** 2))
    rmse_after = np.sqrt(np.mean((yr - pred) ** 2))
    return float(rmse_before - rmse_after), int(ok.sum()), n_tot


def run_split(split, cals, mov_times):
    t0 = time.time()
    print(f"\n===== split={split} =====")
    Xtr_raw, ytr, pool_meta = D.load_part(split, "train", FEATURESET,
                                          filter_target=True, outlier_policy=POLICY)
    Xva_raw, yva, meta_va = D.load_part(split, "val", FEATURESET, filter_target=False)
    Xte_raw, yte, meta_te = D.load_part(split, "test", FEATURESET, filter_target=False)
    Xva = D.align_columns(Xtr_raw, Xva_raw)
    Xte = D.align_columns(Xtr_raw, Xte_raw)
    cats = [c for c in CATEGORICAL if c in Xtr_raw.columns]
    ytr = np.asarray(ytr, dtype="float64")
    yva = np.asarray(yva, dtype="float64")
    yte = np.asarray(yte, dtype="float64")

    import lightgbm as lgb
    seed_everything(SEED)
    m = PerAirportLGBM(PARAMS, NUM_BOOST_ROUND, PATIENCE, cats)
    m.fit(Xtr_raw, ytr, pool_meta, Xva, yva, meta_va, verbose=False, lgb=lgb)
    pred_te = np.clip(m.predict(Xte, meta_te), config.PREDICTION_CLIP_MIN_SEC,
                      config.PREDICTION_CLIP_MAX_SEC)
    resid = yte - pred_te
    print(f"  baseline per_airport test RMSE: {np.sqrt(np.mean(resid**2)):.2f}  "
          f"({time.time()-t0:.0f}s)")

    ap = meta_te["ADEP_mvt"].to_numpy()
    idx = Xte.index.to_numpy()
    dep_times = mov_times.reindex(idx)
    is_hol, is_before, is_after = holiday_flags(dep_times, pd.Series(ap, index=Xte.index), cals)
    print(f"  holiday rate: {is_hol.mean()*100:.2f}%  eve rate: {is_before.mean()*100:.2f}%  "
          f"after rate: {is_after.mean()*100:.2f}%")
    regressors = np.column_stack([is_hol, is_before, is_after]).astype("float64")

    rows = []
    for a in sorted(pd.unique(ap)):
        sel = ap == a
        ceil, n_ok, n_tot = ols_ceiling(resid[sel], regressors[sel])
        rows.append({"split": split, "airport": a, "n": n_tot,
                    "n_holiday_rows": int(regressors[sel, 0].sum()),
                    "ceiling_rmse_reduction_sec": ceil})
        print(f"  {a:6s} n={n_tot:6d} n_hol={int(regressors[sel,0].sum()):4d}  "
             f"ceiling={ceil:+7.3f}s")
    ceil_p, n_ok_p, n_tot_p = ols_ceiling(resid, regressors)
    print(f"  {'POOLED':6s} n={n_tot_p:6d}  ceiling={ceil_p:+7.3f}s")
    rows.append({"split": split, "airport": "POOLED", "n": n_tot_p,
                "n_holiday_rows": int(regressors[:, 0].sum()),
                "ceiling_rmse_reduction_sec": ceil_p})
    return rows


def main() -> int:
    t0 = time.time()
    cals = build_holiday_calendars()
    print(f"built holiday calendars for {len(cals)} airports ({time.time()-t0:.0f}s)")

    mov = cio.load_training(columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"])
    dep = mov[mov["PHASE_mvt"].eq("DEP")].set_index("MVT_ID_mvt")
    mov_times = dep["MVT_TIME_UTC_mvt"].dt.tz_localize(None)

    all_rows = []
    for split in ["forward", "seasonal"]:
        all_rows.extend(run_split(split, cals, mov_times))

    df = pd.DataFrame(all_rows)
    df.to_csv(HERE / "summary.csv", index=False)
    print(f"\nwrote {HERE/'summary.csv'}")

    print("\n" + "=" * 78)
    print("Ceiling on the decisive `seasonal` test surface (bar: %.1fs), all airports:" % BAR_SEC)
    seasonal = df[df.split == "seasonal"]
    for _, v in seasonal.iterrows():
        verdict = "CLEARS bar" if v["ceiling_rmse_reduction_sec"] >= BAR_SEC else "does not clear"
        print(f"  {v['airport']:6s}: ceiling {v['ceiling_rmse_reduction_sec']:+.3f}s "
             f"(n_holiday={int(v['n_holiday_rows'])}, n={int(v['n'])}) -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
