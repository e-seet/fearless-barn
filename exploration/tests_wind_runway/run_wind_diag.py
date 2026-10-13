#!/usr/bin/env python3
"""Diagnostic (no retrain, no features.py edit): headwind/crosswind component
of a departure's OWN assigned runway, from real wind-direction data.

Follow-up to SS2z (runway-configuration/active-arrival-runway), which closed
on a ceiling but flagged an open question: the tested constructions were
proxies for "which runway configuration is active" inferred indirectly from
arrival traffic, and the equality flag was near-degenerate at EGLL/EHAM
(segregated-mode ops). This is a more DIRECT and mechanistically real
signal: aircraft take off and land into the wind (crosswind/tailwind limits
are a hard operational constraint), so the actual wind direction at
departure time, resolved against the departure's own runway heading, gives
a headwind/crosswind component without needing to infer anything about
concurrent arrivals.

Data: the existing `prc-2026-datasets-wx/` METAR feed (already used in
SS2r/SS2w) has wind SPEED (`sknt`) but not DIRECTION. Wind direction (`drct`)
was fetched fresh from the same open provider (Iowa Environmental Mesonet
ASOS archive, mesonet.agron.iastate.edu) for all 10 airports, confirmed at
the same 30-min cadence as the existing feed and covering through the real
scored period (checked EGLL through 2026-07-30). Cached under
`artifacts/external/wind_direction/<ICAO>.csv`.

Runway heading is NOT a fitted or guessed constant -- it's read directly off
the runway's own name (`RUNWAY_mvt`), which encodes heading in tens of
degrees by standard ICAO convention ("09" = 090 deg, "27" = 270 deg, "36" =
360/000 deg; L/C/R suffixes don't change heading). This is public,
deterministic, and the same information a pilot reads off the runway
number -- not an airport-topology guess like the dropped
`cross_runway_conflict` idea.

Method: same "ceiling before training" pattern as SS2h/SS2j/SS2p/SS2r/SS2w/
SS2z -- in-sample joint OLS of [headwind_component, crosswind_component]
against a quick single-seed `per_airport` baseline's residual, per airport,
on `forward` test (secondary) and `seasonal` test (pre-registered decisive,
bar 5s per SS2o).
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
sys.path.insert(0, str(ROOT))

import config
from common import io as cio
from common import dataset as D
from common.bootstrap import seed_everything
from common.features import CATEGORICAL
from pipeline_per_airport.model import PerAirportLGBM

SEED = 42
FEATURESET, POLICY = "with_aobt", "keep_all"
JOIN_TOLERANCE_MIN = 35  # matches the existing METAR join tolerance (SS2r)
AIRPORTS_OF_INTEREST = ("EGLL", "EHAM")
BAR_SEC = 5.0
WIND_DIR = HERE.parent / "artifacts" / "external" / "wind_direction"

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

RWY_RE = re.compile(r"^(\d{1,2})")


def runway_heading_deg(rwy: "pd.Series") -> "pd.Series":
    """'09R' -> 90.0, '27L' -> 270.0, '36' -> 360.0. NaN for non-numeric (helipads,
    'NA' sentinel, etc)."""
    def parse(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        m = RWY_RE.match(str(v))
        if not m:
            return np.nan
        n = int(m.group(1))
        if not (1 <= n <= 36):
            return np.nan
        return float(n * 10)
    return rwy.map(parse)


def load_wind() -> "pd.DataFrame":
    frames = []
    for f in sorted(WIND_DIR.glob("*.csv")):
        df = pd.read_csv(f, na_values=["M"])
        df["valid"] = pd.to_datetime(df["valid"], utc=True).dt.tz_localize(None)
        df["drct"] = pd.to_numeric(df["drct"], errors="coerce")
        df["sknt"] = pd.to_numeric(df["sknt"], errors="coerce")
        frames.append(df[["station", "valid", "drct", "sknt"]])
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["station", "valid"]).reset_index(drop=True)
    return out


def join_wind(dep_times: "pd.Series", dep_airports: "pd.Series", wind: "pd.DataFrame"
             ) -> "pd.DataFrame":
    """merge_asof backward, per airport, within JOIN_TOLERANCE_MIN."""
    left = pd.DataFrame({"t": dep_times.to_numpy(), "ap": dep_airports.to_numpy()})
    left = left.reset_index().rename(columns={"index": "_orig_idx"})
    left = left.sort_values("t")
    out_drct = np.full(len(left), np.nan)
    out_sknt = np.full(len(left), np.nan)
    tol = pd.Timedelta(minutes=JOIN_TOLERANCE_MIN)
    for ap, grp in left.groupby("ap"):
        w = wind[wind.station == ap]
        if w.empty:
            continue
        merged = pd.merge_asof(grp.sort_values("t"), w.sort_values("valid"),
                               left_on="t", right_on="valid",
                               direction="backward", tolerance=tol)
        out_drct[merged["_orig_idx"].to_numpy()] = merged["drct"].to_numpy()
        out_sknt[merged["_orig_idx"].to_numpy()] = merged["sknt"].to_numpy()
    res = pd.DataFrame({"drct": out_drct, "sknt": out_sknt}, index=left["_orig_idx"])
    return res.sort_index()


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


def run_split(split, wind, mov_times):
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
    wj = join_wind(dep_times, pd.Series(ap, index=Xte.index), wind)
    cov = wj["drct"].notna().mean()
    print(f"  wind join coverage: {cov*100:.1f}%")

    rwy_heading = runway_heading_deg(Xte["RUNWAY_mvt"].astype("string"))
    ang_diff = np.radians(wj["drct"].to_numpy() - rwy_heading.to_numpy())
    headwind = wj["sknt"].to_numpy() * np.cos(ang_diff)
    crosswind = wj["sknt"].to_numpy() * np.abs(np.sin(ang_diff))
    regressors = np.column_stack([headwind, crosswind])

    rows = []
    for a in sorted(pd.unique(ap)):
        sel = ap == a
        ceil, n_ok, n_tot = ols_ceiling(resid[sel], regressors[sel])
        cov_a = n_ok / n_tot if n_tot else float("nan")
        rows.append({"split": split, "airport": a, "n": n_tot, "coverage": cov_a,
                    "ceiling_rmse_reduction_sec": ceil})
        flag = "  <-- of interest" if a in AIRPORTS_OF_INTEREST else ""
        print(f"  {a:6s} n={n_tot:6d} cov={cov_a*100:5.1f}%  ceiling={ceil:+7.3f}s{flag}")
    ceil_p, n_ok_p, n_tot_p = ols_ceiling(resid, regressors)
    print(f"  {'POOLED':6s} n={n_tot_p:6d} cov={n_ok_p/n_tot_p*100:5.1f}%  ceiling={ceil_p:+7.3f}s")
    rows.append({"split": split, "airport": "POOLED", "n": n_tot_p,
                "coverage": n_ok_p / n_tot_p, "ceiling_rmse_reduction_sec": ceil_p})
    return rows


def main() -> int:
    t0 = time.time()
    wind = load_wind()
    print(f"loaded wind: {len(wind)} rows across {wind.station.nunique()} stations "
          f"({time.time()-t0:.0f}s)")

    mov = cio.load_training(columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"])
    dep = mov[mov["PHASE_mvt"].eq("DEP")].set_index("MVT_ID_mvt")
    mov_times = dep["MVT_TIME_UTC_mvt"].dt.tz_localize(None)

    all_rows = []
    for split in ["forward", "seasonal"]:
        all_rows.extend(run_split(split, wind, mov_times))

    df = pd.DataFrame(all_rows)
    df.to_csv(HERE / "summary.csv", index=False)
    print(f"\nwrote {HERE/'summary.csv'}")

    print("\n" + "=" * 78)
    print("Ceiling on the decisive `seasonal` test surface (bar: %.1fs), all airports:" % BAR_SEC)
    seasonal = df[df.split == "seasonal"]
    for _, v in seasonal.iterrows():
        verdict = "CLEARS bar" if v["ceiling_rmse_reduction_sec"] >= BAR_SEC else "does not clear"
        print(f"  {v['airport']:6s}: ceiling {v['ceiling_rmse_reduction_sec']:+.3f}s "
             f"(coverage {v['coverage']*100:.1f}%, n={int(v['n'])}) -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
