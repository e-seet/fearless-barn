#!/usr/bin/env python3
"""Diagnostic (no retrain, no features.py edit): does the runway configuration
of trailing ARRIVALS carry residual signal at EGLL/EHAM, where taxi-out is
plausibly dominated by active-runway-mode and crossing conflicts?

Scope, deliberately narrowed on advisor review: `cross_runway_conflict` was
DROPPED. It would need a hardcoded per-airport runway-topology table (which
physical runway pairs cross), and that is exactly the kind of magic-constant
artifact `airport_debiased_multimodel` was quarantined for (PIPELINE_REVIEW.md
tier E) -- worse, EGLL's runway-alternation ("segregated mode") is specifically
DESIGNED to avoid crossings, so a naive conflict table risks encoding the wrong
physical premise. Only two features, both pure data or uncontroversial public
fact:

  * `active_arr_runway` -- the runway of the most recent ARRIVAL (PHASE_mvt !=
    'DEP') at the same airport in the trailing 30 minutes (a nearest-neighbour
    proxy for "which runway configuration is currently live" -- simpler than a
    true windowed mode, and equivalent to it whenever the configuration is
    stable within the window, which is the regime the hypothesis is about).
  * `dep_rwy_eq_active_arr` -- 1 if this departure's own RUNWAY_mvt matches
    that active arrival runway, else 0. NaN (excluded from the fit, coverage
    reported) when no arrival landed in the trailing 30 min.
  * `polderbaan_flag` -- EHAM-specific: 1 if RUNWAY_mvt in {'18R','36L'} (the
    Polderbaan, a well-documented long-taxi runway), else 0.

Method: an in-sample joint OLS of these 2 regressors against a quick
single-seed `per_airport` baseline's residual -- the same "ceiling before
training" pattern as SS2h/SS2j/SS2p/SS2r/SS2w. Reported PER AIRPORT (not just
pooled -- SS0a's failure mode was exactly a pooled gain attributable to the
wrong airport), on both `forward` test (secondary) and `seasonal` test (the
pre-registered decisive, board-composition-matched surface), with EGLL/EHAM
as the airports the hypothesis is actually about.

Verified before writing this: `RUNWAY_mvt` is 0% null on BOTH DEP and ARR
rows in the real `ranking.parquet` (344,841 DEP / 344,693 ARR rows checked
directly) -- so this feature family is buildable at scoring time if the
ceiling here justifies it.
"""
from __future__ import annotations

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
from common import evaluate as E
from common.bootstrap import seed_everything
from common.features import _to_ns, CATEGORICAL
from pipeline_per_airport.model import PerAirportLGBM

SEED = 42
FEATURESET, POLICY = "with_aobt", "keep_all"
WINDOW_MIN = 30
AIRPORTS_OF_INTEREST = ("EGLL", "EHAM")
BAR_SEC = 5.0

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


def build_active_arr_runway(mov: "pd.DataFrame") -> "pd.DataFrame":
    """Per-DEP-row: runway of the most recent arrival at the same airport within
    the trailing WINDOW_MIN minutes. Returns a frame indexed by MVT_ID_mvt."""
    is_dep = mov["PHASE_mvt"].eq("DEP").to_numpy()
    airport = np.where(is_dep, mov["ADEP_mvt"].to_numpy(), mov["ADES_mvt"].to_numpy())
    t_ns = _to_ns(mov["MVT_TIME_UTC_mvt"])
    rwy_all = mov["RUNWAY_mvt"].astype("string").to_numpy()
    mvt_id = mov["MVT_ID_mvt"].to_numpy()

    dep_idx = np.flatnonzero(is_dep)
    n = dep_idx.size
    active_arr_rwy = np.full(n, None, dtype=object)
    window_ns = WINDOW_MIN * 60 * 1_000_000_000

    for ap in pd.unique(airport):
        if ap is None or (isinstance(ap, float) and np.isnan(ap)):
            continue
        at_ap = airport == ap
        ap_dep = np.flatnonzero(at_ap & is_dep)
        ap_arr = np.flatnonzero(at_ap & ~is_dep)
        if ap_dep.size == 0:
            continue
        p = np.searchsorted(dep_idx, ap_dep)  # position of these rows within dep_idx
        q = t_ns[ap_dep]
        if ap_arr.size == 0:
            continue
        arr_order = np.argsort(t_ns[ap_arr], kind="stable")
        arr_sorted_t = t_ns[ap_arr][arr_order]
        arr_sorted_rwy = rwy_all[ap_arr][arr_order]

        # index of the nearest preceding arrival (side='left' gives insertion
        # point; subtract 1 for the last event strictly before q)
        idx = np.searchsorted(arr_sorted_t, q, side="left") - 1
        ok = idx >= 0
        if ok.any():
            within_window = (q[ok] - arr_sorted_t[idx[ok]]) <= window_ns
            sel = np.flatnonzero(ok)[within_window]
            active_arr_rwy[p[sel]] = arr_sorted_rwy[idx[sel]]

    out = pd.DataFrame({
        "MVT_ID_mvt": mvt_id[dep_idx],
        "dep_runway": rwy_all[dep_idx],
        "active_arr_runway": active_arr_rwy,
    })
    out["dep_rwy_eq_active_arr"] = np.where(
        pd.isna(out["active_arr_runway"]), np.nan,
        (out["dep_runway"] == out["active_arr_runway"]).astype("float64"))
    return out.set_index("MVT_ID_mvt")


def ols_ceiling(resid: "np.ndarray", regressors: "np.ndarray") -> float:
    """In-sample joint OLS: variance of resid explained by regressors, as an
    RMSE reduction (optimistic ceiling, same method as SS2h/SS2j)."""
    ok = np.isfinite(regressors).all(axis=1) & np.isfinite(resid)
    if ok.sum() < 20:
        return float("nan"), 0, int(ok.sum())
    Xr = np.column_stack([np.ones(ok.sum()), regressors[ok]])
    yr = resid[ok]
    coef, *_ = np.linalg.lstsq(Xr, yr, rcond=None)
    pred = Xr @ coef
    rmse_before = np.sqrt(np.mean(yr ** 2))
    rmse_after = np.sqrt(np.mean((yr - pred) ** 2))
    return float(rmse_before - rmse_after), int(ok.sum()), int(len(resid))


def run_split(split, runway_df):
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
    rmse = np.sqrt(np.mean(resid ** 2))
    print(f"  baseline per_airport test RMSE: {rmse:.2f}  ({time.time()-t0:.0f}s)")

    ap = meta_te["ADEP_mvt"].to_numpy()
    dep_rwy = Xte["RUNWAY_mvt"].astype("string").to_numpy()
    idx = Xte.index.to_numpy()
    rd = runway_df.reindex(idx)
    dep_eq = rd["dep_rwy_eq_active_arr"].to_numpy()
    polder = ((ap == "EHAM") & (pd.Series(dep_rwy).isin(["18R", "36L"]).to_numpy())
             ).astype("float64")
    regressors = np.column_stack([dep_eq, polder])

    rows = []
    for a in sorted(pd.unique(ap)):
        sel = ap == a
        ceil, n_ok, n_tot = ols_ceiling(resid[sel], regressors[sel])
        cov = n_ok / n_tot if n_tot else float("nan")
        rows.append({"split": split, "airport": a, "n": n_tot, "coverage": cov,
                    "ceiling_rmse_reduction_sec": ceil})
        flag = "  <-- of interest" if a in AIRPORTS_OF_INTEREST else ""
        print(f"  {a:6s} n={n_tot:6d} cov={cov*100:5.1f}%  ceiling={ceil:+7.3f}s{flag}")
    ceil_pooled, n_ok_p, n_tot_p = ols_ceiling(resid, regressors)
    print(f"  {'POOLED':6s} n={n_tot_p:6d} cov={n_ok_p/n_tot_p*100:5.1f}%  ceiling={ceil_pooled:+7.3f}s")
    rows.append({"split": split, "airport": "POOLED", "n": n_tot_p,
                "coverage": n_ok_p / n_tot_p, "ceiling_rmse_reduction_sec": ceil_pooled})
    return rows


def main() -> int:
    t0 = time.time()
    mov = cio.load_training(columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt",
                                     "RUNWAY_mvt", "MVT_TIME_UTC_mvt"])
    print(f"loaded raw movements: {len(mov)} rows ({time.time()-t0:.0f}s)")
    runway_df = build_active_arr_runway(mov)
    cov_all = runway_df["dep_rwy_eq_active_arr"].notna().mean()
    print(f"active_arr_runway coverage (whole year, all airports pooled): {cov_all*100:.1f}%  "
          f"({time.time()-t0:.0f}s)")

    all_rows = []
    for split in ["forward", "seasonal"]:
        all_rows.extend(run_split(split, runway_df))

    df = pd.DataFrame(all_rows)
    df.to_csv(HERE / "summary.csv", index=False)
    print(f"\nwrote {HERE/'summary.csv'}")

    print("\n" + "=" * 78)
    print("EGLL/EHAM ceiling on the decisive `seasonal` test surface (bar: %.1fs):" % BAR_SEC)
    for a in AIRPORTS_OF_INTEREST:
        row = df[(df.split == "seasonal") & (df.airport == a)]
        if len(row):
            v = row.iloc[0]
            verdict = "CLEARS bar" if v["ceiling_rmse_reduction_sec"] >= BAR_SEC else "does NOT clear bar"
            print(f"  {a}: ceiling {v['ceiling_rmse_reduction_sec']:+.3f}s "
                 f"(coverage {v['coverage']*100:.1f}%, n={int(v['n'])}) -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
