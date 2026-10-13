#!/usr/bin/env python3
"""Label-correction experiment on LIRF's AOBT-unmatched population.

Background (`tests_stand_turnover/RESULT.md`): 59% of LIRF's large-delay
(>3600s), no-flight-record training rows have a training LABEL that is
physically impossible -- another aircraft is recorded at the same stand
while this one's recorded `BLOCK_TIME_UTC_mvt` claims it is still parked
there. The physical bound (time to that contradicting event) averages just
28.5% of the recorded `TAXITIME_SEC_mvt`. This population is 0.08% of rows
but 20.7% of pooled SSE (`TAXI_OUT_DRIVERS.md` sec.2q) -- the single
worst-behaved segment in the dataset.

This experiment caps each contradicted row's TRAINING label at its physical
bound (floored at `config.PREDICTION_CLIP_MIN_SEC`) and re-trains `per_airport`
LightGBM -- chosen specifically because it fits one independent booster per
airport (`pipeline_per_airport/model.py`), so touching only LIRF's training
rows is GUARANTEED to leave every other airport's booster byte-identical.
Evaluation always uses the ORIGINAL, uncorrected labels (val/test are never
touched) -- only the training supervision changes.

Three variants, three seeds, two splits:
  - l2ctl:   unmodified labels (control)
  - phys:    contradicted rows capped at their TRUE physical bound
  - placebo: the SAME 439 rows capped, but bounds SHUFFLED among them
             (a fresh random permutation per seed) -- tests whether the
             effect (if any) comes from correctly matching row to bound, or
             just from shrinking arbitrary extreme LIRF labels. Per this
             project's standing discipline (permutation controls for any
             real-looking effect before trusting it).

Splits: `forward` (primary honest split) and `seasonal` (board-composition-
matched, per SS2o) -- a real effect should show on both, not just one.

Pre-registered bar (decided before running): `phys` must beat `l2ctl` on
pooled test RMSE by more than the arm's seed-sd, on BOTH splits, AND must
beat `placebo` by a comparable margin (attributing any gain to the specific
physical correction, not generic outlier-shrinkage). LIRF-only test RMSE is
also reported since the intervention is LIRF-scoped by construction.

IMPORTANT precedent (`TAXI_OUT_DRIVERS.md` sec.2l): a related idea -- dropping
15 heuristically-flagged rows from this same territory -- was decisively
net-negative on test (+5.45s) via a shared per-airport BLEND-weight shift,
even though the training-pool-local logic looked sound and val even improved
(-2.34s). This script measures `per_airport`'s OWN val/test only, which
cannot show that failure mode (no blend re-solve happens here) -- a positive
result here is necessary but not sufficient; the blend-level check is a
separate, later step if this looks promising.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import config
from common import dataset as D
from common import evaluate as E
from common import io as cio
from common.bootstrap import run_dir, seed_everything
from pipeline_per_airport.model import PerAirportLGBM, AIRPORT_COL
from common import features as F

FEATURESET = "with_aobt"
SPLITS = ["forward", "seasonal"]
SEEDS = [42, 43, 44]
VARIANTS = ["l2ctl", "phys", "placebo"]
CLIP_LO, CLIP_HI = config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC
AIRPORT = "LIRF"
LARGE_DELAY_THRESHOLD_SEC = 3600

PARAMS = {  # identical base hyperparams to pipeline_per_airport/train.py PARAMS
    "objective": "l2", "metric": "rmse",
    "learning_rate": 0.05, "num_leaves": 255, "min_data_in_leaf": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "max_cat_threshold": 64, "cat_smooth": 20.0,
    "verbosity": -1, "num_threads": 0,
}
NUM_BOOST_ROUND, PATIENCE = 5000, 200


def _params(seed):
    p = dict(PARAMS)
    p.update({"seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed,
              "device_type": "cpu"})
    return p


def build_contradiction_table() -> "pd.DataFrame":
    """Reproduces tests_stand_turnover's detection exactly: for each LIRF DEP
    row with no flight record and TAXITIME > threshold, find whether another
    movement occupies the same stand strictly inside [BLOCK_TIME, takeoff).
    Returns one row per CONTRADICTED training row: mvt_id, claimed, bound_sec.
    """
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "STAND_mvt",
            "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", config.TARGET,
            "AIRCRAFT_OPERATOR_flt"]
    mov = cio.load_training(columns=cols)
    for c in ("MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"):
        mov[c] = pd.to_datetime(mov[c], utc=True).dt.tz_localize(None)

    is_dep = mov["PHASE_mvt"].eq("DEP")
    is_arr = mov["PHASE_mvt"].eq("ARR")
    at_lirf = (is_dep & mov["ADEP_mvt"].eq(AIRPORT)) | (is_arr & mov["ADES_mvt"].eq(AIRPORT))
    lirf = mov[at_lirf].copy()
    lirf["event_time"] = lirf["MVT_TIME_UTC_mvt"]
    lirf_stand = lirf.dropna(subset=["STAND_mvt"]).sort_values("event_time")

    dep_lirf = lirf[lirf["PHASE_mvt"].eq("DEP")].copy()
    dep_lirf["has_flight_record"] = dep_lirf["AIRCRAFT_OPERATOR_flt"].notna()
    pop = dep_lirf[~dep_lirf["has_flight_record"]].copy()
    pop_large = pop[pop[config.TARGET] > LARGE_DELAY_THRESHOLD_SEC].copy()
    pop_large = pop_large.dropna(subset=["STAND_mvt", "BLOCK_TIME_UTC_mvt"])

    by_stand = {s: g["event_time"].to_numpy() for s, g in lirf_stand.groupby("STAND_mvt")}
    by_stand_id = {s: g["MVT_ID_mvt"].to_numpy() for s, g in lirf_stand.groupby("STAND_mvt")}

    rows = []
    for row in pop_large.itertuples():
        times = by_stand.get(row.STAND_mvt)
        ids = by_stand_id.get(row.STAND_mvt)
        if times is None:
            continue
        b = np.datetime64(row.BLOCK_TIME_UTC_mvt, "ns")
        e = np.datetime64(row.MVT_TIME_UTC_mvt, "ns")
        lo = np.searchsorted(times, b, side="right")
        hi = np.searchsorted(times, e, side="left")
        cand = [(t, i) for t, i in zip(times[lo:hi], ids[lo:hi]) if i != row.MVT_ID_mvt]
        if cand:
            first_t = min(t for t, _ in cand)
            bound_sec = (first_t - b) / np.timedelta64(1, "s")
            rows.append({"mvt_id": row.MVT_ID_mvt,
                        "claimed_sec": float(getattr(row, config.TARGET)),
                        "bound_sec": float(bound_sec)})
    return pd.DataFrame(rows)


def corrected_labels(ytr: "pd.Series", contra: "pd.DataFrame", variant: str,
                     rng: "np.random.Generator") -> "pd.Series":
    if variant == "l2ctl":
        return ytr
    present = contra[contra["mvt_id"].isin(ytr.index)].copy()
    if not len(present):
        return ytr
    y = ytr.copy()
    if variant == "phys":
        bounds = present["bound_sec"].to_numpy()
    elif variant == "placebo":
        bounds = rng.permutation(present["bound_sec"].to_numpy())
    else:
        raise ValueError(variant)
    capped = np.minimum(y.loc[present["mvt_id"]].to_numpy(), bounds)
    capped = np.clip(capped, CLIP_LO, None)
    y.loc[present["mvt_id"]] = capped
    return y


def clip(p):
    return np.clip(p, CLIP_LO, CLIP_HI)


def main() -> int:
    import lightgbm as lgb
    t0 = time.time()
    print("[label-correction] building contradiction table...")
    contra = build_contradiction_table()
    print("  {} contradicted rows found (full training pool)".format(len(contra)))

    rows = []
    for split in SPLITS:
        Xtr_raw, ytr, pool_meta = D.load_part(split, "train", FEATURESET,
                                              filter_target=True, outlier_policy="keep_all")
        Xva_raw, yva, meta_va = D.load_part(split, "val", FEATURESET, filter_target=False)
        Xte_raw, yte, meta_te = D.load_part(split, "test", FEATURESET, filter_target=False)
        Xva = D.align_columns(Xtr_raw, Xva_raw)
        Xte = D.align_columns(Xtr_raw, Xte_raw)
        cats = [c for c in F.CATEGORICAL if c in Xtr_raw.columns]
        n_present = contra["mvt_id"].isin(ytr.index).sum()
        print("\n===== split {} ===== ({} contradicted rows in this split's train pool)".format(
            split, n_present))

        for seed in SEEDS:
            seed_everything(seed)
            rng = np.random.default_rng(seed)
            for variant in VARIANTS:
                y_corr = corrected_labels(ytr, contra, variant, rng)
                m = PerAirportLGBM(_params(seed), NUM_BOOST_ROUND, PATIENCE, cats)
                m.fit(Xtr_raw, y_corr.to_numpy(), pool_meta, Xva, yva.to_numpy(), meta_va,
                      lgb=lgb, verbose=False)
                pv = clip(m.predict(Xva, meta_va))
                pt = clip(m.predict(Xte, meta_te))

                rv = E.evaluate(yva.to_numpy(), pv, meta_va, ytr.to_numpy(), part="val",
                                arm="per_airport", split=split, featureset=FEATURESET, seed=seed)
                rt = E.evaluate(yte.to_numpy(), pt, meta_te, ytr.to_numpy(), part="test",
                                arm="per_airport", split=split, featureset=FEATURESET, seed=seed)

                lirf_va = meta_va[AIRPORT_COL].to_numpy() == AIRPORT
                lirf_te = meta_te[AIRPORT_COL].to_numpy() == AIRPORT
                lirf_va_rmse = float(np.sqrt(np.mean((yva.to_numpy()[lirf_va] - pv[lirf_va]) ** 2))) if lirf_va.any() else float("nan")
                lirf_te_rmse = float(np.sqrt(np.mean((yte.to_numpy()[lirf_te] - pt[lirf_te]) ** 2))) if lirf_te.any() else float("nan")

                row = {"split": split, "seed": seed, "variant": variant,
                       "val_rmse": round(rv["rmse"], 3), "val_inband": round(rv["rmse_in_band"], 3),
                       "test_rmse": round(rt["rmse"], 3), "test_inband": round(rt["rmse_in_band"], 3),
                       "lirf_val_rmse": round(lirf_va_rmse, 2), "lirf_test_rmse": round(lirf_te_rmse, 2),
                       "n_corrected": int(n_present) if variant != "l2ctl" else 0}
                rows.append(row)
                print("  {} seed{} {:8s}  val {:8.2f}/ib{:7.2f}  test {:8.2f}/ib{:7.2f}  "
                      "LIRF val {:9.2f} test {:9.2f}".format(
                          split, seed, variant, row["val_rmse"], row["val_inband"],
                          row["test_rmse"], row["test_inband"], lirf_va_rmse, lirf_te_rmse))

    with (HERE / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def agg(split, variant, key):
        xs = [r[key] for r in rows if r["split"] == split and r["variant"] == variant]
        return (float(np.mean(xs)), float(np.std(xs)), xs)

    lines = ["label-correction experiment (per_airport, LIRF-only intervention)",
             "=" * 78]
    for split in SPLITS:
        lines.append("\n-- split: {} --".format(split))
        lines.append("{:20s} {:>10s} {:>10s} {:>10s}  {:>9s} {:>9s}  verdict".format(
            "metric", "l2ctl", "phys", "placebo", "phys-ctl", "phys-plb"))
        for key, label in [("test_rmse", "test RMSE"), ("test_inband", "test in-band"),
                           ("lirf_test_rmse", "LIRF test RMSE"),
                           ("val_rmse", "val RMSE"), ("lirf_val_rmse", "LIRF val RMSE")]:
            ctl_m, ctl_sd, _ = agg(split, "l2ctl", key)
            phy_m, phy_sd, _ = agg(split, "phys", key)
            plb_m, plb_sd, _ = agg(split, "placebo", key)
            d_ctl = phy_m - ctl_m
            d_plb = phy_m - plb_m
            decisive_ctl = abs(d_ctl) > max(ctl_sd, phy_sd)
            decisive_plb = abs(d_plb) > max(plb_sd, phy_sd)
            verdict = "phys {} ctl ({}); phys {} placebo ({})".format(
                "better" if d_ctl < 0 else "worse", "decisive" if decisive_ctl else "noise",
                "better" if d_plb < 0 else "worse", "decisive" if decisive_plb else "noise")
            lines.append("{:20s} {:10.2f} {:10.2f} {:10.2f}  {:+9.2f} {:+9.2f}  {}".format(
                label, ctl_m, phy_m, plb_m, d_ctl, d_plb, verdict))
    txt = "\n".join(lines)
    (HERE / "summary.txt").write_text(txt + "\n", encoding="utf-8")
    print("\n" + txt)
    print("\n[{:.0f}s] wrote {} and summary.txt".format(time.time() - t0, HERE / "summary.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
