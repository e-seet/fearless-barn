#!/usr/bin/env python3
"""Stage 1 -- data audit. Runs on the FULL real data, before any modelling.

Contract
  Trigger : the raw challenge parquets exist under prc-2026-datasets/
  Action  : validate schema, the ranking-availability contract, target sanity,
            coverage, cross-split identity overlap, categorical drift, and the
            strength of the AOBT_3 near-target proxy
  Output  : artifacts/audit/audit_report.json  (+ a printed summary)
  Gate    : G1 no MVT_ID overlap between training and ranking
            G2 the blanked-on-ranking-DEP column set is EXACTLY what config declares
            G3 submitting.parquet's ID set == ranking DEP's ID set
            G4 the target's own definition holds on training DEP rows
  A failing gate HALTS the build -- it does not warn and continue.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from common import io as cio
from common.bootstrap import write_json_guarded
from common.metrics import rmse

FAILURES: "list[str]" = []


def gate(name: str, ok: bool, detail: str) -> bool:
    status = "PASS" if ok else "FAIL"
    print("  [{}] {}: {}".format(status, name, detail))
    if not ok:
        FAILURES.append("{}: {}".format(name, detail))
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="overwrite an existing report")
    args = ap.parse_args(argv)

    config.AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    rep: dict = {"stage": "stage1_audit"}

    # ---------------------------------------------------------------- files
    print("\n== files ==")
    files = cio.training_files()
    rep["training_files"] = [f.name for f in files]
    per_file = {}
    for f in files:
        md = pq.ParquetFile(f).metadata
        per_file[f.name] = {"rows": md.num_rows, "cols": md.num_columns}
    rep["per_file"] = per_file
    print("  {} training files, {} rows total".format(
        len(files), sum(v["rows"] for v in per_file.values())))

    train = cio.load_training()
    rank = cio.load_ranking()
    sub = pq.read_table(config.SUBMITTING_PARQUET).to_pandas()
    tr_dep = train[train.PHASE_mvt.eq("DEP")]
    rk_dep = rank[rank.PHASE_mvt.eq("DEP")]
    rep["counts"] = {"train_rows": len(train), "train_dep": len(tr_dep),
                     "rank_rows": len(rank), "rank_dep": len(rk_dep),
                     "submitting_rows": len(sub)}
    print("  train DEP {} | ranking DEP {} | submitting {}".format(
        len(tr_dep), len(rk_dep), len(sub)))

    # ------------------------------------------------- G2 ranking contract
    print("\n== G2: ranking-availability contract ==")
    blanked = sorted(c for c in rank.columns
                     if c not in ("month",) and rk_dep[c].isna().all())
    rep["blanked_on_ranking_dep"] = blanked
    gate("G2_blanked_columns", set(blanked) == set(config.BLANKED_ON_RANKING_DEP),
         "observed {} vs declared {}".format(blanked, list(config.BLANKED_ON_RANKING_DEP)))
    avail = {c: float(rk_dep[c].notna().mean()) for c in rank.columns
             if c not in blanked and c != "month"}
    rep["rank_dep_availability"] = {k: round(v, 5) for k, v in sorted(avail.items())}

    # ------------------------------------------------------- G3 submission
    print("\n== G3: submission template ==")
    same = set(sub[config.ID]) == set(rk_dep[config.ID])
    gate("G3_submitting_ids", same and len(sub) == len(rk_dep),
         "submitting IDs {} ranking-DEP IDs (n={} vs {})".format(
             "==" if same else "!=", len(sub), len(rk_dep)))
    rep["submitting_target_all_null"] = bool(sub[config.TARGET].isna().all())

    # ------------------------------------------------- G1 identity overlap
    print("\n== G1: cross-split identity overlap ==")
    ov_id = len(set(train[config.ID]) & set(rank[config.ID]))
    gate("G1_mvt_id_overlap", ov_id == 0, "{} shared MVT_ID".format(ov_id))
    fid_tr = set(train.FLIGHT_ID_mvt.dropna())
    fid_rk = set(rank.FLIGHT_ID_mvt.dropna())
    rep["flight_id_overlap"] = len(fid_tr & fid_rk)
    rep["dup_mvt_id_in_train"] = int(train[config.ID].duplicated().sum())
    rep["dup_mvt_id_in_rank"] = int(rank[config.ID].duplicated().sum())
    print("  FLIGHT_ID overlap train/rank: {} | dup MVT_ID train {} rank {}".format(
        rep["flight_id_overlap"], rep["dup_mvt_id_in_train"], rep["dup_mvt_id_in_rank"]))

    # ------------------------------------------------ G4 target definition
    print("\n== G4: target definition ==")
    d = (tr_dep.MVT_TIME_UTC_mvt - tr_dep.BLOCK_TIME_UTC_mvt).dt.total_seconds()
    exact = float((d == tr_dep[config.TARGET]).mean())
    gate("G4_target_identity", exact == 1.0,
         "TAXITIME == MVT_TIME - BLOCK_TIME on {:.4%} of train DEP rows".format(exact))
    rep["target_identity_match_frac"] = exact

    # ---------------------------------------------------- target quality
    print("\n== target quality (train DEP) ==")
    y = tr_dep[config.TARGET].astype("float64")
    q = {
        "n": int(y.size), "n_null": int(y.isna().sum()),
        "min": float(y.min()), "max": float(y.max()),
        "mean": float(y.mean()), "std": float(y.std()),
        "n_le_0": int((y <= 0).sum()),
        "n_lt_60": int((y < config.TRAIN_TARGET_MIN_SEC).sum()),
        "n_gt_7200": int((y > config.TRAIN_TARGET_MAX_SEC).sum()),
        "n_gt_21600": int((y > 21600).sum()),
        "pct": {str(p): float(np.percentile(y, p))
                for p in (0.01, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 99.99, 100)},
    }
    q["frac_outside_train_bounds"] = (q["n_lt_60"] + q["n_gt_7200"]) / q["n"]
    rep["target_quality"] = q
    print("  n={} min={:.0f} max={:.0f} mean={:.1f}".format(q["n"], q["min"], q["max"], q["mean"]))
    print("  <=0s: {} | <60s: {} | >2h: {} | >6h: {}  ({:.4%} outside training bounds)".format(
        q["n_le_0"], q["n_lt_60"], q["n_gt_7200"], q["n_gt_21600"], q["frac_outside_train_bounds"]))

    # ------------------------------------------------------ coverage matrix
    print("\n== airport x month coverage ==")
    tr_cov = pd.crosstab(tr_dep.ADEP_mvt, tr_dep.month)
    rk_cov = pd.crosstab(rk_dep.ADEP_mvt, rk_dep.month)
    rep["train_airports"] = sorted(tr_dep.ADEP_mvt.unique().tolist())
    rep["rank_airports"] = sorted(rk_dep.ADEP_mvt.unique().tolist())
    rep["train_coverage"] = tr_cov.to_dict()
    rep["rank_coverage"] = rk_cov.to_dict()
    rep["train_zero_cells"] = int((tr_cov == 0).sum().sum())
    rep["rank_zero_cells"] = int((rk_cov == 0).sum().sum())
    print(rk_cov.to_string())
    print("  train zero cells: {} | ranking zero cells: {}".format(
        rep["train_zero_cells"], rep["rank_zero_cells"]))

    # leaderboard composition weights (share of scored rows per airport-month)
    comp = (rk_dep.groupby([rk_dep.ADEP_mvt, rk_dep.month], observed=True)
            .size().rename("n").reset_index())
    comp["weight"] = comp.n / comp.n.sum()
    rep["leaderboard_composition"] = {
        "{}|{}".format(r.ADEP_mvt, r.month): {"n": int(r.n), "weight": float(r.weight)}
        for r in comp.itertuples()}
    rep["leaderboard_airport_weights"] = (
        rk_dep.ADEP_mvt.value_counts(normalize=True).round(6).to_dict())

    # ------------------------------------------------------- proxy strength
    print("\n== AOBT_3 near-target proxy ==")
    proxy_stats = {}
    for col in ("AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt", "SCHED_TIME_UTC_mvt"):
        p = (tr_dep.MVT_TIME_UTC_mvt - tr_dep[col]).dt.total_seconds()
        m = p.notna() & y.notna()
        yy, pp = y[m], p[m]
        # clip the proxy to the same sane band a model would be trained on
        ppc = pp.clip(config.TRAIN_TARGET_MIN_SEC, config.TRAIN_TARGET_MAX_SEC)
        proxy_stats[col] = {
            "coverage": float(m.mean()),
            "rmse_raw": rmse(yy, pp),
            "rmse_clipped": rmse(yy, ppc),
            "median_signed_error": float((pp - yy).median()),
            "frac_within_60s": float(((pp - yy).abs() <= 60).mean()),
        }
        print("  {:22s} cov={:.3f} rmse_raw={:9.1f} rmse_clip={:8.1f} within60s={:.3f}".format(
            col, proxy_stats[col]["coverage"], proxy_stats[col]["rmse_raw"],
            proxy_stats[col]["rmse_clipped"], proxy_stats[col]["frac_within_60s"]))
    rep["proxy_stats"] = proxy_stats
    rep["proxy_coverage_on_ranking"] = {
        c: float(rk_dep[c].notna().mean()) for c in config.PROXY_COLS}

    # ------------------------------------------------- flight-join failures
    print("\n== flight-record join ==")
    flt_cols = [c for c in train.columns if c.endswith("_flt")]
    rep["no_flight_record"] = {
        "train_dep_frac": float(tr_dep[flt_cols].isna().all(axis=1).mean()),
        "rank_dep_frac": float(rk_dep[flt_cols].isna().all(axis=1).mean()),
    }
    print("  rows with NO flight record: train DEP {:.4%} | ranking DEP {:.4%}".format(
        rep["no_flight_record"]["train_dep_frac"], rep["no_flight_record"]["rank_dep_frac"]))

    # documented mvt-vs-flt inconsistencies
    incon = {}
    for a, b in (("ADEP_mvt", "ADEP_flt"), ("ADES_mvt", "ADES_flt"),
                 ("AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt"),
                 ("FLIGHT_RULE_mvt", "FLIGHT_RULE_flt")):
        both = tr_dep[[a, b]].dropna()
        incon[a + "_vs_" + b] = {"n_compared": int(len(both)),
                                 "disagree_frac": float((both[a] != both[b]).mean())}
        print("  {:24s} vs {:24s} disagree {:.4%} of {} comparable rows".format(
            a, b, incon[a + "_vs_" + b]["disagree_frac"], len(both)))
    rep["mvt_flt_inconsistency"] = incon

    # ------------------------------------------------- categorical drift
    print("\n== categorical drift: unseen levels in ranking ==")
    drift = {}
    for col in ("STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt",
                "AIRCRAFT_OPERATOR_flt", "ADES_mvt", "MARKET_SEGMENT_flt"):
        seen = set(tr_dep[col].dropna().unique())
        rk_vals = rk_dep[col].dropna()
        unseen_rows = float((~rk_vals.isin(seen)).mean()) if len(rk_vals) else float("nan")
        drift[col] = {
            "train_cardinality": len(seen),
            "rank_cardinality": int(rk_vals.nunique()),
            "rank_rows_with_unseen_level": unseen_rows,
            "rank_null_frac": float(rk_dep[col].isna().mean()),
        }
        print("  {:24s} train_card={:6d} rank_card={:6d} unseen_rows={:.4%} null={:.4%}".format(
            col, drift[col]["train_cardinality"], drift[col]["rank_cardinality"],
            unseen_rows, drift[col]["rank_null_frac"]))
    rep["categorical_drift"] = drift

    # STAND is the key congestion/geometry proxy -- how usable is it?
    rep["stand_null_frac"] = {"train_dep": float(tr_dep.STAND_mvt.isna().mean()),
                              "rank_dep": float(rk_dep.STAND_mvt.isna().mean())}

    # ------------------------------------------------------ numeric drift
    print("\n== numeric drift: hour-of-day and taxi-in (ARR) ==")
    tr_arr = train[train.PHASE_mvt.eq("ARR")]
    rk_arr = rank[rank.PHASE_mvt.eq("ARR")]
    rep["arr_taxiin"] = {
        "train_mean": float(tr_arr[config.TARGET].mean()),
        "rank_mean": float(rk_arr[config.TARGET].mean()),
        "train_median": float(tr_arr[config.TARGET].median()),
        "rank_median": float(rk_arr[config.TARGET].median()),
    }
    print("  ARR taxi-in mean: train {:.1f}s -> ranking {:.1f}s".format(
        rep["arr_taxiin"]["train_mean"], rep["arr_taxiin"]["rank_mean"]))
    # per-airport monthly mean taxi-out, 2025 vs the seasonally-matched 2026 proxy
    tr_month_mean = tr_dep.groupby(["ADEP_mvt", "month"], observed=True)[config.TARGET].mean()
    rep["train_airport_month_mean_taxiout"] = {
        "{}|{}".format(a, m): float(v) for (a, m), v in tr_month_mean.items()}

    # --------------------------------------------------------------- verdict
    rep["gates_failed"] = FAILURES
    rep["passed"] = not FAILURES
    out = config.AUDIT_DIR / "audit_report.json"
    write_json_guarded(out, rep, force=args.force)
    print("\n== verdict ==")
    if FAILURES:
        print("  AUDIT FAILED ({} gate(s)):".format(len(FAILURES)))
        for f in FAILURES:
            print("    - " + f)
        print("  report: {}".format(out))
        return 1
    print("  all gates PASS. report: {}".format(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
