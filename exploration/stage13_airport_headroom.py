#!/usr/bin/env python3
"""Stage 13: where is the REDUCIBLE error, once the metric is scored the way the
leaderboard scores it?

`stage12` established that the tail carries ~15-21 % of pooled squared error from
~0.02 % of rows, and `tests_composition_correction/` established that val's
airport mix is not the scored set's, so the corrected metric
(`sqrt(sum_g w_g * MSE_g)`, w = scored-set share) is the one that matters.

Neither answered the question that decides what to build next: **within the
corrected metric, which airport x band cell holds error that a model could still
remove?** Three quantities have been conflated in this build's history:

  * a big RMSE                 -- may just mean long taxi times (EGLL's mean is
                                  1392 s vs EHAM's 727 s; its RMSE/mean ratio is
                                  the second BEST of the ten)
  * a big share of the metric  -- weight x MSE, which is what actually moves the
                                  leaderboard
  * REDUCIBLE error            -- the part a better model could take, as opposed
                                  to tail rows that are unlearnable artifacts

An airport can be large on the second and near-zero on the third. Targeting the
wrong one is how effort gets spent where it cannot pay.

The skill column is the guard against the first confusion: model RMSE against the
naive_offblock floor on the SAME rows. A cell where the model is already far
below the trivial floor has little left regardless of how big its RMSE looks.

Usage
    python3 stage13_airport_headroom.py
    python3 stage13_airport_headroom.py --arm airport_operator_multimodel --tag msmlp
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from common import dataset as D
from common.predict import predict_arm
from common.bootstrap import write_json_guarded

OUT_DIR = config.ARTIFACTS / "audit"
AIRPORT_COL = "ADEP_mvt"
IN_BAND_MAX = 7200.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="airport_operator_multimodel")
    ap.add_argument("--tag", default="msmlp")
    ap.add_argument("--baseline-arm", default="naive_offblock",
                    help="trivial floor used for the skill column")
    ap.add_argument("--split", default="forward", choices=sorted(config.SPLITS))
    ap.add_argument("--part", default="val", choices=["val", "test"],
                    help="val by design -- this is a decision input, and "
                         "selection must not read test")
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if args.part == "test":
        print("  WARNING: --part test. This stage is a DECISION input; reading "
              "test to choose what to build next is selection on test.")

    weights = json.loads((config.AUDIT_DIR / "audit_report.json").read_text()
                         )["leaderboard_airport_weights"]

    Xtr, _, _ = D.load_part(args.split, "train", args.featureset,
                            filter_target=True, outlier_policy=args.outlier_policy)
    X, y, meta = D.load_part(args.split, args.part, args.featureset, filter_target=False)
    y = np.asarray(y, dtype="float64")
    ap_arr = meta[AIRPORT_COL].astype("object").to_numpy()

    p = predict_arm(args.arm, args.split, args.part, args.featureset, args.seed,
                    args.outlier_policy, X=X, meta=meta, Xtr=Xtr, tag=args.tag)
    p0 = predict_arm(args.baseline_arm, args.split, args.part, args.featureset,
                     args.seed, args.outlier_policy, X=X, meta=meta, Xtr=Xtr)

    in_band = y <= IN_BAND_MAX
    rows = []
    for a, w in weights.items():
        m = (ap_arr == a)
        n = int(m.sum())
        mse = float(np.mean((y[m] - p[m]) ** 2))
        for band, bm in (("in_band", m & in_band), ("tail", m & ~in_band)):
            nb = int(bm.sum())
            if nb == 0:
                continue
            se = float(np.sum((y[bm] - p[bm]) ** 2))
            se0 = float(np.sum((y[bm] - p0[bm]) ** 2))
            # contribution to the corrected metric: w * (SE_cell / n_airport)
            contrib = w * se / n
            rows.append({
                "airport": a, "band": band, "weight": w,
                "n_rows": nb, "n_airport": n,
                "rmse": math.sqrt(se / nb),
                "baseline_rmse": math.sqrt(se0 / nb),
                "skill_vs_naive": 1.0 - math.sqrt(se / nb) / math.sqrt(se0 / nb),
                "contrib_to_corrected_mse": contrib,
            })
        rows[-1]["airport_mse"] = mse

    total = sum(r["contrib_to_corrected_mse"] for r in rows)
    for r in rows:
        r["pct_of_corrected_metric"] = 100.0 * r["contrib_to_corrected_mse"] / total

    print("=" * 92)
    print("Corrected val metric decomposed by airport x band   (total RMSE {:.2f})".format(
        math.sqrt(total)))
    print("=" * 92)
    print("{:6s} {:8s} {:>7s} {:>8s} {:>9s} {:>10s} {:>8s} {:>8s}".format(
        "adep", "band", "weight", "n", "rmse", "naive", "skill", "% metric"))
    for r in sorted(rows, key=lambda r: -r["pct_of_corrected_metric"]):
        print("{:6s} {:8s} {:7.4f} {:8d} {:9.1f} {:10.1f} {:7.1%} {:7.1f}%".format(
            r["airport"], r["band"], r["weight"], r["n_rows"], r["rmse"],
            r["baseline_rmse"], r["skill_vs_naive"], r["pct_of_corrected_metric"]))

    ib = [r for r in rows if r["band"] == "in_band"]
    tl = [r for r in rows if r["band"] == "tail"]
    print("\n  in-band total: {:5.1f}% of the corrected metric across {:,} rows".format(
        sum(r["pct_of_corrected_metric"] for r in ib), sum(r["n_rows"] for r in ib)))
    print("  tail total:    {:5.1f}% of the corrected metric across {:,} rows".format(
        sum(r["pct_of_corrected_metric"] for r in tl), sum(r["n_rows"] for r in tl)))
    print("\n  Read the two columns together: a cell with a big '% metric' AND a")
    print("  LOW skill still has room; a big '% metric' with HIGH skill is mostly")
    print("  irreducible and spending effort there is how the ladder happened.")

    report = {"arm": args.arm, "tag": args.tag, "split": args.split,
              "part": args.part, "seed": args.seed,
              "corrected_rmse": math.sqrt(total), "cells": rows}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_guarded(OUT_DIR / "airport_headroom.json", report, force=args.force)
    print("\nwrote {}".format(OUT_DIR / "airport_headroom.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
