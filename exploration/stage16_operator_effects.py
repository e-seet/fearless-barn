#!/usr/bin/env python3
"""Stage 16 -- test TAXI_OUT_DRIVERS.md hypothesis 2, operator effects sub-item.

Does AIRCRAFT_OPERATOR_flt identity explain taxi-out variance beyond stand
geometry? `pipeline_per_operator/` (tier B, "real but subsumed") already shows
operator carries SOME signal as a model feature, but that's a modeling result,
not a direct read of how much and where -- this measures it directly, the same
way stage15's part_b measured the business-segment effect.

Method, per airport
  1. Raw eta-squared: variance of taxi-out EXPLAINED by operator identity
     (between-operator SS / total SS), operators with n>=MIN_N pooled into
     "Other" so a few huge fleets don't get outvoted by hundreds of 3-flight
     operators.
  2. Stand-residualized eta-squared: subtract each (airport, stand)'s own mean
     taxi-out first (the model's strongest geometry signal, per features.py),
     THEN compute operator's eta-squared on what's left. This is the same
     "control for where the aircraft sits" logic stage15's part_b used for
     business segment, generalized from a 2-group diff to a many-level
     categorical via eta-squared instead of a mean difference.
  3. Report the 3 fastest and 3 slowest operators (by stand-residualized mean),
     among those meeting MIN_N, per airport -- for a concrete read, not just a
     variance number.

Usage
    python3 stage16_operator_effects.py
"""
from __future__ import annotations

import sys as _s, pathlib as _p; _r = _p.Path(__file__).resolve().parents[1]; exec(open(_r / "tools" / "repo_layout.py").read()); dual(_r / "exploration")
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from common import dataset as D
from common import io as cio
from common.bootstrap import write_json_guarded

OUT_DIR = config.ARTIFACTS / "audit"
AIRPORT_COL = "ADEP_mvt"
STAND_COL = "STAND_mvt"
OPERATOR_COL = "AIRCRAFT_OPERATOR_flt"
MIN_N = 100


def eta_squared(values: "np.ndarray", groups: "np.ndarray") -> float:
    """Fraction of variance in `values` explained by `groups` (between-group SS / total SS)."""
    total_ss = float(np.sum((values - values.mean()) ** 2))
    if total_ss == 0:
        return 0.0
    between_ss = 0.0
    for g in pd.unique(groups):
        m = groups == g
        gv = values[m]
        between_ss += len(gv) * (gv.mean() - values.mean()) ** 2
    return float(between_ss / total_ss)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--min-n", type=int, default=MIN_N)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    X = D.build_scope("train2025", args.featureset)
    raw = cio.load_training(columns=[config.ID, "PHASE_mvt", config.TARGET])
    dep = raw[raw["PHASE_mvt"].eq("DEP")]
    y_by_id = pd.Series(dep[config.TARGET].astype("float64").to_numpy(),
                        index=dep[config.ID].to_numpy())
    ids = X[config.ID].to_numpy()
    y = y_by_id.reindex(ids).to_numpy()
    valid = np.isfinite(y) & (y > config.TARGET_IMPOSSIBLE_MAX_SEC)

    airport = X[AIRPORT_COL].astype("object").to_numpy()
    stand = X[STAND_COL].astype("object").to_numpy()
    operator = X[OPERATOR_COL].astype("object").to_numpy()

    rows = []
    for ap_code in sorted(pd.unique(airport[valid])):
        m = valid & (airport == ap_code)
        yy = y[m]
        op = operator[m].copy()
        st = stand[m]
        n = int(m.sum())

        # pool small operators into "Other" so eta-squared isn't dominated by noise
        counts = pd.Series(op).value_counts()
        keep = set(counts[counts >= args.min_n].index)
        op_grouped = np.array([o if o in keep else "__other__" for o in op], dtype=object)

        raw_eta = eta_squared(yy, op_grouped)

        # stand-residualized: subtract each stand's own mean. A handful of rows
        # have a null STAND_mvt (0.002-0.006% at some airports) -- groupby drops
        # NaN keys, so their transform() comes back NaN, and a plain numpy
        # .mean() (unlike pandas') propagates that ONE NaN through the whole
        # array's sum, silently NaN-ing eta-squared for the entire airport.
        # Drop just those few rows for this computation rather than losing the
        # airport's number to a 1-in-50,000 missing field.
        stand_df = pd.DataFrame({"stand": st, "y": yy})
        stand_mean = stand_df.groupby("stand")["y"].transform("mean").to_numpy()
        stand_ok = np.isfinite(stand_mean)
        resid = yy[stand_ok] - stand_mean[stand_ok]
        resid_eta = eta_squared(resid, op_grouped[stand_ok])

        # top/bottom operators by stand-residualized mean, among kept operators.
        # Index against the SAME stand_ok-filtered arrays resid/op_grouped[stand_ok]
        # came from -- `om` must be built on the filtered op array, not the
        # full-length one, or this mismatches resid's now-shorter length.
        op_grouped_ok = op_grouped[stand_ok]
        yy_ok = yy[stand_ok]
        op_means = []
        for o in keep:
            om = op_grouped_ok == o
            if om.sum() < args.min_n:
                continue
            op_means.append((o, int(om.sum()), float(resid[om].mean()), float(yy_ok[om].mean())))
        op_means.sort(key=lambda t: t[2])

        rows.append({
            "airport": ap_code, "n": n, "n_operators_kept": len(keep),
            "mean_taxiout": float(yy.mean()),
            "raw_eta_squared": raw_eta, "stand_residualized_eta_squared": resid_eta,
            "fastest_operators": op_means[:3],
            "slowest_operators": op_means[-3:][::-1],
        })

    rows.sort(key=lambda r: -r["stand_residualized_eta_squared"])

    print("=" * 100)
    print("Operator effect on taxi-out, per airport (min n={} per operator)".format(args.min_n))
    print("=" * 100)
    print("{:6s} {:>8s} {:>8s} {:>10s} {:>12s}".format(
        "adep", "n", "n_ops", "raw_eta2", "stand_eta2"))
    for r in rows:
        print("{:6s} {:8d} {:8d} {:10.4f} {:12.4f}".format(
            r["airport"], r["n"], r["n_operators_kept"],
            r["raw_eta_squared"], r["stand_residualized_eta_squared"]))

    print("\n  eta-squared = fraction of taxi-out variance explained by operator identity.")
    print("  raw_eta2 includes geometry (different operators favor different stands);")
    print("  stand_eta2 is AFTER subtracting each stand's own mean -- the residual is")
    print("  closer to a genuine SOP/handling-agent effect, not just where they park.")

    print("\nFastest/slowest operator (stand-residualized), 2 example airports:")
    for r in rows[:1] + rows[-1:]:
        print("  {}: fastest={}, slowest={}".format(
            r["airport"],
            [(o, n, round(m, 1)) for o, n, m, _ in r["fastest_operators"]],
            [(o, n, round(m, 1)) for o, n, m, _ in r["slowest_operators"]]))

    report = {"featureset": args.featureset, "min_n": args.min_n, "airports": rows}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_guarded(OUT_DIR / "operator_effects.json", report, force=args.force)
    print("\nwrote {}".format(OUT_DIR / "operator_effects.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
