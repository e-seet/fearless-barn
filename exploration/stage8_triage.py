#!/usr/bin/env python3
"""Stage 8: triage the arm ranking against the band it is actually decided on.

The finding this stage exists to make un-ignorable
--------------------------------------------------
On forward/val, 93 of 369,624 rows have y > 7,200 s and ONE has y >= 80,000 s.
On forward/test, 61 and THREE. Under the `keep_all` policy the tail carries
10-50% of every arm's squared error, and the *differences between arms* are
almost entirely differences on those few rows.

Concretely, on test: `mlp_antigravity` beats `lightgbm` by 6.8e9 of squared
error in total, while the three y>=80,000 s rows alone account for 1.0e10 of
that difference -- so on the other 328,006 rows LightGBM is the better model and
the ranking is reversed. The MLP is not identifying those rows: every tail row in
either part is the same cell (LIRF, no flight record, no AOBT_3), the cell's
conditional distribution runs from ~1,000 s to ~87,600 s, and the MLP simply
predicts higher and with more variance inside it (31k-84k where LightGBM predicts
20k-26k). It got closest on the largest row and furthest on two others.

So this stage reports every ranking twice: once pooled (the leaderboard's own
metric, which must remain the headline) and once on `decidable` rows only -- the
artifact band excluded. Where the two disagree, the pooled ranking is a coin flip
on a handful of date errors and must be reported as such, not as a method result.

This is diagnosis, not selection: `--part test` is allowed here because nothing
is chosen from the output. The winner is still whatever compare.py crowned on val.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from common import dataset as D
from common.bootstrap import run_dir, write_json_guarded
from stage6_clip_sweep import BANDS, band_sse

ROOT = Path(__file__).resolve().parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
ARMS = list(config.ARMS)   # one list, in config (see config.ARMS)


def _predict(arm, split, part, featureset, seed, policy, X, Xtr, meta=None):
    """Clipped predictions for one arm, via this build's own dispatch.

    This used to call `pipeline_ensemble_antigravity.train.predict_base_arm`
    directly. That module is third-party to this build and its arm list is a
    closed set, so the first arm added to `config.ARMS` made this stage die with
    `ValueError: Unknown base arm 'per_airport'` -- after it had loaded the
    features and scored five arms. `common/predict.py` owns the dispatch now and
    delegates only the *_antigravity arms.
    """
    from common.predict import predict_arm
    return predict_arm(arm, split, part, featureset, seed, policy,
                       X=X, meta=meta, Xtr=Xtr)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    Xtr, ytr, _ = D.load_part(args.split, "train", args.featureset,
                              filter_target=True, outlier_policy=args.outlier_policy)

    out = {"split": args.split, "featureset": args.featureset,
           "outlier_policy": args.outlier_policy, "seed": args.seed,
           "bands": [{"name": n, "lo": lo, "hi": (None if not np.isfinite(hi) else hi)}
                     for n, lo, hi in BANDS],
           "parts": {}}

    for part in ("val", "test"):
        X, y, meta = D.load_part(args.split, part, args.featureset, filter_target=False)
        y = np.asarray(y, dtype="float64")
        rows = []
        for arm in ARMS:
            d = run_dir(config.ARTIFACTS / "runs", arm, protocol=proto,
                        seed=args.seed, create=False)
            if not d.exists():
                print("  (skip {}: no run at {})".format(arm, d))
                continue
            pred = _predict(arm, args.split, part, args.featureset, args.seed,
                            args.outlier_policy, X, Xtr, meta=meta)
            b = band_sse(y, np.asarray(pred, dtype="float64"))
            rows.append({"arm": arm, "pooled_rmse": float(np.sqrt(b["total_sse"] / y.size)),
                         "decidable_rmse": b["decidable_rmse"], "bands": b})

        by_pooled = [r["arm"] for r in sorted(rows, key=lambda r: r["pooled_rmse"])]
        by_dec = [r["arm"] for r in sorted(rows, key=lambda r: r["decidable_rmse"])]
        n_art = rows[0]["bands"]["artifact"]["n"] if rows else 0

        print("\n=== {} / {} ({} rows; {} in the artifact band) ===".format(
            args.split, part, y.size, n_art))
        print("  {:22s} {:>9s} {:>11s} {:>9s} {:>10s} {:>12s}".format(
            "arm", "pooled", "decidable", "in_band", "mid_tail", "artifact_SSE"))
        for r in sorted(rows, key=lambda r: r["pooled_rmse"]):
            b = r["bands"]
            print("  {:22s} {:9.2f} {:11.2f} {:9.2f} {:10.1f} {:12.3e}".format(
                r["arm"], r["pooled_rmse"], r["decidable_rmse"],
                b["in_band"]["rmse"], b["mid_tail"]["rmse"] or float("nan"),
                b["artifact"]["sse"]))
        agree = by_pooled == by_dec
        print("  ranking pooled     : {}".format(" > ".join(by_pooled)))
        print("  ranking decidable  : {}".format(" > ".join(by_dec)))
        print("  {}".format(
            "orderings AGREE -- the ranking does not rest on the artifact rows"
            if agree else
            "orderings DISAGREE -- the pooled ranking is decided by {} artifact "
            "row(s) and is not a method result".format(n_art)))
        out["parts"][part] = {"n": int(y.size), "n_artifact": int(n_art),
                              "arms": rows, "ranking_pooled": by_pooled,
                              "ranking_decidable": by_dec,
                              "orderings_agree": bool(agree)}

    out["headline"] = (
        "pooled RMSE is the leaderboard metric and stays the headline; the "
        "decidable column says how much of any gap between arms is a handful of "
        "one-day date errors rather than method")
    write_json_guarded(
        config.ARTIFACTS / "triage" / "bands__{}__seed{}.json".format(proto, args.seed),
        out, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
