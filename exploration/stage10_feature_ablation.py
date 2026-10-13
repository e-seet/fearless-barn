#!/usr/bin/env python3
"""Stage 10: drop feature groups one at a time and see whether val RMSE improves.

Why bother -- and why "improves" is the right word rather than "degrades"
------------------------------------------------------------------------
A GBM is not harmed much by an irrelevant feature, but it is not free either:
every useless column is another axis `feature_fraction` can waste a split on, and
this build already has direct evidence of inert columns. `stage5_ablations.py`
showed the arm ranking is unchanged when the absolute-time features are dropped,
and `lightgbm_unsup` measured its ten unsupervised columns at 1.1% of total gain
while making test 3.1 s WORSE. So the question "does removing something help?" is
live rather than rhetorical.

Two groups are the obvious suspects, for reasons that are specific rather than
general:

* **absolute time** (`month_num`, `doy`, `doy_sin`, `doy_cos`) -- under the
  `forward` split every test row is out of the training range on these, so they
  sit directly on the extrapolation axis. A tree that splits on `doy` learns a
  boundary that cannot generalise forward, which is the split artifact
  `stage5_ablations.py` was written to detect.
* **near-duplicate schedule deltas** -- `delta_sched_sec` and `abs_delta_sched_sec`
  (likewise the EOBT pair) carry the same quantity twice, once signed and once
  folded. That is deliberate and may help, but it is worth checking rather than
  assuming.

Selection surface: VAL. Reports the drop that helps most, if any does; a result
where every drop hurts is the useful outcome that the feature set is not padded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from common import dataset as D
from common import features as F
from common.metrics import rmse
from common.bootstrap import declare_selection_set, write_json_guarded

GROUPS = {
    "absolute_time": ["month_num", "doy", "doy_sin", "doy_cos"],
    "abs_delta_dupes": ["abs_delta_sched_sec", "abs_delta_eobt_sec"],
    "signed_delta_dupes": ["delta_sched_sec", "delta_eobt_sec"],
    "next_windows": ["dep_next_15m", "dep_next_30m", "dep_next_60m",
                     "same_rwy_next_15m", "same_rwy_next_30m"],
    "coarse_geo": ["dest_region"],
    "aobt_derived": ["aobt_minus_eobt_sec", "aobt_minus_sched_sec",
                     "aobt_minus_lobt_sec"],
}

# Five of the six groups above improved val RMSE individually, each by an amount
# near the 1.67 s seed spread. Individually that is ambiguous; the question that
# is NOT ambiguous is whether the drops compound. If removing four groups at once
# gains materially more than the best single drop, the feature set is genuinely
# padded rather than four separate coin flips landing the same way.
GROUPS["combo_all_helpful"] = (GROUPS["absolute_time"] + GROUPS["abs_delta_dupes"]
                               + GROUPS["signed_delta_dupes"] + GROUPS["next_windows"])
GROUPS["combo_deltas_only"] = GROUPS["abs_delta_dupes"] + GROUPS["signed_delta_dupes"]
GROUPS["combo_time_and_next"] = GROUPS["absolute_time"] + GROUPS["next_windows"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--num-boost-round", type=int, default=5000)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--groups", default="", help="comma-separated subset of GROUPS")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    import lightgbm as lgb
    from pipeline_lightgbm.train import PARAMS

    if args.split in config.SELECTION_FORBIDDEN_SPLITS:
        raise SystemExit("split {!r} is not a selection surface".format(args.split))

    Xtr, ytr, _ = D.load_part(args.split, "train", args.featureset,
                              filter_target=True, outlier_policy=args.outlier_policy)
    Xva, yva, _ = D.load_part(args.split, "val", args.featureset, filter_target=False)
    Xva = D.align_columns(Xtr, Xva)
    ytr_a = np.asarray(ytr, dtype="float64")
    yva_a = np.asarray(yva, dtype="float64")

    names = [g.strip() for g in args.groups.split(",") if g.strip()] or list(GROUPS)
    params = dict(PARAMS)
    params.update({"seed": args.seed, "bagging_seed": args.seed,
                   "feature_fraction_seed": args.seed})

    def fit(drop):
        keep = [c for c in Xtr.columns if c not in drop]
        cats = [c for c in F.CATEGORICAL if c in keep]
        dtr = lgb.Dataset(Xtr[keep], label=ytr_a, categorical_feature=cats,
                          free_raw_data=False)
        dva = lgb.Dataset(Xva[keep], label=yva_a, categorical_feature=cats,
                          reference=dtr, free_raw_data=False)
        b = lgb.train(params, dtr, num_boost_round=args.num_boost_round,
                      valid_sets=[dva], valid_names=["val"],
                      callbacks=[lgb.early_stopping(args.patience, verbose=False),
                                 lgb.log_evaluation(0)])
        p = np.clip(b.predict(Xva[keep], num_iteration=b.best_iteration),
                    config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
        return rmse(yva_a, p), int(b.best_iteration), len(keep)

    base_rmse, base_it, base_n = fit(set())
    print("baseline (all {} features): val RMSE {:.3f}  best_iter {}".format(
        base_n, base_rmse, base_it))
    rows = [{"group": "(baseline)", "dropped": [], "n_features": base_n,
             "val_rmse": base_rmse, "delta_vs_baseline": 0.0, "best_iteration": base_it}]
    for g in names:
        cols = [c for c in GROUPS[g] if c in Xtr.columns]
        if not cols:
            print("  (skip {}: none of its columns are present)".format(g))
            continue
        r, it, nf = fit(set(cols))
        rows.append({"group": g, "dropped": cols, "n_features": nf, "val_rmse": r,
                     "delta_vs_baseline": r - base_rmse, "best_iteration": it})
        print("  drop {:20s} ({} cols) -> val {:.3f}  ({:+.3f})  best_iter {}".format(
            g, len(cols), r, r - base_rmse, it))

    helpful = [r for r in rows[1:] if r["delta_vs_baseline"] < 0]
    print()
    if helpful:
        best = min(helpful, key=lambda r: r["delta_vs_baseline"])
        print("  dropping {} HELPS by {:.3f} s of val RMSE".format(
            best["group"], -best["delta_vs_baseline"]))
        print("  NOTE: LightGBM's val seed spread on this protocol is ~1.67 s. A gain "
              "smaller than that is noise, not a finding.")
    else:
        print("  no group helped -- the feature set is not padded on these axes.")

    sel = declare_selection_set("feature-group ablation", selected_on="val",
                                value=[r["group"] for r in rows[1:]])
    write_json_guarded(
        config.ARTIFACTS / "ablations" / "feature_groups__{}__seed{}.json".format(
            config.protocol_key(args.split, args.featureset, args.outlier_policy),
            args.seed),
        {"baseline_val_rmse": base_rmse, "results": rows, "selection": dict(sel),
         "seed_spread_reference_sec": 1.67,
         "caveat": ("single seed; a delta smaller than the arm's seed spread is "
                    "noise. Re-run across seeds before acting on one.")},
        force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
