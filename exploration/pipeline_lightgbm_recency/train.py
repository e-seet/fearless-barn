#!/usr/bin/env python3
"""Arm lightgbm_recency: LightGBM with exponentially recency-weighted rows.

The hypothesis
--------------
The real task extrapolates **13 months forward**: train on 2025, score Jan and Jul
2026. The audit measured that the process drifts over exactly that horizon --
LTFM's mean taxi-out moved +20.2% between 2025 and Jan 2026, and the
scored-share-weighted mean absolute drift across airports is 43.2 s, which is
comparable to a sixth of the whole error budget. A pooled model weights January
2025 and August 2025 equally when predicting the future; if the process is
drifting, the older months are describing a system that no longer exists.

So: weight each training row by `0.5 ** (months_before_pool_end / half_life)`.
A short half-life discards history and raises variance; a long one converges to
the unweighted arm. `half_life=inf` is included in the sweep as the exact
no-weighting control, so this arm re-measures its own baseline rather than
comparing against a number produced by different code.

What it can and cannot address
------------------------------
It targets drift, which is a *bulk* effect spread over every row -- not the
date-artifact rows, which are not a trend and cannot be down-weighted away. So
the place to look for a gain is the in-band band, and `stage8_triage.py`'s
`decidable` column is the honest read on whether one appeared.

Selection surface: VAL. The half-life is chosen on validation RMSE and declared
through `declare_selection_set`, exactly like the ridge arm's alpha, and it is
recorded in `model.json` so the `full` refit can freeze it.
"""
from __future__ import annotations

import sys as _s, pathlib as _p; _r = _p.Path(__file__).resolve().parents[2]; exec(open(_r / "tools" / "repo_layout.py").read()); dual(_r / "exploration")
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common import dataset as D
from common import features as F
from common.metrics import rmse
from common.bootstrap import (convergence_evidence, declare_selection_set, run_dir,
                              seed_everything, write_json_guarded)

ARM = "lightgbm_recency"

PARAMS = {
    "objective": "l2", "metric": "rmse",
    "learning_rate": 0.05, "num_leaves": 255, "min_data_in_leaf": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "max_cat_threshold": 64, "cat_smooth": 20.0,
    "verbosity": -1, "num_threads": 0,
}

# Months. `inf` is the unweighted control and must stay in the grid: a sweep that
# cannot reproduce its own starting point cannot tell a real gain from a changed
# environment.
HALF_LIVES = [1.0, 2.0, 3.0, 6.0, 12.0, float("inf")]


def recency_weights(months: "pd.Series", half_life: float) -> np.ndarray:
    """0.5 ** (months_before_the_newest_training_month / half_life)."""
    m = pd.PeriodIndex(pd.to_datetime(months.astype(str) + "-01"), freq="M")
    # Subtracting two PeriodIndexes yields an Index of offset objects, not a
    # number -- `.n` exists on a single offset, not on the Index. Work in period
    # ordinals instead. (Caught by the tiny-input smoke run that MISTAKES.md
    # #77/#78 prescribe, before any training time was spent.)
    ords = m.asi8.astype("float64")
    age = ords.max() - ords                          # 0 for the newest month
    if not np.isfinite(half_life):
        return np.ones(len(age), dtype="float64")
    return np.power(0.5, age / float(half_life))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--num-boost-round", type=int, default=5000,
                    help="SAFETY CAP, not the stopping condition")
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--half-lives", default=None,
                    help="comma-separated months, or 'inf'; selected on VAL")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu", "cuda"])
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--freeze-from", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    import lightgbm as lgb

    seed_everything(args.seed)
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed,
                  extra=({"tag": args.tag} if args.tag else None))

    Xtr, ytr, pool_meta = D.load_part(args.split, "train", args.featureset,
                                      filter_target=True,
                                      outlier_policy=args.outlier_policy)
    cats = [c for c in F.CATEGORICAL if c in Xtr.columns]
    ytr_a = np.asarray(ytr, dtype="float64")

    refit_only = not config.SPLITS[args.split]["val"]
    frozen = None
    if refit_only:
        if not args.freeze_from:
            raise SystemExit(
                "split '{}' has no validation set, so neither the half-life nor "
                "the round count can be selected on it. Pass --freeze-from."
                .format(args.split))
        frozen = json.loads(Path(args.freeze_from).read_text())
        if frozen.get("split") in config.SELECTION_FORBIDDEN_SPLITS:
            raise SystemExit(
                "--freeze-from points at split '{}', not a valid selection "
                "surface.".format(frozen["split"]))
        grid = [float(frozen["half_life_months"])]
    else:
        Xva, yva, _ = D.load_part(args.split, "val", args.featureset,
                                  filter_target=False)
        Xva = D.align_columns(Xtr, Xva)
        grid = ([float(x) for x in args.half_lives.split(",")]
                if args.half_lives else list(HALF_LIVES))

    params = dict(frozen["params"]) if frozen else dict(PARAMS)
    params.update({"seed": args.seed, "bagging_seed": args.seed,
                   "feature_fraction_seed": args.seed, "device_type": args.device})

    print("[{}] train n={} val n={} feats={} cats={} half-lives={}".format(
        ARM, len(Xtr), 0 if refit_only else len(Xva), Xtr.shape[1], len(cats), grid))

    best = None
    sweep = []
    for hl in grid:
        w = recency_weights(pool_meta["month"], hl)
        dtr = lgb.Dataset(Xtr, label=ytr_a, weight=w, categorical_feature=cats,
                          free_raw_data=False)
        if refit_only:
            ratio = len(Xtr) / max(1, frozen["n_train"])
            rounds = max(1, int(round(frozen["best_iteration"] * ratio)))
            print("  frozen best_iteration {} x pool ratio {:.3f} -> {} rounds "
                  "(heuristic, not a measured optimum)".format(
                      frozen["best_iteration"], ratio, rounds))
            booster = lgb.train(params, dtr, num_boost_round=rounds,
                                callbacks=[lgb.log_evaluation(0)])
            booster.best_iteration = rounds
            conv = {"refit_only": True, "hit_max_epochs": False, "rounds": rounds,
                    "frozen_from": args.freeze_from,
                    "note": "no early stopping: this protocol has no val by design"}
            best = {"half_life": hl, "val_rmse": None, "booster": booster,
                    "conv": conv, "weights_stats": None}
            break

        dva = lgb.Dataset(Xva, label=np.asarray(yva, dtype="float64"),
                          categorical_feature=cats, reference=dtr,
                          free_raw_data=False)
        hist = {}
        booster = lgb.train(params, dtr, num_boost_round=args.num_boost_round,
                            valid_sets=[dva], valid_names=["val"],
                            callbacks=[lgb.early_stopping(args.patience, verbose=False),
                                       lgb.log_evaluation(0),
                                       lgb.record_evaluation(hist)])
        curve = hist["val"]["rmse"]
        conv = convergence_evidence(
            [{"epoch": i + 1, "val_rmse": v} for i, v in enumerate(curve)],
            max_epochs=args.num_boost_round, patience=args.patience,
            metric_key="val_rmse", mode="min")
        vp = np.clip(booster.predict(Xva, num_iteration=booster.best_iteration),
                     config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
        v = rmse(yva, vp)
        # effective sample size: how much data a half-life actually leaves behind
        ess = float(w.sum() ** 2 / np.square(w).sum())
        sweep.append({"half_life_months": (None if not np.isfinite(hl) else hl),
                      "val_rmse": v, "best_iteration": int(booster.best_iteration),
                      "early_stop_fired": bool(not conv.get("hit_max_epochs", True)),
                      "effective_n": ess, "effective_frac": ess / len(Xtr)})
        print("   half_life={:>6s}  val RMSE={:8.3f}  best_iter={:5d}  "
              "effective n={:9.0f} ({:.1%})  {}".format(
                  "inf" if not np.isfinite(hl) else "{:g}".format(hl), v,
                  booster.best_iteration, ess, ess / len(Xtr),
                  "early-stopped" if not conv.get("hit_max_epochs", True) else "HIT CAP"))
        if best is None or v < best["val_rmse"]:
            best = {"half_life": hl, "val_rmse": v, "booster": booster,
                    "conv": conv, "weights_stats": {"effective_n": ess}}

    if not refit_only:
        ctrl = next((r for r in sweep if r["half_life_months"] is None), None)
        if ctrl:
            print("  best half-life {} vs the UNWEIGHTED control {:.3f}: {:+.3f} s "
                  "of val RMSE".format(
                      "inf" if not np.isfinite(best["half_life"]) else best["half_life"],
                      ctrl["val_rmse"], ctrl["val_rmse"] - best["val_rmse"]))

    sel = declare_selection_set("lightgbm_recency half-life + best_iteration",
                                selected_on="frozen" if refit_only else "val",
                                value={"half_life_months":
                                       (None if not np.isfinite(best["half_life"])
                                        else best["half_life"]),
                                       "best_iteration": int(best["booster"].best_iteration)})

    booster = best["booster"]
    booster.save_model(str(out / "model.txt"), num_iteration=booster.best_iteration)
    write_json_guarded(out / "model.json", {
        "arm": ARM, "split": args.split, "featureset": args.featureset,
        "seed": args.seed, "params": params,
        "half_life_months": (None if not np.isfinite(best["half_life"])
                             else best["half_life"]),
        "half_life_is_unweighted_control": bool(not np.isfinite(best["half_life"])),
        "sweep": sweep,
        "num_boost_round_cap": args.num_boost_round, "patience": args.patience,
        "best_iteration": int(booster.best_iteration),
        "val_rmse_clipped": best["val_rmse"],
        "refit_only": bool(refit_only),
        "convergence": best["conv"], "selection": dict(sel),
        "n_train": int(len(Xtr)), "n_features": int(Xtr.shape[1]),
        "train_pool_ids_sha": D.pool_fingerprint(Xtr.index),
        "weighting_note": ("row weights change the LOSS, not the training pool -- "
                           "every row is still present, so the pool fingerprint is "
                           "identical to the unweighted arm's and compare.py's "
                           "equality gate still applies"),
        "outlier_policy": dict(pool_meta.attrs.get("outlier_policy", {})),
        "categorical_features": cats,
    }, force=args.force)
    print("[{}] saved -> {}".format(ARM, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
