#!/usr/bin/env python3
"""Stage 4 -- negative controls, run on the split AS BUILT.

Root CLAUDE.md, Mode 2 step 4: run the label-permutation control right after the
split is built, not only at evaluation. A leak introduced by the split or the
feature builder is far cheaper to find here than after a full GPU sweep.

Two controls:
  * label permutation -- shuffle the TRAINING target, refit a fast LightGBM, and
    require val RMSE to collapse to the no-information floor. It must NOT retain
    predictive power: if it does, something in the feature frame carries the
    target through a channel other than the labels.
  * overfit-one-batch -- a small model on a few thousand rows must be able to
    drive TRAINING error far below the floor. If it cannot, the pipeline is
    broken in a way no amount of tuning will fix.

The no-information floor for RMSE is measured, not assumed: it is the RMSE of
predicting the (permuted) training mean on the untouched validation set.
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
from common.bootstrap import (label_permutation_control, run_dir, seed_everything,
                              write_json_guarded)

FAST = {"objective": "l2", "metric": "rmse", "learning_rate": 0.1,
        "num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--subsample", type=int, default=300_000,
                    help="rows used for the control fits; the control tests the "
                         "pipeline's wiring, not the final model's capacity")
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES),
                    help="how the TRAINING tail is handled; part of the artifact "
                         "key. Evaluation sets are never filtered.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    import lightgbm as lgb

    seed_everything(args.seed)
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "controls", "controls", protocol=proto, seed=args.seed)

    Xtr, ytr, _ = D.load_part(args.split, "train", args.featureset,
                                 filter_target=True,
                                 outlier_policy=args.outlier_policy)
    Xva, yva, _ = D.load_part(args.split, "val", args.featureset, filter_target=False)
    Xva = D.align_columns(Xtr, Xva)
    cats = [c for c in F.CATEGORICAL if c in Xtr.columns]

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(Xtr), size=min(args.subsample, len(Xtr)), replace=False)
    Xs, ys = Xtr.iloc[idx], np.asarray(ytr, dtype="float64")[idx]
    yva_np = np.asarray(yva, dtype="float64")

    def fit_score(labels, n_rounds=args.rounds):
        d = lgb.Dataset(Xs, label=np.asarray(labels, dtype="float64"),
                        categorical_feature=cats, free_raw_data=False)
        b = lgb.train(dict(FAST, seed=args.seed), d, num_boost_round=n_rounds)
        return b, rmse(yva_np, np.clip(b.predict(Xva), config.PREDICTION_CLIP_MIN_SEC,
                                       config.PREDICTION_CLIP_MAX_SEC))

    # ---- reference levels, measured ----
    print("\n== reference levels (measured, not assumed) ==")
    floor = rmse(yva_np, np.full_like(yva_np, float(np.mean(ys))))
    _, real_score = fit_score(ys)
    print("  no-information floor (predict train mean) : {:.3f}".format(floor))
    print("  honest arm on the same subsample          : {:.3f}".format(real_score))

    # ---- control 1: label permutation ----
    print("\n== control 1: label permutation ({} repeats) ==".format(args.n_repeats))

    def score_permuted(permuted, repeat_index):
        _, s = fit_score(permuted)
        print("    repeat {}: permuted val RMSE = {:.3f}".format(repeat_index, s))
        return s

    # Design-gate finding S4: tolerance and real_arm_margin default to 0.05 --
    # in METRIC UNITS, against an RMSE around 500 s. That makes `near_reference`
    # hypersensitive (a 0.06 s fluctuation fails) while `below_real` accepts a
    # real arm beating noise by 0.06 s, i.e. the control had almost no power.
    # Scale both to the metric.
    tol = 0.02 * floor          # ~10 s
    perm = label_permutation_control(
        ys, score_permuted, measured_chance=floor, n_repeats=args.n_repeats,
        real_arm_score=real_score, real_arm_margin=25.0, tolerance=tol,
        seed=args.seed, higher_is_better=False,
        name="label_permutation__" + proto,
    )
    # ...and the helper bounds the permuted score from below only. A permuted run
    # scoring FAR ABOVE the floor is also a broken pipeline (it means the fit is
    # actively harmful), so bound it explicitly from above too.
    # ControlResult is a dataclass, not a mapping: the numbers live in
    # .measured and the JSON-safe form is .to_dict(). Treating it as a dict
    # (.get / dict(...)) raises only AFTER the permutation repeats have run.
    perm_mean = float(perm.measured.get("mean_permuted", float("nan")))
    upper_ok = bool(np.isfinite(perm_mean) and perm_mean <= floor + 4 * tol)
    print("  permuted mean {:.3f} vs floor {:.3f} (+/- {:.1f} s band) -> upper "
          "bound {}".format(perm_mean, floor, tol, "OK" if upper_ok else "EXCEEDED"))
    print("  verdict: {}".format("PASS" if (perm.passed and upper_ok) else "FAIL"))
    for k, v in perm.to_dict().items():
        if k not in ("passed", "name"):
            print("    {}: {}".format(k, v))

    # ---- control 2: overfit one batch ----
    print("\n== control 2: overfit a small batch ==")
    n_small = 4000
    Xb, yb = Xtr.iloc[:n_small], np.asarray(ytr, dtype="float64")[:n_small]
    db = lgb.Dataset(Xb, label=yb, categorical_feature=cats, free_raw_data=False)
    # feature_fraction/bagging_fraction inherited from FAST would handicap the
    # control: a failure must be unambiguous evidence of a broken pipeline, not
    # of deliberate subsampling. Turn both off for this control only.
    bb = lgb.train(dict(FAST, learning_rate=0.2, num_leaves=255, min_data_in_leaf=1,
                        feature_fraction=1.0, bagging_fraction=1.0, bagging_freq=0,
                        seed=args.seed), db, num_boost_round=400)
    train_rmse = rmse(yb, bb.predict(Xb))
    batch_floor = rmse(yb, np.full_like(yb, float(np.mean(yb))))
    ratio = train_rmse / batch_floor
    ofb_pass = ratio < 0.20
    print("  train RMSE on the batch : {:.3f}".format(train_rmse))
    print("  batch mean floor        : {:.3f}".format(batch_floor))
    print("  ratio (want < 0.20)     : {:.4f} -> {}".format(
        ratio, "PASS" if ofb_pass else "FAIL"))

    # ---- control 3: target-encoding leak probe (design-gate finding S5) ----
    # The out-of-fold target encoder is the highest leak-risk component here and
    # nothing tested it. Note WHY control 1 cannot: a leaky in-fold encoding
    # inflates the TRAINING fit while leaving val at the floor, so a val-side
    # permutation check is structurally blind to it. This one watches train.
    print("\n== control 3: target-encoding leak probe (ridge, train-side) ==")
    from pipeline_ridge.train import RidgePipeline
    perm_labels = np.random.default_rng(args.seed).permutation(ys)
    te_pipe = RidgePipeline(args.seed, 10.0).fit(Xs, pd.Series(perm_labels, index=Xs.index))
    te_train_rmse = rmse(perm_labels, te_pipe.predict(Xs))
    te_floor = rmse(perm_labels, np.full_like(perm_labels, float(np.mean(perm_labels))))
    te_ratio = te_train_rmse / te_floor
    # A leak-free encoder cannot fit permuted labels: train RMSE must stay near
    # the permuted-train floor. A leaky in-fold encoding drives it far below.
    te_pass = te_ratio > 0.90
    print("  permuted TRAIN RMSE      : {:.3f}".format(te_train_rmse))
    print("  permuted-train floor     : {:.3f}".format(te_floor))
    print("  ratio (want > 0.90)      : {:.4f} -> {}".format(
        te_ratio, "PASS" if te_pass else "FAIL -- the encoder is leaking the target"))

    rep = {"split": args.split, "featureset": args.featureset, "seed": args.seed,
           "outlier_policy": args.outlier_policy,
           "target_encoding_leak_probe": {
               "permuted_train_rmse": te_train_rmse, "permuted_train_floor": te_floor,
               "ratio": te_ratio, "passed": bool(te_pass)},
           "permutation_upper_bound_ok": upper_ok,
           "no_information_floor_rmse": floor, "honest_arm_rmse": real_score,
           "label_permutation": perm.to_dict(),
           "overfit_one_batch": {"n": n_small, "train_rmse": train_rmse,
                                 "batch_mean_floor": batch_floor, "ratio": ratio,
                                 "passed": bool(ofb_pass)},
           "passed": bool(perm.passed and upper_ok and ofb_pass and te_pass)}
    write_json_guarded(out / "controls.json", rep, force=args.force)
    print("\n  controls {} -> {}".format("PASS" if rep["passed"] else "FAIL", out))
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
