#!/usr/bin/env python3
"""Arm airport_operator_catboost_multimodel: PER-AIRPORT NNLS blend of
catboost_antigravity + airport_operator_blend + mlp_antigravity.

Where this comes from
----------------------
pipeline_airport_catboost_multimodel (catboost_antigravity + mlp_antigravity,
per-airport convex weight, swapping OUT airport_operator_blend entirely) beat
the MLP alone robustly (-7.70s, SD 1.11 over 3 seeds) but only tied our
existing best blend airport_operator_multimodel (-0.47s, SD 1.16, direction
inconsistent across seeds -- see exploration/TAXI_OUT_DRIVERS.md SS2l). That
swap discarded airport_operator_blend's own internal tree-ensemble structure
(per_airport + per_operator) entirely. This arm tests the fairer question:
does CatBoost's diversity ADD to the existing blend, not substitute for it.

Why per-airport, not a single global weight vector
----------------------------------------------------
tri_blend already measured the alternative directly: ONE global NNLS weight
over 3 arms (per_airport + per_operator + mlp_antigravity, no per-airport
structure) LOST clearly -- val 263 vs MLP alone's ~254 -- because "the
tree/neural mix must vary by airport, which a single global weight can't
express" (pipeline_review.csv). airport_operator_multimodel's per-airport
2-arm weight is the best-performing blend in this build for exactly that
reason. So this arm combines what's separately already validated here:
per-airport structure (from airport_operator_multimodel) with NNLS for more
than 2 arms (from tri_blend) -- not a novel design, a principled combination
of two things this codebase has already tested independently.

Free-parameter caution
------------------------
Per-airport NNLS over 3 arms is up to 2 effective degrees of freedom per
airport (weights are NOT renormalised to sum to 1, same convention as
tri_blend) x 10 airports = up to 20 free parameters. MISTAKES.md's lever-3
lesson (a 9-cell x 3-arm grid -- 27 params -- measured 246.94 in-sample vs
254.24 honest, WORSE than baseline) is a direct warning that this is in a
range where overfitting the weight-solve itself is a real risk, not just a
theoretical one. Mitigated the same way airport_operator_multimodel already
is: weights are solved ONLY on forward_inner (a holdout carved from TRAIN,
months the final models never trained on), never on val, so an overfit
weight-solve would show up as forward_inner beating val by an implausible
margin -- both are reported so this is visible, not just asserted.

Requires catboost_antigravity, airport_operator_blend and mlp_antigravity to
all be trained under BOTH the forward and forward_inner protocols for the
target seed (same rule pipeline_airport_operator_multimodel and tri_blend
already impose on their own base arms).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls, minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common import dataset as D
from common.metrics import rmse
from common.predict import predict_arm
from common.bootstrap import (declare_selection_set, run_dir, seed_everything,
                              write_json_guarded)
from pipeline_per_airport.model import AIRPORT_COL
from pipeline_airport_operator_multimodel.train import parse_seeds, mlp_prediction
from pipeline_airport_catboost_multimodel.train import catboost_prediction

ARM = "airport_operator_catboost_multimodel"
TREE_ARM = "airport_operator_blend"


def tree_prediction(split, part, featureset, seed, outlier_policy, X, meta, Xtr, tag=None):
    return predict_arm(TREE_ARM, split, part, featureset, seed, outlier_policy,
                       X=X, meta=meta, Xtr=Xtr, tag=tag)


def solve_weights_per_airport_nnls(y, p_cb, p_tree, p_mlp, airport) -> dict[str, list]:
    """Per-airport NNLS over 3 predictors. Weights are NOT renormalised to
    sum to 1 -- same convention as tri_blend, for the same reason: a
    non-negative unconstrained-sum fit is the more honest read of what NNLS
    actually solved, and forcing renormalisation would silently change the
    fitted model without changing what's reported.

    Measured unstable at one seed/airport (LIRF, seed44: solved weights
    summed to 1.15 with the MLP weight pushed past 1.0 -- NNLS chasing noise
    in LIRF's small, extreme-outlier-driven local sample, not a sample-size
    problem: LIRF's forward_inner holdout (30,670 rows) isn't even the
    smallest airport's). See solve_weights_per_airport_simplex for the fix.
    """
    weights = {}
    for a in sorted(pd.unique(airport)):
        m = (airport == a)
        P = np.column_stack([p_cb[m], p_tree[m], p_mlp[m]])
        w, _ = nnls(P, y[m])
        weights[str(a)] = [float(x) for x in w]
    return weights


def solve_weights_per_airport_simplex(y, p_cb, p_tree, p_mlp, airport,
                                      l2_shrinkage: float = 0.0,
                                      ftol: float | None = None) -> dict[str, list]:
    """Per-airport constrained least squares: w >= 0, sum(w) = 1 (a true
    convex combination, unlike plain NNLS which can push an individual
    weight past 1 as long as the fit improves -- exactly what broke at
    LIRF/seed44). Optional `l2_shrinkage` adds a penalty toward equal
    weights (1/3 each), a second, independent safety net against a small or
    outlier-heavy local sample pushing the fit to a simplex CORNER (e.g.
    100% on one arm) that doesn't generalise, even though the simplex
    constraint alone already rules out the specific overshoot-past-1 failure
    mode seen at seed44.

    minimise  mean((X w - y)^2) + l2_shrinkage * sum((w - 1/3)^2)
    subject to  w >= 0, sum(w) = 1

    Mean (not sum) squared error keeps the data term's scale comparable
    across airports of very different row counts, so l2_shrinkage means the
    same thing at every airport rather than implicitly being stronger at
    small airports and weaker at large ones.

    The objective is additionally normalised by mean(y^2) for that airport
    BEFORE handing it to SLSQP -- found necessary by direct debugging, not a
    hypothetical: at LIRF (MSE scale ~1.1e6, since a chunk of its rows are
    unmatched with targets in the thousands of seconds), SLSQP's default
    finite-difference gradient step underflowed against that scale and it
    reported false convergence AT THE STARTING POINT (0.33/0.33/0.33) --
    `success=True` while `obj([0,0,1])` was 44% lower, a real, findable
    improvement it never looked for. Normalising the objective per airport
    fixes this without changing what's being optimised (scaling by a
    positive constant doesn't move the optimum) and was confirmed directly:
    same problem, unscaled -> false convergence at w0; scaled -> converges to
    [0, 0, 1], the expected all-MLP weight for LIRF.
    """
    # ftol=None keeps SLSQP's default (1e-6), which every shipped solve used. On the
    # mean(y^2)-scaled objective that tolerance stops early: 2026-09-26 grid check found
    # LEMD and EDDF left at the (1/3,1/3,1/3) start and LEMD 0.23 RMSE above its optimum
    # on the solve surface. Pass ftol (e.g. 1e-12) for a converged solve; the default
    # stays so that recorded weights of v25-v35 re-solve identically. AND: out of sample the
    # tight solve was NOT better (forward test +0.14, Jan+Jul +0.03 in-band) -- stopping near
    # 1/3 acts as shrinkage. Keep the default for shipped solves.
    weights = {}
    n_arms = 3
    w0 = np.full(n_arms, 1.0 / n_arms)
    bounds = [(0.0, 1.0)] * n_arms
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    for a in sorted(pd.unique(airport)):
        m = (airport == a)
        P = np.column_stack([p_cb[m], p_tree[m], p_mlp[m]])
        yv = y[m]
        scale = float(np.mean(yv ** 2))
        if scale <= 0 or not np.isfinite(scale):
            scale = 1.0

        def objective(w, P=P, yv=yv, scale=scale):
            resid = P @ w - yv
            mse = float(np.mean(resid ** 2))
            shrink = l2_shrinkage * float(np.sum((w - 1.0 / n_arms) ** 2))
            return (mse + shrink) / scale

        res = minimize(objective, w0, method="SLSQP", bounds=bounds,
                       constraints=constraints,
                       options={"maxiter": 300, **({"ftol": ftol} if ftol else {})})
        if not res.success:
            print("  WARNING: simplex solve did not converge for {} ({}) -- "
                  "falling back to equal weights".format(a, res.message))
            w = w0
        else:
            w = res.x
        weights[str(a)] = [float(x) for x in w]
    return weights


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--mlp-tag", default=None)
    ap.add_argument("--catboost-tag", default=None)
    ap.add_argument("--tree-tag", default=None)
    ap.add_argument("--mlp-seeds", default="")
    ap.add_argument("--catboost-seeds", default="")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--weight-split", default="forward_inner", choices=["forward_inner", "val"])
    ap.add_argument("--weight-method", default="nnls", choices=["nnls", "simplex"],
                    help="'nnls' is the original unconstrained-sum solve (measured "
                         "unstable at LIRF/seed44 -- see module docstring). "
                         "'simplex' constrains w>=0, sum(w)=1 per airport, "
                         "optionally with --l2-shrinkage toward equal weights.")
    ap.add_argument("--l2-shrinkage", type=float, default=0.0,
                    help="only used with --weight-method simplex. Penalty "
                         "strength pulling each airport's weights toward "
                         "(1/3, 1/3, 1/3); 0 = pure simplex constraint, no "
                         "extra shrinkage.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed,
                  extra=({"tag": args.tag} if args.tag else None))

    mlp_seeds = parse_seeds(args.mlp_seeds, args.seed)
    cb_seeds = parse_seeds(args.catboost_seeds, args.seed)
    print(f"[{ARM}] MLP seeds: {mlp_seeds}  CatBoost seeds: {cb_seeds}  tree seed: {args.seed}")

    Xtr, _, _ = D.load_part(args.split, "train", args.featureset, filter_target=True,
                            outlier_policy=args.outlier_policy)

    def load_three(wsplit):
        Xw_tr, _, _ = D.load_part(wsplit, "train", args.featureset, filter_target=True,
                                  outlier_policy=args.outlier_policy)
        Xw, yw, meta_w = D.load_part(wsplit, "val", args.featureset, filter_target=False)
        yw = np.asarray(yw, dtype="float64")
        p_cb = catboost_prediction(wsplit, "val", args.featureset, cb_seeds,
                                   args.outlier_policy, Xw, meta_w, Xw_tr, tag=args.catboost_tag)
        p_tree = tree_prediction(wsplit, "val", args.featureset, args.seed,
                                 args.outlier_policy, Xw, meta_w, Xw_tr, tag=args.tree_tag)
        p_mlp = mlp_prediction(wsplit, "val", args.featureset, mlp_seeds,
                               args.outlier_policy, Xw, meta_w, Xw_tr, tag=args.mlp_tag)
        ap_w = meta_w[AIRPORT_COL].astype("object").to_numpy()
        return yw, p_cb, p_tree, p_mlp, ap_w

    wsplit = "forward_inner" if args.weight_split == "forward_inner" else args.split
    yw, pcb_w, ptree_w, pmlp_w, apw = load_three(wsplit)
    if args.weight_method == "simplex":
        weights_per_airport = solve_weights_per_airport_simplex(
            yw, pcb_w, ptree_w, pmlp_w, apw, l2_shrinkage=args.l2_shrinkage)
    else:
        weights_per_airport = solve_weights_per_airport_nnls(yw, pcb_w, ptree_w, pmlp_w, apw)

    inner_blend = np.zeros(len(yw), dtype="float64")
    for a in pd.unique(apw):
        m = (apw == a)
        w_cb, w_tree, w_mlp = weights_per_airport[str(a)]
        inner_blend[m] = w_cb * pcb_w[m] + w_tree * ptree_w[m] + w_mlp * pmlp_w[m]
    inner_rmse_blend = rmse(yw, np.clip(inner_blend, config.PREDICTION_CLIP_MIN_SEC,
                                        config.PREDICTION_CLIP_MAX_SEC))

    print(f"[{ARM}] Solved weights (catboost, airport_operator_blend, mlp) on {wsplit}:")
    for k, v in weights_per_airport.items():
        print(f"  {k}: cb={v[0]:.4f} tree={v[1]:.4f} mlp={v[2]:.4f} sum={sum(v):.4f}")
    print(f"[{ARM}] {wsplit} blend RMSE: {inner_rmse_blend:.4f}")

    has_val = bool(config.SPLITS[args.split]["val"])
    val_rmse = None
    if has_val:
        Xva, yva, meta_va = D.load_part(args.split, "val", args.featureset, filter_target=False)
        yva = np.asarray(yva, dtype="float64")
        pcb_val = catboost_prediction(args.split, "val", args.featureset, cb_seeds,
                                      args.outlier_policy, Xva, meta_va, Xtr, tag=args.catboost_tag)
        ptree_val = tree_prediction(args.split, "val", args.featureset, args.seed,
                                    args.outlier_policy, Xva, meta_va, Xtr, tag=args.tree_tag)
        pmlp_val = mlp_prediction(args.split, "val", args.featureset, mlp_seeds,
                                  args.outlier_policy, Xva, meta_va, Xtr, tag=args.mlp_tag)
        ap_val = meta_va[AIRPORT_COL].astype("object").to_numpy()
        p_val_blend = np.zeros(len(yva), dtype="float64")
        for a in pd.unique(ap_val):
            m = (ap_val == a)
            w_cb, w_tree, w_mlp = weights_per_airport.get(str(a), [0.0, 0.0, 1.0])
            p_val_blend[m] = w_cb * pcb_val[m] + w_tree * ptree_val[m] + w_mlp * pmlp_val[m]
        p_val_blend = np.clip(p_val_blend, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
        val_rmse = rmse(yva, p_val_blend)
        clip = lambda p: np.clip(p, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
        print(f"[{ARM}] Forward Validation Blend RMSE: {val_rmse:.4f}  "
              f"[catboost alone {rmse(yva, clip(pcb_val)):.3f} | "
              f"tree alone {rmse(yva, clip(ptree_val)):.3f} | "
              f"mlp alone {rmse(yva, clip(pmlp_val)):.3f}]  "
              f"(inner-vs-val gap: {inner_rmse_blend - val_rmse:+.3f})")

    sel = declare_selection_set(f"{ARM} per-airport NNLS weights",
                                selected_on=args.weight_split, value=weights_per_airport)

    meta = {
        "arm": ARM, "split": args.split, "featureset": args.featureset,
        "outlier_policy": args.outlier_policy, "seed": args.seed,
        "base_arms": ["catboost_antigravity", TREE_ARM, "mlp_antigravity"],
        "weights_per_airport": weights_per_airport,
        "weight_order": ["catboost", "tree", "mlp"],
        "weight_split": args.weight_split,
        "weight_method": args.weight_method,
        "l2_shrinkage": args.l2_shrinkage,
        "inner_rmse_blend": inner_rmse_blend,
        "mlp_seeds": mlp_seeds, "catboost_seeds": cb_seeds,
        "catboost_tag": args.catboost_tag, "mlp_tag": args.mlp_tag, "tree_tag": args.tree_tag,
        "val_rmse": float(val_rmse) if val_rmse is not None else None,
        "selection": dict(sel),
        "train_pool_ids_sha": D.pool_fingerprint(Xtr.index),
    }
    write_json_guarded(out / "model.json", meta, force=args.force)
    print(f"[{ARM}] saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
