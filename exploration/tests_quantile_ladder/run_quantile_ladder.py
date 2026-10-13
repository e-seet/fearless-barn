#!/usr/bin/env python3
"""Experiment: distributional regression via a quantile ladder, numerically
integrated to recover E[y|x], versus plain L2 point regression.

Why this construction and not "LightGBM objective=quantile, use the median."
That would just be a pinball-loss point estimate -- pinball is L1-family,
STRICTLY MORE tail-de-weighting than Huber, which TAXI_OUT_DRIVERS.md sec.2i
already closed at +82.2s. Testing the median would re-run an axis this
project has already closed three times (Huber, log1p, Poisson -- sec.2ac)
under a new name. The genuinely distinct construction is: fit the full
conditional distribution (a ladder of quantiles) and recover the MEAN by
integrating the quantile function, Mean = integral_0^1 Q(u) du. RMSE is
minimized by the conditional mean, not the median, so if this construction
adds anything over plain L2 it has to come from a better-shaped estimate of
the mean under a heavy-tailed conditional distribution -- not from dodging
the tail the way every previously-closed variant did.

The known risk, stated before running: naive trapezoidal integration of a
finite quantile ladder implicitly treats the untracked mass beyond the
highest fitted quantile as constant at that quantile's value -- i.e. it CAPS
the tail, exactly the failure mode config.OUTLIER_POLICIES and the
prediction-cap probe (submissions.csv v18, v27) already showed is
catastrophic for this metric (a hard cap at 7200s cost +185 RMSE on the
actual board). To give this construction a fair test rather than a rigged
loss, the tail beyond the last fitted quantile is NOT truncated -- it is
extrapolated with an exponential-tail (per-row Value-at-Risk/Expected-
Shortfall style) fit from the top two quantiles, the standard way to
recover a mean from a truncated quantile ladder over a heavy right tail.

Design: one seed first (42), matching this project's "cheap single-seed
check before committing to a multi-seed retrain" convention (used for the
weather feature, tests_weather_metar/RESULT.md) -- 14 quantile regressors is
already ~14x the compute of one pooled model. Compare against the SAME
lightgbm l2ctl recipe (not the shipped blend) as the fair like-for-like
baseline, matching how Huber/Poisson/log1p were each screened before
deciding whether a multi-seed or blend-level test was warranted.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
sys.path.insert(0, str(ROOT))

import config
from common import dataset as D
from common import evaluate as E
from common import features as F
from common.bootstrap import run_dir, seed_everything

SPLIT, FEATURESET, POLICY = "forward", "with_aobt", "keep_all"
SEED = 42
CLIP_LO, CLIP_HI = config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC

BASE_PARAMS = {
    "learning_rate": 0.05, "num_leaves": 255, "min_data_in_leaf": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "max_cat_threshold": 64, "cat_smooth": 20.0,
    "verbosity": -1, "num_threads": 0,
}
NUM_BOOST_ROUND, PATIENCE = 5000, 200

# Fine near both tails -- the mean of a heavy-tailed distribution is highly
# sensitive to the upper quantiles specifically, which is exactly the part a
# coarse ladder (e.g. decile-only) would get wrong.
QUANTILES = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80,
             0.90, 0.95, 0.99, 0.999]


def clip(p):
    return np.clip(p, CLIP_LO, CLIP_HI)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, dtype="float64") - p) ** 2)))


def train_quantile(lgb, Xtr, ytr, Xva, yva, cats, seed, alpha):
    params = dict(BASE_PARAMS)
    params.update({"objective": "quantile", "metric": "quantile", "alpha": alpha,
                   "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed,
                   "device_type": "cpu"})
    dtr = lgb.Dataset(Xtr, label=ytr, categorical_feature=cats, free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva, categorical_feature=cats, reference=dtr,
                      free_raw_data=False)
    booster = lgb.train(params, dtr, num_boost_round=NUM_BOOST_ROUND,
                        valid_sets=[dva], valid_names=["val"],
                        callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    best = booster.best_iteration or NUM_BOOST_ROUND
    return booster, best, bool(best >= NUM_BOOST_ROUND)


def integrate_mean(Q):
    """Q: (n_rows, n_quantiles) predicted values at QUANTILES, ASCENDING order
    already enforced (monotone per row). Returns E[y] per row.

    Trapezoid between fitted levels; [0, q_1] assumed ~constant at the
    lowest fitted quantile (little mass/variance down there -- the target's
    lower reaches are not heavy-tailed, only the upper ones are); [q_K, 1)
    extrapolated with a per-row exponential tail fit from the last two
    quantiles (the standard finite-ladder-heavy-tail correction): if
    1-F(v) ~ (1-q) shape, solving for scale b from (q_{K-1}, v_{K-1}) and
    (q_K, v_K) gives b = (v_K - v_{K-1}) / ln((1-q_{K-1})/(1-q_K)), and the
    tail's contribution to the mean is (1-q_K) * (v_K + b) (mean of an
    exponential with scale b, shifted to start at v_K).
    """
    qs = np.asarray(QUANTILES, dtype="float64")
    n, k = Q.shape
    mean = np.zeros(n, dtype="float64")

    # low end: [0, q_1] contributes ~q_1 * v_1
    mean += qs[0] * Q[:, 0]

    # trapezoid segments between consecutive fitted quantiles
    for i in range(k - 1):
        dq = qs[i + 1] - qs[i]
        mean += dq * (Q[:, i] + Q[:, i + 1]) / 2.0

    # upper tail beyond the last quantile: exponential extrapolation
    v_km1, v_k = Q[:, -2], Q[:, -1]
    q_km1, q_k = qs[-2], qs[-1]
    gap = np.maximum(v_k - v_km1, 1e-6)              # monotonicity already enforced
    log_ratio = np.log((1.0 - q_km1) / (1.0 - q_k))  # constant, > 0
    b = gap / log_ratio
    mean += (1.0 - q_k) * (v_k + b)

    return mean


def main() -> int:
    import lightgbm as lgb
    t0 = time.time()

    Xtr_raw, ytr, pool_meta = D.load_part(SPLIT, "train", FEATURESET,
                                          filter_target=True, outlier_policy=POLICY)
    Xva_raw, yva, meta_va = D.load_part(SPLIT, "val", FEATURESET, filter_target=False)
    Xte_raw, yte, meta_te = D.load_part(SPLIT, "test", FEATURESET, filter_target=False)
    Xva = D.align_columns(Xtr_raw, Xva_raw)
    Xte = D.align_columns(Xtr_raw, Xte_raw)
    cats = [c for c in F.CATEGORICAL if c in Xtr_raw.columns]
    ytr = np.asarray(ytr, dtype="float64")
    yva = np.asarray(yva, dtype="float64")
    yte = np.asarray(yte, dtype="float64")
    print(f"loaded: train={len(Xtr_raw)} val={len(Xva)} test={len(Xte)}  "
          f"({time.time()-t0:.0f}s)")

    seed_everything(SEED)

    # ---- L2 control (identical recipe to the registered l2ctl arm) ----
    params_l2 = dict(BASE_PARAMS)
    params_l2.update({"objective": "l2", "metric": "rmse", "seed": SEED,
                      "bagging_seed": SEED, "feature_fraction_seed": SEED,
                      "device_type": "cpu"})
    dtr = lgb.Dataset(Xtr_raw, label=ytr, categorical_feature=cats, free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva, categorical_feature=cats, reference=dtr,
                      free_raw_data=False)
    b_l2 = lgb.train(params_l2, dtr, num_boost_round=NUM_BOOST_ROUND,
                     valid_sets=[dva], valid_names=["val"],
                     callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    best_l2 = b_l2.best_iteration or NUM_BOOST_ROUND
    p_val_l2 = clip(b_l2.predict(Xva, num_iteration=best_l2))
    p_test_l2 = clip(b_l2.predict(Xte, num_iteration=best_l2))
    print(f"L2 control: val RMSE {rmse(yva, p_val_l2):.2f}  test RMSE {rmse(yte, p_test_l2):.2f}"
          f"  (best_iter {best_l2})  [{time.time()-t0:.0f}s]")

    # ---- quantile ladder ----
    Qva = np.empty((len(yva), len(QUANTILES)), dtype="float64")
    Qte = np.empty((len(yte), len(QUANTILES)), dtype="float64")
    hit_caps = []
    for i, alpha in enumerate(QUANTILES):
        booster, best, hit_cap = train_quantile(lgb, Xtr_raw, ytr, Xva, yva, cats, SEED, alpha)
        hit_caps.append(hit_cap)
        Qva[:, i] = booster.predict(Xva, num_iteration=best)
        Qte[:, i] = booster.predict(Xte, num_iteration=best)
        print(f"  q={alpha:6.3f}  best_iter={best:5d}  hit_cap={hit_cap}  "
              f"[{time.time()-t0:.0f}s]")

    # enforce monotonicity per row (Chernozhukov rearrangement: sort, don't clip)
    Qva_mono = np.sort(Qva, axis=1)
    Qte_mono = np.sort(Qte, axis=1)
    n_crossed_va = int(np.any(Qva != Qva_mono, axis=1).sum())
    n_crossed_te = int(np.any(Qte != Qte_mono, axis=1).sum())
    print(f"quantile crossing: val {n_crossed_va}/{len(yva)} rows reordered, "
          f"test {n_crossed_te}/{len(yte)} rows reordered")

    p_val_qmean = clip(integrate_mean(Qva_mono))
    p_test_qmean = clip(integrate_mean(Qte_mono))

    rv_l2 = E.evaluate(yva, p_val_l2, meta_va, ytr, part="val", arm="lightgbm",
                       split=SPLIT, featureset=FEATURESET, seed=SEED)
    rt_l2 = E.evaluate(yte, p_test_l2, meta_te, ytr, part="test", arm="lightgbm",
                       split=SPLIT, featureset=FEATURESET, seed=SEED)
    rv_q = E.evaluate(yva, p_val_qmean, meta_va, ytr, part="val", arm="qladder",
                      split=SPLIT, featureset=FEATURESET, seed=SEED)
    rt_q = E.evaluate(yte, p_test_qmean, meta_te, ytr, part="test", arm="qladder",
                      split=SPLIT, featureset=FEATURESET, seed=SEED)

    proto = config.protocol_key(SPLIT, FEATURESET, POLICY)
    d = run_dir(config.ARTIFACTS / "runs", "qladder", protocol=proto, seed=SEED,
               extra={"tag": "meanintegrated"})
    (d / "model.json").write_text(json.dumps({
        "arm": "qladder", "split": SPLIT, "featureset": FEATURESET, "seed": SEED,
        "quantiles": QUANTILES, "hit_caps": hit_caps,
        "note": "tests_quantile_ladder/run_quantile_ladder.py -- diagnostic only",
    }, indent=2))
    (d / "eval_val.json").write_text(json.dumps(rv_q, indent=2))
    (d / "eval_test.json").write_text(json.dumps(rt_q, indent=2))

    lines = [
        "quantile-ladder-to-mean vs L2 point regression  "
        "(forward / with_aobt / keep_all, seed 42)",
        "=" * 70,
        f"{'metric':22s} {'L2 control':>12s} {'q-ladder mean':>14s} {'delta':>9s}",
        f"{'val RMSE':22s} {rv_l2['rmse']:12.2f} {rv_q['rmse']:14.2f} "
        f"{rv_q['rmse']-rv_l2['rmse']:+9.2f}",
        f"{'val in-band':22s} {rv_l2['rmse_in_band']:12.2f} {rv_q['rmse_in_band']:14.2f} "
        f"{rv_q['rmse_in_band']-rv_l2['rmse_in_band']:+9.2f}",
        f"{'test RMSE':22s} {rt_l2['rmse']:12.2f} {rt_q['rmse']:14.2f} "
        f"{rt_q['rmse']-rt_l2['rmse']:+9.2f}",
        f"{'test in-band':22s} {rt_l2['rmse_in_band']:12.2f} {rt_q['rmse_in_band']:14.2f} "
        f"{rt_q['rmse_in_band']-rt_l2['rmse_in_band']:+9.2f}",
        f"{'test pred mean':22s} {np.mean(p_test_l2):12.2f} {np.mean(p_test_qmean):14.2f} "
        f"{'(true mean='+format(float(np.mean(yte)), '.2f')+')'}",
        "",
        f"quantile crossing (pre-sort): val {n_crossed_va}/{len(yva)}  test {n_crossed_te}/{len(yte)}",
        f"any quantile regressor hit the round cap: {any(hit_caps)}",
    ]
    txt = "\n".join(lines)
    (HERE / "summary.txt").write_text(txt + "\n", encoding="utf-8")
    print("\n" + txt)
    print(f"\n[{time.time()-t0:.0f}s] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
