"""RMSE and the breakdowns this competition's scored-set composition demands.

The leaderboard is a single pooled RMSE over a set that is 71% January (10
airports) and 29% July (3 airports). A pooled number on a differently-composed
local split is therefore NOT an estimate of the leaderboard -- reweighting is,
which is what `composition_reweighted_rmse` computes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    if y_true.shape != y_pred.shape:
        raise ValueError("shape mismatch: {} vs {}".format(y_true.shape, y_pred.shape))
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not m.any():
        raise ValueError("no finite (y_true, y_pred) pairs to score")
    if m.sum() != y_true.size:
        raise ValueError(
            "{} of {} predictions are non-finite. The leaderboard scores every row; "
            "a NaN prediction is a scoring failure, not a row to skip.".format(
                int(y_true.size - m.sum()), int(y_true.size))
        )
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def rmse_by(y_true, y_pred, keys) -> "pd.DataFrame":
    """Per-group RMSE with support counts -- the subgroup view a pooled RMSE hides."""
    df = pd.DataFrame({"y": np.asarray(y_true, dtype="float64"),
                       "p": np.asarray(y_pred, dtype="float64"),
                       "k": np.asarray(keys)})
    g = df.groupby("k", observed=True).apply(
        lambda d: pd.Series({"n": len(d),
                             "rmse": float(np.sqrt(np.mean((d.y - d.p) ** 2))),
                             "mean_true": float(d.y.mean())}),
        include_groups=False,
    )
    return g.sort_values("n", ascending=False)


def composition_reweighted_rmse(y_true, y_pred, keys, target_weights) -> float:
    """Pooled RMSE re-weighted so each group carries its leaderboard share.

    RMSE pools in squared space, so the reweighting is over mean squared error:
        sqrt( sum_g w_g * MSE_g ),  sum_g w_g = 1
    Groups present locally but absent from `target_weights` get weight 0; groups
    in `target_weights` with no local rows raise, because silently dropping them
    would understate the estimate.
    """
    df = pd.DataFrame({"y": np.asarray(y_true, dtype="float64"),
                       "p": np.asarray(y_pred, dtype="float64"),
                       "k": np.asarray(keys)})
    mse = df.groupby("k", observed=True).apply(
        lambda d: float(np.mean((d.y - d.p) ** 2)), include_groups=False)
    missing = [k for k in target_weights if k not in mse.index]
    if missing:
        raise ValueError(
            "target_weights names group(s) with no local rows to score: {}. "
            "Reweighting would silently drop them.".format(missing)
        )
    total = float(sum(target_weights.values()))
    if total <= 0:
        raise ValueError("target_weights sum to {}".format(total))
    acc = sum((w / total) * mse[k] for k, w in target_weights.items())
    return float(np.sqrt(acc))


def regression_trivial_baselines(y_train, y_eval) -> dict:
    """Constant-predictor floors, fitted on TRAIN and scored on EVAL.

    Fitting the constant on the evaluation set itself would be an oracle floor,
    not a trivial baseline (evaluation-and-baselines.md -> test discipline).
    """
    y_train = np.asarray(y_train, dtype="float64")
    y_eval = np.asarray(y_eval, dtype="float64")
    out = {}
    for name, const in (("train_mean", float(np.mean(y_train))),
                        ("train_median", float(np.median(y_train)))):
        out[name] = {"constant": const,
                     "rmse": rmse(y_eval, np.full_like(y_eval, const))}
    return out
