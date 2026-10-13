"""Shared evaluation: one implementation of the metric block every arm reports.

Three arms computing their own RMSE, their own baselines and their own subgroup
tables is how two of them end up quietly measuring different things. This module
is the single implementation; a pipeline's evaluate.py only supplies predictions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common.metrics import (composition_reweighted_rmse, regression_trivial_baselines,
                            rmse, rmse_by)


def leaderboard_weights() -> dict:
    """Airport share of the scored set, read from the audit report (not hardcoded)."""
    import json
    p = config.AUDIT_DIR / "audit_report.json"
    if not p.exists():
        raise FileNotFoundError(
            "{} missing -- run stage1_audit.py first; the leaderboard composition "
            "is measured from ranking.parquet, never assumed.".format(p))
    return json.loads(p.read_text())["leaderboard_airport_weights"]


def hard_floor_rmse(y_true, *, cap: float = None) -> float:
    """The RMSE no model can beat, given predictions are clipped at `cap`.

    Design-gate finding B3. The plan originally read the disagreement between the
    two off-block sources (RMS 255-557 s per airport) as a "label-noise floor".
    That is not a floor: the disagreement is largely SYSTEMATIC, hence learnable.
    A per-stand_runway median correction fitted on Jan-Aug and scored on Sep-Oct
    takes the raw proxy from 396.25 to 354.02 -- 42 s below the claimed floor. So
    Var(proxy - label) bounds nothing; only the unpredictable component does.

    What IS a hard bound is the prediction cap: with predictions clipped at
    `cap`, every row above it contributes at least (y - cap)^2 no matter what the
    model does. That is a real, per-part, arithmetic floor -- and it differs by
    ~42 s between forward/val and forward/test purely from tail composition, so
    a train/test gap must be read against it rather than against a shared number.
    """
    cap = config.PREDICTION_CLIP_MAX_SEC if cap is None else cap
    y = np.asarray(y_true, dtype="float64")
    return float(np.sqrt(np.mean((y - np.minimum(y, cap)) ** 2)))


def evaluate(y_true, y_pred, meta, y_train, *, part: str, arm: str,
             split: str, featureset: str, seed: int, extra=None) -> dict:
    """Full metric block for one arm on one evaluation part."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")

    res = {
        "arm": arm, "part": part, "split": split, "featureset": featureset,
        "seed": seed, "n": int(y_true.size), "metric": config.METRIC,
        "higher_is_better": config.HIGHER_IS_BETTER,
        "rmse": rmse(y_true, y_pred),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "bias_mean_signed_error": float(np.mean(y_pred - y_true)),
        "trivial_baselines": regression_trivial_baselines(y_train, y_true),
        # design-gate B3: the arithmetic floor implied by the prediction cap, on
        # THIS part. Per-part because tail composition differs between parts.
        "hard_floor_rmse_at_cap": hard_floor_rmse(y_true),
        "prediction_cap_sec": config.PREDICTION_CLIP_MAX_SEC,
        "n_above_cap": int(np.sum(y_true > config.PREDICTION_CLIP_MAX_SEC)),
    }

    # design-gate S2: the pooled RMSE is dominated by a few dozen extreme rows,
    # so early stopping / alpha selection / arm ranking driven by it alone are
    # driven by those rows' luck. The in-band figure is the low-variance
    # companion -- reported beside the headline, never instead of it.
    in_band = y_true <= 7200
    res["n_in_band"] = int(in_band.sum())
    res["rmse_in_band"] = rmse(y_true[in_band], y_pred[in_band]) if in_band.any() else None

    by_ap = rmse_by(y_true, y_pred, meta["ADEP_mvt"].to_numpy())
    res["rmse_by_airport"] = by_ap.to_dict(orient="index")
    if "month" in meta.columns:
        res["rmse_by_month"] = rmse_by(
            y_true, y_pred, meta["month"].to_numpy()).to_dict(orient="index")

    # The pooled local RMSE is NOT a leaderboard estimate: the scored set has a
    # different airport mix. Reweight to the measured composition.
    try:
        w = {k: v for k, v in leaderboard_weights().items()
             if k in set(meta["ADEP_mvt"].dropna().unique())}
        res["leaderboard_weighted_rmse"] = composition_reweighted_rmse(
            y_true, y_pred, meta["ADEP_mvt"].to_numpy(), w)
        res["leaderboard_weights_used"] = w
    except Exception as exc:                       # noqa: BLE001 - reported, not raised
        res["leaderboard_weighted_rmse"] = None
        res["leaderboard_weighted_rmse_error"] = str(exc)

    # RMSE is squared-error pooled, so the worst rows dominate it. Report how much.
    err2 = (y_true - y_pred) ** 2
    order = np.argsort(-err2)
    for frac in (0.001, 0.01, 0.05):
        k = max(1, int(frac * err2.size))
        res["sse_share_worst_{:g}pct".format(frac * 100)] = float(
            err2[order[:k]].sum() / err2.sum())

    if extra:
        res.update(extra)
    return res


def print_result(res: dict) -> None:
    tb = res["trivial_baselines"]
    print("\n== {} / {} / {} / seed {} -- {} ==".format(
        res["arm"], res["split"], res["featureset"], res["seed"], res["part"]))
    print("  n            : {}".format(res["n"]))
    print("  RMSE         : {:.2f} s".format(res["rmse"]))
    print("  MAE          : {:.2f} s".format(res["mae"]))
    print("  mean signed  : {:+.2f} s".format(res["bias_mean_signed_error"]))
    print("  in-band RMSE : {} s  (n={}, y<=7200s; low-variance companion)".format(
        "{:.2f}".format(res["rmse_in_band"]) if res.get("rmse_in_band") else "n/a",
        res.get("n_in_band")))
    print("  hard floor   : {:.2f} s  (arithmetic, from the {} s prediction cap; "
          "{} rows above it)".format(res["hard_floor_rmse_at_cap"],
                                     res["prediction_cap_sec"], res["n_above_cap"]))
    print("  trivial floors: train_mean {:.2f} | train_median {:.2f}".format(
        tb["train_mean"]["rmse"], tb["train_median"]["rmse"]))
    if res.get("leaderboard_weighted_rmse") is not None:
        print("  LB-weighted  : {:.2f} s  (reweighted to the scored airport mix)".format(
            res["leaderboard_weighted_rmse"]))
    print("  SSE share of worst 0.1% / 1% of rows: {:.1%} / {:.1%}".format(
        res["sse_share_worst_0.1pct"], res["sse_share_worst_1pct"]))
    ap = pd.DataFrame(res["rmse_by_airport"]).T.sort_values("rmse", ascending=False)
    print("  per-airport RMSE (worst first):")
    for k, r in ap.iterrows():
        print("    {:6s} n={:7.0f} rmse={:8.2f} mean_true={:8.1f}".format(
            str(k), r["n"], r["rmse"], r["mean_true"]))
