#!/usr/bin/env python3
"""Stage 6: choose the PREDICTION CAP on val, per outlier policy.

Why this stage exists
---------------------
`config.PREDICTION_CLIP_MAX_SEC` was set once, at p99.99 of the training target
(10,712 s), and treated as a constant. It is not a constant -- it is a
hyperparameter, and one that interacts directly with the outlier policy: a model
trained under `winsor_21600` learns to emit values a 10,712 s cap then truncates.
Measured on the pre-existing runs: `winsor_21600` scored val RMSE 311.01 RAW and
324.92 CLIPPED, so the mismatched cap cost ~14 s of RMSE by itself -- larger than
the gap between any two arms in the build.

What it is honest about
-----------------------
This cap is chosen on a couple of dozen rows. On forward/val only a handful of
rows sit above any candidate cap, and RMSE is a squared-error pool, so the whole
sweep is decided by the tail's luck. That does not make selecting it on val
illegitimate -- val is the selection surface -- but it does make it HIGH VARIANCE,
so this stage reports, for every cap: the number of val rows above it, the
arithmetic hard floor it implies, and the in-band (y <= 7200 s) RMSE, which the
cap cannot touch at all. A cap whose gain does not survive the repeat-seed sweep
is noise dressed as a lever, and `--check-seeds` re-runs the sweep on the other
seeds' models to say so.

Selection surface: VAL. Never test.
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
from common.bootstrap import declare_selection_set, run_dir, write_json_guarded
from common.metrics import rmse

ROOT = Path(__file__).resolve().parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)

# Cap grid, seconds. `None` means "no upper cap at all", which is a real
# candidate: nothing forces a cap to exist, and its absence is the only setting
# under which the arithmetic hard floor is zero.
# The configured cap is always in the grid: this stage compares against "what we
# currently do", and after the first sweep moved config.PREDICTION_CLIP_MAX_SEC to
# 86,400 s the hardcoded grid no longer contained it, so the comparison row could
# not be found and the stage crashed. A sweep must include its own starting point.
CAP_GRID = sorted({7200, 9000, 10712, 12600, 14400, 18000, 21600, 28800, 43200,
                   int(config.PREDICTION_CLIP_MAX_SEC)}) + [None]


def raw_predictions(arm: str, split: str, part: str, featureset: str,
                    outlier_policy: str, seed: int):
    """UNclipped predictions for one arm on one part, plus (y, Xtr-pool y)."""
    proto = config.protocol_key(split, featureset, outlier_policy)
    d = run_dir(config.ARTIFACTS / "runs", arm, protocol=proto, seed=seed, create=False)
    if not d.exists():
        raise SystemExit("no run at {} -- train the arm first".format(d))

    Xtr, ytr, _ = D.load_part(split, "train", featureset, filter_target=True,
                              outlier_policy=outlier_policy)
    X, y, meta = D.load_part(split, part, featureset, filter_target=False)
    X = D.align_columns(Xtr, X)

    if arm == "lightgbm":
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(d / "model.txt"))
        best = json.loads((d / "model.json").read_text()).get("best_iteration")
        pred = booster.predict(X, num_iteration=best)
    elif arm == "ridge":
        import pickle
        with open(d / "model.pkl", "rb") as fh:
            model = pickle.load(fh)
        pred = model.predict(X)
    else:
        raise SystemExit(
            "arm {!r} is not supported by this stage. It needs UNCLIPPED "
            "predictions, and only this build's own arms expose a load path that "
            "returns them; the *_antigravity arms clip inside their evaluate.py "
            "and are not this build's to edit.".format(arm))
    return np.asarray(pred, dtype="float64"), np.asarray(y, dtype="float64"), ytr, meta


# The three bands the pooled RMSE decomposes into. The split at 80,000 s is not
# arbitrary: exactly 15 DEP rows in the whole 2.085M-row training set sit in
# [80000, 95000), every one of them at 86,400 s + a normal taxi time, with the
# off-block timestamp one calendar day before the takeoff timestamp. Three of
# them land in forward/test, where they carry 52% of the pooled squared error on
# their own, and one lands in forward/val carrying 14%. Nothing in the feature set
# distinguishes them beforehand, so no arm can win them on signal.
#
# They are NOT, however, policy-invariant -- an earlier version of this comment
# claimed they were and was wrong. Raising the cap lets predictions climb toward
# that single 87,543 s val row, moving the band's error from 83,933 to 68,962 s:
# ~2.3e9 of squared error out of ONE row, comparable to the entire in-band cost of
# the same change. So a policy or cap chosen on the pooled number is partly chosen
# on one date error. `decidable_sse` below (in_band + mid_tail) is the ranking that
# excludes it, and the choice is only trustworthy where the two agree -- which they
# do here: see artifacts/clip/policy_cap_decision.json.
BANDS = [("in_band", 0.0, 7200.0),
         ("mid_tail", 7200.0, 80000.0),
         ("artifact", 80000.0, np.inf)]


def band_sse(y, p) -> dict:
    """Squared-error mass per band, so the policy decision is not made on 3 rows."""
    out = {}
    e2 = (y - p) ** 2
    for name, lo, hi in BANDS:
        m = (y > lo) & (y <= hi) if np.isfinite(hi) else (y > lo)
        out[name] = {"n": int(m.sum()), "sse": float(e2[m].sum()),
                     "rmse": float(np.sqrt(e2[m].mean())) if m.any() else None}
    out["total_sse"] = float(e2.sum())
    out["decidable_sse"] = out["in_band"]["sse"] + out["mid_tail"]["sse"]
    out["decidable_rmse"] = float(np.sqrt(
        out["decidable_sse"] / max(1, out["in_band"]["n"] + out["mid_tail"]["n"])))
    return out


def sweep_caps(y, pred, floor=config.PREDICTION_CLIP_MIN_SEC) -> "list[dict]":
    rows = []
    for cap in CAP_GRID:
        hi = np.inf if cap is None else float(cap)
        p = np.clip(pred, floor, hi)
        in_band = y <= 7200
        rows.append({
            "cap_sec": cap,
            "val_rmse": rmse(y, p),
            "val_rmse_in_band": rmse(y[in_band], p[in_band]),
            "n_true_above_cap": int(np.sum(y > hi)) if cap is not None else 0,
            "n_pred_clipped": int(np.sum(pred > hi)) if cap is not None else 0,
            # the arithmetic bound this cap imposes on ANY model (design-gate B3)
            "hard_floor_rmse": (0.0 if cap is None else
                                float(np.sqrt(np.mean((y - np.minimum(y, hi)) ** 2)))),
            "bands": band_sse(y, p),
        })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="lightgbm", choices=["lightgbm", "ridge"])
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--check-seeds", default="",
                    help="comma-separated extra seeds whose models exist; the "
                         "chosen cap must win on those too or it is seed noise")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if args.split in config.SELECTION_FORBIDDEN_SPLITS:
        raise SystemExit(
            "split {!r} is not a selection surface (config.SELECTION_FORBIDDEN_"
            "SPLITS); a cap chosen on it would be chosen on the wrong data."
            .format(args.split))

    pred, y, ytr, _meta = raw_predictions(args.arm, args.split, "val",
                                          args.featureset, args.outlier_policy,
                                          args.seed)
    rows = sweep_caps(y, pred)
    best = min(rows, key=lambda r: r["val_rmse"])
    current = next(r for r in rows if r["cap_sec"] == config.PREDICTION_CLIP_MAX_SEC)

    print("\n== cap sweep: {} / {} / {} / {} / seed {} (val, n={}) ==".format(
        args.arm, args.split, args.featureset, args.outlier_policy, args.seed, y.size))
    print("  {:>9s} {:>10s} {:>11s} {:>9s} {:>9s} {:>11s}".format(
        "cap", "val RMSE", "in-band", "y>cap", "pred>cap", "hard floor"))
    for r in rows:
        mark = " <-- best" if r is best else (
            " (current)" if r["cap_sec"] == config.PREDICTION_CLIP_MAX_SEC else "")
        print("  {:>9s} {:10.2f} {:11.2f} {:9d} {:9d} {:11.2f}{}".format(
            "none" if r["cap_sec"] is None else str(r["cap_sec"]),
            r["val_rmse"], r["val_rmse_in_band"], r["n_true_above_cap"],
            r["n_pred_clipped"], r["hard_floor_rmse"], mark))
    print("  best cap {} beats the current {} s cap by {:.2f} s of val RMSE, "
          "decided by {} val rows above the current cap.".format(
              best["cap_sec"], config.PREDICTION_CLIP_MAX_SEC,
              current["val_rmse"] - best["val_rmse"], current["n_true_above_cap"]))

    # ---- does the choice survive other seeds? ----
    per_seed = {}
    seeds = [s for s in (args.check_seeds.split(",") if args.check_seeds else []) if s]
    for s in seeds:
        try:
            p2, y2, _, _ = raw_predictions(args.arm, args.split, "val",
                                           args.featureset, args.outlier_policy, int(s))
        except SystemExit as exc:
            print("  (seed {}: {})".format(s, exc))
            continue
        r2 = sweep_caps(y2, p2)
        b2 = min(r2, key=lambda r: r["val_rmse"])
        per_seed[s] = {"best_cap": b2["cap_sec"], "rows": r2}
        print("  seed {}: best cap {}".format(s, b2["cap_sec"]))
    agrees = (not per_seed) or all(v["best_cap"] == best["cap_sec"]
                                   for v in per_seed.values())
    if per_seed and not agrees:
        print("  WARNING: the best cap is NOT stable across seeds -- treat the "
              "gain as noise and keep the current cap.")

    sel = declare_selection_set("prediction cap (upper clip)", selected_on="val",
                                value=best["cap_sec"])
    out_dir = config.ARTIFACTS / "clip"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_guarded(
        out_dir / "clip__{}__seed{}__{}.json".format(
            config.protocol_key(args.split, args.featureset, args.outlier_policy),
            args.seed, args.arm),
        {"arm": args.arm, "split": args.split, "featureset": args.featureset,
         "seed": args.seed, "outlier_policy": args.outlier_policy,
         "selection": dict(sel), "grid": rows, "best": best,
         "current_cap_sec": config.PREDICTION_CLIP_MAX_SEC,
         "current_row": current,
         "gain_vs_current_sec": current["val_rmse"] - best["val_rmse"],
         "decided_by_n_val_rows_above_current_cap": current["n_true_above_cap"],
         "seed_agreement": {"checked": list(per_seed), "agrees": bool(agrees),
                            "per_seed_best": {k: v["best_cap"]
                                              for k, v in per_seed.items()}},
         "caveat": ("this cap is decided by a few dozen extreme val rows; the "
                    "in-band column is the part of the metric it cannot touch")},
        force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
