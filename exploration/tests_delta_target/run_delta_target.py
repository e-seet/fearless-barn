#!/usr/bin/env python3
"""Experiment: does reformulating the TARGET (not just having it as a feature)
as a delta/residual on top of `delta_aobt_sec` help the tree side?

Context (see solution/TAXI_OUT_DRIVERS.md SS2j, SS2q): `delta_aobt_sec` =
`MVT_TIME_UTC_mvt - AOBT_3_flt` is already the #1-importance input feature in
every model checked this session. A community note in conversation.md (S4,
"Piyush Patil") claims a top team's single biggest lever was "anchoring the
TARGET on takeoff minus the NM off-block time" -- i.e. training on
`Delta = y - delta_aobt_sec` and inverting at predict time, not just handing
the model the raw difference as one more column. SS2j's "closed on the
argument" reasoning doesn't transfer cleanly here: that argument was about a
training-target AGGREGATE of an already-embedded categorical (`stand_runway`
floor), whereas this anchor is a per-row OBSERVED quantity, and SS2u already
found a case in this exact repo where reformulating what the model sees
(target statistic vs raw categorical) bought a real 2.6s from representation
alone -- so "the model can already represent this" is not a safe assumption
to reuse without checking.

Design
------
* Two variants, same data/params/seed, only the label differs:
  - `raw`   : plain per-airport LightGBM on y (control, byte-for-byte the
              existing per_airport arm's shape).
  - `delta` : trained ONLY on rows with a valid AOBT (`has_aobt==1`) using
              label `y - delta_aobt_sec`; predictions inverted back
              (`pred + delta_aobt_sec`) on matched rows. Rows without a valid
              AOBT (delta is undefined there -- the AOBT-unmatched population,
              SS2q, already closed as a separate lever) fall back to the
              CONTROL model's raw-target prediction, byte-identical, so the
              pooled comparison isolates the reformulation's effect on the
              98.5% of rows where it's even defined and doesn't let the
              already-closed unmatched population confound the read.
* Single seed (42): `per_airport`'s own seed range is 0.02s (PIPELINE_REVIEW.md),
  so a one-seed tree read is interpretable here, unlike an MLP read.
* `common/features.py` untouched -- `delta_aobt_sec`/`has_aobt` already exist
  under featureset="with_aobt". `config.feature_code_fingerprint()` unaffected.
* Run on BOTH splits: `forward` (secondary read, Nov-Dec test) and `seasonal`
  (PRE-REGISTERED DECISIVE surface -- board-composition-matched Jan+Jul test,
  per SS2o's established ~5s board-visibility bar). Decision is made on the
  `seasonal` test number, not `forward`.

Protocol: with_aobt / keep_all, both `forward` and `seasonal` splits.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import config
from common import dataset as D
from common import evaluate as E
from common.bootstrap import run_dir, seed_everything
from pipeline_per_airport.model import PerAirportLGBM
from common import features as F

SPLITS = ["forward", "seasonal"]
FEATURESET, POLICY = "with_aobt", "keep_all"
SEED = 42
CLIP_LO, CLIP_HI = config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC
BAR_SEC = 5.0  # pre-registered board-visibility floor (SS2o), checked on `seasonal` test

PARAMS = {  # identical to pipeline_per_airport/train.py PARAMS
    "objective": "l2", "metric": "rmse",
    "learning_rate": 0.05, "num_leaves": 255, "min_data_in_leaf": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "max_cat_threshold": 64, "cat_smooth": 20.0,
    "verbosity": -1, "num_threads": 0,
}
NUM_BOOST_ROUND, PATIENCE = 5000, 200


def _params(seed):
    p = dict(PARAMS)
    p.update({"seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed,
              "device_type": "cpu"})
    return p


def clip(p):
    return np.clip(np.asarray(p, dtype="float64"), CLIP_LO, CLIP_HI)


def write_run(split, arm, seed, tag, conv, res_val, res_test):
    proto = config.protocol_key(split, FEATURESET, POLICY)
    d = run_dir(config.ARTIFACTS / "runs", arm, protocol=proto, seed=seed,
                extra={"tag": tag})
    (d / "model.json").write_text(json.dumps({
        "arm": arm, "split": split, "featureset": FEATURESET, "seed": seed,
        "tag": tag,
        "target_transform": ("delta_aobt_with_fallback" if tag == "delta_fb" else "identity"),
        "params": _params(seed), "convergence": conv,
        "selection_surface": "val",
        "note": "tests_delta_target/run_delta_target.py -- delta/residual target "
                "experiment; raw control and delta+fallback variant trained by the "
                "same script, single seed (tree side).",
    }, indent=2))
    (d / "eval_val.json").write_text(json.dumps(res_val, indent=2))
    (d / "eval_test.json").write_text(json.dumps(res_test, indent=2))
    return d


def run_split(split, lgb, rows):
    t0 = time.time()
    print(f"\n===== split={split} =====")
    Xtr_raw, ytr, pool_meta = D.load_part(split, "train", FEATURESET,
                                          filter_target=True, outlier_policy=POLICY)
    Xva_raw, yva, meta_va = D.load_part(split, "val", FEATURESET, filter_target=False)
    Xte_raw, yte, meta_te = D.load_part(split, "test", FEATURESET, filter_target=False)
    Xva = D.align_columns(Xtr_raw, Xva_raw)
    Xte = D.align_columns(Xtr_raw, Xte_raw)
    cats = [c for c in F.CATEGORICAL if c in Xtr_raw.columns]
    ytr = pd.Series(np.asarray(ytr, dtype="float64"), index=Xtr_raw.index)
    yva = pd.Series(np.asarray(yva, dtype="float64"), index=Xva.index)
    yte = pd.Series(np.asarray(yte, dtype="float64"), index=Xte.index)
    print(f"loaded: train={len(Xtr_raw)} val={len(Xva)} test={len(Xte)} "
          f"cats={len(cats)}  ({time.time()-t0:.0f}s)")

    seed_everything(SEED)

    # ---- control: raw-target per_airport ----
    m_raw = PerAirportLGBM(_params(SEED), NUM_BOOST_ROUND, PATIENCE, cats)
    m_raw.fit(Xtr_raw, ytr.to_numpy(), pool_meta, Xva, yva.to_numpy(), meta_va,
              lgb=lgb, verbose=False)
    conv_raw = {"best_iteration": {k: int(v) for k, v in m_raw.best_iter_.items()},
                "any_hit_cap": any(c["hit_max_rounds"] for c in m_raw.convergence_.values())}
    pv_raw = clip(m_raw.predict(Xva, meta_va))
    pt_raw = clip(m_raw.predict(Xte, meta_te))

    # ---- treatment: delta-target on matched rows, fallback to m_raw elsewhere ----
    mtr = (Xtr_raw["has_aobt"] == 1)
    mva = (Xva["has_aobt"] == 1)
    mte = (Xte["has_aobt"] == 1)
    print(f"matched (has_aobt==1): train {mtr.mean()*100:.2f}%  "
          f"val {mva.mean()*100:.2f}%  test {mte.mean()*100:.2f}%")

    ytr_delta = (ytr[mtr] - Xtr_raw.loc[mtr, "delta_aobt_sec"]).to_numpy()
    yva_delta = (yva[mva] - Xva.loc[mva, "delta_aobt_sec"]).to_numpy()

    m_delta = PerAirportLGBM(_params(SEED), NUM_BOOST_ROUND, PATIENCE, cats)
    m_delta.fit(Xtr_raw.loc[mtr], ytr_delta, pool_meta.loc[mtr],
                Xva.loc[mva], yva_delta, meta_va.loc[mva],
                lgb=lgb, verbose=False)
    conv_delta = {"best_iteration": {k: int(v) for k, v in m_delta.best_iter_.items()},
                  "any_hit_cap": any(c["hit_max_rounds"] for c in m_delta.convergence_.values()),
                  "train_matched_frac": float(mtr.mean())}

    def combined(X, meta, m_bool, pred_raw):
        out = pred_raw.copy()
        if m_bool.any():
            pd_ = m_delta.predict(X.loc[m_bool], meta.loc[m_bool])
            out[m_bool.to_numpy()] = pd_ + X.loc[m_bool, "delta_aobt_sec"].to_numpy()
        return clip(out)

    pv_fb = combined(Xva, meta_va, mva, pv_raw)
    pt_fb = combined(Xte, meta_te, mte, pt_raw)

    def score(arm, tag, p_val, p_test, conv):
        rv = E.evaluate(yva.to_numpy(), p_val, meta_va, ytr.to_numpy(), part="val",
                        arm=arm, split=split, featureset=FEATURESET, seed=SEED)
        rt = E.evaluate(yte.to_numpy(), p_test, meta_te, ytr.to_numpy(), part="test",
                        arm=arm, split=split, featureset=FEATURESET, seed=SEED)
        write_run(split, arm, SEED, tag, conv, rv, rt)
        row = {"split": split, "arm": arm, "tag": tag,
               "val_rmse": round(rv["rmse"], 3), "val_inband": round(rv["rmse_in_band"], 3),
               "test_rmse": round(rt["rmse"], 3), "test_inband": round(rt["rmse_in_band"], 3),
               "test_lb_weighted": round(rt.get("leaderboard_weighted_rmse", float("nan")), 3)}
        rows.append(row)
        print(f"  {arm:12s} {tag:10s}  val {row['val_rmse']:8.2f} / ib {row['val_inband']:7.2f}   "
              f"test {row['test_rmse']:8.2f} / ib {row['test_inband']:7.2f} / "
              f"lbw {row['test_lb_weighted']:7.2f}")
        return row

    r_raw = score("per_airport", "raw", pv_raw, pt_raw, conv_raw)
    r_fb = score("per_airport", "delta_fb", pv_fb, pt_fb, conv_delta)
    print(f"[{time.time()-t0:.0f}s] split={split} done")
    return r_raw, r_fb


def main() -> int:
    import lightgbm as lgb
    rows = []
    results = {}
    for split in SPLITS:
        r_raw, r_fb = run_split(split, lgb, rows)
        results[split] = (r_raw, r_fb)

    with (HERE / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    lines = ["delta-target experiment  (with_aobt / keep_all, seed 42, per_airport)",
             "=" * 78]
    for split in SPLITS:
        r_raw, r_fb = results[split]
        d_test = r_fb["test_rmse"] - r_raw["test_rmse"]
        d_val = r_fb["val_rmse"] - r_raw["val_rmse"]
        lines.append(f"[{split}]  val: raw {r_raw['val_rmse']:.2f} -> fb {r_fb['val_rmse']:.2f}"
                     f"  (delta {d_val:+.2f})   "
                     f"test: raw {r_raw['test_rmse']:.2f} -> fb {r_fb['test_rmse']:.2f}"
                     f"  (delta {d_test:+.2f})")
        if split == "seasonal":
            verdict = ("CLEARS bar" if d_test <= -BAR_SEC else "does NOT clear bar")
            lines.append(f"  pre-registered decisive check: seasonal test delta "
                         f"{d_test:+.2f}s vs bar -{BAR_SEC:.1f}s -> {verdict}")
    txt = "\n".join(lines)
    (HERE / "summary.txt").write_text(txt + "\n", encoding="utf-8")
    print("\n" + txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
