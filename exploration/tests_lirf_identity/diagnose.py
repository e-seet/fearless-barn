#!/usr/bin/env python3
"""Per-row diagnostic for airport_lirf_identity_multimodel: exactly which rows the
zero-parameter override touches, and how much of the RMSE gain each one carries.

The arm replaces the base prediction with raw delta_sched_sec for
(LIRF & has_flight_record==0 & delta_sched_sec > 25200). This checks whether the
gain is a checked identity or a few lucky tail rows.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
sys.path.insert(0, str(ROOT))
import config
from common import dataset as D
from common.metrics import rmse
from common.predict import predict_arm
from pipeline_airport_lirf_identity_multimodel.train import BASE_ARM, THRESHOLD_SEC, apply_lirf_identity_override

SPLIT, FS, POL, SEED = "forward", "with_aobt", "keep_all", 42


def band(v):
    if v > 50000: return ">50k (artifact/rollover)"
    if v > 25200: return "25.2-50k (checked identity)"
    return "<=25.2k"


def do(part):
    Xtr, ytr, _ = D.load_part(SPLIT, "train", FS, filter_target=True, outlier_policy=POL)
    X, y, meta = D.load_part(SPLIT, part, FS, filter_target=False)
    y = np.asarray(y, dtype="float64")
    pbase = predict_arm(BASE_ARM, SPLIT, part, FS, SEED, POL, X=X, meta=meta, Xtr=Xtr)
    pbase = np.clip(np.asarray(pbase, float), config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
    ap = meta["ADEP_mvt"].astype("object").to_numpy()
    pnew, n = apply_lirf_identity_override(pbase, X, ap, THRESHOLD_SEC)

    m = (ap == "LIRF") & (X["has_flight_record"].to_numpy() == 0) & (X["delta_sched_sec"].fillna(0).to_numpy() > THRESHOLD_SEC)
    ds = X["delta_sched_sec"].fillna(0).to_numpy()

    print(f"\n================  {part.upper()}  (n={len(y)})  ================")
    print(f"pooled RMSE : base {rmse(y,pbase):8.3f}  ->  override {rmse(y,pnew):8.3f}   "
          f"delta {rmse(y,pnew)-rmse(y,pbase):+.3f}")
    ib = y <= 7200
    print(f"in-band RMSE: base {rmse(y[ib],pbase[ib]):8.3f}  ->  override {rmse(y[ib],pnew[ib]):8.3f}   "
          f"delta {rmse(y[ib],pnew[ib])-rmse(y[ib],pbase[ib]):+.3f}   (must be ~0: override only hits y>25200)")
    print(f"rows overridden: {m.sum()}")
    tot_sse_gain = np.sum((y[m]-pbase[m])**2 - (y[m]-pnew[m])**2)
    print(f"total SSE reduced by override: {tot_sse_gain:.3e}   "
          f"(= {np.sqrt(np.sum((y-pbase)**2)/len(y)) - np.sqrt(np.sum((y-pnew)**2)/len(y)):+.3f} s pooled RMSE)")
    print(f"\n{'#':>2} {'band':28s} {'y_true':>10s} {'delta_sched':>12s} {'base_pred':>10s} "
          f"{'new_pred':>10s} {'|err|base':>10s} {'|err|new':>9s} {'SSE gain':>11s}")
    idx = np.where(m)[0]
    for k, i in enumerate(sorted(idx, key=lambda i: -abs(y[i]-pbase[i]))):
        eb, en = abs(y[i]-pbase[i]), abs(y[i]-pnew[i])
        g = (y[i]-pbase[i])**2 - (y[i]-pnew[i])**2
        print(f"{k+1:>2} {band(ds[i]):28s} {y[i]:>10.0f} {ds[i]:>12.0f} {pbase[i]:>10.0f} "
              f"{pnew[i]:>10.0f} {eb:>10.0f} {en:>9.0f} {g:>+11.2e}")
    # what if we DROP the >50k rows from the override (keep only the checked-identity band)?
    m_id = m & (ds <= 50000)
    p_idonly = pbase.copy(); p_idonly[m_id] = ds[m_id]
    p_idonly = np.clip(p_idonly, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
    print(f"\nif override applied ONLY to the 25.2-50k checked-identity band "
          f"({m_id.sum()} rows), skipping the {int((m & (ds>50000)).sum())} unpredictable >50k rows:")
    print(f"  pooled RMSE {rmse(y,p_idonly):.3f}  (base {rmse(y,pbase):.3f}, full override {rmse(y,pnew):.3f})")


for part in ("val", "test"):
    do(part)
