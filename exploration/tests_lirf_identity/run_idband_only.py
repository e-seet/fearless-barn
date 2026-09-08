#!/usr/bin/env python3
"""airport_lirf_identity_multimodel, checked-identity band ONLY.

The full arm overrides two bands:
  - 25.2k-50k : y = delta_sched_sec to +/-3 s over 34 full-year rows -- a real identity
  - >50k      : NOT an identity (full-year std 13,488 s); a motivated tail bet

This variant applies the override ONLY to the 25.2k-50k band and leaves the >50k
rows to the base model. Writes a tagged run dir (tag=idband) so harvest folds it
into results.csv, and prints base vs full-override vs idband-only.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from common import dataset as D
from common import evaluate as E
from common.metrics import rmse
from common.predict import predict_arm
from common.bootstrap import run_dir

BASE_ARM = "airport_multiseed_multimodel"
ARM = "airport_lirf_identity_multimodel"
TAG = "idband"
SPLIT, FS, POL, SEED = "forward", "with_aobt", "keep_all", 42
LO, HI = 25200.0, 50000.0
CLIP = (config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)


def overrides(pbase, X, ap):
    ds = X["delta_sched_sec"].fillna(0).to_numpy()
    hasrec = X["has_flight_record"].to_numpy()
    base_cond = (ap == "LIRF") & (hasrec == 0)
    m_id = base_cond & (ds > LO) & (ds <= HI)          # checked identity only
    m_full = base_cond & (ds > LO)                     # what the shipped arm does
    p_id = np.clip(np.where(m_id, ds, pbase), *CLIP)
    p_full = np.clip(np.where(m_full, ds, pbase), *CLIP)
    return np.clip(pbase, *CLIP), p_id, p_full, int(m_id.sum()), int(m_full.sum())


def main():
    Xtr, ytr, _ = D.load_part(SPLIT, "train", FS, filter_target=True, outlier_policy=POL)
    rows = {}
    d = run_dir(config.ARTIFACTS / "runs", ARM, protocol=config.protocol_key(SPLIT, FS, POL),
                seed=SEED, extra={"tag": TAG})
    for part in ("val", "test"):
        X, y, meta = D.load_part(SPLIT, part, FS, filter_target=False)
        y = np.asarray(y, float)
        pbase = predict_arm(BASE_ARM, SPLIT, part, FS, SEED, POL, X=X, meta=meta, Xtr=Xtr)
        ap = meta["ADEP_mvt"].astype("object").to_numpy()
        p_base, p_id, p_full, n_id, n_full = overrides(pbase, X, ap)
        res = E.evaluate(y, p_id, meta, ytr, part=part, arm=ARM, split=SPLIT,
                         featureset=FS, seed=SEED,
                         extra={"variant": "checked-identity band only (25.2k-50k)",
                                "band": [LO, HI], "n_overridden": n_id, "base_arm": BASE_ARM})
        (d / f"eval_{part}.json").write_text(json.dumps(res, indent=2))
        ib = y <= 7200
        rows[part] = dict(
            base=rmse(y, p_base), idband=rmse(y, p_id), full=rmse(y, p_full),
            base_ib=rmse(y[ib], p_base[ib]), idband_ib=rmse(y[ib], p_id[ib]), full_ib=rmse(y[ib], p_full[ib]),
            base_lbw=E.evaluate(y, p_base, meta, ytr, part=part, arm="b", split=SPLIT, featureset=FS, seed=SEED)["leaderboard_weighted_rmse"],
            idband_lbw=res["leaderboard_weighted_rmse"],
            full_lbw=E.evaluate(y, p_full, meta, ytr, part=part, arm="f", split=SPLIT, featureset=FS, seed=SEED)["leaderboard_weighted_rmse"],
            n_id=n_id, n_full=n_full)

    (d / "model.json").write_text(json.dumps({
        "arm": ARM, "tag": TAG, "base_arm": BASE_ARM, "seed": SEED,
        "variant": "checked-identity band only", "band_sec": [LO, HI],
        "n_free_params": 0, "weights_are_val_fitted": False,
        "selection_surface": "", "train_pool_ids_sha": D.pool_fingerprint(Xtr.index),
    }, indent=2))

    w = 10
    print(f"\n{'':16s}{'base':>{w}}{'idband':>{w}}{'FULL arm':>{w}}   (base = {BASE_ARM})")
    for part in ("val", "test"):
        r = rows[part]
        print(f"-- {part} --  n_override: idband={r['n_id']}  full={r['n_full']}")
        print(f"  pooled RMSE  {r['base']:>{w}.2f}{r['idband']:>{w}.2f}{r['full']:>{w}.2f}"
              f"     idband delta {r['idband']-r['base']:+.2f} | full delta {r['full']-r['base']:+.2f}")
        print(f"  in-band RMSE {r['base_ib']:>{w}.2f}{r['idband_ib']:>{w}.2f}{r['full_ib']:>{w}.2f}"
              f"     (both must be ~0 vs base)")
        print(f"  LB-weighted  {r['base_lbw']:>{w}.2f}{r['idband_lbw']:>{w}.2f}{r['full_lbw']:>{w}.2f}"
              f"     idband delta {r['idband_lbw']-r['base_lbw']:+.2f} | full delta {r['full_lbw']-r['base_lbw']:+.2f}")
    print(f"\nwrote {d}")


if __name__ == "__main__":
    main()
