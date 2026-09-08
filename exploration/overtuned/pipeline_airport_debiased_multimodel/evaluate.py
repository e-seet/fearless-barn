#!/usr/bin/env python3
"""Evaluate pipeline_airport_debiased_multimodel arm."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common import dataset as D
from common import evaluate as E
from common.predict import predict_arm
from common.bootstrap import run_dir, seed_everything, write_json_guarded
from pipeline_airport_debiased_multimodel.train import (
    apply_alignment, blend_post_aligned, get_multiseed_preds,
    apply_seasonal_calibration,
    DEFAULT_THRESHOLD_SEC, DEFAULT_THRESHOLD_MID_SEC, DEFAULT_SHRINKAGE_LAMBDA,
    DEFAULT_SEASONAL_GAMMA, DEFAULT_SEASONAL_GAMMA_POS, DEFAULT_SEASONAL_GAMMA_NEG, DEFAULT_SEEDS
)

ARM = "airport_debiased_multimodel"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--part", default="val", choices=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed)
    meta_json = json.loads((out / "model.json").read_text())

    shrunk_alphas = meta_json.get("shrunk_alphas")
    global_alpha = meta_json.get("global_alpha", 0.720)
    coef_ext = meta_json.get("lirf_align_coef_ext", meta_json.get("lirf_align_coef"))
    coef_mid = meta_json.get("lirf_align_coef_mid")
    thresh_ext = float(meta_json.get("threshold_ext_sec", meta_json.get("threshold_sec", DEFAULT_THRESHOLD_SEC)))
    thresh_mid = meta_json.get("threshold_mid_sec")
    if thresh_mid is not None:
        thresh_mid = float(thresh_mid)

    seasonal_slopes = meta_json.get("seasonal_slopes")
    seasonal_gamma_pos = float(meta_json.get("seasonal_gamma_pos", meta_json.get("seasonal_gamma", DEFAULT_SEASONAL_GAMMA)))
    seasonal_gamma_neg = float(meta_json.get("seasonal_gamma_neg", DEFAULT_SEASONAL_GAMMA_NEG))

    Xtr, ytr, _ = D.load_part(args.split, "train", args.featureset,
                              filter_target=True, outlier_policy=args.outlier_policy)
    X, y, meta = D.load_part(args.split, args.part, args.featureset, filter_target=False)
    ap_arr = meta["ADEP_mvt"].astype("object").to_numpy()
    has_rec_arr = X["has_flight_record"].to_numpy()

    if shrunk_alphas is not None:
        seeds = meta_json.get("ensemble_seeds", DEFAULT_SEEDS)
        p_gbdt = get_multiseed_preds("airport_operator_blend", args.split, args.part, args.featureset,
                                     args.outlier_policy, X=X, meta=meta, Xtr=Xtr, seeds=seeds)
        p_mlp = get_multiseed_preds("mlp_antigravity", args.split, args.part, args.featureset,
                                    args.outlier_policy, X=X, meta=meta, Xtr=Xtr, seeds=seeds)
        p_base = blend_post_aligned(p_gbdt, p_mlp, ap_arr, shrunk_alphas, global_alpha=global_alpha)
    else:
        base_arm = meta_json.get("base_arm", "airport_multiseed_multimodel")
        p_base = predict_arm(base_arm, args.split, args.part,
                             args.featureset, args.seed, args.outlier_policy,
                             X=X, meta=meta, Xtr=Xtr)

    pred = apply_alignment(p_base, X, ap_arr, has_rec_arr, coef_ext=coef_ext,
                           threshold_ext=thresh_ext, coef_mid=coef_mid, threshold_mid=thresh_mid)

    if seasonal_slopes and "doy_cos" in X.columns:
        pred = apply_seasonal_calibration(pred, ap_arr, X["doy_cos"].to_numpy(), seasonal_slopes,
                                          gamma_pos=seasonal_gamma_pos, gamma_neg=seasonal_gamma_neg)

    extra_eval = {
        "lirf_align_coef_ext": coef_ext,
        "lirf_align_coef_mid": coef_mid,
        "threshold_ext_sec": thresh_ext,
        "threshold_mid_sec": thresh_mid,
        "base_arm": meta_json.get("base_arm", "airport_multiseed_multimodel"),
        "global_alpha": global_alpha,
        "shrunk_alphas": shrunk_alphas,
        "seasonal_slopes": seasonal_slopes,
        "seasonal_gamma_pos": seasonal_gamma_pos,
        "seasonal_gamma_neg": seasonal_gamma_neg,
        "seasonal_gamma": seasonal_gamma_pos,
    }
    res = E.evaluate(y, pred, meta, ytr, part=args.part, arm=ARM, split=args.split,
                     featureset=args.featureset, seed=args.seed,
                     extra=extra_eval)
    E.print_result(res)
    write_json_guarded(out / "eval_{}.json".format(args.part), res, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
