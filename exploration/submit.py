#!/usr/bin/env python3
"""Build and validate a competition submission file. Does NOT upload.

Submission contract (ansperformance.eu/study/data-challenge/dc2026/ranking.html):
  * filename  : <team-name>_v<incremental integer>.parquet
  * columns   : MVT_ID_mvt, TAXITIME_SEC_mvt
  * rows      : exactly submitting.parquet's rows -- none missing, none extra,
                every MVT_ID matching the template. The ranking system rejects
                the file otherwise.
  * scoring   : pooled RMSE over the ranking DEP set.

Upload is deliberately NOT performed by this script. It prints the exact `mc`
command instead, so sending a submission stays an explicit human action -- and
so a half-finished model cannot burn a submission slot by accident.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from common import dataset as D
from common.bootstrap import run_dir, seed_everything

SUBMIT_DIR = config.SOLUTION_DIR / "submissions"


def predict_ranking(arm: str, split: str, featureset: str, seed: int,
                    outlier_policy: str = None, tag: str = None):
    """Predictions for the scored set from one trained run.

    `tag` selects a run keyed by `train.py --tag` -- a configuration the protocol
    key does not name, such as a tuned hyperparameter set. It applies ONLY to the
    arm being asked for: a blend's base arms always resolve to the untagged runs
    its weights were fitted against, because swapping a base arm underneath frozen
    weights silently changes a model that was never validated in that combination.
    """
    proto = config.protocol_key(split, featureset, outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", arm, protocol=proto, seed=seed,
                  extra=({"tag": tag} if tag else None), create=False)
    if not out.exists():
        raise FileNotFoundError(
            "no trained run at {}. Train the arm first.".format(out))

    Xtr, _, _ = D.load_part(split, "train", featureset, filter_target=True,
                            outlier_policy=outlier_policy)
    Xrk, meta = D.load_ranking_features(featureset)
    Xrk = D.align_columns(Xtr, Xrk)

    if arm == "lightgbm":
        import lightgbm as lgb
        pred = lgb.Booster(model_file=str(out / "model.txt")).predict(Xrk)
    elif arm == "ridge":
        with open(out / "model.pkl", "rb") as fh:
            pred = pickle.load(fh).predict(Xrk)
    elif arm in ("per_airport", "per_airport_antigravity"):
        with open(out / "model.pkl", "rb") as fh:
            pred = pickle.load(fh).predict(Xrk, meta)
    elif arm == "per_airport_stack":
        from pipeline_per_airport_stack.evaluate import blend
        with open(out / "model.pkl", "rb") as fh:
            pred = blend(pickle.load(fh), Xrk, meta)
    elif arm in ("lightgbm_recency",):
        import lightgbm as lgb
        pred = lgb.Booster(model_file=str(out / "model.txt")).predict(Xrk)
    elif arm == "lightgbm_unsup":
        import lightgbm as lgb
        with open(out / "unsup.pkl", "rb") as fh:
            unsup = pickle.load(fh)
        _, _, tr_meta = D.load_part(split, "train", featureset, filter_target=True,
                                    outlier_policy=outlier_policy)
        Xtr_u = unsup.transform(Xtr, tr_meta)
        Xrk_u = D.align_columns(Xtr_u, unsup.transform(Xrk, meta))
        pred = lgb.Booster(model_file=str(out / "model.txt")).predict(Xrk_u)
    elif arm == "naive_offblock":
        from pipeline_naive_offblock.evaluate import load_model
        from pipeline_naive_offblock.train import predict as naive_predict
        pred, _ = naive_predict(load_model(out), Xrk, featureset)
    elif arm == "xgboost_antigravity":
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(str(out / "model.ubj"))
        meta_json = json.loads((out / "model.json").read_text())
        best_iter = meta_json.get("best_iteration")
        iter_range = (0, best_iter + 1) if best_iter is not None else None
        dmat = xgb.DMatrix(Xrk, enable_categorical=True)
        pred = booster.predict(dmat, iteration_range=iter_range)
    elif arm == "mlp_antigravity":
        import torch
        from torch.utils.data import DataLoader
        from pipeline_mlp_antigravity.model import TabularMLP
        from pipeline_mlp_antigravity.train import TabularDataset, encode_data

        with open(out / "preprocessor.pkl", "rb") as fh:
            prep = pickle.load(fh)
        cat_tensors, num_tensor = encode_data(Xrk, prep["cat_cols"], prep["num_cols"],
                                              prep["cat_maps"], prep["scaler"], prep["medians"])
        ds = TabularDataset(cat_tensors, num_tensor)
        loader = DataLoader(ds, batch_size=4096, shuffle=False, num_workers=0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prior_mean = prep.get("prior_mean", 990.0)
        model = TabularMLP(prep["cat_cardinalities"], num_features=len(prep["num_cols"]), prior_mean=prior_mean).to(device)
        model.load_state_dict(torch.load(out / "model.pt", map_location=device))
        model.eval()
        mlp_preds = []
        with torch.no_grad():
            for x_cat, x_num in loader:
                x_cat = {c: x_cat[c].to(device) for c in x_cat}
                x_num = x_num.to(device)
                mlp_preds.append(model(x_cat, x_num).cpu().numpy())
        pred = np.concatenate(mlp_preds)
    elif arm == "ensemble_antigravity":
        meta_json = json.loads((out / "model.json").read_text())
        weights = meta_json["weights"]
        base_arms = meta_json["base_arms"]
        preds = []
        arm_weights = []
        for b_arm in base_arms:
            w = weights.get(b_arm, 0.0)
            if w <= 0:
                continue
            b_pred, _ = predict_ranking(b_arm, split, featureset, seed, outlier_policy)
            preds.append(b_pred)
            arm_weights.append(w)
        total_w = sum(arm_weights)
        norm_weights = [w / total_w for w in arm_weights] if total_w > 0 else [1.0 / len(arm_weights)] * len(arm_weights)
        pred = np.zeros(len(Xrk), dtype="float64")
        for w, p in zip(norm_weights, preds):
            pred += w * p
    elif arm == "multimodel_antigravity":
        meta_json = json.loads((out / "model.json").read_text())
        weights_per_airport = meta_json["weights_per_airport"]
        p_ap, meta_ap = predict_ranking("per_airport", split, featureset, seed, outlier_policy)
        p_mlp, _ = predict_ranking("mlp_antigravity", split, featureset, seed, outlier_policy)
        ap_arr = meta_ap[config.ID.replace("MVT_ID_mvt", "ADEP_mvt") if "ADEP_mvt" in meta_ap else "ADEP_mvt"].astype("object").to_numpy()
        pred = np.zeros(len(p_ap), dtype="float64")
        for a in pd.unique(ap_arr):
            m = (ap_arr == a)
            w = weights_per_airport.get(str(a), 0.5)
            pred[m] = w * p_ap[m] + (1.0 - w) * p_mlp[m]
    elif arm == "feature_pruned_antigravity":
        import lightgbm as lgb
        from pipeline_feature_pruned_antigravity.train import DROP_FEATURES
        Xrk_pruned = Xrk.drop(columns=[c for c in DROP_FEATURES if c in Xrk.columns])
        pred = lgb.Booster(model_file=str(out / "model.txt")).predict(Xrk_pruned)
    elif arm == "delay_twostage_antigravity":
        with open(out / "model.pkl", "rb") as fh:
            model = pickle.load(fh)
        pred = model.predict(Xrk)
    elif arm == "per_operator":
        with open(out / "model.pkl", "rb") as fh:
            model = pickle.load(fh)
        pred = model.predict(Xrk, Xrk)
    elif arm == "airport_operator_blend":
        if (out / "model.pkl").exists():
            with open(out / "model.pkl", "rb") as fh:
                bundle = pickle.load(fh)
            p_a = bundle["per_airport"].predict(Xrk, meta)
            p_b = bundle["per_operator"].predict(Xrk, Xrk)
            pred = bundle["alpha"] * p_a + (1.0 - bundle["alpha"]) * p_b
        else:
            meta_json = json.loads((out / "model.json").read_text())
            w_ap = meta_json.get("alpha_per_airport", 0.53)
            p_ap, meta_ap = predict_ranking("per_airport", split, featureset, seed, outlier_policy)
            p_op, _ = predict_ranking("per_operator", split, featureset, seed, outlier_policy)
            pred = w_ap * p_ap + (1.0 - w_ap) * p_op
    elif arm == "airport_operator_multimodel":
        # This branch was MISSING while the arm was in config.ARMS, so argparse
        # accepted `--arm airport_operator_multimodel` and then predict_ranking
        # raised "unknown arm" -- the val-selected winner could not be submitted
        # at all. Exactly the drift config.py's ARMS comment describes (a
        # per-file copy of a list that must agree), and the same shape as the
        # bug that comment says it was written to fix. Its five bucketed
        # descendants below all had branches; the honest parent did not.
        meta_json = json.loads((out / "model.json").read_text())
        weights = meta_json["weights_per_airport"]
        p_aob, meta_aob = predict_ranking("airport_operator_blend", split, featureset,
                                          seed, outlier_policy)
        # The MLP seed set comes from the RUN's own model.json, never from
        # `seed`. Under --mlp-seeds the per-airport weights were solved against
        # an AVERAGED MLP, so pairing those weights with a single MLP draw here
        # would ship a combination that was never validated anywhere -- exactly
        # the "swapping a base arm underneath frozen weights" failure this
        # function's own docstring warns about. Absent the key (a pre-msmlp run)
        # this falls back to [seed], which is the original single-draw path.
        mlp_seeds = meta_json.get("mlp_seeds") or [seed]
        mlp_preds = [predict_ranking("mlp_antigravity", split, featureset, s, outlier_policy)[0]
                     for s in mlp_seeds]
        p_mlp = np.mean(mlp_preds, axis=0)
        ap_arr = meta_aob["ADEP_mvt"].astype("object").to_numpy()
        pred = np.zeros(len(p_aob), dtype="float64")
        for a in pd.unique(ap_arr):
            m_a = (ap_arr == a)
            w = weights.get(str(a), 0.5)
            pred[m_a] = w * p_aob[m_a] + (1.0 - w) * p_mlp[m_a]
    elif arm == "airport_wtc_multimodel":
        meta_json = json.loads((out / "model.json").read_text())
        cell_alpha = meta_json["cell_alpha"]
        ap_alpha = meta_json["airport_alpha_fallback"]
        p_aob, meta_aob = predict_ranking("airport_operator_blend", split, featureset, seed, outlier_policy)
        p_mlp, _ = predict_ranking("mlp_antigravity", split, featureset, seed, outlier_policy)
        ap_col = config.ID.replace("MVT_ID_mvt", "ADEP_mvt") if "ADEP_mvt" in meta_aob else "ADEP_mvt"
        ap_arr = meta_aob[ap_col].astype("object").to_numpy()
        from pipeline_airport_wtc_multimodel.train import blend_with_cell_alpha
        pred = blend_with_cell_alpha(p_aob, p_mlp, ap_arr, Xrk, cell_alpha, ap_alpha)
    elif arm == "airport_wtc_rec_multimodel":
        meta_json = json.loads((out / "model.json").read_text())
        cell_alpha = meta_json["cell_alpha"]
        ap_alpha = meta_json["airport_alpha_fallback"]
        p_aob, meta_aob = predict_ranking("airport_operator_blend", split, featureset, seed, outlier_policy)
        p_mlp, _ = predict_ranking("mlp_antigravity", split, featureset, seed, outlier_policy)
        ap_col = config.ID.replace("MVT_ID_mvt", "ADEP_mvt") if "ADEP_mvt" in meta_aob else "ADEP_mvt"
        ap_arr = meta_aob[ap_col].astype("object").to_numpy()
        from pipeline_airport_wtc_rec_multimodel.train import blend_with_cell_alpha
        pred = blend_with_cell_alpha(p_aob, p_mlp, ap_arr, Xrk, cell_alpha, ap_alpha)
    elif arm == "airport_unsup_multimodel":
        meta_json = json.loads((out / "model.json").read_text())
        cell_alpha = meta_json["cell_alpha"]
        ap_alpha = meta_json["airport_alpha_fallback"]
        with open(out / "cluster_models.pkl", "rb") as fh:
            cmodels = pickle.load(fh)
        p_aob, meta_aob = predict_ranking("airport_operator_blend", split, featureset, seed, outlier_policy)
        p_mlp, _ = predict_ranking("mlp_antigravity", split, featureset, seed, outlier_policy)
        ap_col = config.ID.replace("MVT_ID_mvt", "ADEP_mvt") if "ADEP_mvt" in meta_aob else "ADEP_mvt"
        ap_arr = meta_aob[ap_col].astype("object").to_numpy()
        from pipeline_airport_unsup_multimodel.train import assign_clusters, _bucket_key, blend_with_cell_alpha
        clusters = assign_clusters(Xrk, ap_arr, cmodels["scalers"], cmodels["kmeans_models"])
        cell_arr = _bucket_key(ap_arr, Xrk, clusters)
        pred = blend_with_cell_alpha(p_aob, p_mlp, cell_arr, ap_arr, cell_alpha, ap_alpha, meta_json.get("parent_alpha", {}))
    elif arm == "airport_multiseed_multimodel":
        meta_json = json.loads((out / "model.json").read_text())
        b_arm = meta_json.get("base_arm", "airport_unsup_multimodel")
        seeds = meta_json.get("ensemble_seeds", [42, 43, 44])
        preds = []
        meta_rk = None
        for s in seeds:
            p_s, meta_rk = predict_ranking(b_arm, split, featureset, s, outlier_policy)
            preds.append(p_s)
        pred = np.mean(preds, axis=0)
        meta = meta_rk
    elif arm == "airport_debiased_multimodel":
        from pipeline_per_airport.model import AIRPORT_COL
        from pipeline_airport_debiased_multimodel.train import apply_alignment, blend_post_aligned, DEFAULT_THRESHOLD_SEC
        meta_json = json.loads((out / "model.json").read_text())
        b_arm = meta_json.get("base_arm", "airport_multiseed_multimodel")
        coef_ext = meta_json.get("lirf_align_coef_ext", meta_json.get("lirf_align_coef"))
        coef_mid = meta_json.get("lirf_align_coef_mid")
        thresh_ext = float(meta_json.get("threshold_ext_sec", meta_json.get("threshold_sec", DEFAULT_THRESHOLD_SEC)))
        thresh_mid = meta_json.get("threshold_mid_sec")
        if thresh_mid is not None:
            thresh_mid = float(thresh_mid)

        if "shrunk_alphas" in meta_json:
            seeds = meta_json.get("ensemble_seeds", [42, 43, 44])
            preds_gbdt = []
            preds_mlp = []
            meta_rk = None
            for s in seeds:
                p_g, meta_rk = predict_ranking("airport_operator_blend", split, featureset, s, outlier_policy)
                p_m, _ = predict_ranking("mlp_antigravity", split, featureset, s, outlier_policy)
                preds_gbdt.append(p_g)
                preds_mlp.append(p_m)
            p_gbdt = np.mean(preds_gbdt, axis=0)
            p_mlp = np.mean(preds_mlp, axis=0)
            ap_arr = meta_rk[AIRPORT_COL].astype("object").to_numpy()
            p_base = blend_post_aligned(p_gbdt, p_mlp, ap_arr, meta_json["shrunk_alphas"], meta_json.get("global_alpha", 0.720))
        else:
            p_base, meta_rk = predict_ranking(b_arm, split, featureset, seed, outlier_policy)
            ap_arr = meta_rk[AIRPORT_COL].astype("object").to_numpy()

        has_rec_arr = Xrk["has_flight_record"].to_numpy()
        pred = apply_alignment(p_base, Xrk, ap_arr, has_rec_arr, coef_ext=coef_ext,
                               threshold_ext=thresh_ext, coef_mid=coef_mid, threshold_mid=thresh_mid)
        if "seasonal_slopes" in meta_json and "doy_cos" in Xrk.columns:
            from pipeline_airport_debiased_multimodel.train import apply_seasonal_calibration
            pred = apply_seasonal_calibration(
                pred, ap_arr, Xrk["doy_cos"].to_numpy(),
                meta_json["seasonal_slopes"],
                gamma_pos=float(meta_json.get("seasonal_gamma_pos", meta_json.get("seasonal_gamma", 0.50))),
                gamma_neg=float(meta_json.get("seasonal_gamma_neg", 0.10))
            )
        meta = meta_rk
    elif arm == "per_airport_mlp":
        import torch
        from torch.utils.data import DataLoader
        from pipeline_per_airport.model import AIRPORT_COL
        from pipeline_per_airport_mlp.model import AirportTabularMLP
        from pipeline_per_airport_mlp.train import TabularAirportDataset, encode_airport_data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dicts = torch.load(out / "models.pt", map_location="cpu")
        with open(out / "preprocessors.pkl", "rb") as fh:
            preps = pickle.load(fh)
        pred = np.zeros(len(Xrk), dtype="float64")
        for a in sorted(preps.keys()):
            m = (meta[AIRPORT_COL] == a)
            if m.sum() == 0:
                continue
            X_ap = Xrk[m].drop(columns=[AIRPORT_COL], errors="ignore")
            prep = preps[a]
            cat_tensors, num_tensor = encode_airport_data(
                X_ap, prep["cat_cols"], prep["num_cols"],
                prep["cat_maps"], prep["scaler"], prep["medians"]
            )
            model = AirportTabularMLP(prep["cat_cardinalities"], num_features=len(prep["num_cols"]),
                                      hidden_dims=(128, 64), prior_mean=prep["prior_mean"]).to(device)
            model.load_state_dict(state_dicts[a])
            model.eval()
            ap_preds = []
            n_ap = len(num_tensor)
            with torch.no_grad():
                for b_start in range(0, n_ap, 2048):
                    x_cat = cat_tensors[b_start: b_start + 2048].to(device)
                    x_num = num_tensor[b_start: b_start + 2048].to(device)
                    ap_preds.append(model(x_cat, x_num).cpu().numpy())
            pred[m.to_numpy()] = np.concatenate(ap_preds)
    else:
        raise ValueError("unknown arm {!r}".format(arm))
    return np.clip(pred, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC), meta


def validate(sub: "pd.DataFrame") -> "list[str]":
    """Every rejection reason the ranking system lists, checked before upload."""
    tmpl = pd.read_parquet(config.SUBMITTING_PARQUET)
    errs = []
    if list(sub.columns) != [config.ID, config.TARGET]:
        errs.append("columns are {}, must be exactly ['{}', '{}']".format(
            list(sub.columns), config.ID, config.TARGET))
    if len(sub) != len(tmpl):
        errs.append("row count {} != template {}".format(len(sub), len(tmpl)))
    missing = set(tmpl[config.ID]) - set(sub[config.ID])
    extra = set(sub[config.ID]) - set(tmpl[config.ID])
    if missing:
        errs.append("{} MVT_ID present in template but missing from submission".format(len(missing)))
    if extra:
        errs.append("{} MVT_ID in submission but not in template".format(len(extra)))
    if sub[config.ID].duplicated().any():
        errs.append("{} duplicated MVT_ID".format(int(sub[config.ID].duplicated().sum())))
    n_bad = int((~np.isfinite(sub[config.TARGET].astype("float64"))).sum())
    if n_bad:
        errs.append("{} non-finite predictions (NaN/inf) -- these score as failures".format(n_bad))
    neg = int((sub[config.TARGET] < 0).sum())
    if neg:
        errs.append("{} negative taxi-time predictions".format(neg))
    return errs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # config.ARMS, not a local copy: this list drifted from the real set of arms
    # and argparse then refused the val-selected winner outright, even though
    # predict_ranking() above has supported all six arms all along.
    ap.add_argument("--arm", required=True, choices=list(config.ARMS))
    ap.add_argument("--tag", default=None,
                    help="submit from a run keyed by train.py --tag (e.g. a tuned "
                         "configuration). Applies to --arm only; a blend's base "
                         "arms always resolve untagged.")
    ap.add_argument("--split", default="full", choices=sorted(config.SPLITS),
                    help="which trained run to submit from. Defaults to 'full' "
                         "-- the all-12-months refit. Submitting from 'forward' "
                         "ships a model that never saw Sep-Dec, the four months "
                         "closest to the scored January (design-gate B4).")
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--team", default=config.TEAM,
                    help="team name, as registered with the PRC. Defaults to "
                         "config.TEAM (%(default)s); pass it only to override.")
    ap.add_argument("--version", type=int, required=True,
                    help="incremental integer; increment for every new submission")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    if args.split in config.SELECTION_FORBIDDEN_SPLITS and args.split != "full":
        print("  WARNING: submitting from split '{}', which is not a valid "
              "selection surface.".format(args.split))
    pred, meta = predict_ranking(args.arm, args.split, args.featureset, args.seed,
                                 args.outlier_policy, tag=args.tag)

    tmpl = pd.read_parquet(config.SUBMITTING_PARQUET)
    sub = pd.DataFrame({config.ID: tmpl[config.ID].to_numpy(),
                        config.TARGET: np.rint(pred).astype("int32")})

    print("\n== prediction summary ==")
    s = pd.Series(pred)
    print("  n={} min={:.0f} p25={:.0f} median={:.0f} p75={:.0f} max={:.0f} mean={:.1f}".format(
        len(s), s.min(), s.quantile(.25), s.median(), s.quantile(.75), s.max(), s.mean()))
    print("  per-airport median prediction:")
    for ap_, v in pd.Series(pred, index=meta["ADEP_mvt"].to_numpy()).groupby(level=0).median().items():
        print("    {:6s} {:7.0f} s".format(str(ap_), v))

    print("\n== submission validation ==")
    errs = validate(sub)
    if errs:
        print("  REJECTED -- the ranking system would refuse this file:")
        for e in errs:
            print("    - " + e)
        return 1
    print("  [PASS] columns, row count, ID set, uniqueness, finiteness, sign")

    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    name = "{}_v{}.parquet".format(args.team, args.version)
    path = SUBMIT_DIR / name
    if path.exists() and not args.force:
        print("\n  {} already exists. Bump --version (submissions are versioned, "
              "never overwritten) or pass --force.".format(path))
        return 1
    sub.to_parquet(path, index=False)
    # Provenance: every run this file's predictions came from, by path, with the
    # feature-code fingerprint and (for boosters) the round count. Without this a
    # submission cannot be tied back to the models on disk, and a later --force
    # refit of a shared run path changes the file's meaning with nothing to show
    # for it -- which is exactly what happened to v1 before this was added.
    def _provenance(a, tag=None):
        d = run_dir(config.ARTIFACTS / "runs", a,
                    protocol=config.protocol_key(args.split, args.featureset,
                                                 args.outlier_policy),
                    seed=args.seed, extra=({"tag": tag} if tag else None),
                    create=False)
        m = json.loads((d / "model.json").read_text()) if (d / "model.json").exists() else {}
        return {"run_dir": str(d.relative_to(config.SOLUTION_DIR)),
                "feature_fingerprint": m.get("feature_fingerprint"),
                "best_iteration": m.get("best_iteration"),
                "params": {k: m.get("params", {}).get(k) for k in
                           ("learning_rate", "num_leaves", "min_data_in_leaf")}
                if m.get("params") else None,
                "weights": m.get("weights")}

    prov = {args.arm: _provenance(args.arm, args.tag)}
    comps = prov[args.arm].get("weights") or {}
    for b, w in comps.items():
        if w > 0:
            prov[b] = _provenance(b)          # base arms resolve untagged, by design
    (SUBMIT_DIR / (name + ".meta.json")).write_text(json.dumps({
        "arm": args.arm, "split": args.split, "featureset": args.featureset,
        "outlier_policy": args.outlier_policy, "tag": args.tag,
        "prediction_cap_sec": config.PREDICTION_CLIP_MAX_SEC,
        "seed": args.seed, "team": args.team, "version": args.version,
        "n_rows": int(len(sub)),
        "pred_median": float(s.median()), "pred_mean": float(s.mean()),
        "provenance": prov,
    }, indent=2))
    print("\n  wrote {}".format(path))
    print("\n== upload (run this yourself -- this script does not upload) ==")
    print("  Credentials live in ../credentials.json (accessKey / secretKey / url / path).")
    print("  1) one-time alias:")
    print('       mc alias set dc26 "$(jq -r .url ../credentials.json)" \\')
    print('         "$(jq -r .accessKey ../credentials.json)" "$(jq -r .secretKey ../credentials.json)"')
    print("  2) upload:")
    print('       mc cp "{}" "dc26/$(jq -r .path ../credentials.json)/{}"'.format(path, name))
    print("  3) check the leaderboard at the REST endpoint in credentials.json (.api)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
