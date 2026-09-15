"""One place that turns a trained run into predictions, for every arm.

Why this exists
---------------
`stage8_triage.py` originally borrowed `pipeline_ensemble_antigravity.train.
predict_base_arm` because that module already had a loader for the six arms of
the day. That was a fragile dependency in two directions: it is third-party to
this build (so it is not this build's to extend), and its arm list is a *closed*
set — adding `per_airport` to `config.ARMS` made the triage stage die with
`ValueError: Unknown base arm 'per_airport'` after it had already loaded the
features and scored five arms.

So the dispatch lives here, in code this build owns, and it degrades sensibly:
arms this build wrote are loaded directly, the `*_antigravity` arms are delegated
to their own module (read-only), and an unknown arm raises with the list of what
is supported rather than a bare KeyError.

Predictions are returned **clipped exactly as each arm's own `evaluate.py` clips
them**, so a number computed here matches the one in that arm's `eval_*.json`.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common import dataset as D
from common.bootstrap import run_dir

OWN_ARMS = ("naive_offblock", "ridge", "lightgbm", "per_airport",
            "per_airport_stack", "lightgbm_recency", "lightgbm_unsup",
            "cross_blend", "lightgbm_arrfeat", "per_airport_arrfeat",
            "per_airport_tuned", "feature_pruned_antigravity",
            "per_airport_antigravity", "multimodel_antigravity",
            "delay_twostage_antigravity", "per_operator", "airport_operator_blend",
            "airport_operator_multimodel", "tri_blend", "per_airport_flighttype",
            "airport_wtc_multimodel", "airport_runway_multimodel",
            "airport_wtc_rec_multimodel", "airport_unsup_multimodel",
            "airport_multiseed_multimodel", "airport_debiased_multimodel",
            "per_airport_mlp", "airport_lirf_identity_multimodel",
            "catboost_antigravity")
THIRD_PARTY_ARMS = ("xgboost_antigravity", "mlp_antigravity", "ensemble_antigravity")


def _clip(p):
    return np.clip(np.asarray(p, dtype="float64"),
                   config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)


def predict_arm(arm, split, part, featureset, seed, outlier_policy,
                X=None, y=None, meta=None, Xtr=None, tag=None):
    """Clipped predictions for one arm on one part.

    X/meta/Xtr may be passed in to avoid reloading features per arm; anything
    omitted is loaded here.
    """
    proto = config.protocol_key(split, featureset, outlier_policy)
    d = run_dir(config.ARTIFACTS / "runs", arm, protocol=proto, seed=seed,
                extra=({"tag": tag} if tag else None), create=False)
    if not d.exists():
        raise FileNotFoundError("no trained run at {}".format(d))

    if Xtr is None:
        Xtr, _, _ = D.load_part(split, "train", featureset, filter_target=True,
                                outlier_policy=outlier_policy)
    if X is None:
        X, y, meta = D.load_part(split, part, featureset, filter_target=False)
    Xa = D.align_columns(Xtr, X)

    if arm in THIRD_PARTY_ARMS:
        from pipeline_ensemble_antigravity.train import predict_base_arm
        if arm != "ensemble_antigravity":
            # `tag` must go through: `d` above was resolved WITH it, so dropping
            # it here would score the untagged run while reporting the tagged one.
            return _clip(predict_base_arm(arm, split, part, featureset, seed,
                                          outlier_policy, X, Xtr=Xtr, tag=tag))
        # the blend is not one of its own base arms; rebuild it from its weights
        m = json.loads((d / "model.json").read_text())
        acc = None
        for base, w in m["weights"].items():
            if w <= 0:
                continue
            p = np.asarray(predict_base_arm(base, split, part, featureset, seed,
                                            outlier_policy, X, Xtr=Xtr),
                           dtype="float64") * float(w)
            acc = p if acc is None else acc + p
        return _clip(acc)

    if arm in ("lightgbm", "lightgbm_recency"):
        import lightgbm as lgb
        return _clip(lgb.Booster(model_file=str(d / "model.txt")).predict(Xa))

    if arm == "catboost_antigravity":
        from catboost import CatBoostRegressor
        from common import features as F
        from pipeline_catboost_antigravity.train import _cat_frame
        cats = [c for c in F.CATEGORICAL if c in Xa.columns]
        model = CatBoostRegressor()
        model.load_model(str(d / "model.cbm"))
        return _clip(model.predict(_cat_frame(Xa, cats)))

    if arm == "lightgbm_unsup":
        import lightgbm as lgb
        with open(d / "unsup.pkl", "rb") as fh:
            unsup = pickle.load(fh)
        _, _, tr_meta = D.load_part(split, "train", featureset, filter_target=True,
                                    outlier_policy=outlier_policy)
        Xtr_u = unsup.transform(Xtr, tr_meta)
        Xu = D.align_columns(Xtr_u, unsup.transform(Xa, meta))
        return _clip(lgb.Booster(model_file=str(d / "model.txt")).predict(Xu))

    if arm == "lightgbm_arrfeat":
        import lightgbm as lgb
        from pipeline_lightgbm_arrfeat.train import attach
        Xtr_u = attach(Xtr, "train2025", featureset)
        Xu = D.align_columns(Xtr_u, attach(Xa, "train2025", featureset))
        return _clip(lgb.Booster(model_file=str(d / "model.txt")).predict(Xu))

    if arm == "per_airport_arrfeat":
        # same router, but its models were trained on the extra columns
        from pipeline_lightgbm_arrfeat.train import attach
        Xtr_u = attach(Xtr, "train2025", featureset)
        Xu = D.align_columns(Xtr_u, attach(Xa, "train2025", featureset))
        with open(d / "model.pkl", "rb") as fh:
            return _clip(pickle.load(fh).predict(Xu, meta))

    if arm in ("per_airport", "per_airport_tuned", "per_airport_antigravity"):
        with open(d / "model.pkl", "rb") as fh:
            model = pickle.load(fh)
        return _clip(model.predict(Xa, meta))

    if arm == "per_operator":
        with open(d / "model.pkl", "rb") as fh:
            model = pickle.load(fh)
        return _clip(model.predict(Xa, Xa))  # OPERATOR_COL lives in X, not meta

    if arm == "airport_operator_blend":
        with open(d / "model.pkl", "rb") as fh:
            bundle = pickle.load(fh)
        # The bundle records whether it was trained with the congestion columns.
        # Predicting without them would feed the boosters a frame missing
        # features they split on -- LightGBM would not raise, it would just
        # score a different model.
        if bundle.get("congest"):
            from pipeline_per_airport_congest.extra import attach as attach_congest
            Xa = D.align_columns(attach_congest(Xtr, "train2025", featureset),
                                 attach_congest(Xa, "train2025", featureset))
        p_a = bundle["per_airport"].predict(Xa, meta)
        p_b = bundle["per_operator"].predict(Xa, Xa)
        return _clip(bundle["alpha"] * p_a + (1 - bundle["alpha"]) * p_b)

    if arm == "feature_pruned_antigravity":
        import lightgbm as lgb
        from pipeline_feature_pruned_antigravity.train import DROP_FEATURES
        Xu = Xa.drop(columns=[c for c in DROP_FEATURES if c in Xa.columns])
        return _clip(lgb.Booster(model_file=str(d / "model.txt")).predict(Xu))

    if arm == "delay_twostage_antigravity":
        with open(d / "model.pkl", "rb") as fh:
            return _clip(pickle.load(fh).predict(Xa))

    if arm == "multimodel_antigravity":
        # Per-airport blend of per_airport and mlp_antigravity; rebuild it from
        # the weights it recorded (same lever as cross_blend, different parents).
        from pipeline_per_airport.model import AIRPORT_COL
        m = json.loads((d / "model.json").read_text())
        weights_per_airport = m["weights_per_airport"]
        p_ap = predict_arm("per_airport", split, part, featureset, seed, outlier_policy,
                           X=X, meta=meta, Xtr=Xtr)
        p_mlp = predict_arm("mlp_antigravity", split, part, featureset, seed, outlier_policy,
                            X=X, meta=meta, Xtr=Xtr)
        ap_arr = meta[AIRPORT_COL].astype("object").to_numpy()
        pred = np.zeros(len(ap_arr), dtype="float64")
        for a in pd.unique(ap_arr):
            sel = (ap_arr == a)
            w = weights_per_airport.get(str(a), 0.5)
            pred[sel] = w * p_ap[sel] + (1.0 - w) * p_mlp[sel]
        return _clip(pred)

    if arm == "airport_operator_multimodel":
        # Per-airport blend of airport_operator_blend and mlp_antigravity; same
        # lever as multimodel_antigravity with a different tree-side parent.
        from pipeline_per_airport.model import AIRPORT_COL
        m = json.loads((d / "model.json").read_text())
        weights_per_airport = m["weights_per_airport"]
        p_aob = predict_arm("airport_operator_blend", split, part, featureset, seed, outlier_policy,
                            X=X, meta=meta, Xtr=Xtr, tag=m.get("tree_tag"))
        # The MLP seed set comes from the RUN's own model.json. Under --mlp-seeds
        # the per-airport weights were solved against an AVERAGED MLP, so pairing
        # them with a single draw returns a model that was never fitted or
        # validated -- silently, which is the worst way to be wrong. Absent the
        # key (a pre-msmlp run) this is [seed], the original single-draw path.
        #
        # train.py, evaluate.py and submit.py already read this list; this branch
        # did not, so predict_arm(..., tag="msmlp") disagreed with the arm's own
        # eval_*.json. Caught when a measured LIRF in-band RMSE came back 434.3
        # for a blend whose LIRF alpha is exactly 0.0 -- i.e. pure MLP, which
        # scores 429.3. A convex blend at alpha=0 must equal its parent.
        mlp_seeds = m.get("mlp_seeds") or [seed]
        p_mlp = np.mean([predict_arm("mlp_antigravity", split, part, featureset, s,
                                     outlier_policy, X=X, meta=meta, Xtr=Xtr,
                                     tag=m.get("mlp_tag"))
                         for s in mlp_seeds], axis=0)
        ap_arr = meta[AIRPORT_COL].astype("object").to_numpy()
        pred = np.zeros(len(ap_arr), dtype="float64")
        for a in pd.unique(ap_arr):
            sel = (ap_arr == a)
            w = weights_per_airport.get(str(a), 0.5)
            pred[sel] = w * p_aob[sel] + (1.0 - w) * p_mlp[sel]
        return _clip(pred)

    if arm == "airport_lirf_identity_multimodel":
        # Zero-parameter override on top of `base_arm`'s prediction. Both the
        # base arm AND its tag come from this run's own model.json: the override
        # was validated against one specific base, and silently applying it to a
        # different one is the "swap a parent under frozen weights" failure.
        from pipeline_airport_lirf_identity_multimodel.train import apply_lirf_identity_override
        m = json.loads((d / "model.json").read_text())
        p_base = predict_arm(m["base_arm"], split, part, featureset, seed,
                             outlier_policy, X=X, meta=meta, Xtr=Xtr,
                             tag=m.get("base_tag"))
        ap_arr = meta["ADEP_mvt"].astype("object").to_numpy()
        pred, _ = apply_lirf_identity_override(
            p_base, Xa, ap_arr, m["threshold_sec"],
            band_hi=m.get("identity_band_hi_sec"))
        return _clip(pred)

    if arm == "tri_blend":
        # Global NNLS blend over per_airport, per_operator, mlp_antigravity;
        # weights recorded at train time (unconstrained sum, not renormalised).
        m = json.loads((d / "model.json").read_text())
        base_arms = m["base_arms"]
        weights = m["weights"]
        w = np.array([weights[a] for a in base_arms], dtype="float64")
        P = np.column_stack([
            predict_arm(a, split, part, featureset, seed, outlier_policy,
                       X=X, meta=meta, Xtr=Xtr)
            for a in base_arms])
        return _clip(P @ w)

    if arm == "per_airport_flighttype":
        with open(d / "model.pkl", "rb") as fh:
            model = pickle.load(fh)
        return _clip(model.predict(Xa, meta))

    if arm == "airport_wtc_multimodel":
        from pipeline_per_airport.model import AIRPORT_COL
        from pipeline_airport_wtc_multimodel.train import blend_with_cell_alpha
        m = json.loads((d / "model.json").read_text())
        p_aob = predict_arm("airport_operator_blend", split, part, featureset, seed, outlier_policy,
                            X=X, meta=meta, Xtr=Xtr)
        p_mlp = predict_arm("mlp_antigravity", split, part, featureset, seed, outlier_policy,
                            X=X, meta=meta, Xtr=Xtr)
        ap_arr = meta[AIRPORT_COL].astype("object").to_numpy()
        pred = blend_with_cell_alpha(p_aob, p_mlp, ap_arr, X, m["cell_alpha"], m["airport_alpha_fallback"])
        return _clip(pred)

    if arm == "airport_runway_multimodel":
        from pipeline_per_airport.model import AIRPORT_COL
        from pipeline_airport_runway_multimodel.train import blend_with_cell_alpha as blend_rwy
        m = json.loads((d / "model.json").read_text())
        p_aob = predict_arm("airport_operator_blend", split, part, featureset, seed, outlier_policy,
                            X=X, meta=meta, Xtr=Xtr)
        p_mlp = predict_arm("mlp_antigravity", split, part, featureset, seed, outlier_policy,
                            X=X, meta=meta, Xtr=Xtr)
        ap_arr = meta[AIRPORT_COL].astype("object").to_numpy()
        pred = blend_rwy(p_aob, p_mlp, ap_arr, X, m["primary_runway_map"],
                         m["cell_alpha"], m["airport_alpha_fallback"])
        return _clip(pred)

    if arm == "airport_wtc_rec_multimodel":
        from pipeline_per_airport.model import AIRPORT_COL
        from pipeline_airport_wtc_rec_multimodel.train import blend_with_cell_alpha
        m = json.loads((d / "model.json").read_text())
        p_aob = predict_arm("airport_operator_blend", split, part, featureset, seed, outlier_policy,
                            X=X, meta=meta, Xtr=Xtr)
        p_mlp = predict_arm("mlp_antigravity", split, part, featureset, seed, outlier_policy,
                            X=X, meta=meta, Xtr=Xtr)
        ap_arr = meta[AIRPORT_COL].astype("object").to_numpy()
        pred = blend_with_cell_alpha(p_aob, p_mlp, ap_arr, X, m["cell_alpha"], m["airport_alpha_fallback"])
        return _clip(pred)

    if arm == "airport_unsup_multimodel":
        from pipeline_per_airport.model import AIRPORT_COL
        from pipeline_airport_unsup_multimodel.train import assign_clusters, _bucket_key, blend_with_cell_alpha
        m = json.loads((d / "model.json").read_text())
        with open(d / "cluster_models.pkl", "rb") as fh:
            cmodels = pickle.load(fh)
        p_aob = predict_arm("airport_operator_blend", split, part, featureset, seed, outlier_policy,
                            X=X, meta=meta, Xtr=Xtr)
        p_mlp = predict_arm("mlp_antigravity", split, part, featureset, seed, outlier_policy,
                            X=X, meta=meta, Xtr=Xtr)
        ap_arr = meta[AIRPORT_COL].astype("object").to_numpy()
        clusters = assign_clusters(X, ap_arr, cmodels["scalers"], cmodels["kmeans_models"])
        cell_arr = _bucket_key(ap_arr, X, clusters)
        pred = blend_with_cell_alpha(p_aob, p_mlp, cell_arr, ap_arr, m["cell_alpha"], m["airport_alpha_fallback"], m.get("parent_alpha", {}))
        return _clip(pred)

    if arm == "airport_multiseed_multimodel":
        m = json.loads((d / "model.json").read_text())
        b_arm = m.get("base_arm", "airport_unsup_multimodel")
        seeds = m.get("ensemble_seeds", [42, 43, 44])
        preds = []
        for s in seeds:
            p_s = predict_arm(b_arm, split, part, featureset, s, outlier_policy,
                              X=X, meta=meta, Xtr=Xtr)
            preds.append(p_s)
        return _clip(np.mean(preds, axis=0))

    if arm == "airport_debiased_multimodel":
        from pipeline_airport_debiased_multimodel.train import apply_alignment, DEFAULT_THRESHOLD_SEC
        m = json.loads((d / "model.json").read_text())
        b_arm = m.get("base_arm", "airport_multiseed_multimodel")
        coef_ext = m.get("lirf_align_coef_ext", m.get("lirf_align_coef"))
        coef_mid = m.get("lirf_align_coef_mid")
        thresh_ext = float(m.get("threshold_ext_sec", m.get("threshold_sec", DEFAULT_THRESHOLD_SEC)))
        thresh_mid = m.get("threshold_mid_sec")
        if thresh_mid is not None:
            thresh_mid = float(thresh_mid)
        ap_arr = meta["ADEP_mvt"].astype("object").to_numpy()
        has_rec_arr = X["has_flight_record"].to_numpy()

        if "shrunk_alphas" in m:
            from pipeline_airport_debiased_multimodel.train import blend_post_aligned, get_multiseed_preds
            seeds = m.get("ensemble_seeds", [42, 43, 44])
            p_gbdt = get_multiseed_preds("airport_operator_blend", split, part, featureset, outlier_policy,
                                         X=X, meta=meta, Xtr=Xtr, seeds=seeds)
            p_mlp = get_multiseed_preds("mlp_antigravity", split, part, featureset, outlier_policy,
                                        X=X, meta=meta, Xtr=Xtr, seeds=seeds)
            p_base = blend_post_aligned(p_gbdt, p_mlp, ap_arr, m["shrunk_alphas"], m.get("global_alpha", 0.720))
        else:
            p_base = predict_arm(b_arm, split, part, featureset, seed, outlier_policy,
                                 X=X, meta=meta, Xtr=Xtr)

        pred = apply_alignment(p_base, X, ap_arr, has_rec_arr, coef_ext=coef_ext,
                               threshold_ext=thresh_ext, coef_mid=coef_mid, threshold_mid=thresh_mid)
        if "seasonal_slopes" in m and "doy_cos" in X.columns:
            from pipeline_airport_debiased_multimodel.train import apply_seasonal_calibration
            pred = apply_seasonal_calibration(pred, ap_arr, X["doy_cos"].to_numpy(),
                                              m["seasonal_slopes"], gamma=float(m.get("seasonal_gamma", 0.50)))
        return _clip(pred)

    if arm == "per_airport_mlp":
        import torch
        from torch.utils.data import DataLoader
        from pipeline_per_airport.model import AIRPORT_COL
        from pipeline_per_airport_mlp.model import AirportTabularMLP
        from pipeline_per_airport_mlp.train import TabularAirportDataset, encode_airport_data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dicts = torch.load(d / "models.pt", map_location="cpu")
        with open(d / "preprocessors.pkl", "rb") as fh:
            preps = pickle.load(fh)
        pred = np.zeros(len(Xa), dtype="float64")
        for a in sorted(preps.keys()):
            sel = (meta[AIRPORT_COL] == a)
            if sel.sum() == 0:
                continue
            X_ap = Xa[sel].drop(columns=[AIRPORT_COL], errors="ignore")
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
            pred[sel.to_numpy()] = np.concatenate(ap_preds)
        return _clip(pred)

    if arm == "per_airport_stack":
        from pipeline_per_airport_stack.evaluate import blend
        with open(d / "model.pkl", "rb") as fh:
            bundle = pickle.load(fh)
        return _clip(blend(bundle, Xa, meta))

    if arm == "cross_blend":
        # A blend of two other arms; rebuild it from the weights it recorded.
        from pipeline_cross_blend.train import apply_weights
        from pipeline_per_airport.model import AIRPORT_COL
        m = json.loads((d / "model.json").read_text())
        b = m["base_arms"]
        w = float(m["weights"]) if m["weight_mode"] == "global" else m["weights"]
        pa = predict_arm(b["a"], split, part, featureset, seed, outlier_policy,
                         X=X, meta=meta, Xtr=Xtr, tag=b["a_tag"])
        pb = predict_arm(b["b"], split, part, featureset, seed, outlier_policy,
                         X=X, meta=meta, Xtr=Xtr, tag=b["b_tag"])
        return _clip(apply_weights(pa, pb, w,
                                   meta[AIRPORT_COL].astype("object").to_numpy()))

    if arm == "ridge":
        with open(d / "model.pkl", "rb") as fh:
            return _clip(pickle.load(fh).predict(Xa))

    if arm == "naive_offblock":
        from pipeline_naive_offblock.evaluate import load_model
        from pipeline_naive_offblock.train import predict as naive_predict
        pred, _ = naive_predict(load_model(d), Xa, featureset)
        return _clip(pred)

    raise ValueError(
        "no prediction path for arm {!r}. Supported: {}. Add it here rather than "
        "in a pipeline module -- a per-pipeline loader is how the triage stage "
        "came to depend on a third-party arm's closed arm list."
        .format(arm, sorted(OWN_ARMS + THIRD_PARTY_ARMS)))
