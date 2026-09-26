#!/usr/bin/env python3
"""Predict taxi-out time for ANY movements file with the shipped recipe, from saved models.

The public scored set is re-predicted by `submit.py`, which replays some members from
predictions cached for that set (FTT and TabNet `.npz`). A hidden final stage would bring
new rows, so this script predicts every member from its *saved model*:

  tree   airport_operator_blend (full)       -> submit.predict_ranking, same code path
  MLP    mlp_antigravity, recorded seeds      -> submit.predict_ranking, same code path
  FTT    tests_ftt/model_seed{s}_full_{tag}.pt, the recorded members and weights
  TabNet tests_tabnet/build/rebuilt_final_seed{s}_model.zip + _encoders.pkl (--tabnet models)
         or the cached npz, valid only for the current scored set (--tabnet npz)

then applies the recorded per-airport weights and the LIRF TabNet override.

Input: a parquet with ranking.parquet's schema (movements; DEP rows need not carry the
target, which is blanked anyway). Output: (MVT_ID_mvt, TAXITIME_SEC_mvt) for every DEP row,
or for the ids in --ids (e.g. a submitting template), in that order.

usage:
  python predict_new.py --movements <file.parquet> --out <pred.parquet> [--ids <template.parquet>]
  python predict_new.py --selftest            # current ranking.parquet vs the uploaded file
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config
import submit
from common import dataset as D
from common import features as F
from common import io as cio
from common.bootstrap import run_dir
from pipeline_ftt_multimodel.train import blend_per_airport
from pipeline_mlp_antigravity.train import encode_data
from pipeline_tabnet_lirf_override.train import apply_override, seed_preds_path

FS, POL, SPLIT, SEED = "with_aobt", "keep_all", "full", 42
DEFAULT_RECIPE = "lirf_w0.459_c2"          # v36 / v38


def build_features(path: Path, ids=None):
    mov = cio._with_month(cio.load_raw([path]))
    mov = cio.apply_ranking_contract(mov)
    cio.assert_ranking_computable(F.source_columns(FS), name=f"{path.name}/{FS}")
    X = F.build_features(mov, featureset=FS)
    X.insert(0, config.ID, mov.loc[X.index, config.ID].to_numpy())
    dep = mov[mov["PHASE_mvt"].eq("DEP")]
    ids = dep[config.ID].to_numpy() if ids is None else np.asarray(ids)
    X = X.set_index(config.ID).reindex(ids)
    if X.isna().all(axis=1).any():
        raise SystemExit(f"{int(X.isna().all(axis=1).sum())} requested ids have no DEP row in {path}")
    meta = dep.set_index(config.ID).reindex(ids)
    return X, meta


def run_json(arm, tag):
    d = run_dir(config.ARTIFACTS / "runs", arm, protocol=config.protocol_key(SPLIT, FS, POL),
                seed=SEED, extra={"tag": tag}, create=False)
    return json.loads((d / "model.json").read_text())


def ftt_member(seed, tag, X, Xtr):
    import torch
    from tests_ftt.model import FTTransformer
    ck = torch.load(ROOT / "tests_ftt" / f"model_seed{seed}_full_{tag}.pt", map_location="cpu", weights_only=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = FTTransformer(ck["card"], num_features=len(ck["num_cols"]), d_token=ck.get("d_token", 64),
                      n_blocks=ck.get("n_blocks", 3), n_heads=ck.get("n_heads", 8)).to(dev)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    cat, num = encode_data(D.align_columns(Xtr, X), ck["cat_cols"], ck["num_cols"], ck["cat_maps"],
                           ck["scaler"], ck["medians"])
    amp = "d192" in tag                    # the d192 members were trained and predicted under bf16 autocast
    out = []
    with torch.no_grad():
        for i in range(0, len(num), 4096):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp and dev.type == "cuda"):
                o = m({c: t[i:i + 4096].to(dev) for c, t in cat.items()}, num[i:i + 4096].to(dev))
            out.append(o.float().cpu().numpy())
    return np.clip(np.concatenate(out), config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)


def tabnet_member(seed, X, Xtr):
    from pytorch_tabnet.tab_model import TabNetRegressor
    b = ROOT / "tests_tabnet" / "build"
    m = TabNetRegressor()
    m.load_model(str(b / f"rebuilt_final_seed{seed}_model.zip"))
    with open(b / f"rebuilt_final_seed{seed}_encoders.pkl", "rb") as fh:
        e = pickle.load(fh)
    Xa = D.align_columns(Xtr, X)
    ca = np.empty((len(Xa), len(e["cat_cols"])), dtype="int64")
    for j, c in enumerate(e["cat_cols"]):
        ca[:, j] = Xa[c].astype("object").map(e["maps"][c]).fillna(len(e["maps"][c])).to_numpy(dtype="int64")
    na = e["scaler"].transform(Xa[e["num_cols"]].astype("float64").fillna(e["medians"]).to_numpy()).astype("float32")
    p = m.predict(np.concatenate([ca, na], axis=1)).reshape(-1)
    return np.clip(p, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)


def predict(X, meta, recipe=DEFAULT_RECIPE, tabnet="models"):
    ov = run_json("tabnet_lirf_override", recipe)
    base = run_json(ov["base_arm"], ov["base_tag"])
    assert ov["base_arm"] == "ftt_multimodel", "only the ftt_multimodel base is supported"
    # feed the new frames through submit.py's own tree/MLP code path
    D.load_ranking_features = lambda fs: (X.copy(), meta.copy())
    tree, _ = submit.predict_ranking("airport_operator_blend", SPLIT, FS, SEED, POL)
    mlp = np.mean([submit.predict_ranking("mlp_antigravity", SPLIT, FS, s, POL)[0] for s in base["mlp_seeds"]], axis=0)
    Xtr, _, _ = D.load_part(SPLIT, "train", FS, filter_target=True, outlier_policy=POL)
    members = base.get("ftt_members") or [[s, base.get("ftt_tag", "lr1e-3")] for s in base["ftt_seeds"]]
    fp = [ftt_member(int(s), tg, X, Xtr) for s, tg in members]
    mw = base.get("ftt_member_weights")
    ftt = np.average(fp, axis=0, weights=mw) if mw else np.mean(fp, axis=0)
    ap = meta["ADEP_mvt"].astype(str).to_numpy()
    blend = blend_per_airport(ftt, tree, mlp, ap, base["weights_per_airport"])
    if tabnet == "models":
        tab = np.mean([tabnet_member(int(s), X, Xtr) for s in ov["tabnet_seeds"]], axis=0)
    else:
        zs = [np.load(seed_preds_path(int(s)), allow_pickle=True) for s in ov["tabnet_seeds"]]
        if not all(np.array_equal(z["rank_ids"], X.index.to_numpy()) for z in zs):
            raise SystemExit("--tabnet npz only works for the current scored set")
        tab = np.mean([z["pr"] for z in zs], axis=0)
    pred, _ = apply_override(blend, tab, ap, float(ov["w"]))
    return pd.DataFrame({config.ID: X.index.to_numpy(), config.TARGET: np.rint(pred).astype("int32")}), \
        {"ftt": ftt, "tree": tree, "mlp": mlp, "tabnet": tab}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--movements", type=Path, default=config.RANKING_PARQUET)
    ap.add_argument("--ids", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--recipe", default=DEFAULT_RECIPE)
    ap.add_argument("--tabnet", choices=["models", "npz"], default="models")
    ap.add_argument("--selftest", action="store_true",
                    help="predict current ranking.parquet from scratch and compare with the uploaded file")
    ap.add_argument("--selftest-version", type=int, default=36)
    a = ap.parse_args()
    ids = pd.read_parquet(a.ids)[config.ID].to_numpy() if a.ids else None
    if a.selftest:
        ids = pd.read_parquet(config.SUBMITTING_PARQUET)[config.ID].to_numpy()
    X, meta = build_features(a.movements, ids)
    print(f"features built from {a.movements.name}: {X.shape}", flush=True)
    sub, parts = predict(X, meta, a.recipe, a.tabnet)
    if a.selftest:
        up = pd.read_parquet(ROOT / "submissions" / f"fearless_barn_v{a.selftest_version}.parquet")
        m = sub.merge(up, on=config.ID, suffixes=("_new", "_up"))
        d = (m[f"{config.TARGET}_new"] - m[f"{config.TARGET}_up"]).to_numpy()
        L = meta["ADEP_mvt"].astype(str).reindex(m[config.ID].to_numpy()).to_numpy() == "LIRF"
        print(f"SELFTEST vs v{a.selftest_version}: identical {(d == 0).sum()} / {len(d)} | LIRF identical "
              f"{(d[L] == 0).sum()} / {L.sum()} | non-LIRF identical {(d[~L] == 0).sum()} / {(~L).sum()} | "
              f"max |diff| {np.abs(d).max()} (tabnet from {a.tabnet})")
    if a.out:
        errs = submit.validate(sub) if a.selftest else []
        sub.to_parquet(a.out, index=False)
        print(f"wrote {a.out} ({len(sub)} rows) {errs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
