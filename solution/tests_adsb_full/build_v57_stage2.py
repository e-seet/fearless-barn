#!/usr/bin/env python3
"""v57 stage 2 (round 10 variant K: 1500 trees, 63 leaves, lr 0.02) = the v48 recipe with K capacity. Was: v48 = v42 + corrector S (PREREGISTERED_v48.md PASS): the v46 corrector design (v46 inputs and params, 7 airports),
averaged over seeds 42-46, trained on all 7 held-out months (Jan, Jul, Aug, Sep, Oct, Nov, Dec 2025).

Writes model_v57/ (booster + config), v57_stage2_candidate.parquet, build_report_v47.txt. Reloads the saved booster
and re-predicts to check the file is reproducible from the saved model. Does not upload.
"""
import json, sys, filecmp
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import features
from common import dataset as D
from common import io as cio
from tests_tail_diagnosis.diagnose import CACHE

cfg = json.loads((HERE / "model_v46" / "config.json").read_text())
A7, FEAT_A, FEAT_ADSB, PARAMS = cfg["airports"], cfg["features"], cfg["adsb_features"], cfg["params"]
PARAMS = {**PARAMS, "n_estimators": 1500, "num_leaves": 63, "learning_rate": 0.02, "min_data_in_leaf": 50}   # round 10 variant K
BOARD_V46 = 272.2619
SEEDS = (42, 43, 44, 45, 46)
lines = []
def say(s):
    print(s, flush=True); lines.append(s)


def est(prefixes):
    days = sorted(f.stem for f in (HERE / "points").glob("*.parquet") if f.stem[:7] in prefixes)
    e = pd.concat([features.load_day(d) for d in days]).sort_values("liftoff_dt")
    return e[~e.index.duplicated()], days


def to_str_cats(X):
    X = X.copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(str)
    return X


# ---------------- training: 7 held-out months ----------------
raw = cio.load_training(columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]).set_index("MVT_ID_mvt")
z = np.load(ROOT / "tests_ftt" / "preds_seed42_lr1e-3.npz")
ftt_sf = np.mean([np.load(ROOT / "tests_ftt" / f"preds_seed{s}_seasonal_full_lr1e-3.npz")["pt"] for s in range(42, 51)], axis=0)
ftt_fi = np.mean([np.load(ROOT / "tests_ftt" / f"preds_seed{s}_forward_inner_lr1e-3.npz")["pv"] for s in range(42, 51)], axis=0)
SURF = [("seasonal_full", "test", (np.load(CACHE / "tree_seasonal_full_test.npy") + np.load(CACHE / "mlp6_seasonal_full_test.npy") + ftt_sf) / 3, {"2025-01", "2025-07"}),
        ("forward_inner", "val", (np.load(CACHE / "tree_forward_inner_val.npy") + np.load(CACHE / "mlp18_forward_inner_val.npy") + ftt_fi) / 3, {"2025-08"}),
        ("forward", "val", (np.load(CACHE / "tree_forward_val.npy") + np.load(CACHE / "mlp12_forward_val.npy") + z["pv"]) / 3, {"2025-09", "2025-10"}),
        ("forward", "test", (np.load(CACHE / "tree_forward_test.npy") + np.load(CACHE / "mlp12_forward_test.npy") + z["pt"]) / 3, {"2025-11", "2025-12"})]
frames, all_days = [], []
for split, part, base, prefixes in SURF:
    X, y, meta = D.load_part(split, part, "with_aobt", filter_target=False)
    t = pd.to_datetime(raw.MVT_TIME_UTC_mvt.reindex(X.index), utc=True)
    df = to_str_cats(X)
    TAB = list(df.columns)
    df["y"] = np.asarray(y, float); df["base"] = base; df["ap"] = np.asarray(meta["ADEP_mvt"]).astype(str)
    df["day"] = np.asarray(t.dt.strftime("%Y-%m-%d")); df["hour"] = np.asarray(t.dt.hour)
    e, days = est(prefixes); all_days += days
    df = df[df.day.isin(days) & np.isfinite(df.y)].join(e[FEAT_ADSB])
    frames.append(df)
tr = pd.concat(frames); tr = tr[~tr.index.duplicated()]
tr["has"] = tr.e_first.notna() & tr.ap.isin(A7)
tr["apc"] = pd.Categorical(tr.ap.where(tr.ap.isin(A7)), categories=A7)
STRCOLS = [c for c in TAB if tr[c].dtype == object or pd.api.types.is_string_dtype(tr[c])]
LEVELS = {c: sorted(tr[c].dropna().astype(str).unique().tolist()) for c in STRCOLS}
for c in STRCOLS:
    tr[c] = pd.Categorical(tr[c].astype(str), categories=LEVELS[c])
FEAT_B = FEAT_A   # S: v46 inputs only (no tabular columns)
trc = tr[tr.has]
say(f"training: {len(all_days)} days, {len(tr)} rows, {len(trc)} covered at {A7}; {len(FEAT_B)} inputs")
MD = HERE / "model_v57"; MD.mkdir(exist_ok=True)
models = []
for sd in SEEDS:
    m_ = lgb.LGBMRegressor(**{**PARAMS, "random_state": sd}).fit(trc[FEAT_B], trc.y - trc.base)
    m_.booster_.save_model(str(MD / f"adsb_corrector_seed{sd}.txt")); models.append(m_)
(MD / "config.json").write_text(json.dumps({"airports": A7, "features": FEAT_B, "adsb_features": FEAT_ADSB, "tabular": TAB,
    "categorical_levels": LEVELS, "params": PARAMS, "seeds": list(SEEDS), "train_days": all_days, "featureset": "with_aobt",
    "base_file": "submissions/fearless_barn_v42.parquet"}, indent=1))

# ---------------- 2026 on top of v42 ----------------
v42 = pd.read_parquet(ROOT / "submissions" / "fearless_barn_v42.parquet")
Xr, mr = D.load_ranking_features("with_aobt")
ids = v42.MVT_ID_mvt.astype("int64").to_numpy()
assert (Xr.index.astype("int64").to_numpy() == ids).all(), "ranking feature order != v42 order"
te = to_str_cats(Xr)
for c in STRCOLS:
    te[c] = pd.Categorical(te[c].astype(str), categories=LEVELS[c])
te.index = ids
te["base"] = v42.TAXITIME_SEC_mvt.astype(float).to_numpy()
te["ap"] = np.asarray(mr["ADEP_mvt"]).astype(str)
te["hour"] = pd.to_datetime(mr["MVT_TIME_UTC_mvt"], utc=True).dt.hour.to_numpy()
e26, days26 = est({"2026-01", "2026-07"}); e26.index = e26.index.astype("int64")
te = te.join(e26[FEAT_ADSB])
te["has"] = te.e_first.notna() & te.ap.isin(A7)
te["apc"] = pd.Categorical(te.ap.where(te.ap.isin(A7)), categories=A7)
c = te.has.to_numpy()


def predict_with(booster_predict):
    corr = np.zeros(len(te)); corr[c] = booster_predict(te.loc[c, FEAT_B])
    return corr, np.clip(np.rint(te.base.to_numpy() + corr), 0, 86400).astype("int32")


corr, new = predict_with(lambda X: np.mean([m_.predict(X) for m_ in models], axis=0))
out = v42.copy(); out["TAXITIME_SEC_mvt"] = new
out.to_parquet(HERE / "v57_stage2_candidate.parquet", index=False)
# reproducibility: reload the saved booster and predict again
bs = [lgb.Booster(model_file=str(MD / f"adsb_corrector_seed{sd}.txt")) for sd in SEEDS]
_, new2 = predict_with(lambda X: np.mean([b.predict(X) for b in bs], axis=0))
out2 = v42.copy(); out2["TAXITIME_SEC_mvt"] = new2
out2.to_parquet(HERE / "v57_stage2_rebuilt.parquet", index=False)
same = filecmp.cmp(HERE / "v57_stage2_candidate.parquet", HERE / "v57_stage2_rebuilt.parquet", shallow=False)
v46 = pd.read_parquet(ROOT / "submissions" / "fearless_barn_v46.parquet")
say(f"2026: {len(days26)} days; corrected rows {int(c.sum())} of {len(te)} ({c.mean():.1%}); "
    f"correction mean {corr[c].mean():+.1f}, mean |.| {np.abs(corr[c]).mean():.1f}, 1-99% [{np.percentile(corr[c], 1):.0f}, {np.percentile(corr[c], 99):.0f}]")
say(f"rows changed vs v42 {(new != v42.TAXITIME_SEC_mvt.to_numpy()).sum()}, vs v46 {(new != v46.TAXITIME_SEC_mvt.to_numpy()).sum()}; "
    f"mean |v48 - v46| on corrected rows {np.abs(new - v46.TAXITIME_SEC_mvt.to_numpy())[c].mean():.1f} s")
say(f"reload-from-saved-model rebuild byte-identical: {same}")
say("EXPECTATION (written before upload): 271.9-272.4 vs v46 272.2619 (a near-tie is likely). Basis: LOMO pooled -0.05 (S vs A), 7/7 months; "
    "plus Aug 2025 added to training. If above 272.2619: re-upload v46 last as a byte copy.")
(HERE / "build_report_v57_stage2.txt").write_text("\n".join(lines) + "\n")
