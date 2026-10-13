#!/usr/bin/env python3
"""PREREGISTERED_tree.md decision: X (old tree base + corrector on old bases) vs Y (adsb tree base + corrector on
new bases), both correctors trained on Aug-Dec 2025 only, scored on seasonal_full Jan + Jul 2025."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import features
from common import dataset as D
from common import io as cio
from tests_tail_diagnosis.diagnose import CACHE

cfg = json.loads((HERE / "model_v47" / "config.json").read_text())
A7, FEAT_B, FEAT_ADSB, PARAMS = cfg["airports"], cfg["features"], cfg["adsb_features"], cfg["params"]
TP = HERE / "tree_preds"
rm = lambda r: float(np.sqrt(np.mean(np.square(r))))
raw = cio.load_training(columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]).set_index("MVT_ID_mvt")
z = np.load(ROOT / "tests_ftt" / "preds_seed42_lr1e-3.npz")
ftt_sf = np.mean([np.load(ROOT / "tests_ftt" / f"preds_seed{s}_seasonal_full_lr1e-3.npz")["pt"] for s in range(42, 51)], axis=0)
ftt_fi = np.mean([np.load(ROOT / "tests_ftt" / f"preds_seed{s}_forward_inner_lr1e-3.npz")["pv"] for s in range(42, 51)], axis=0)
NN = {("seasonal_full", "test"): np.load(CACHE / "mlp6_seasonal_full_test.npy") + ftt_sf,
      ("forward_inner", "val"): np.load(CACHE / "mlp18_forward_inner_val.npy") + ftt_fi,
      ("forward", "val"): np.load(CACHE / "mlp12_forward_val.npy") + z["pv"],
      ("forward", "test"): np.load(CACHE / "mlp12_forward_test.npy") + z["pt"]}
PREF = {("seasonal_full", "test"): {"2025-01", "2025-07"}, ("forward_inner", "val"): {"2025-08"},
        ("forward", "val"): {"2025-09", "2025-10"}, ("forward", "test"): {"2025-11", "2025-12"}}

# fairness control: the regenerated old tree must match the cached tree the earlier rounds used
for (s, p), cname in ((("seasonal_full", "test"), "tree_seasonal_full_test"), (("forward", "test"), "tree_forward_test")):
    d = np.abs(np.load(TP / f"old_{s}_{p}.npy") - np.load(CACHE / f"{cname}.npy"))
    print(f"control: regenerated old tree vs cache on {s} {p}: max |diff| {d.max():.4f}")

frames = []
for (split, part), pref in PREF.items():
    X, y, meta = D.load_part(split, part, "with_aobt", filter_target=False)
    t = pd.to_datetime(raw.MVT_TIME_UTC_mvt.reindex(X.index), utc=True)
    df = X.copy()
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype(str)
    TAB = list(X.columns)
    df["y"] = np.asarray(y, float); df["ap"] = np.asarray(meta["ADEP_mvt"]).astype(str)
    df["day"] = np.asarray(t.dt.strftime("%Y-%m-%d")); df["hour"] = np.asarray(t.dt.hour)
    df["tree_old"] = np.load(TP / f"old_{split}_{part}.npy"); df["tree_new"] = np.load(TP / f"adsb_{split}_{part}.npy")
    df["base_X"] = (df.tree_old + NN[(split, part)]) / 3; df["base_Y"] = (df.tree_new + NN[(split, part)]) / 3
    days = sorted(f.stem for f in (HERE / "points").glob("*.parquet") if f.stem[:7] in pref)
    e = pd.concat([features.load_day(d) for d in days]).sort_values("liftoff_dt"); e = e[~e.index.duplicated()]
    df = df[df.day.isin(days) & np.isfinite(df.y)].join(e[FEAT_ADSB])
    frames.append(df)
df = pd.concat(frames); df = df[~df.index.duplicated()]
df["mon"] = df.day.str[5:7]
df["has"] = df.e_first.notna() & df.ap.isin(A7)
for c in TAB:
    if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
        df[c] = df[c].astype("category")
df["apc"] = pd.Categorical(df.ap.where(df.ap.isin(A7)), categories=A7)
tr, te = df[df.mon.isin(["08", "09", "10", "11", "12"])], df[df.mon.isin(["01", "07"])]
print(f"corrector training Aug-Dec: {len(tr)} rows ({int(tr.has.sum())} covered); test Jan+Jul: {len(te)} rows")


def corrected(bcol):
    t = tr[tr.has].copy(); t["base"] = t[bcol]
    m = lgb.LGBMRegressor(**PARAMS).fit(t[FEAT_B], t.y - t.base)
    s = te.copy(); s["base"] = s[bcol]
    p = s.base.to_numpy().copy(); c = s.has.to_numpy()
    p[c] += m.predict(s.loc[c, FEAT_B])
    return p


pX, pY = corrected("base_X"), corrected("base_Y")
res = {}
for lab, m in (("pooled", np.ones(len(te), bool)), ("Jan", (te.mon == "01").to_numpy()), ("Jul", (te.mon == "07").to_numpy())):
    y = te.y.to_numpy()[m]
    res[lab] = {"tree_old": rm(y - te.tree_old.to_numpy()[m]), "tree_new": rm(y - te.tree_new.to_numpy()[m]),
                "base_X": rm(y - te.base_X.to_numpy()[m]), "base_Y": rm(y - te.base_Y.to_numpy()[m]),
                "X": rm(y - pX[m]), "Y": rm(y - pY[m])}
    r = res[lab]
    print(f"{lab:6s}: tree old {r['tree_old']:.2f} new {r['tree_new']:.2f} | base X {r['base_X']:.2f} Y(no corrector) {r['base_Y']:.2f} | "
          f"X {r['X']:.2f}  Y {r['Y']:.2f} ({r['Y'] - r['X']:+.2f})")
ok = all(res[k]["Y"] < res[k]["X"] for k in ("pooled", "Jan", "Jul"))
res["PASS"] = bool(ok)
print(f"PRE-REGISTERED: Y < X pooled, Jan and Jul => {'PASS' if ok else 'FAIL'}")
(HERE / "evaluate_tree.json").write_text(json.dumps(res, indent=2))
