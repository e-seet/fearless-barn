#!/usr/bin/env python3
"""tests_febjun/PREREGISTERED.md: does adding Feb-Jun (held-out holdout_febjun base) to the corrector's training
improve the 7 scored months? A = v48 design trained on the other 6 scored months; A5 = same + Feb-Jun."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests_adsb_full"))
import features
from common import dataset as D
from common import io as cio
from tests_tail_diagnosis.diagnose import CACHE

cfg = json.loads((ROOT / "tests_adsb_full" / "model_v48" / "config.json").read_text())
A7, FEAT_A, FEAT_ADSB, PARAMS, SEEDS = cfg["airports"], cfg["features"], cfg["adsb_features"], cfg["params"], cfg["seeds"]
rm = lambda r: float(np.sqrt(np.mean(np.square(r))))
raw = cio.load_training(columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]).set_index("MVT_ID_mvt")
P = HERE / "preds"
z = np.load(ROOT / "tests_ftt" / "preds_seed42_lr1e-3.npz")
ftt_sf = np.mean([np.load(ROOT / "tests_ftt" / f"preds_seed{s}_seasonal_full_lr1e-3.npz")["pt"] for s in range(42, 51)], axis=0)
ftt_fi = np.mean([np.load(ROOT / "tests_ftt" / f"preds_seed{s}_forward_inner_lr1e-3.npz")["pv"] for s in range(42, 51)], axis=0)
ftt_fj = np.mean([np.load(ROOT / "tests_ftt" / f"preds_seed{s}_holdout_febjun_lr1e-3.npz")["pt"] for s in (42, 43, 44)], axis=0)
mlp_fj = np.mean([np.load(P / f"mlp_holdout_febjun_test_seed{s}.npy") for s in range(42, 48)], axis=0)
SURF = [("seasonal_full", "test", (np.load(CACHE / "tree_seasonal_full_test.npy") + np.load(CACHE / "mlp6_seasonal_full_test.npy") + ftt_sf) / 3, {"2025-01", "2025-07"}),
        ("forward_inner", "val", (np.load(CACHE / "tree_forward_inner_val.npy") + np.load(CACHE / "mlp18_forward_inner_val.npy") + ftt_fi) / 3, {"2025-08"}),
        ("forward", "val", (np.load(CACHE / "tree_forward_val.npy") + np.load(CACHE / "mlp12_forward_val.npy") + z["pv"]) / 3, {"2025-09", "2025-10"}),
        ("forward", "test", (np.load(CACHE / "tree_forward_test.npy") + np.load(CACHE / "mlp12_forward_test.npy") + z["pt"]) / 3, {"2025-11", "2025-12"}),
        ("holdout_febjun", "test", (np.load(P / "tree_holdout_febjun_test.npy") + mlp_fj + ftt_fj) / 3, {"2025-02", "2025-03", "2025-04", "2025-05", "2025-06"})]
frames = []
for split, part, base, prefixes in SURF:
    X, y, meta = D.load_part(split, part, "with_aobt", filter_target=False)
    assert len(base) == len(X), (split, part)
    t = pd.to_datetime(raw.MVT_TIME_UTC_mvt.reindex(X.index), utc=True)
    df = pd.DataFrame({"y": np.asarray(y, float), "base": base, "ap": np.asarray(meta["ADEP_mvt"]).astype(str),
                       "day": np.asarray(t.dt.strftime("%Y-%m-%d")), "hour": np.asarray(t.dt.hour)}, index=X.index)
    days = sorted(f.stem for f in (ROOT / "tests_adsb_full" / "points").glob("*.parquet") if f.stem[:7] in prefixes)
    e = pd.concat([features.load_day(d) for d in days]).sort_values("liftoff_dt"); e = e[~e.index.duplicated()]
    frames.append(df[df.day.isin(days) & np.isfinite(df.y)].join(e[FEAT_ADSB]))
df = pd.concat(frames); df = df[~df.index.duplicated()]
df["mon"] = df.day.str[5:7]
df["covA"] = df.e_first.notna() & df.ap.isin(A7)
df["apc"] = pd.Categorical(df.ap.where(df.ap.isin(A7)), categories=A7)
FJ = ["02", "03", "04", "05", "06"]
print("Feb-Jun base RMSE", round(rm((df.y - df.base)[df.mon.isin(FJ)]), 2), "| covered share by month:",
      df.groupby("mon").covA.mean().round(2).to_dict())


def corr(trc, te):
    c = te.covA.to_numpy(); p = te.base.to_numpy().copy()
    p[c] += np.mean([lgb.LGBMRegressor(**{**PARAMS, "random_state": s}).fit(trc[FEAT_A], trc.y - trc.base).predict(te.loc[c, FEAT_A]) for s in SEEDS], axis=0)
    return p


res, se, gains = {}, {"A": [], "A5": []}, []
for mon in ["01", "07", "08", "09", "10", "11", "12"]:
    te = df[df.mon == mon]
    trA = df[~df.mon.isin(FJ + [mon]) & df.covA]
    trB = df[(df.mon != mon) & df.covA]                      # + Feb-Jun
    pA, pB = corr(trA, te), corr(trB, te)
    res[mon] = {"A": rm(te.y - pA), "A5": rm(te.y - pB), "n_train_A": len(trA), "n_train_A5": len(trB)}
    se["A"].append(((te.y - pA) ** 2).to_numpy()); se["A5"].append(((te.y - pB) ** 2).to_numpy())
    gains.append(((te.y - pA) ** 2 - (te.y - pB) ** 2).to_numpy())
    print(f"  month {mon}: A {res[mon]['A']:.3f} | A5 {res[mon]['A5']:.3f} ({res[mon]['A5'] - res[mon]['A']:+.3f}) | "
          f"training rows {len(trA)} -> {len(trB)}", flush=True)
pooled = {k: float(np.sqrt(np.concatenate(v).mean())) for k, v in se.items()}
g = np.concatenate(gains); seA = np.concatenate(se["A"])
keep = np.ones(len(g), bool); keep[np.argsort(-g)[:5]] = False
drop5 = (float(np.sqrt(seA[keep].mean())), float(np.sqrt((seA - g)[keep].mean())))
wins = sum(r["A5"] < r["A"] for r in res.values())
ok = pooled["A5"] < pooled["A"] and wins >= 5 and drop5[1] < drop5[0]
print(f"\nPOOLED: A {pooled['A']:.3f} | A5 {pooled['A5']:.3f} ({pooled['A5'] - pooled['A']:+.3f}) | {wins}/7 months | "
      f"best 5 removed {drop5[0]:.3f} -> {drop5[1]:.3f}")
print(f"PRE-REGISTERED => {'PASS' if ok else 'FAIL'}")
(HERE / "lomo_febjun.json").write_text(json.dumps({"months": res, "pooled": pooled, "wins": wins, "drop5": drop5, "PASS": bool(ok)}, indent=2))
