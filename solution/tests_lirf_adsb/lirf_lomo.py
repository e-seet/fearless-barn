#!/usr/bin/env python3
"""tests_lirf_adsb/PREREGISTERED.md: LIRF ADS-B corrector on rows with an NM record, leave-one-month-out Jul-Dec 2025."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests_adsb_full"))
import features
from common import dataset as D
from common import io as cio
from tests_tail_diagnosis.diagnose import CACHE

cfg = json.loads((ROOT / "tests_adsb_full" / "model_v48" / "config.json").read_text())
FEAT_ADSB, PARAMS, SEEDS = cfg["adsb_features"], cfg["params"], cfg["seeds"]
FEAT = FEAT_ADSB + ["base", "hour"]
TW = 0.459
rm = lambda r: float(np.sqrt(np.mean(np.square(r))))
raw = cio.load_training(columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "FLIGHT_ID_mvt"]).drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
TB = ROOT / "tests_tabnet"
tab_sf = np.mean([np.load(TB / f"preds_seed{s}_sf_lr2e-3p.npz")["pt"] for s in (42, 43, 44)], axis=0)
tab_in = np.mean([np.load(TB / "build" / f"inner_seed{s}.npz")["pv"] for s in (42, 43, 44)], axis=0)
tab_fv = np.mean([np.load(TB / f"preds_seed{s}_lr2e-3p.npz")["pv"] for s in (42, 43, 44)], axis=0)
tab_ft = np.mean([np.load(TB / f"preds_seed{s}_lr2e-3p.npz")["pt"] for s in (42, 43, 44)], axis=0)
SURF = [("seasonal_full", "test", np.load(CACHE / "mlp6_seasonal_full_test.npy"), tab_sf, {"2025-07"}),
        ("forward_inner", "val", np.load(CACHE / "mlp18_forward_inner_val.npy"), tab_in, {"2025-08"}),
        ("forward", "val", np.load(CACHE / "mlp12_forward_val.npy"), tab_fv, {"2025-09", "2025-10"}),
        ("forward", "test", np.load(CACHE / "mlp12_forward_test.npy"), tab_ft, {"2025-11", "2025-12"})]
frames = []
for split, part, mlp, tab, prefixes in SURF:
    X, y, meta = D.load_part(split, part, "with_aobt", filter_target=False)
    assert len(mlp) == len(tab) == len(X), (split, part)
    t = pd.to_datetime(raw.MVT_TIME_UTC_mvt.reindex(X.index), utc=True)
    df = pd.DataFrame({"y": np.asarray(y, float), "base": (1 - TW) * mlp + TW * tab, "ap": np.asarray(meta["ADEP_mvt"]).astype(str),
                       "day": np.asarray(t.dt.strftime("%Y-%m-%d")), "hour": np.asarray(t.dt.hour),
                       "rec": raw.FLIGHT_ID_mvt.reindex(X.index).notna().to_numpy()}, index=X.index)
    df = df[df.ap.eq("LIRF")]
    days = sorted(f.stem for f in (ROOT / "tests_adsb_full" / "points").glob("*.parquet") if f.stem[:7] in prefixes)
    e = pd.concat([features.load_day(d) for d in days]).sort_values("liftoff_dt"); e = e[~e.index.duplicated()]
    frames.append(df[df.day.isin(days) & np.isfinite(df.y)].join(e[FEAT_ADSB]))
df = pd.concat(frames); df = df[~df.index.duplicated()]
df["mon"] = df.day.str[5:7]
df["elig"] = df.rec & df.e_first.notna()
print("LIRF rows by month:", df.groupby("mon").size().to_dict(), "| eligible share:", df.groupby("mon").elig.mean().round(2).to_dict())


def shuffled(part, seed):
    part = part.copy(); rng = np.random.default_rng(seed); idx = np.flatnonzero(part.elig.to_numpy())
    part.iloc[idx, [part.columns.get_loc(c) for c in FEAT_ADSB]] = part.iloc[rng.permutation(idx)][FEAT_ADSB].to_numpy()
    return part


def predict(tr, te):
    trc = tr[tr.elig]; c = te.elig.to_numpy(); p = te.base.to_numpy().copy()
    if c.any() and len(trc):
        p[c] += np.mean([lgb.LGBMRegressor(**{**PARAMS, "random_state": s}).fit(trc[FEAT], trc.y - trc.base).predict(te.loc[c, FEAT]) for s in SEEDS], axis=0)
    return p


res, se = {}, {"base": [], "corr": [], "perm0": [], "perm1": [], "perm2": []}
for mon in sorted(df.mon.unique()):
    tr, te = df[df.mon != mon], df[df.mon == mon]
    P = {"base": te.base.to_numpy(), "corr": predict(tr, te)}
    for k in range(3):
        P[f"perm{k}"] = predict(shuffled(tr, 10 + k), shuffled(te, 20 + k))
    res[mon] = {k: rm(te.y - v) for k, v in P.items()}; res[mon]["n_elig"] = int(te.elig.sum())
    for k, v in P.items():
        se[k].append(((te.y - v) ** 2).to_numpy())
    r = res[mon]
    print(f"  month {mon}: LIRF base {r['base']:.1f} -> corrected {r['corr']:.1f} ({r['corr'] - r['base']:+.1f}) | perms "
          f"{r['perm0']:.1f}/{r['perm1']:.1f}/{r['perm2']:.1f} | eligible rows {r['n_elig']}", flush=True)
pooled = {k: float(np.sqrt(np.concatenate(v).mean())) for k, v in se.items()}
g = np.concatenate(se["base"]) - np.concatenate(se["corr"]); s0 = np.concatenate(se["base"])
keep = np.ones(len(g), bool); keep[np.argsort(-g)[:5]] = False
drop5 = (float(np.sqrt(s0[keep].mean())), float(np.sqrt((s0 - g)[keep].mean())))
wins = sum(r["corr"] < r["base"] for r in res.values())
best_perm = min(pooled[f"perm{k}"] for k in range(3))
ok = pooled["corr"] < pooled["base"] and wins >= 4 and drop5[1] < drop5[0] and pooled["corr"] < best_perm
print(f"\nPOOLED LIRF: base {pooled['base']:.2f} -> corrected {pooled['corr']:.2f} ({pooled['corr'] - pooled['base']:+.2f}) | {wins}/6 months | "
      f"best 5 removed {drop5[0]:.2f} -> {drop5[1]:.2f} | best permutation {best_perm:.2f}")
print(f"PRE-REGISTERED => {'PASS' if ok else 'FAIL'}")
(HERE / "lirf_lomo.json").write_text(json.dumps({"months": res, "pooled": pooled, "wins": wins, "drop5": drop5, "PASS": bool(ok)}, indent=2))
