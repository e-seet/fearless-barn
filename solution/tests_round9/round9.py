#!/usr/bin/env python3
"""tests_round9/PREREGISTERED.md: A vs R (record rows only), C (+LFPG corrector), G (EGLL own model); LOMO over 7 months."""
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
A7, FEAT_A, FEAT_ADSB, PARAMS, SEEDS = cfg["airports"], cfg["features"], cfg["adsb_features"], cfg["params"], cfg["seeds"]
FEAT_1 = FEAT_ADSB + ["base", "hour"]          # single-airport models: no airport column
rm = lambda r: float(np.sqrt(np.mean(np.square(r))))


def frame():
    raw = cio.load_training(columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "FLIGHT_ID_mvt"]).drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
    z = np.load(ROOT / "tests_ftt" / "preds_seed42_lr1e-3.npz")
    ftt_sf = np.mean([np.load(ROOT / "tests_ftt" / f"preds_seed{s}_seasonal_full_lr1e-3.npz")["pt"] for s in range(42, 51)], axis=0)
    ftt_fi = np.mean([np.load(ROOT / "tests_ftt" / f"preds_seed{s}_forward_inner_lr1e-3.npz")["pv"] for s in range(42, 51)], axis=0)
    SURF = [("seasonal_full", "test", (np.load(CACHE / "tree_seasonal_full_test.npy") + np.load(CACHE / "mlp6_seasonal_full_test.npy") + ftt_sf) / 3, {"2025-01", "2025-07"}),
            ("forward_inner", "val", (np.load(CACHE / "tree_forward_inner_val.npy") + np.load(CACHE / "mlp18_forward_inner_val.npy") + ftt_fi) / 3, {"2025-08"}),
            ("forward", "val", (np.load(CACHE / "tree_forward_val.npy") + np.load(CACHE / "mlp12_forward_val.npy") + z["pv"]) / 3, {"2025-09", "2025-10"}),
            ("forward", "test", (np.load(CACHE / "tree_forward_test.npy") + np.load(CACHE / "mlp12_forward_test.npy") + z["pt"]) / 3, {"2025-11", "2025-12"})]
    frames = []
    for split, part, base, prefixes in SURF:
        X, y, meta = D.load_part(split, part, "with_aobt", filter_target=False)
        t = pd.to_datetime(raw.MVT_TIME_UTC_mvt.reindex(X.index), utc=True)
        df = pd.DataFrame({"y": np.asarray(y, float), "base": base, "ap": np.asarray(meta["ADEP_mvt"]).astype(str),
                           "day": np.asarray(t.dt.strftime("%Y-%m-%d")), "hour": np.asarray(t.dt.hour),
                           "rec": raw.FLIGHT_ID_mvt.reindex(X.index).notna().to_numpy()}, index=X.index)
        days = sorted(f.stem for f in (ROOT / "tests_adsb_full" / "points").glob("*.parquet") if f.stem[:7] in prefixes)
        e = pd.concat([features.load_day(d) for d in days]).sort_values("liftoff_dt"); e = e[~e.index.duplicated()]
        frames.append(df[df.day.isin(days) & np.isfinite(df.y)].join(e[FEAT_ADSB]))
    df = pd.concat(frames); df = df[~df.index.duplicated()]
    df["mon"] = df.day.str[5:7]
    df["trk"] = df.e_first.notna()
    df["apc"] = pd.Categorical(df.ap.where(df.ap.isin(A7)), categories=A7)
    return df


def fit_predict(tr, te, feat, train_mask, apply_mask):
    trc = tr[train_mask(tr)]; c = apply_mask(te).to_numpy(); out = np.zeros(len(te))
    if c.any() and len(trc) >= 200:
        out[c] = np.mean([lgb.LGBMRegressor(**{**PARAMS, "random_state": s}).fit(trc[feat], trc.y - trc.base).predict(te.loc[c, feat]) for s in SEEDS], axis=0)
    return out


def shuffled(part, mask_fn, seed):
    part = part.copy(); rng = np.random.default_rng(seed); idx = np.flatnonzero(mask_fn(part).to_numpy())
    part.iloc[idx, [part.columns.get_loc(c) for c in FEAT_ADSB]] = part.iloc[rng.permutation(idx)][FEAT_ADSB].to_numpy()
    return part


in7 = lambda d: d.ap.isin(A7) & d.trk
in7_rec = lambda d: d.ap.isin(A7) & d.trk & d.rec
in6 = lambda d: d.ap.isin([a for a in A7 if a != "EGLL"]) & d.trk
lfpg = lambda d: d.ap.eq("LFPG") & d.trk & d.rec
egll = lambda d: d.ap.eq("EGLL") & d.trk
egll_rec = lambda d: d.ap.eq("EGLL") & d.trk & d.rec

VARIANTS = {
    "A": lambda tr, te: fit_predict(tr, te, FEAT_A, in7, in7),
    "R": lambda tr, te: fit_predict(tr, te, FEAT_A, in7_rec, in7_rec),
    "C": lambda tr, te: fit_predict(tr, te, FEAT_A, in7, in7) + fit_predict(tr, te, FEAT_1, lfpg, lfpg),
    "G": lambda tr, te: fit_predict(tr, te, FEAT_A, in6, in6) + fit_predict(tr, te, FEAT_1, egll, egll),
    "RG": lambda tr, te: (fit_predict(tr, te, FEAT_A, lambda d: in6(d) & d.rec, lambda d: in6(d) & d.rec)
                          + fit_predict(tr, te, FEAT_1, egll_rec, egll_rec)),
}
CHANGED = {"R": in7, "C": lfpg, "G": egll}        # rows whose prediction the test changes (for its permutation control)

if __name__ == "__main__":
    df = frame()
    print(f"{len(df)} rows; tracked at 7 airports {int(in7(df).sum())} (no record {int((in7(df) & ~df.rec).sum())}); "
          f"LFPG eligible {int(lfpg(df).sum())} by month {df[lfpg(df)].mon.value_counts().sort_index().to_dict()}; EGLL tracked {int(egll(df).sum())}")
    months = ["01", "07", "08", "09", "10", "11", "12"]
    pred = {k: {} for k in list(VARIANTS) + [f"{t}_perm{i}" for t in CHANGED for i in range(3)]}
    for mon in months:
        tr, te = df[df.mon != mon], df[df.mon == mon]
        for k, f in VARIANTS.items():
            pred[k][mon] = te.base.to_numpy() + f(tr, te)
        for t, m in CHANGED.items():
            for i in range(3):
                pred[f"{t}_perm{i}"][mon] = te.base.to_numpy() + VARIANTS[t](shuffled(tr, m, 10 + i), shuffled(te, m, 20 + i))
        print(f"  month {mon}: " + " | ".join(f"{k} {rm(te.y - pred[k][mon]):.3f}" for k in VARIANTS), flush=True)
    y = {mon: df[df.mon == mon].y.to_numpy() for mon in months}
    pooled = {k: float(np.sqrt(np.concatenate([(y[m] - pred[k][m]) ** 2 for m in months]).mean())) for k in pred}
    out = {"pooled": pooled, "tests": {}}
    for t in ("R", "C", "G"):
        per = {m: rm(y[m] - pred[t][m]) - rm(y[m] - pred["A"][m]) for m in months}
        changed_months = [m for m in months if np.any(pred[t][m] != pred["A"][m])]
        wins = sum(per[m] < 0 for m in (changed_months if t == "C" else months))
        need = (2 / 3 * len(changed_months)) if t == "C" else 5
        g = np.concatenate([(y[m] - pred["A"][m]) ** 2 - (y[m] - pred[t][m]) ** 2 for m in months])
        sA = np.concatenate([(y[m] - pred["A"][m]) ** 2 for m in months])
        keep = np.ones(len(g), bool); keep[np.argsort(-g)[:5]] = False
        drop5 = (float(np.sqrt(sA[keep].mean())), float(np.sqrt((sA - g)[keep].mean())))
        best_perm = min(pooled[f"{t}_perm{i}"] for i in range(3))
        ok = pooled[t] < pooled["A"] and wins >= need and len(changed_months) > 0 and drop5[1] < drop5[0] and pooled[t] < best_perm
        out["tests"][t] = {"pooled_delta": pooled[t] - pooled["A"], "per_month": per, "wins": wins, "months_counted": len(changed_months) if t == "C" else 7,
                           "drop5": drop5, "best_perm": best_perm, "PASS": bool(ok)}
        print(f"{t}: pooled {pooled[t] - pooled['A']:+.3f} | wins {wins}/{len(changed_months) if t == 'C' else 7} | best-5 removed {drop5[0]:.3f}->{drop5[1]:.3f} | "
              f"perm best {best_perm - pooled['A']:+.3f} => {'PASS' if ok else 'FAIL'}")
    print(f"\nPOOLED: A {pooled['A']:.3f} | R {pooled['R']:.3f} | C {pooled['C']:.3f} | G {pooled['G']:.3f} | RG {pooled['RG']:.3f}")
    if out["tests"]["R"]["PASS"] and out["tests"]["G"]["PASS"]:
        out["RG_beats_both"] = bool(pooled["RG"] < min(pooled["R"], pooled["G"]))
        print("RG combination beats R and G:", out["RG_beats_both"])
    (HERE / "round9.json").write_text(json.dumps(out, indent=2, default=float))
