#!/usr/bin/env python3
"""tests_weather_validated/PREREGISTERED.md: leave-one-month-out test of weather offsets on rows without ADS-B."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; REPO = ROOT.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests_adsb_full"))
import features
from common import dataset as D
from common import io as cio
from tests_tail_diagnosis.diagnose import CACHE

cfg = json.loads((ROOT / "tests_adsb_full" / "model_v48" / "config.json").read_text())
A7, FEAT_A, FEAT_ADSB, PARAMS, SEEDS = cfg["airports"], cfg["features"], cfg["adsb_features"], cfg["params"], cfg.get("seeds", [42])
COND = ["ts", "fz", "frost", "fog", "rain"]
K = 200.0
rm = lambda r: float(np.sqrt(np.mean(np.square(r))))
num = lambda s: pd.to_numeric(s.replace({"M": np.nan, "T": "0.0001"}), errors="coerce")


def weather_table():
    out = []
    for f in sorted((REPO / "prc-2026-datasets-wx").glob("[A-Z]*.csv")):
        if f.stem.startswith("_"):
            continue
        d = pd.read_csv(f, dtype=str)
        w = d.wxcodes.fillna("").str.upper()
        t = pd.DataFrame({"valid": pd.to_datetime(d.valid, utc=True, errors="coerce"), "station": f.stem,
                          "tmpf": num(d.tmpf), "dwpf": num(d.dwpf), "vsby": num(d.vsby),
                          "w_ts": w.str.contains("TS"), "w_fz": w.str.contains("FZ"),
                          "w_fog": w.str.contains(r"\bFG\b|\bBR\b|FZFG|MIFG", regex=True),
                          "w_rain": w.str.contains(r"\bRA\b|\bDZ\b|\+RA|-RA|\+DZ|-DZ", regex=True)})
        out.append(t.dropna(subset=["valid", "tmpf"]))
    return pd.concat(out).sort_values("valid")


def condition(ids, times, aps, wx):
    left = pd.DataFrame({"id": ids, "valid": times, "station": aps}).sort_values("valid")
    m = pd.merge_asof(left, wx, on="valid", by="station", direction="backward", tolerance=pd.Timedelta("90min")).set_index("id").reindex(ids)
    frost = ((m.tmpf <= 34) & ((m.tmpf - m.dwpf) <= 3)).fillna(False).to_numpy()
    ts = m.w_ts.fillna(False).astype(bool).to_numpy(); fz = m.w_fz.fillna(False).astype(bool).to_numpy()
    fog = (~frost) & (m.w_fog.fillna(False).astype(bool).to_numpy() | (m.vsby < 1).fillna(False).to_numpy())
    rain = (~ts) & m.w_rain.fillna(False).astype(bool).to_numpy()
    c = np.full(len(ids), "", dtype=object)
    for name, mask in (("rain", rain), ("fog", fog), ("frost", frost), ("fz", fz), ("ts", ts)):   # reverse order: first match wins
        c[mask] = name
    return c


if __name__ == "__main__":
    raw = cio.load_training(columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]).set_index("MVT_ID_mvt")
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
                           "time": t.to_numpy(), "day": np.asarray(t.dt.strftime("%Y-%m-%d")), "hour": np.asarray(t.dt.hour)}, index=X.index)
        days = sorted(f.stem for f in (ROOT / "tests_adsb_full" / "points").glob("*.parquet") if f.stem[:7] in prefixes)
        e = pd.concat([features.load_day(d) for d in days]).sort_values("liftoff_dt"); e = e[~e.index.duplicated()]
        frames.append(df[df.day.isin(days) & np.isfinite(df.y)].join(e[FEAT_ADSB]))
    df = pd.concat(frames); df = df[~df.index.duplicated()]
    df["mon"] = df.day.str[5:7]
    df["covA"] = df.e_first.notna() & df.ap.isin(A7)
    df["apc"] = pd.Categorical(df.ap.where(df.ap.isin(A7)), categories=A7)
    df["cond"] = condition(df.index.to_numpy(), pd.to_datetime(df.time, utc=True), df.ap.to_numpy(), weather_table())
    print("condition counts on uncovered rows:", df[~df.covA].cond.value_counts().to_dict())

    # A+ per fold (ADS-B corrector, v48 design), then the residual r = y - A+ used by the weather offsets
    df["pA"] = np.nan
    for mon in sorted(df.mon.unique()):
        tr, te = df[(df.mon != mon) & df.covA], df[df.mon == mon]
        c = te.covA.to_numpy(); p = te.base.to_numpy().copy()
        p[c] += np.mean([lgb.LGBMRegressor(**{**PARAMS, "random_state": s}).fit(tr[FEAT_A], tr.y - tr.base).predict(te.loc[c, FEAT_A]) for s in SEEDS], axis=0)
        df.loc[te.index, "pA"] = p
    df["r"] = df.y - df.pA
    unc = ~df.covA

    def fit_offsets(tr):
        u = tr[~tr.covA & tr.cond.ne("")]
        g = u.groupby("cond").r.agg(["mean", "size"])
        W = {c: float(g.loc[c, "mean"] * g.loc[c, "size"] / (g.loc[c, "size"] + K)) if c in g.index else 0.0 for c in COND}
        ga = u.groupby(["ap", "cond"]).r.agg(["mean", "size"])
        WA = {(a, c): float((row["mean"] * row["size"] + W[c] * K) / (row["size"] + K)) for (a, c), row in ga.iterrows()}
        return W, WA

    res, se = {}, {k: [] for k in ("A+", "V", "W", "WA")}
    gains = []
    for mon in sorted(df.mon.unique()):
        tr, te = df[df.mon != mon], df[df.mon == mon]
        W, WA = fit_offsets(tr)
        u = (~te.covA).to_numpy(); cnd = te.cond.to_numpy(); ap = te.ap.to_numpy()
        pV = te.pA.to_numpy().copy()
        if mon == "01": pV[u & (cnd == "frost")] += 30
        if mon == "07": pV[u & (cnd == "ts")] += 200
        pW = te.pA.to_numpy() + np.where(u, [W.get(c, 0.0) if c else 0.0 for c in cnd], 0.0)
        pWA = te.pA.to_numpy() + np.where(u, [WA.get((a, c), W.get(c, 0.0)) if c else 0.0 for a, c in zip(ap, cnd)], 0.0)
        P = {"A+": te.pA.to_numpy(), "V": pV, "W": pW, "WA": pWA}
        res[mon] = {k: rm(te.y - v) for k, v in P.items()}; res[mon]["W_offsets"] = W
        for k, v in P.items():
            se[k].append(((te.y - v) ** 2).to_numpy())
        gains.append(((te.y - P["A+"]) ** 2 - (te.y - pW) ** 2).to_numpy())
        r = res[mon]
        print(f"  month {mon}: A+ {r['A+']:.3f} | V {r['V']:.3f} ({r['V'] - r['A+']:+.3f}) | W {r['W']:.3f} ({r['W'] - r['A+']:+.3f}) | "
              f"WA {r['WA']:.3f} ({r['WA'] - r['A+']:+.3f}) | W offsets {({k: round(v) for k, v in W.items()})}", flush=True)
    pooled = {k: float(np.sqrt(np.concatenate(v).mean())) for k, v in se.items()}
    g = np.concatenate(gains); seA = np.concatenate(se["A+"])
    keep = np.ones(len(g), bool); keep[np.argsort(-g)[:5]] = False
    drop5 = (float(np.sqrt(seA[keep].mean())), float(np.sqrt((seA - g)[keep].mean())))
    wW = sum(r["W"] < r["A+"] for r in res.values()); wWA = sum(r["WA"] < r["W"] for r in res.values())
    W_pass = pooled["W"] < pooled["A+"] and wW >= 5 and drop5[1] < drop5[0]
    WA_adopt = W_pass and pooled["WA"] < pooled["W"] and wWA >= 5
    print(f"\nPOOLED: A+ {pooled['A+']:.3f} | V {pooled['V']:.3f} ({pooled['V'] - pooled['A+']:+.3f}) | W {pooled['W']:.3f} ({pooled['W'] - pooled['A+']:+.3f}) | WA {pooled['WA']:.3f} ({pooled['WA'] - pooled['A+']:+.3f})")
    print(f"best 5 rows removed (W): {drop5[0]:.3f} -> {drop5[1]:.3f}")
    print(f"W: pooled {pooled['W'] < pooled['A+']} | {wW}/7 months | best-5 removed {drop5[1] < drop5[0]} => {'PASS' if W_pass else 'FAIL'}; "
          f"WA over W: {wWA}/7 months => {'ADOPT WA' if WA_adopt else 'keep W' if W_pass else '-'}")
    Wall, WAall = fit_offsets(df)
    print("offsets fitted on all 7 months: W", {k: round(v, 1) for k, v in Wall.items()})
    (HERE / "weather_lomo.json").write_text(json.dumps({"months": res, "pooled": pooled, "drop5": drop5, "W_PASS": bool(W_pass), "WA_ADOPT": bool(WA_adopt),
        "W_all": Wall, "WA_all": {f"{a}|{c}": v for (a, c), v in WAall.items()}}, indent=2, default=float))
