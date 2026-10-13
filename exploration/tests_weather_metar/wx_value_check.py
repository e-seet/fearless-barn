#!/usr/bin/env python3
"""Offline value check: does METAR explain forward/test residual variance that
the current (v22-class) model does not already capture?

No fingerprint change, no GPU. Uses existing trained runs only.
"""
from __future__ import annotations
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path("C:/Users/Admin/Desktop/eddie/fearless-barn/solution")
WX = Path("C:/Users/Admin/Desktop/eddie/fearless-barn/prc-2026-datasets-wx")
DATA = Path("C:/Users/Admin/Desktop/eddie/fearless-barn/prc-2026-datasets")
sys.path.insert(0, str(EXP))
import config
from common import dataset as D
from common.predict import predict_arm
from common.metrics import rmse

STATIONS = ["LTFM", "LFPG", "EGLL", "EHAM", "LEMD", "EDDF", "LEBL", "LIRF", "EDDM", "LSZH"]


# --------------------------------------------------------------------------- #
# 1. baseline predictions + residuals on forward/test
# --------------------------------------------------------------------------- #
print("loading forward/test features + baseline predictions ...")
Xtr, _, _ = D.load_part("forward", "train", "with_aobt", filter_target=True,
                        outlier_policy="keep_all")
X, y, meta = D.load_part("forward", "test", "with_aobt", filter_target=False)
y = pd.Series(y, index=X.index).astype("float64")

def try_pred(arm, tag=None):
    try:
        p = predict_arm(arm, "forward", "test", "with_aobt", 42, "keep_all",
                        X=X, y=y, meta=meta, Xtr=Xtr, tag=tag)
        print("  {:32s} tag={:10s} RMSE {:.2f}".format(arm, str(tag), rmse(y, p)))
        return np.asarray(p, dtype="float64")
    except Exception as e:
        print("  {:32s} tag={:10s} FAILED {}".format(arm, str(tag), repr(e)[:120]))
        return None

cands = [("airport_operator_multimodel", "msmlp"),
         ("airport_operator_multimodel", None),
         ("airport_operator_blend", None),
         ("per_airport", None)]
pred = None
for arm, tag in cands:
    p = try_pred(arm, tag)
    if p is not None and pred is None:
        pred, pred_name = p, "{}:{}".format(arm, tag)
print("using baseline:", pred_name)

resid = (y.to_numpy() - pred)                      # +ve => model under-predicts
adep = meta["ADEP_mvt"].astype("string").to_numpy()


# --------------------------------------------------------------------------- #
# 2. movement timestamps for the as-of join
# --------------------------------------------------------------------------- #
print("\nloading movement timestamps ...")
mv = []
for f in sorted(glob.glob(str(DATA / "training_2025-1[12]-01_*.parquet"))):
    d = pd.read_parquet(f, columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "PHASE_mvt"])
    mv.append(d[d.PHASE_mvt.eq("DEP")].drop(columns="PHASE_mvt"))
mv = pd.concat(mv, ignore_index=True).set_index("MVT_ID_mvt")
mt = mv["MVT_TIME_UTC_mvt"].reindex(X.index)
mt = pd.to_datetime(mt, utc=True)
print("  matched timestamps:", int(mt.notna().sum()), "/", len(X))


# --------------------------------------------------------------------------- #
# 3. parse METAR -> numeric weather + derived flags
# --------------------------------------------------------------------------- #
def num(s):
    return pd.to_numeric(s.replace({"M": np.nan, "T": "0.0001"}), errors="coerce")

def parse_station(st):
    fp = WX / (st + ".csv")
    d = pd.read_csv(fp, dtype=str)
    d["valid"] = pd.to_datetime(d["valid"], utc=True, errors="coerce")
    out = pd.DataFrame({"valid": d["valid"]})
    out["wx_tmpf"] = num(d["tmpf"])
    out["wx_vsby_mi"] = num(d["vsby"])
    out["wx_wind_kt"] = num(d["sknt"])
    g = num(d["gust"])
    out["wx_gust_kt"] = g.fillna(out["wx_wind_kt"])
    out["wx_gust_spread"] = (out["wx_gust_kt"] - out["wx_wind_kt"]).clip(lower=0)
    out["wx_relh"] = num(d["relh"])
    out["wx_p01i"] = num(d["p01i"]).fillna(0.0)
    skyc = d["skyc1"].fillna("").str.upper()
    skyl = num(d["skyl1"])
    low_cloud = skyc.isin(["BKN", "OVC", "VV"])
    out["wx_ceiling_ft"] = np.where(low_cloud, skyl, 25000.0)
    wx = d["wxcodes"].fillna("").str.upper()
    out["wx_has_precip"] = ((out["wx_p01i"] > 0) |
                            wx.str.contains("RA|SN|DZ|PL|GR|GS|SG|IC", regex=True)).astype(int)
    out["wx_snow_ice"] = wx.str.contains("SN|PL|GR|GS|SG|IC|FZ", regex=True).astype(int)
    out["wx_freezing"] = (wx.str.contains("FZ") | (out["wx_tmpf"] <= 32)).astype(int)
    out["wx_thunder"] = wx.str.contains("TS").astype(int)
    out["wx_fog_mist"] = wx.str.contains(r"\bFG\b|\bBR\b|FZFG|MIFG", regex=True).astype(int)
    out["wx_lowvis"] = (out["wx_vsby_mi"] < 3).astype(int)
    out["wx_ifr"] = ((out["wx_vsby_mi"] < 3) | (out["wx_ceiling_ft"] < 1000)).astype(int)
    out["wx_lifr"] = ((out["wx_vsby_mi"] < 1) | (out["wx_ceiling_ft"] < 500)).astype(int)
    out["wx_highwind"] = (out["wx_gust_kt"] >= 25).astype(int)
    out = out.dropna(subset=["valid"]).sort_values("valid")
    out["station"] = st
    return out

print("\nparsing METAR ...")
wxall = pd.concat([parse_station(st) for st in STATIONS], ignore_index=True)
WXCOLS = [c for c in wxall.columns if c.startswith("wx_")]


# --------------------------------------------------------------------------- #
# 4. as-of join (nearest METAR at-or-before movement time, <=90 min)
# --------------------------------------------------------------------------- #
print("as-of joining ...")
left = pd.DataFrame({"mvt_id": X.index, "valid": mt.to_numpy(),
                     "station": adep, "resid": resid, "y": y.to_numpy(),
                     "pred": pred})
left = left.dropna(subset=["valid"]).sort_values("valid")
J = pd.merge_asof(left, wxall.sort_values("valid"), on="valid", by="station",
                  direction="backward", tolerance=pd.Timedelta("90min"))
J["month"] = J["valid"].dt.month
matched = J["wx_vsby_mi"].notna() | J["wx_wind_kt"].notna()
print("  rows with a METAR match: {} / {} ({:.1%})".format(
    int(matched.sum()), len(J), matched.mean()))
J = J[matched].copy()
J[WXCOLS] = J[WXCOLS].fillna(J[WXCOLS].median(numeric_only=True))

base_rmse = rmse(J["y"], J["pred"])
print("\nbaseline RMSE on matched forward/test rows: {:.2f}  (n={})".format(base_rmse, len(J)))
print("resid mean {:.1f}  std {:.1f}".format(J.resid.mean(), J.resid.std()))


# --------------------------------------------------------------------------- #
# 5a. bucketed residual RMSE by weather condition
# --------------------------------------------------------------------------- #
def bucket(mask, label):
    m = mask.to_numpy() if hasattr(mask, "to_numpy") else mask
    if m.sum() < 50:
        print("  {:22s} n={:6d}   (too few)".format(label, int(m.sum())))
        return
    r_in = rmse(J["y"][m], J["pred"][m]); r_out = rmse(J["y"][~m], J["pred"][~m])
    bias_in = J.resid[m].mean()
    print("  {:22s} n={:6d}  RMSE {:8.1f}  vs {:7.1f} elsewhere   resid_bias {:+7.1f}".format(
        label, int(m.sum()), r_in, r_out, bias_in))

print("\n=== 5a. residual RMSE by weather bucket (baseline is {:.1f}) ===".format(base_rmse))
bucket(J.wx_lowvis.eq(1), "visibility < 3 mi")
bucket(J.wx_lifr.eq(1), "LIFR (vis<1 or cig<500)")
bucket(J.wx_ifr.eq(1), "IFR (vis<3 or cig<1000)")
bucket(J.wx_fog_mist.eq(1), "fog / mist")
bucket(J.wx_has_precip.eq(1), "any precip")
bucket(J.wx_snow_ice.eq(1), "snow / ice pellets")
bucket(J.wx_freezing.eq(1), "freezing (FZ or <=0C)")
bucket(J.wx_thunder.eq(1), "thunderstorm")
bucket(J.wx_highwind.eq(1), "gust >= 25 kt")
bucket(J.wx_gust_spread.ge(10), "gust spread >= 10 kt")
bucket(J.wx_ceiling_ft.lt(1000), "ceiling < 1000 ft")


# --------------------------------------------------------------------------- #
# 5b. OOS incremental: can weather predict the residual out of sample?
#     train on Nov, test on Dec and vice versa. Compare against congestion
#     features regressed on the SAME residual (they are already in the model).
# --------------------------------------------------------------------------- #
import lightgbm as lgb

CONGEST = ["dep_prev_15m", "dep_next_15m", "arr_prev_15m", "dep_prev_30m",
           "dep_next_30m", "arr_prev_30m", "dep_prev_60m", "dep_next_60m",
           "arr_prev_60m", "same_rwy_prev_15m", "same_rwy_prev_30m",
           "same_rwy_next_15m", "same_rwy_next_30m", "arr_taxiin_mean_prev_60m",
           "n_taxiing_at_pushback", "dep_surge_ratio_15_60"]
cong = X.reindex(J["mvt_id"].to_numpy())[CONGEST].reset_index(drop=True)
cong.columns = ["cg_" + c for c in cong.columns]
J = J.reset_index(drop=True)
J = pd.concat([J, cong], axis=1)
CGCOLS = list(cong.columns)

def oos_r2(feat_cols, tag):
    r2s, rmse_r = [], []
    for tr_m, te_m in [(11, 12), (12, 11)]:
        tr = J[J.month.eq(tr_m)]; te = J[J.month.eq(te_m)]
        g = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03,
                              num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                              min_child_samples=200, n_jobs=-1, verbose=-1)
        g.fit(tr[feat_cols], tr["resid"])
        p = g.predict(te[feat_cols])
        ss_res = np.sum((te["resid"] - p) ** 2)
        ss_tot = np.sum((te["resid"] - te["resid"].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        # RMSE of (y) if we ADD this residual model to the baseline
        new_pred = te["pred"] + p
        rmse_r.append(rmse(te["y"], new_pred))
        r2s.append(r2)
    print("  {:20s} OOS R2 on residual: {:+.4f} / {:+.4f}   "
          "y-RMSE baseline+corr: {:.2f} / {:.2f}  (base {:.1f})".format(
              tag, r2s[0], r2s[1], rmse_r[0], rmse_r[1], base_rmse))
    return np.mean(r2s)

print("\n=== 5b. out-of-sample residual-explaining power (LightGBM, Nov<->Dec) ===")
r2_w = oos_r2(WXCOLS, "weather only")
r2_c = oos_r2(CGCOLS, "congestion only")
r2_wc = oos_r2(WXCOLS + CGCOLS, "weather + congestion")
print("\n  interpretation: weather is additive iff (weather+congest) R2 clearly "
      "exceeds congestion-only R2.")
print("  weather-only {:+.4f} | congest-only {:+.4f} | both {:+.4f} | "
      "weather delta over congest {:+.4f}".format(r2_w, r2_c, r2_wc, r2_wc - r2_c))


# --------------------------------------------------------------------------- #
# 5c. tail: does weather flag the big-residual rows?
# --------------------------------------------------------------------------- #
print("\n=== 5c. tail association (rows with |resid| > 600 s) ===")
big = J.resid.abs().gt(600)
for c in ["wx_ifr", "wx_lowvis", "wx_has_precip", "wx_snow_ice", "wx_freezing",
          "wx_thunder", "wx_highwind", "wx_fog_mist"]:
    rate_big = J[c][big].mean(); rate_rest = J[c][~big].mean()
    lift = rate_big / rate_rest if rate_rest > 0 else float("nan")
    print("  {:16s} rate in big-resid {:.3f}  vs rest {:.3f}   lift x{:.2f}".format(
        c, rate_big, rate_rest, lift))

print("\nDONE")
