#!/usr/bin/env python3
"""Why does January blow up at LFPG and LIRF?

Model-free decomposition on 2025 data (no training, no submission). Baseline is
the proxy (MVT_TIME - AOBT_3) where present, else the airport hour-of-day median
learned from OTHER months -- i.e. what a well-calibrated 'normal' model predicts.
Then characterise the rows whose true taxi-out blows past that baseline:
  - how many, how concentrated in SSE
  - date-clustered (few snow days) or spread across the month
  - AOBT_3 present or missing on them
  - do the METAR flags (snow / freezing / low vis) light up on those days
  - what does v24 predict for the SAME airports/month in the 2026 scored set
"""
from __future__ import annotations
import glob
import sys
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("C:/Users/Admin/Desktop/eddie/fearless-barn/prc-2026-datasets")
WX = Path("C:/Users/Admin/Desktop/eddie/fearless-barn/prc-2026-datasets-wx")
EXP = Path("C:/Users/Admin/Desktop/eddie/fearless-barn/solution"); exec(open(EXP.parent / "tools" / "repo_layout.py").read()); EXP = dual(EXP)

FOCUS = ["LFPG", "LIRF"]
CONTROL = ["EDDF", "EHAM"]
ALL_AP = FOCUS + CONTROL


def load_month(glob_pat):
    f = sorted(glob.glob(str(DATA / glob_pat)))[0]
    d = pd.read_parquet(f, columns=["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt",
                                    "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt",
                                    "AOBT_3_flt", "EOBT_1_flt", "SCHED_TIME_UTC_mvt"])
    d = d[d.PHASE_mvt.eq("DEP") & d.ADEP_mvt.isin(ALL_AP)].copy()
    d["MVT_TIME_UTC_mvt"] = pd.to_datetime(d["MVT_TIME_UTC_mvt"], utc=True)
    d["AOBT_3_flt"] = pd.to_datetime(d["AOBT_3_flt"], utc=True)
    d["proxy"] = (d["MVT_TIME_UTC_mvt"] - d["AOBT_3_flt"]).dt.total_seconds()
    d["has_aobt"] = d["AOBT_3_flt"].notna()
    d["hour"] = d["MVT_TIME_UTC_mvt"].dt.hour
    d["date"] = d["MVT_TIME_UTC_mvt"].dt.date
    d = d[(d.TAXITIME_SEC_mvt > 0) & (d.TAXITIME_SEC_mvt < 200000)]
    return d


print("loading 2025 Jan (held-out month) + Feb..Jul (baseline months) ...")
jan = load_month("training_2025-01-01_*.parquet")
base_months = pd.concat([load_month(f"training_2025-0{m}-01_*.parquet") for m in (2, 3, 4, 5, 6, 7)],
                        ignore_index=True)

# airport x hour median taxi-out, learned on Feb..Jul only
med = base_months.groupby(["ADEP_mvt", "hour"])["TAXITIME_SEC_mvt"].median()
gmed = base_months.groupby("ADEP_mvt")["TAXITIME_SEC_mvt"].median()

def baseline_pred(df):
    # proxy where sane, else airport-hour median, else airport median
    p = df["proxy"].to_numpy(dtype="float64")
    key = list(zip(df["ADEP_mvt"], df["hour"]))
    hm = np.array([med.get(k, np.nan) for k in key])
    am = df["ADEP_mvt"].map(gmed).to_numpy(dtype="float64")
    fb = np.where(np.isfinite(hm), hm, am)
    ok = np.isfinite(p) & (p > 30) & (p < 20000)
    return np.where(ok, p, fb)

jan["pred"] = baseline_pred(jan)
jan["resid"] = jan["TAXITIME_SEC_mvt"] - jan["pred"]

print("\n=== January 2025 baseline error by airport ===")
for ap in ALL_AP:
    s = jan[jan.ADEP_mvt.eq(ap)]
    rmse = np.sqrt((s.resid ** 2).mean())
    tag = "FOCUS" if ap in FOCUS else "ctrl "
    print(f"  {tag} {ap}  n={len(s):6d}  RMSE {rmse:8.1f}  "
          f"median_true {s.TAXITIME_SEC_mvt.median():6.0f}  "
          f"p99_true {s.TAXITIME_SEC_mvt.quantile(.99):7.0f}  "
          f"max_true {s.TAXITIME_SEC_mvt.max():8.0f}")

# --- the blow-up rows: big positive residual at the two focus airports ---
foc = jan[jan.ADEP_mvt.isin(FOCUS)].copy()
tot_sse = (foc.resid ** 2).sum()
for thr in (1800, 3600, 5400):
    big = foc[foc.resid > thr]
    print(f"\n=== LFPG+LIRF Jan rows with resid > {thr}s: n={len(big)} "
          f"({len(big)/len(foc)*100:.2f}% of rows), "
          f"{(big.resid**2).sum()/tot_sse*100:.1f}% of SSE ===")
    if len(big) == 0:
        continue
    print(f"   AOBT_3 missing on them: {(~big.has_aobt).mean()*100:.1f}%  "
          f"(vs {(~foc.has_aobt).mean()*100:.1f}% of all Jan focus rows)")
    print(f"   |true - proxy| on the AOBT-present subset: median "
          f"{(big[big.has_aobt].TAXITIME_SEC_mvt - big[big.has_aobt].proxy).abs().median():.0f}s")
    # date concentration
    by_day = big.groupby(["ADEP_mvt", "date"]).size().sort_values(ascending=False)
    ndays = big["date"].nunique()
    top5 = by_day.head(5)
    print(f"   spread over {ndays} distinct days; top day-airport buckets:")
    for (ap, dt), c in top5.items():
        print(f"      {ap} {dt}: {c} rows")
    share_top = by_day.head(5).sum() / len(big)
    print(f"   top 5 day-airport buckets hold {share_top*100:.0f}% of the blow-up rows")


# --------------------------------------------------------------------------- #
# METAR on the worst days vs normal days
# --------------------------------------------------------------------------- #
def parse_station(st):
    d = pd.read_csv(WX / (st + ".csv"), dtype=str)
    d["valid"] = pd.to_datetime(d["valid"], utc=True, errors="coerce")
    def num(s): return pd.to_numeric(s.replace({"M": np.nan, "T": "0.0001"}), errors="coerce")
    wx = d["wxcodes"].fillna("").str.upper()
    out = pd.DataFrame({"valid": d["valid"], "station": st})
    out["vsby"] = num(d["vsby"])
    out["tmpf"] = num(d["tmpf"])
    out["snow"] = wx.str.contains("SN|PL|GS|SG").astype(int)
    out["freezing"] = (wx.str.contains("FZ") | (out["tmpf"] <= 32)).astype(int)
    out["lowvis"] = (out["vsby"] < 3).astype(int)
    out["date"] = out["valid"].dt.date
    return out.dropna(subset=["valid"])

wx = pd.concat([parse_station(s) for s in FOCUS], ignore_index=True)
wx_jan = wx[(wx.valid >= "2025-01-01") & (wx.valid < "2025-02-01")]
day_wx = wx_jan.groupby(["station", "date"]).agg(
    snow_obs=("snow", "sum"), freezing_obs=("freezing", "sum"),
    lowvis_obs=("lowvis", "sum"), n_obs=("snow", "size")).reset_index()

big = foc[foc.resid > 3600]
big_days = set(zip(big.ADEP_mvt, big.date))
day_wx["is_blowup_day"] = day_wx.apply(lambda r: (r["station"], r["date"]) in big_days, axis=1)
print("\n=== METAR: blow-up days vs the rest of January (LFPG+LIRF) ===")
g = day_wx.groupby("is_blowup_day")[["snow_obs", "freezing_obs", "lowvis_obs"]].mean()
print(g.to_string())
print("\n  per-day detail on blow-up days:")
print(day_wx[day_wx.is_blowup_day].to_string(index=False))


# --------------------------------------------------------------------------- #
# 2026 scored set: what does v24 predict for LFPG/LIRF in January 2026?
# --------------------------------------------------------------------------- #
print("\n=== 2026 ranking set: LFPG/LIRF, and v24's predictions there ===")
rk = pd.read_parquet(DATA / "ranking.parquet",
                     columns=["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"])
rk = rk[rk.PHASE_mvt.eq("DEP") & rk.ADEP_mvt.isin(FOCUS)].copy()
rk["MVT_TIME_UTC_mvt"] = pd.to_datetime(rk["MVT_TIME_UTC_mvt"], utc=True)
rk["month"] = rk["MVT_TIME_UTC_mvt"].dt.month
rk["has_aobt"] = pd.to_datetime(rk["AOBT_3_flt"], utc=True).notna()
v24 = pd.read_parquet(EXP / "submissions/fearless_barn_v24.parquet")
idcol = [c for c in v24.columns if c.upper().startswith("MVT")][0]
predcol = [c for c in v24.columns if c != idcol][0]
v24 = v24.rename(columns={idcol: "MVT_ID_mvt", predcol: "pred"})
rk = rk.merge(v24, on="MVT_ID_mvt", how="left")
for ap in FOCUS:
    for mo in (1, 2, 7):
        s = rk[rk.ADEP_mvt.eq(ap) & rk.month.eq(mo)]
        if not len(s):
            continue
        print(f"  {ap} 2026-{mo:02d}: n={len(s):5d}  AOBT-missing {(~s.has_aobt).mean()*100:4.1f}%  "
              f"v24 pred  median {s.pred.median():6.0f}  p99 {s.pred.quantile(.99):7.0f}  "
              f"max {s.pred.max():8.0f}  n(pred>3600) {int((s.pred>3600).sum())}")

print("\nDONE")
