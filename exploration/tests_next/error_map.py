#!/usr/bin/env python3
"""Exploratory: where is the squared error of the current recipe (held-out base + v48-design ADS-B corrector,
leave-one-month-out) on Jan + Jul 2025? By airport, ADS-B covered or not, and target band."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests_weather_validated")); sys.path.insert(0, str(ROOT / "tests_adsb_full"))
import weather_lomo as W   # reuses its frame + per-fold A+ code path
import lightgbm as lgb
from common import io as cio
src = open(ROOT / "tests_weather_validated" / "weather_lomo.py", encoding="utf-8").read()
src = src.split("    unc = ~df.covA")[0].replace('if __name__ == "__main__":', "if True:")
ns = {"__file__": str(ROOT / "tests_weather_validated" / "weather_lomo.py"), "__name__": "x"}
exec(compile(src, "wl", "exec"), ns)
df = ns["df"]
d = df[df.mon.isin(["01", "07"])].copy()
d["se"] = (d.y - d.pA) ** 2
tot = d.se.sum(); n = len(d)
print(f"Jan+Jul 2025 rows {n}, RMSE after ADS-B corrector {np.sqrt(tot / n):.1f}")
d["band"] = pd.cut(d.y, [-1, 1800, 3600, 7200, 1e9], labels=["<=30m", "30-60m", "1-2h", ">2h"])
d["cov"] = np.where(d.covA, "ADS-B", "no ADS-B")
for keys in (["ap"], ["cov"], ["band"], ["ap", "cov"]):
    g = d.groupby(keys, observed=True).agg(rows=("se", "size"), se=("se", "sum"))
    g["row%"] = (100 * g.rows / n).round(1); g["SE%"] = (100 * g.se / tot).round(1); g["RMSE"] = np.sqrt(g.se / g.rows).round(0)
    print("\n", g[["rows", "row%", "SE%", "RMSE"]].sort_values("SE%", ascending=False).head(14).to_string())
top = d.sort_values("se", ascending=False)
for k in (50, 200, 1000):
    print(f"top {k} rows carry {100 * top.se.head(k).sum() / tot:.1f}% of squared error")
print("top 200 rows by airport:", top.head(200).ap.value_counts().to_dict())
d[["y", "pA", "base", "ap", "cov", "band", "mon", "covA"]].to_parquet(HERE / "error_map_janjul.parquet")
g = d.groupby(["ap", "band"], observed=True).agg(rows=("se", "size"), se=("se", "sum"))
g["SE%"] = (100 * g.se / tot).round(1); g["RMSE"] = np.sqrt(g.se / g.rows).round(0)
print("\n", g[g["SE%"] >= 0.8][["rows", "SE%", "RMSE"]].sort_values("SE%", ascending=False).to_string())
for ap in ("LFPG", "LIRF", "LTFM", "EGLL"):
    s = d[d.ap == ap]
    print(f"{ap}: in-band (<=60m) RMSE {np.sqrt(((s.y - s.pA) ** 2)[s.y <= 3600].mean()):.0f}, mean bias in-band {(s.y - s.pA)[s.y <= 3600].mean():+.1f}, "
          f"Jan {np.sqrt(((s.y - s.pA) ** 2)[s.mon == '01'].mean()):.0f} / Jul {np.sqrt(((s.y - s.pA) ** 2)[s.mon == '07'].mean()):.0f}")
