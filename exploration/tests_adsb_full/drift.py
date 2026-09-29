#!/usr/bin/env python3
"""2025 -> 2026 drift of every ADS-B feature, same month, same airport (forum warning: the receiver
network changed between years). For each feature and airport: coverage and the standardized mean
difference (SMD) of the feature between the 2025 and the 2026 days of the same month.
Rule (PREREGISTERED.md, addendum 2): a feature with pooled |SMD| > 0.2 in either month is drifting.
usage: drift.py  (uses every fetched day)"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import fetch, features

FEAT = ["e_first", "e_disp", "e_move", "n_ground", "liftoff_dt"] + features.MOVE
rows = []
for f in sorted((HERE / "points").glob("*.parquet")):
    day = f.stem
    t = fetch.targets(day); t = t[t["T"] < pd.Timestamp(day).value / 1e9 + 86400].set_index("MVT_ID_mvt")
    e = features.load_day(day).reindex(t.index)
    e["ap"] = t.ADEP_mvt; e["year"] = day[:4]; e["mon"] = day[5:7]
    rows.append(e)
df = pd.concat(rows)
df["covered"] = df.e_first.notna()
out = {}
for mon, g in df.groupby("mon"):
    if g.year.nunique() < 2:
        print(f"month {mon}: only {g.year.unique()} fetched, no comparison"); continue
    a, b = g[g.year == "2025"], g[g.year == "2026"]
    print(f"\n== month {mon}: 2025 {a.index.size} deps, 2026 {b.index.size} deps")
    cov = pd.DataFrame({"2025": a.groupby("ap").covered.mean(), "2026": b.groupby("ap").covered.mean()}).round(2)
    print("coverage by airport:\n", cov.T.to_string())
    res = {}
    for c in FEAT:
        x, y = a[c].dropna(), b[c].dropna()
        sd = np.sqrt((x.var() + y.var()) / 2)
        pooled = float((y.mean() - x.mean()) / sd) if sd > 0 else 0.0
        # within-airport SMD averaged over airports, so a change in airport mix is not counted as drift
        per = []
        for ap in sorted(set(a.ap) & set(b.ap)):
            xa, ya = x[a.ap.reindex(x.index) == ap], y[b.ap.reindex(y.index) == ap]
            if len(xa) >= 50 and len(ya) >= 50:
                s = np.sqrt((xa.var() + ya.var()) / 2)
                per.append((ya.median() - xa.median()) / s if s > 0 else 0.0)
        res[c] = {"smd_pooled": pooled, "smd_within_airport_median": float(np.mean(per)) if per else None,
                  "drifting": abs(pooled) > 0.2}
        print(f"   {c:10s} SMD pooled {pooled:+.2f} | within-airport {res[c]['smd_within_airport_median'] if per else float('nan'):+.2f}"
              f"{'  DRIFT' if abs(pooled) > 0.2 else ''}")
    out[mon] = {"coverage": cov.to_dict(), "features": res}
(HERE / "drift.json").write_text(json.dumps(out, indent=2, default=float))
