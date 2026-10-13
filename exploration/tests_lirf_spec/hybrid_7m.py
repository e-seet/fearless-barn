#!/usr/bin/env python3
"""PREREGISTERED_hybrid.md, surface 1: the hybrid on the 7 months, from the saved S_heldout.parquet."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests_copymix"))
from common import dataset as D
from common import io as cio
from copymix import times, lobt_clip
rm = lambda r: float(np.sqrt(np.mean(np.square(r))))
MONTHS = ["01", "07", "08", "09", "10", "11", "12"]
raw = cio.load_training(columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt"])
raw = raw[raw.PHASE_mvt.eq("DEP")].drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
X, y, _ = D.load_part("full", "train", "with_aobt", filter_target=False); y = pd.Series(np.asarray(y, float), index=X.index)
d = pd.read_parquet(ROOT / "tests_copymix" / "cache" / "F_heldout.parquet"); d["y"] = y.reindex(d.index).to_numpy(); d = d[np.isfinite(d.y)]
mv, sc, lb = times(d.index, raw); g = np.maximum(mv - sc, 0)
Fc = lobt_clip(d.F.to_numpy(), mv, lb); S = pd.read_parquet(HERE / "S_heldout.parquet").S.reindex(d.index).to_numpy()
out = {}
for cut in (21600, 3600, 10800, 43200):
    C = lobt_clip(np.where(np.isfinite(S) & (g > cut), S, Fc), mv, lb)
    a, b = (d.y.to_numpy() - Fc) ** 2, (d.y.to_numpy() - C) ** 2; gg = a - b; o = np.argsort(-gg)
    mons = {m: (rm(np.sqrt(a[d.mon == m])), rm(np.sqrt(b[d.mon == m]))) for m in MONTHS}
    changed = [m for m in MONTHS if np.any(C[(d.mon == m).to_numpy()] != Fc[(d.mon == m).to_numpy()])]
    wins = sum(mons[m][1] < mons[m][0] for m in changed)
    r = {"rows_changed": int((C != Fc).sum()), "pooled": (float(np.sqrt(a.mean())), float(np.sqrt(b.mean()))), "wins": wins, "changed_months": len(changed),
         "months": {m: round(v[1] - v[0], 3) for m, v in mons.items()}}
    jj = d.mon.isin(["01", "07"]).to_numpy(); r["janjul"] = (float(np.sqrt(a[jj].mean())), float(np.sqrt(b[jj].mean())))
    for k in (5, 20, 50):
        keep = np.ones(len(gg), bool); keep[o[:k]] = False; r[f"drop{k}"] = (float(np.sqrt(a[keep].mean())), float(np.sqrt(b[keep].mean())))
    r["PASS"] = r["pooled"][1] < r["pooled"][0] and r["drop20"][1] < r["drop20"][0] and wins >= 5
    out[cut] = r
    tag = "DECIDES" if cut == 21600 else "info only"
    print(f"cut {cut} s ({tag}): rows {r['rows_changed']} | pooled {r['pooled'][0]:.3f} -> {r['pooled'][1]:.3f} ({r['pooled'][1]-r['pooled'][0]:+.3f}) | "
          f"Jan+Jul {r['janjul'][1]-r['janjul'][0]:+.3f} | {wins}/{len(changed)} | drop20 {r['drop20'][1]-r['drop20'][0]:+.3f} | drop50 {r['drop50'][1]-r['drop50'][0]:+.3f} | months {r['months']}"
          + (f" => {'PASS' if r['PASS'] else 'FAIL'}" if cut == 21600 else ""))
(HERE / "hybrid_7m.json").write_text(json.dumps(out, indent=2, default=float))
