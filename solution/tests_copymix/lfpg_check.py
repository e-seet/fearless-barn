#!/usr/bin/env python3
"""Build-construction check (advisor item 2, no pass bar): on LFPG rows where the stage-5 corrector fires,
is the copy blend still useful once the LFPG corrector delta (L0, held out) is added on top?"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
for p in (ROOT, ROOT / "tests_adsb_full", ROOT / "tests_round10", ROOT / "tests_round9", ROOT / "tests_round13"):
    sys.path.insert(0, str(p))
import round9 as R9
rm = lambda r: float(np.sqrt(np.mean(np.square(r))))
src = (ROOT / "tests_round13" / "round13.py").read_text(encoding="utf-8").split("def corr(")[0]
ns = {"__file__": str(ROOT / "tests_round13" / "round13.py"), "__name__": "r13"}
exec(compile(src, "r13", "exec"), ns)
df = ns["df"]
d = pd.read_parquet(HERE / "cache" / "F_heldout.parquet").join(pd.read_parquet(HERE / "cache" / "M_heldout.parquet"))
d = d.join(df[["y", "lfpg"]])
d = d[np.isfinite(d.y)]
d["L"] = 0.0
for m in sorted(d.mon.unique()):
    ltr = df[(df.mon != m) & df.lfpg].copy(); ltr["base"] = ltr.base_eq
    te = df[(df.mon == m) & df.lfpg].copy(); te["base"] = te.base_r4   # applied to the realistic-4 base, as in round 13
    if len(te):
        L = np.mean([lgb.LGBMRegressor(**{**R9.PARAMS, "random_state": s}).fit(ltr[R9.FEAT_1], ltr.y - ltr.base).predict(te[R9.FEAT_1]) for s in (42, 43, 44, 45, 46)], axis=0)
        d.loc[te.index.intersection(d.index), "L"] = pd.Series(L, index=te.index).reindex(te.index.intersection(d.index)).to_numpy()
L = d[d.ap.eq("LFPG")]
print(f"LFPG rows {len(L)}, stage-5 rows {int(L.lfpg.sum())}")
tot = {"a": [], "b": [], "c": []}
for m in sorted(L.mon.unique()):
    o = L[L.mon != m]
    dl = (o.M - o.F).to_numpy(); r = (o.y - o.F - o.L).to_numpy()   # weight refitted with the L delta present
    w = float(np.clip(dl @ r / (dl @ dl), 0, 1))
    t = L[(L.mon == m) & L.lfpg]
    a = t.y - t.F - t.L; b = t.y - ((1 - w) * t.F + w * t.M + t.L)
    tot["a"].append(a.to_numpy() ** 2); tot["b"].append(b.to_numpy() ** 2)
    print(f"  month {m}: w {w:.2f} | stage-5 rows {len(t)} | F+L {rm(a):.2f} | blend+L {rm(b):.2f}")
A, B = np.concatenate(tot["a"]), np.concatenate(tot["b"])
print(f"POOLED stage-5 rows: F+L {np.sqrt(A.mean()):.3f} -> blend+L {np.sqrt(B.mean()):.3f} ({np.sqrt(B.mean()) - np.sqrt(A.mean()):+.3f})")
print("ALL DONE")
