#!/usr/bin/env python3
"""Strategy 3 ceiling (PREREGISTERED.md): what could a routed stand->runway distance gain at most?"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT); sys.path.insert(0, str(ROOT))
from common import dataset as D
from common import io as cio
from tests_tail_diagnosis.diagnose import CACHE

Xtr, _, _ = D.load_part("seasonal_full", "train", "with_aobt", filter_target=True, outlier_policy="keep_all")
X, y, meta = D.load_part("seasonal_full", "test", "with_aobt", filter_target=False)
ftt = np.mean([np.load(ROOT / "tests_ftt" / f"preds_seed{s}_seasonal_full_lr1e-3.npz")["pt"] for s in range(42, 51)], axis=0)
base = (np.load(CACHE / "tree_seasonal_full_test.npy") + np.load(CACHE / "mlp6_seasonal_full_test.npy") + ftt) / 3
y = np.asarray(y, float); ok = np.isfinite(y)
seen = set(Xtr["stand_runway"].astype(str))
pair = X["stand_runway"].astype(str).to_numpy()
unseen = ~np.isin(pair, list(seen))
ap = np.asarray(meta["ADEP_mvt"]).astype(str)
se = (y - base) ** 2
tot = se[ok].sum(); n = ok.sum()
print(f"Jan+Jul 2025 held-out: {unseen[ok].mean():.2%} of rows have an unseen stand_runway pair, carrying {se[ok & unseen].sum() / tot:.2%} of squared error")
# ceiling: unseen rows' squared error replaced by the seen-row mean squared error of the same airport
se_c = se.copy()
for a in np.unique(ap):
    m_seen = ok & ~unseen & (ap == a); m_un = ok & unseen & (ap == a)
    if m_un.any() and m_seen.any():
        se_c[m_un] = np.minimum(se[m_un], se[m_seen].mean())
    if m_un.sum() > 50:
        print(f"   {a}: unseen {m_un.sum():5d} rows ({m_un.sum() / (ok & (ap == a)).sum():.1%}), RMSE {np.sqrt(se[m_un].mean()):7.1f} vs seen {np.sqrt(se[m_seen].mean()):7.1f}")
r0, r1 = np.sqrt(tot / n), np.sqrt(se_c[ok].sum() / n)
print(f"CEILING (perfect routed distance on unseen pairs): pooled {r0:.2f} -> {r1:.2f} ({r1 - r0:+.2f} s)")
# 2026 share of unseen pairs against all of 2025
Xall, _, _ = D.load_part("full", "train", "with_aobt", filter_target=True, outlier_policy="keep_all")
Xr, mr = D.load_ranking_features("with_aobt")
u26 = ~Xr["stand_runway"].astype(str).isin(set(Xall["stand_runway"].astype(str)))
print(f"2026 ranking rows with a stand_runway pair unseen in 2025: {u26.mean():.2%}; by airport:",
      u26.groupby(np.asarray(mr["ADEP_mvt"]).astype(str)).mean().round(3).to_dict())
print("DECISION:", "CLOSE (ceiling < 0.5 s)" if (r0 - r1) < 0.5 else "REPORT (ceiling >= 0.5 s)")
