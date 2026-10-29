#!/usr/bin/env python3
"""Check stability of per-airport mean signed error across different splits:
1. forward_inner val (Jul-Aug 2025)
2. forward val (Sep-Oct 2025)
3. forward test (Nov-Dec 2025)
4. seasonal_full test (Jan + Jul 2025)
If per-airport bias is consistent across periods, it is a persistent structural calibration signal.
If it changes signs or fluctuates wildly, it is seasonal drift.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("c:/Users/Admin/Desktop/eddie/fearless-barn")
EXP = ROOT / "exploration"
sys.path.insert(0, str(EXP))
from common import dataset as D
from tests_tail_diagnosis.diagnose import CACHE

# 1. seasonal_full test (Jan+Jul 2025)
X_sf, y_sf, m_sf = D.load_part("seasonal_full", "test", "with_aobt", filter_target=False)
y_sf = np.asarray(y_sf, float)
p_sf = (np.load(CACHE / "tree_seasonal_full_test.npy") +
        np.load(CACHE / "mlp6_seasonal_full_test.npy") +
        np.mean([np.load(EXP / "tests_ftt" / f"preds_seed{s}_seasonal_full_lr1e-3.npz")["pt"] for s in range(42, 51)], axis=0)) / 3.0
ap_sf = np.asarray(m_sf["ADEP_mvt"]).astype(str)

# 2. forward test (Nov-Dec 2025)
X_fwd, y_fwd, m_fwd = D.load_part("forward", "test", "with_aobt", filter_target=False)
y_fwd = np.asarray(y_fwd, float)
ap_fwd = np.asarray(m_fwd["ADEP_mvt"]).astype(str)

print(f"{'Airport':8s} | {'Jan+Jul 2025 Error':20s} | {'Nov-Dec 2025 Error':20s}")
print("-" * 55)

for ap in sorted(np.unique(ap_sf)):
    m1 = np.isfinite(y_sf) & (ap_sf == ap) & (np.abs(y_sf - p_sf) < 3600)
    err1 = np.mean(y_sf[m1] - p_sf[m1])
    
    # We can check forward test if cached or examine raw target difference
    print(f"{ap:8s} | {err1:+10.2f} s           |")
