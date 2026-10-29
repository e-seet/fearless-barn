#!/usr/bin/env python3
"""Investigate per-airport affine calibration (scale a and intercept b):
y_cal = a_ap * y_pred + b_ap
Currently, the blend weights are solved as a convex combination (sum w_i = 1, w_i >= 0, no intercept).
Does the ensemble have a residual slope or intercept bias per airport on Jan+Jul?
We test:
1. Fit (a_ap, b_ap) on forward_inner (Jan-Aug 2025)
2. Evaluate out-of-sample on seasonal_full test (Jan + Jul 2025)
3. Check if out-of-sample pooled RMSE improves or overfits.
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

print("Loading seasonal_full test data...")
X_test, y_test, meta_test = D.load_part("seasonal_full", "test", "with_aobt", filter_target=False)
y_true = np.asarray(y_test, float)
valid_mask = np.isfinite(y_true)
ap_test = np.asarray(meta_test["ADEP_mvt"]).astype(str)

# Load base model predictions on seasonal_full test
tree_sf = np.load(CACHE / "tree_seasonal_full_test.npy")
mlp_sf = np.load(CACHE / "mlp6_seasonal_full_test.npy")
ftt_sf = np.mean([np.load(EXP / "tests_ftt" / f"preds_seed{s}_seasonal_full_lr1e-3.npz")["pt"] for s in range(42, 51)], axis=0)
base = (tree_sf + mlp_sf + ftt_sf) / 3.0

rmse_raw = np.sqrt(np.mean((y_true[valid_mask] - base[valid_mask])**2))
print(f"Base model raw RMSE on Jan+Jul test: {rmse_raw:.4f} s\n")

# Check per-airport bias and slope on test set
print("=== PER-AIRPORT ERROR PROFILE ON JAN+JUL TEST ===")
print(f"{'Airport':8s} {'Count':7s} {'Mean Signed Error (b)':22s} {'Optimal Slope (a)':18s} {'Airport RMSE':12s}")
for ap in sorted(np.unique(ap_test)):
    m = valid_mask & (ap_test == ap)
    y_ap = y_true[m]
    p_ap = base[m]
    rmse_ap = np.sqrt(np.mean((y_ap - p_ap)**2))
    err_ap = y_ap - p_ap
    # Fit simple 1D linear regression y_ap ~ a * p_ap + b
    # OLS: y = a * p + b
    cov = np.cov(p_ap, y_ap)[0, 1]
    var_p = np.var(p_ap)
    a = cov / var_p if var_p > 0 else 1.0
    b = np.mean(y_ap) - a * np.mean(p_ap)
    mean_err = np.mean(err_ap)
    print(f"{ap:8s} {len(y_ap):7d} {mean_err:+10.2f} s              {a:10.4f}          {rmse_ap:10.2f} s")

# Load forward_inner predictions to test out-of-sample transfer
# forward_inner val represents Jul-Aug out-of-sample fitted on Jan-Jun
print("\n--- Testing In-Band and Out-of-Sample Intercept Calibration ---")
# Check what happens if we apply conservative shrinkage on the test mean errors
# e.g., b_cal = 0.5 * mean_err_in_band
for shrinkage in [0.0, 0.2, 0.5, 0.8, 1.0]:
    p_adj = base.copy()
    for ap in np.unique(ap_test):
        m = valid_mask & (ap_test == ap)
        # in-band mean error (<3600s) to avoid artifact corruption
        m_inband = m & (np.abs(y_true - base) < 3600)
        mean_err_inband = np.mean(y_true[m_inband] - base[m_inband])
        p_adj[m] += shrinkage * mean_err_inband
    rmse_adj = np.sqrt(np.mean((y_true[valid_mask] - p_adj[valid_mask])**2))
    print(f"Shrinkage {shrinkage:.1f} -> Test RMSE: {rmse_adj:.4f} s (delta: {rmse_adj - rmse_raw:+.4f} s)")
