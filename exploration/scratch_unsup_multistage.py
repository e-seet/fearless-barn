#!/usr/bin/env python3
"""Explore Unsupervised Traffic Clustering + Multi-Stage Shrinkage Gating across all 3 seeds.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, r"c:\Users\Admin\Desktop\eddie\fearless-barn\exploration")
from common import dataset as D

scratch_dir = Path(r"c:\Users\Admin\.gemini\antigravity-ide\brain\9f42e2ac-0ba3-4fa4-86ca-b021c7eca06d\scratch")
t = np.load(scratch_dir / "targets.npz", allow_pickle=True)

yw = t["yw"]
yva = t["yva"]
yte = t["yte"]

ap_w = t["ap_w"]
ap_va = t["ap_va"]
ap_te = t["ap_te"]

wtc_w = t["wtc_w"]
wtc_va = t["wtc_va"]
wtc_te = t["wtc_te"]

has_rec_w = t["has_rec_w"]
has_rec_va = t["has_rec_va"]
has_rec_te = t["has_rec_te"]

CLUSTER_FEATURES = [
    "dep_prev_15m", "dep_prev_30m", "dep_prev_60m",
    "arr_prev_15m", "arr_prev_30m", "arr_prev_60m",
    "same_rwy_prev_15m", "same_rwy_prev_30m",
    "hour", "minute_of_day"
]

print("Loading data features...")
X_w, _, _ = D.load_part("forward_inner", "val", "with_aobt")
X_va, _, _ = D.load_part("forward", "val", "with_aobt")
X_te, _, _ = D.load_part("forward", "test", "with_aobt")

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def clip(arr):
    return np.clip(arr, 60, 86400)

def get_base_regime(w, r):
    is_heavy = np.isin(w, ["H", "J"])
    is_norec = (r == 0)
    return np.where(is_heavy, "Heavy+", np.where(is_norec, "NoRecord", "Standard"))

reg_w = get_base_regime(wtc_w, has_rec_w)
reg_va = get_base_regime(wtc_va, has_rec_va)
reg_te = get_base_regime(wtc_te, has_rec_te)

# Load cached predictions for all 3 seeds
preds = {}
for seed in [42, 43, 44]:
    preds[seed] = np.load(scratch_dir / f"preds_seed{seed}.npz", allow_pickle=True)

# Test various K clusters: K=2, K=3, K=4, K=5 with shrinkage lambda
for k in [2, 3, 4, 5]:
    # Fit KMeans per airport
    clusters_w = np.zeros(len(X_w), dtype=int)
    clusters_va = np.zeros(len(X_va), dtype=int)
    clusters_te = np.zeros(len(X_te), dtype=int)
    
    for a in np.unique(ap_w):
        mw = (ap_w == a)
        mva = (ap_va == a)
        mte = (ap_te == a)
        scaler = StandardScaler()
        Xw_s = scaler.fit_transform(X_w.loc[mw, CLUSTER_FEATURES].fillna(0).to_numpy())
        Xva_s = scaler.transform(X_va.loc[mva, CLUSTER_FEATURES].fillna(0).to_numpy())
        Xte_s = scaler.transform(X_te.loc[mte, CLUSTER_FEATURES].fillna(0).to_numpy())
        
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xw_s)
        clusters_w[mw] = km.predict(Xw_s)
        clusters_va[mva] = km.predict(Xva_s)
        clusters_te[mte] = km.predict(Xte_s)
        
    cw = np.array([f"{a}|{r}|c{c}" for a, r, c in zip(ap_w, reg_w, clusters_w)])
    cva = np.array([f"{a}|{r}|c{c}" for a, r, c in zip(ap_va, reg_va, clusters_va)])
    cte = np.array([f"{a}|{r}|c{c}" for a, r, c in zip(ap_te, reg_te, clusters_te)])
    
    # Also parent cell: airport | regime
    pw = np.array([f"{a}|{r}" for a, r in zip(ap_w, reg_w)])
    pva = np.array([f"{a}|{r}" for a, r in zip(ap_va, reg_va)])
    pte = np.array([f"{a}|{r}" for a, r in zip(ap_te, reg_te)])
    
    for lam in [0, 50, 100, 200, 500]:
        scores_va = []
        scores_te = []
        
        for seed in [42, 43, 44]:
            p = preds[seed]
            pa_w = p["airport_operator_blend_w"]
            pa_va = p["airport_operator_blend_va"]
            pa_te = p["airport_operator_blend_te"]
            
            pb_w = p["mlp_antigravity_w"]
            pb_va = p["mlp_antigravity_va"]
            pb_te = p["mlp_antigravity_te"]
            
            # Level 1: airport alpha
            airport_alpha = {}
            for a in np.unique(ap_w):
                m = (ap_w == a)
                d = pa_w[m] - pb_w[m]
                r = yw[m] - pb_w[m]
                denom = float(np.dot(d, d))
                airport_alpha[str(a)] = 0.5 if denom <= 1e-12 else float(np.clip(np.dot(d, r) / denom, 0.0, 1.0))
                
            # Level 2: parent (airport, regime) alpha
            parent_alpha = {}
            for parent in np.unique(pw):
                m = (pw == parent)
                a = parent.split("|", 1)[0]
                if m.sum() < 200:
                    parent_alpha[parent] = airport_alpha[a]
                    continue
                d = pa_w[m] - pb_w[m]
                r = yw[m] - pb_w[m]
                denom = float(np.dot(d, d))
                parent_alpha[parent] = 0.5 if denom <= 1e-12 else float(np.clip(np.dot(d, r) / denom, 0.0, 1.0))
                
            # Level 3: child (airport, regime, cluster) alpha with shrinkage
            cell_alpha = {}
            for cell in np.unique(cw):
                m = (cw == cell)
                parent = cell.rsplit("|", 1)[0]
                n_m = m.sum()
                if n_m < 50:
                    cell_alpha[cell] = parent_alpha[parent]
                    continue
                d = pa_w[m] - pb_w[m]
                r = yw[m] - pb_w[m]
                denom = float(np.dot(d, d))
                raw_alpha = 0.5 if denom <= 1e-12 else float(np.clip(np.dot(d, r) / denom, 0.0, 1.0))
                
                # Empirical Bayes / Stein shrinkage towards parent alpha
                shrink_weight = float(n_m) / (float(n_m) + lam)
                cell_alpha[cell] = shrink_weight * raw_alpha + (1.0 - shrink_weight) * parent_alpha[parent]
                
            # Apply to val and test
            pred_va = np.zeros(len(pa_va), dtype="float64")
            for c in np.unique(cva):
                m = (cva == c)
                parent = c.rsplit("|", 1)[0]
                a = c.split("|", 1)[0]
                w = cell_alpha.get(c, parent_alpha.get(parent, airport_alpha.get(a, 0.5)))
                pred_va[m] = w * pa_va[m] + (1.0 - w) * pb_va[m]
                
            pred_te = np.zeros(len(pa_te), dtype="float64")
            for c in np.unique(cte):
                m = (cte == c)
                parent = c.rsplit("|", 1)[0]
                a = c.split("|", 1)[0]
                w = cell_alpha.get(c, parent_alpha.get(parent, airport_alpha.get(a, 0.5)))
                pred_te[m] = w * pa_te[m] + (1.0 - w) * pb_te[m]
                
            s_va = rmse(yva, clip(pred_va))
            s_te = rmse(yte, clip(pred_te))
            scores_va.append(s_va)
            scores_te.append(s_te)
            
        m_va = np.mean(scores_va)
        m_te = np.mean(scores_te)
        print(f"K={k:1d}, Lam={lam:3d} | Val: s42={scores_va[0]:.2f}, s43={scores_va[1]:.2f}, s44={scores_va[2]:.2f} (mean {m_va:.2f}) | Test: s42={scores_te[0]:.2f}, s43={scores_te[1]:.2f}, s44={scores_te[2]:.2f} (mean {m_te:.2f})")
