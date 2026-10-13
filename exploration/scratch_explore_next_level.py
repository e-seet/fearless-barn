#!/usr/bin/env python3
"""Rapid evaluation of Next-Level Architectures:
1. Continuous GMM Soft Gating (soft posterior responsibilities)
2. Logistic / Neural Mixture-of-Experts Gating Network (continuous instance-level routing)
across all 3 seeds (42, 43, 44).
"""
import sys as _s, pathlib as _p; _r = _p.Path(__file__).resolve().parents[1]; exec(open(_r / "tools" / "repo_layout.py").read()); dual(_r / "exploration")
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

sys.path.insert(0, r"c:\Users\Admin\Desktop\eddie\fearless-barn\exploration")
import config
from common import dataset as D

scratch_dir = Path(r"c:\Users\Admin\.gemini\antigravity-ide\brain\9f42e2ac-0ba3-4fa4-86ca-b021c7eca06d\scratch")
t = np.load(scratch_dir / "targets.npz", allow_pickle=True)

yw, yva, yte = t["yw"], t["yva"], t["yte"]
ap_w, ap_va, ap_te = t["ap_w"], t["ap_va"], t["ap_te"]
wtc_w, wtc_va, wtc_te = t["wtc_w"], t["wtc_va"], t["wtc_te"]
has_rec_w, has_rec_va, has_rec_te = t["has_rec_w"], t["has_rec_va"], t["has_rec_te"]

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

preds = {}
for seed in [42, 43, 44]:
    preds[seed] = np.load(scratch_dir / f"preds_seed{seed}.npz", allow_pickle=True)

print("\n--- BASELINE: Reigning Champion (airport_unsup_multimodel K=6, Lam=1500) ---")
# Baseline numbers:
# Val: 247.56 / 248.29 / 251.30 (mean 249.05)
# Test: 244.35 / 241.67 / 239.87 (mean 241.96)

# =========================================================================
# EXPERIMENT 1: Continuous GMM Soft Gating
# =========================================================================
print("\n=== EXPERIMENT 1: Continuous GMM Soft Posterior Gating ===")
for k in [4, 5, 6]:
    gmms = {}
    scalers = {}
    gamma_w = {}
    gamma_va = {}
    gamma_te = {}
    
    for a in np.unique(ap_w):
        mw = (ap_w == a)
        mva = (ap_va == a)
        mte = (ap_te == a)
        scaler = StandardScaler()
        Xw_s = scaler.fit_transform(X_w.loc[mw, CLUSTER_FEATURES].fillna(0).to_numpy())
        Xva_s = scaler.transform(X_va.loc[mva, CLUSTER_FEATURES].fillna(0).to_numpy())
        Xte_s = scaler.transform(X_te.loc[mte, CLUSTER_FEATURES].fillna(0).to_numpy())
        
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=3, reg_covar=1e-3)
        gmm.fit(Xw_s)
        gmms[a] = gmm
        scalers[a] = scaler
        gamma_w[a] = gmm.predict_proba(Xw_s)
        gamma_va[a] = gmm.predict_proba(Xva_s)
        gamma_te[a] = gmm.predict_proba(Xte_s)
        
    for lam in [500, 1500, 3000]:
        val_scores, test_scores = [], []
        for seed in [42, 43, 44]:
            p = preds[seed]
            pa_w, pa_va, pa_te = p["airport_operator_blend_w"], p["airport_operator_blend_va"], p["airport_operator_blend_te"]
            pb_w, pb_va, pb_te = p["mlp_antigravity_w"], p["mlp_antigravity_va"], p["mlp_antigravity_te"]
            
            pred_va = np.zeros(len(pa_va), dtype="float64")
            pred_te = np.zeros(len(pa_te), dtype="float64")
            
            for a in np.unique(ap_w):
                mw = (ap_w == a)
                mva = (ap_va == a)
                mte = (ap_te == a)
                
                d_ap = pa_w[mw] - pb_w[mw]
                r_ap = yw[mw] - pb_w[mw]
                denom_ap = float(np.dot(d_ap, d_ap))
                a_ap = 0.5 if denom_ap <= 1e-12 else float(np.clip(np.dot(d_ap, r_ap) / denom_ap, 0.0, 1.0))
                
                reg_sub_w = reg_w[mw]
                reg_sub_va = reg_va[mva]
                reg_sub_te = reg_te[mte]
                
                probs_w = gamma_w[a]
                probs_va = gamma_va[a]
                probs_te = gamma_te[a]
                
                alpha_cell = np.zeros((len(np.unique(reg_w)), k))
                reg_list = sorted(list(np.unique(reg_w)))
                
                a_parent = {}
                for r_idx, r_name in enumerate(reg_list):
                    m_r = (reg_sub_w == r_name)
                    if m_r.sum() < 200:
                        a_parent[r_name] = a_ap
                        continue
                    d_r = pa_w[mw][m_r] - pb_w[mw][m_r]
                    r_val = yw[mw][m_r] - pb_w[mw][m_r]
                    denom_r = float(np.dot(d_r, d_r))
                    a_parent[r_name] = 0.5 if denom_r <= 1e-12 else float(np.clip(np.dot(d_r, r_val) / denom_r, 0.0, 1.0))
                    
                for r_idx, r_name in enumerate(reg_list):
                    m_r = (reg_sub_w == r_name)
                    p_parent = a_parent[r_name]
                    for c_idx in range(k):
                        w_weights = probs_w[m_r, c_idx]
                        eff_n = float(np.sum(w_weights))
                        if eff_n < 30:
                            alpha_cell[r_idx, c_idx] = p_parent
                            continue
                        d_c = pa_w[mw][m_r] - pb_w[mw][m_r]
                        r_c = yw[mw][m_r] - pb_w[mw][m_r]
                        denom_c = float(np.sum(w_weights * (d_c ** 2)))
                        numer_c = float(np.sum(w_weights * d_c * r_c))
                        raw_a = 0.5 if denom_c <= 1e-12 else float(np.clip(numer_c / denom_c, 0.0, 1.0))
                        
                        shrink = eff_n / (eff_n + lam)
                        alpha_cell[r_idx, c_idx] = shrink * raw_a + (1.0 - shrink) * p_parent
                        
                alpha_va_sub = np.zeros(mva.sum(), dtype="float64")
                for r_idx, r_name in enumerate(reg_list):
                    m_r_va = (reg_sub_va == r_name)
                    if not np.any(m_r_va):
                        continue
                    alpha_va_sub[m_r_va] = probs_va[m_r_va] @ alpha_cell[r_idx]
                    
                alpha_te_sub = np.zeros(mte.sum(), dtype="float64")
                for r_idx, r_name in enumerate(reg_list):
                    m_r_te = (reg_sub_te == r_name)
                    if not np.any(m_r_te):
                        continue
                    alpha_te_sub[m_r_te] = probs_te[m_r_te] @ alpha_cell[r_idx]
                    
                pred_va[mva] = alpha_va_sub * pa_va[mva] + (1.0 - alpha_va_sub) * pb_va[mva]
                pred_te[mte] = alpha_te_sub * pa_te[mte] + (1.0 - alpha_te_sub) * pb_te[mte]
                
            val_scores.append(rmse(yva, clip(pred_va)))
            test_scores.append(rmse(yte, clip(pred_te)))
            
        print(f"GMM K={k}, Lam={lam:4d} | Val: mean {np.mean(val_scores):.2f} (s42={val_scores[0]:.2f}, s43={val_scores[1]:.2f}, s44={val_scores[2]:.2f}) | Test: mean {np.mean(test_scores):.2f} (s42={test_scores[0]:.2f}, s43={test_scores[1]:.2f}, s44={test_scores[2]:.2f})")

# =========================================================================
# EXPERIMENT 2: Continuous Parametric Mixture-of-Experts Gating Network
# =========================================================================
print("\n=== EXPERIMENT 2: Parametric Mixture-of-Experts (MoE) Gating Network ===")
gate_cols = CLUSTER_FEATURES
scaler_g = StandardScaler()
Xw_gate = scaler_g.fit_transform(X_w[gate_cols].fillna(0).to_numpy())
Xva_gate = scaler_g.transform(X_va[gate_cols].fillna(0).to_numpy())
Xte_gate = scaler_g.transform(X_te[gate_cols].fillna(0).to_numpy())

reg_cats = ["Heavy+", "NoRecord", "Standard"]
R_w = np.column_stack([(reg_w == r).astype(float) for r in reg_cats])
R_va = np.column_stack([(reg_va == r).astype(float) for r in reg_cats])
R_te = np.column_stack([(reg_te == r).astype(float) for r in reg_cats])

for C_reg in [0.01, 0.1, 1.0, 10.0]:
    val_moe, test_moe = [], []
    for seed in [42, 43, 44]:
        p = preds[seed]
        pa_w, pa_va, pa_te = p["airport_operator_blend_w"], p["airport_operator_blend_va"], p["airport_operator_blend_te"]
        pb_w, pb_va, pb_te = p["mlp_antigravity_w"], p["mlp_antigravity_va"], p["mlp_antigravity_te"]
        
        pred_va = np.zeros(len(pa_va), dtype="float64")
        pred_te = np.zeros(len(pa_te), dtype="float64")
        
        for a in np.unique(ap_w):
            mw = (ap_w == a)
            mva = (ap_va == a)
            mte = (ap_te == a)
            
            Xa_w = np.column_stack([Xw_gate[mw], R_w[mw]])
            Xa_va = np.column_stack([Xva_gate[mva], R_va[mva]])
            Xa_te = np.column_stack([Xte_gate[mte], R_te[mte]])
            
            r_vec = yw[mw] - pb_w[mw]
            d_vec = pa_w[mw] - pb_w[mw]
            
            denom_ap = float(np.dot(d_vec, d_vec))
            a_scalar = 0.5 if denom_ap <= 1e-12 else float(np.clip(np.dot(d_vec, r_vec) / denom_ap, 0.05, 0.95))
            b_init = np.log(a_scalar / (1.0 - a_scalar))
            theta_init = np.zeros(Xa_w.shape[1] + 1)
            theta_init[-1] = b_init
            
            sub_idx = np.random.RandomState(42).choice(len(r_vec), size=min(len(r_vec), 25000), replace=False)
            Xa_sub = Xa_w[sub_idx]
            r_sub = r_vec[sub_idx]
            d_sub = d_vec[sub_idx]
            
            def loss_fn(theta):
                w_vec = theta[:-1]
                bias = theta[-1]
                logits = np.clip(Xa_sub @ w_vec + bias, -10.0, 10.0)
                g = 1.0 / (1.0 + np.exp(-logits))
                res = r_sub - g * d_sub
                reg = 0.5 * (1.0 / C_reg) * np.sum(w_vec ** 2)
                return np.mean(res ** 2) + reg / len(r_sub)
                
            res = minimize(loss_fn, theta_init, method="L-BFGS-B", options={"maxiter": 80})
            w_opt = res.x[:-1]
            b_opt = res.x[-1]
            
            g_va = 1.0 / (1.0 + np.exp(-np.clip(Xa_va @ w_opt + b_opt, -10.0, 10.0)))
            g_te = 1.0 / (1.0 + np.exp(-np.clip(Xa_te @ w_opt + b_opt, -10.0, 10.0)))
            
            pred_va[mva] = g_va * pa_va[mva] + (1.0 - g_va) * pb_va[mva]
            pred_te[mte] = g_te * pa_te[mte] + (1.0 - g_te) * pb_te[mte]
            
        val_moe.append(rmse(yva, clip(pred_va)))
        test_moe.append(rmse(yte, clip(pred_te)))
        
    print(f"Parametric MoE C={C_reg:5.2f} | Val: mean {np.mean(val_moe):.2f} (s42={val_moe[0]:.2f}, s43={val_moe[1]:.2f}, s44={val_moe[2]:.2f}) | Test: mean {np.mean(test_moe):.2f} (s42={test_moe[0]:.2f}, s43={test_moe[1]:.2f}, s44={test_moe[2]:.2f})")
