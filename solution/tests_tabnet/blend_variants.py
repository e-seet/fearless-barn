#!/usr/bin/env python3
"""TabNet blend-variant matrix. ALL weights/params solved ONLY on forward_inner val
(Jul-Aug 2025, the project's permitted solve surface); evaluated on two held-out
surfaces: forward test (Nov-Dec 2025) and seasonal_full test (Jan+Jul 2025).

Neural inputs: forward_inner MLP = registered seeds 42-59 (the 18 v25 used); forward
test MLP = 24-seed fast pool (tests_lirf_seeds); seasonal_full MLP = registered 42-47.
TabNet: 3 seeds per surface (inner: build/inner_seed*, forward: preds_seed*_lr2e-3p,
seasonal_full: preds_seed*_sf_lr2e-3p). Tree = registered airport_operator_blend seed 42.

Variants
 V0  shipped blend: v25 per-airport w_a*tree + (1-w_a)*MLP (weights frozen from v25)
 V1  V0 + LIRF override  pred_LIRF = (1-w)*V0 + w*TabNet, w grid-solved on inner LIRF,
     guarded to [0.3, 0.8] (cross-surface optimum range seen: 0.5-1.0)
 V2  V0 + LIRF override with fixed w = 0.5
 V3  full 3-way per-airport SIMPLEX (tree, MLP, TabNet), SLSQP w/ mean(y^2)-normalised
     objective (project solver, sec.2m), re-solved for every airport
 V4  global scalar g: (1-g)*V0 + g*TabNet at all airports, g grid-solved
 V5  gated LIRF mix (exploratory): w = w_hi if V0 pred > tau else w_lo; (tau,w_lo,w_hi) grid

PRE-REGISTERED BARS (before any TabNet forward_inner number was seen):
 B1 V1 must beat V0 pooled by >= 2s on seasonal_full test AND by > 0 on forward test.
 B2 V3, V4, V5 each replace V1 only if they beat V1 pooled by >= 2s on BOTH surfaces;
    otherwise V1 (simplest) stays the recipe.
 Reported also: scored-mix RMSE (leaderboard airport weights), LIRF, non-LIRF mix.
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from common import dataset as D
from common.predict import predict_arm

W = json.load(open(ROOT / "artifacts/audit/audit_report.json"))["leaderboard_airport_weights"]
v25 = json.load(open(ROOT / "artifacts/runs/airport_operator_multimodel__full-with_aobt-keep_all__seed42__tag=msmlp18fi/model.json"))
WPA = {a: float(v25["weights_per_airport"].get(a, 0.5)) for a in W}
rm = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))
LS = ROOT / "tests_lirf_seeds"


def surface(split, part, mlp_fn, tab_fn):
    cache = HERE / "build" / f"surface_cache_{split}_{part}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        yy, ap, tree, mlp, ids = z["y"], z["ap"].astype(str), z["tree"], z["mlp"], z["ids"]
        month = z["month"] if bool(z["has_month"]) else None
    else:
        Xtr, _, _ = D.load_part(split, "train", "with_aobt", filter_target=True, outlier_policy="keep_all")
        X, y, meta = D.load_part(split, part, "with_aobt", filter_target=False)
        X = D.align_columns(Xtr, X)
        ap = meta["ADEP_mvt"].astype(str).to_numpy()
        tree = predict_arm("airport_operator_blend", split, part, "with_aobt", 42, "keep_all", X=X, meta=meta, tag=None)
        yy = np.asarray(y, float)
        mlp = mlp_fn(X, meta, split, part)
        ids = X.index.to_numpy()
        month = meta["month"].to_numpy() if "month" in meta.columns else None
        np.savez(cache, y=yy, ap=ap, tree=tree, mlp=mlp, ids=ids, month=(month if month is not None else np.zeros(1)), has_month=month is not None)
    return dict(y=yy, ap=ap, tree=tree, mlp=mlp, tab=tab_fn(pd.Index(ids), yy, mlp), month=month)


def mlp_registered(seeds):
    return lambda X, meta, split, part: np.mean(
        [predict_arm("mlp_antigravity", split, part, "with_aobt", s, "keep_all", X=X, meta=meta, tag=None) for s in seeds], axis=0)


def mlp_fastpool(X, meta, split, part):
    return np.mean([np.load(LS / f"preds_seed{s}.npz")["pt"] for s in range(48, 72)], axis=0)


def tab_inner(idx, y, mlp):
    # health gate (final seed 42 collapsed: val 482, best epoch 2): a TabNet seed is used only if it
    # early-stopped below the cap, best epoch >= 10, and val RMSE within +40s of the MLP ensemble on the same rows.
    ref = rm(y, mlp)
    good, bad = [], []
    import os
    gate_on = os.environ.get("TAB_INNER_GATE", "on") != "off"
    print(f"inner-seed health gate: {'ON' if gate_on else 'OFF (sensitivity run: every inner seed used)'}")
    for f in sorted((HERE / "build").glob("inner_seed*.npz")):
        q = np.load(f, allow_pickle=True)
        ok_ = (not gate_on) or ((not bool(q["hit_cap"])) and int(q["best_epoch"]) >= 10 and float(q["val_rmse"]) <= ref + 40)
        (good if ok_ else bad).append((f.stem, q, float(q["val_rmse"]), int(q["best_epoch"])))
    print(f"inner TabNet seeds: MLP-ens18 val RMSE on these rows = {ref:.2f}")
    for n, _, v, e in good: print(f"   ACCEPT {n}: val {v:.2f} best_epoch {e}")
    for n, _, v, e in bad: print(f"   REJECT {n}: val {v:.2f} best_epoch {e}")
    if len(good) < 3:
        raise SystemExit(f"only {len(good)} healthy inner TabNet seeds (need 3) -- train replacements (inner seeds 45+)")
    return np.mean([pd.Series(q["pv"], index=q["val_ids"]).reindex(idx).to_numpy() for _, q, _, _ in good], axis=0)


def tab_pref(tag, key):
    return lambda idx, y=None, mlp=None: np.mean([np.load(HERE / f"preds_seed{s}_{tag}.npz")[key] for s in (42, 43, 44)], axis=0)


inner = surface("forward_inner", "val", mlp_registered(range(42, 60)), tab_inner)
fwd = surface("forward", "test", mlp_fastpool, tab_pref("lr2e-3p", "pt"))
sf = surface("seasonal_full", "test", mlp_registered(range(42, 48)), tab_pref("sf_lr2e-3p", "pt"))


def v0(S):
    w = np.array([WPA[a] for a in S["ap"]])
    return w * S["tree"] + (1 - w) * S["mlp"]


def simplex(y, P):
    sc = float(np.mean(y ** 2))
    w0 = np.full(3, 1 / 3)
    r = minimize(lambda w: float(np.mean((P @ w - y) ** 2)) / sc, w0, method="SLSQP", bounds=[(0, 1)] * 3,
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"maxiter": 300})
    return r.x if r.success else w0


# ---- solve everything on forward_inner ----
Li = inner["ap"] == "LIRF"
b0 = v0(inner)
grid = np.round(np.arange(0, 1.0001, 0.05), 2)
w_raw = float(grid[int(np.argmin([rm(inner["y"][Li], (1 - w) * b0[Li] + w * inner["tab"][Li]) for w in grid]))])
w_lirf = float(np.clip(w_raw, 0.3, 0.8))
g = float(grid[int(np.argmin([rm(inner["y"], (1 - g_) * b0 + g_ * inner["tab"]) for g_ in grid]))])
W3 = {}
for a in W:
    m = inner["ap"] == a
    W3[a] = simplex(inner["y"][m], np.column_stack([inner["tree"][m], inner["mlp"][m], inner["tab"][m]]))
cands = []
for t in (1000, 1500, 2000, 3000, 4000):
    for lo in (0, .25, .5, .75, 1):
        for hi in (0, .25, .5, .75, 1):
            w = np.where(b0[Li] > t, hi, lo)
            cands.append((rm(inner["y"][Li], (1 - w) * b0[Li] + w * inner["tab"][Li]), t, lo, hi))
_, T5, LO5, HI5 = min(cands)
print(f"SOLVED on forward_inner: LIRF w raw {w_raw} -> guarded {w_lirf} | global g {g} | gated (tau {T5}, w_lo {LO5}, w_hi {HI5})")
print("3-way simplex weights (tree, MLP, TabNet) per airport:")
for a in sorted(W, key=lambda a: -W[a]):
    print(f"   {a}: " + " ".join(f"{x:.2f}" for x in W3[a]))


def variants(S):
    b = v0(S)
    L = S["ap"] == "LIRF"
    out = {"V0 shipped": b}
    x = b.copy(); x[L] = (1 - w_lirf) * b[L] + w_lirf * S["tab"][L]; out[f"V1 LIRF w={w_lirf}"] = x
    x = b.copy(); x[L] = 0.5 * b[L] + 0.5 * S["tab"][L]; out["V2 LIRF w=0.5"] = x
    p3 = np.zeros_like(b)
    for a in W:
        m = S["ap"] == a
        p3[m] = np.column_stack([S["tree"][m], S["mlp"][m], S["tab"][m]]) @ W3[a]
    out["V3 3-way simplex"] = p3
    out[f"V4 global g={g}"] = (1 - g) * b + g * S["tab"]
    x = b.copy(); w = np.where(b[L] > T5, HI5, LO5); x[L] = (1 - w) * b[L] + w * S["tab"][L]; out["V5 gated LIRF"] = x
    return out


def metrics(S, p):
    y, ap = S["y"], S["ap"]
    mix = np.sqrt(sum(W[a] * rm(y[ap == a], p[ap == a]) ** 2 for a in W))
    non = np.sqrt(sum(W[a] * rm(y[ap == a], p[ap == a]) ** 2 for a in W if a != "LIRF") / (1 - W["LIRF"]))
    return rm(y, p), mix, rm(y[ap == "LIRF"], p[ap == "LIRF"]), non


R = {}
for name, S in (("forward test (Nov-Dec)", fwd), ("seasonal_full test (Jan+Jul)", sf)):
    V = variants(S)
    R[name] = {k: metrics(S, p) for k, p in V.items()}
    v1k = [k for k in V if k.startswith("V1")][0]
    print(f"\n===== {name} =====\n  {'variant':22s} {'pooled':>8s} {'scored-mix':>10s} {'LIRF':>8s} {'non-LIRF':>9s}   d-pooled vs V0 | vs V1")
    for k, m_ in R[name].items():
        print(f"  {k:22s} {m_[0]:8.2f} {m_[1]:10.2f} {m_[2]:8.1f} {m_[3]:9.2f}   {m_[0]-R[name]['V0 shipped'][0]:+7.2f} | {m_[0]-R[name][v1k][0]:+7.2f}")
    if S["month"] is not None and name.startswith("seasonal"):
        for mo in np.unique(S["month"]):
            ix = S["month"] == mo
            print(f"   month {mo}: " + "  ".join(f"{k.split()[0]} {rm(S['y'][ix], p[ix]):.1f}" for k, p in V.items()))

print("\n===== PRE-REGISTERED VERDICTS =====")
f, s = R["forward test (Nov-Dec)"], R["seasonal_full test (Jan+Jul)"]
v1k = [k for k in f if k.startswith("V1")][0]
g1 = (s["V0 shipped"][0] - s[v1k][0], f["V0 shipped"][0] - f[v1k][0])
print(f"B1 V1 vs V0: seasonal_full gain {g1[0]:+.2f} (need >=2), forward gain {g1[1]:+.2f} (need >0) -> {'PASS' if g1[0] >= 2 and g1[1] > 0 else 'FAIL'}")
for k in f:
    if k[:2] in ("V3", "V4", "V5"):
        gf, gs = f[v1k][0] - f[k][0], s[v1k][0] - s[k][0]
        print(f"B2 {k:20s} vs V1: forward {gf:+.2f}, seasonal_full {gs:+.2f} (need >=2 on BOTH) -> {'REPLACES V1' if gf >= 2 and gs >= 2 else 'V1 stays'}")
