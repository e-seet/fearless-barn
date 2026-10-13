#!/usr/bin/env python3
"""Month-fold residual corrector screen (PREREGISTERED.md)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT); sys.path.insert(0, str(ROOT))
from common import dataset as D
from pipeline_airport_operator_catboost_multimodel.train import solve_weights_per_airport_simplex
from tests_ftt.solve_vs_strong_mlp import surface
from tests_ftt.check_6seed import blend, ftt_mean
F, T = ROOT / "tests_ftt", ROOT / "tests_tabnet"
rm = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
S6, TW = range(42, 48), 0.459
fi = lambda s: F / f"preds_seed{s}_forward_inner_lr1e-3.npz"
fw = lambda s: F / (f"preds_seed{s}_lr1e-3_e132.npz" if s <= 44 else f"preds_seed{s}_lr1e-3_e175.npz")
sf = lambda s: F / f"preds_seed{s}_seasonal_full_lr1e-3.npz"
cfg = json.loads((F / "confirm_seasonal_full.json").read_text())


def base_for(split, part, key, mlp_seeds, fpath, tabtag):
    yw, apw, tw, mw = surface("forward_inner", "val", mlp_seeds)
    W = solve_weights_per_airport_simplex(yw, ftt_mean(S6, fi, "pv"), tw, mw, apw, l2_shrinkage=0.0)
    y, ap, tree, mlp = surface(split, part, mlp_seeds)
    b = blend(W, ftt_mean(S6, fpath, key), tree, mlp, ap)
    tab = np.mean([np.load(T / f"preds_seed{s}_{tabtag}.npz")[key] for s in (42, 43, 44)], axis=0)
    L = ap == "LIRF"; b[L] = (1 - TW) * b[L] + TW * tab[L]
    Xtr, _, _ = D.load_part(split, "train", "with_aobt", filter_target=True, outlier_policy="keep_all")
    X, _, meta = D.load_part(split, part, "with_aobt", filter_target=False)
    X = D.align_columns(Xtr, X).copy()
    X["base_pred"] = b
    return X, y, b, meta["month"].astype(str).to_numpy()


M12 = list(range(42, 54))
Xv, yv, bv, mv = base_for("forward", "val", "pv", M12, fw, "lr2e-3p")
Xt, yt, bt, _ = base_for("forward", "test", "pt", M12, fw, "lr2e-3p")
Xs, ys, bs, _ = base_for("seasonal_full", "test", "pt", cfg["mlp_seeds"], sf, "sf_lr2e-3p")
tr, es = mv == "2025-09", mv == "2025-10"
ok = np.isfinite(yv)
params = dict(objective="l2", num_leaves=15, min_data_in_leaf=500, learning_rate=0.03, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbosity=-1, seed=42)
dtr = lgb.Dataset(Xv[tr & ok], (yv - bv)[tr & ok]); des = lgb.Dataset(Xv[es & ok], (yv - bv)[es & ok], reference=dtr)
m = lgb.train(params, dtr, 3000, valid_sets=[des], callbacks=[lgb.early_stopping(100, verbose=False)])
print(f"corrector best iteration {m.best_iteration} (Sep train {int((tr&ok).sum())}, Oct early-stop {int((es&ok).sum())})")
out = {"best_iteration": m.best_iteration}
for name, X, y, b in (("Nov-Dec (forward test)", Xt, yt, bt), ("Jan+Jul (seasonal_full test)", Xs, ys, bs)):
    X = X.reindex(columns=Xv.columns)
    c = b + m.predict(X, num_iteration=m.best_iteration)
    okk = np.isfinite(y); ib = okk & (y <= 7200)
    r = {"inband": (rm(y[ib], b[ib]), rm(y[ib], c[ib])), "pooled": (rm(y[okk], b[okk]), rm(y[okk], c[okk])),
         "pooled_half": rm(y[okk], (b + 0.5 * (c - b))[okk])}
    g = (y - b) ** 2 - (y - c) ** 2; g[~okk] = 0; j = int(np.argmax(g)); k = okk.copy(); k[j] = False
    r["pooled_drop_best_row"] = (rm(y[k], b[k]), rm(y[k], c[k]))
    out[name] = r
    print(f"{name}: in-band {r['inband'][0]:.3f} -> {r['inband'][1]:.3f} ({r['inband'][1]-r['inband'][0]:+.3f}) | "
          f"pooled {r['pooled'][0]:.3f} -> {r['pooled'][1]:.3f} ({r['pooled'][1]-r['pooled'][0]:+.3f}) | half-strength pooled {r['pooled_half']:.3f} | "
          f"best row removed {r['pooled_drop_best_row'][1]-r['pooled_drop_best_row'][0]:+.3f}")
ok_ = lambda k: out[k]["inband"][1] < out[k]["inband"][0] and out[k]["pooled"][1] < out[k]["pooled"][0]
s = out["Jan+Jul (seasonal_full test)"]
out["PASS"] = bool(ok_("Nov-Dec (forward test)") and ok_("Jan+Jul (seasonal_full test)") and s["pooled_drop_best_row"][1] < s["pooled_drop_best_row"][0])
print("PRE-REGISTERED:", "PASS" if out["PASS"] else "FAIL")
(HERE / "result.json").write_text(json.dumps(out, indent=2, default=float))
