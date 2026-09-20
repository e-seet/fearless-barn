#!/usr/bin/env python3
"""Assemble the CANDIDATE v28 = v25 with LIRF rows mixed with the TabNet ensemble.

    pred_LIRF = clip((1 - w) * v25_pred + w * mean_seeds(TabNet_final_pred))
    every other row: v25's prediction, byte-identical.

v25's LIRF prediction is 100% MLP (tree weight 0.0, 18 seeds), so this is exactly
the "MLP + TabNet at LIRF" recipe validated on forward and seasonal_full.

usage: python build_candidate.py W [--seeds 42 43 44 45 46 47]

Writes tests_tabnet/build/candidate_v28.parquet (+ .json report). It does NOT write
into submissions/ and does NOT upload anything; submit/upload stays a separate,
user-confirmed step.
"""
import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import config
from common import dataset as D

ap = argparse.ArgumentParser()
ap.add_argument("w", type=float)
ap.add_argument("--seeds", type=int, nargs="+", default=None, help="default: every final_seed*.npz that passes the health gate")
ap.add_argument("--min-seeds", type=int, default=5)
a = ap.parse_args()
W_MIX = a.w
assert 0.0 <= W_MIX <= 1.0

v25 = pd.read_parquet(ROOT / "submissions/fearless_barn_v25.parquet")
X, meta = D.load_ranking_features("with_aobt")            # submitting.parquet order
# Health gate (added after final seed 42 collapsed: Dec val 482, best epoch 2, train loss stuck ~265k):
# a seed is used only if Dec-val RMSE <= 260 (validated TabNet Dec val on seasonal_full was 233-241),
# best epoch >= 10, and early stopping fired below the cap. Rejected seeds are reported, never silently dropped.
GATE_VAL, GATE_EPOCH = 260.0, 10
avail = sorted(int(p.stem.split("seed")[1]) for p in (HERE / "build").glob("final_seed*.npz"))
cand = a.seeds if a.seeds else avail
Zall = {s: np.load(HERE / "build" / f"final_seed{s}.npz", allow_pickle=True) for s in cand}
healthy = lambda z: ("labels_aligned" in z.files) and float(z["val_rmse"]) <= GATE_VAL and int(z["best_epoch"]) >= GATE_EPOCH and not bool(z["hit_cap"])
accepted = [s for s in cand if healthy(Zall[s])]
rejected = [s for s in cand if s not in accepted]
for s in rejected:
    z = Zall[s]
    print(f"REJECTED seed {s}: Dec val {float(z['val_rmse']):.1f} (gate <= {GATE_VAL}), best_epoch {int(z['best_epoch'])} (gate >= {GATE_EPOCH}), hit_cap {bool(z['hit_cap'])}")
a.seeds = accepted
Z = [Zall[s] for s in accepted]

report = {"w": W_MIX, "seeds": accepted, "rejected_seeds": rejected, "checks": {}}
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    report["checks"][name] = {"pass": bool(cond), "detail": str(detail)}
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


# ---- alignment ----
ids_v25 = v25[config.ID].to_numpy()
ids_rank = X.index.to_numpy()
check(f"enough healthy seeds (>= {a.min_seeds})", len(accepted) >= a.min_seeds, f"accepted {accepted}, rejected {rejected}")
check("v25 ids == ranking-feature ids (same order)", np.array_equal(ids_v25, ids_rank), f"n={len(ids_v25)}")
for s, z in zip(a.seeds, Z):
    check(f"seed {s}: rank_ids match", np.array_equal(z["rank_ids"], ids_rank))
    check(f"seed {s}: healthy (Dec val <= {GATE_VAL}, best_epoch >= {GATE_EPOCH}, early-stopped)", healthy(z),
          f"best_epoch {int(z['best_epoch'])}, ran {int(z['epochs_run'])}, Dec val RMSE {float(z['val_rmse']):.2f}")

P = np.array([z["pr"] for z in Z])                        # (n_seeds, n_rows)
tab = P.mean(0)
lirf = (meta["ADEP_mvt"].astype(str).to_numpy() == "LIRF")
base = v25[config.TARGET].to_numpy(dtype="float64")
check("LIRF rows present", lirf.sum() > 10000, f"n_LIRF={int(lirf.sum())} of {len(lirf)} ({lirf.mean():.2%}; scored weight 5.1%)")

new = base.copy()
new[lirf] = np.clip((1 - W_MIX) * base[lirf] + W_MIX * tab[lirf], config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)

# ---- integrity ----
check("no NaN/inf", np.isfinite(new).all())
check("within clip range", (new >= config.PREDICTION_CLIP_MIN_SEC).all() and (new <= config.PREDICTION_CLIP_MAX_SEC).all(),
      f"min {new.min():.1f} max {new.max():.1f}")
changed = new != base
check("only LIRF rows changed", not changed[~lirf].any(), f"{int(changed.sum())} rows changed, all LIRF: {bool(changed.sum() <= lirf.sum())}")
check("non-LIRF rows byte-identical to v25", np.array_equal(new[~lirf], base[~lirf]))

# ---- sanity of the TabNet contribution at LIRF ----
b, t = base[lirf], tab[lirf]
pair = [np.corrcoef(P[i][lirf], P[j][lirf])[0, 1] for i in range(len(P)) for j in range(i + 1, len(P))]
report["lirf"] = {
    "corr_v25_vs_tabnet": float(np.corrcoef(b, t)[0, 1]), "mean_v25": float(b.mean()), "mean_tabnet": float(t.mean()),
    "median_v25": float(np.median(b)), "median_tabnet": float(np.median(t)),
    "max_v25": float(b.max()), "max_tabnet": float(t.max()), "max_new": float(new[lirf].max()),
    "seed_pairwise_corr_min": float(min(pair)), "seed_pairwise_corr_mean": float(np.mean(pair)),
    "mean_abs_change": float(np.abs(new[lirf] - b).mean()), "p99_abs_change": float(np.percentile(np.abs(new[lirf] - b), 99)),
}
for k, v in report["lirf"].items():
    print(f"   LIRF {k}: {v:.4f}")
check("TabNet vs v25 correlated at LIRF (>0.8)", report["lirf"]["corr_v25_vs_tabnet"] > 0.8)
check("TabNet mean within 25% of v25 mean at LIRF", abs(t.mean() / b.mean() - 1) < 0.25)
check("TabNet seeds agree with each other (min pairwise corr > 0.7)", report["lirf"]["seed_pairwise_corr_min"] > 0.7)
check("no runaway tail (new LIRF max <= 1.5x max(v25, tabnet))", new[lirf].max() <= 1.5 * max(b.max(), t.max()))

out = v25.copy()
out[config.TARGET] = new.astype(v25[config.TARGET].dtype)
check("dtype/columns/rows identical to v25", list(out.columns) == list(v25.columns) and out.dtypes.equals(v25.dtypes) and len(out) == len(v25))
report["overall"] = {"mean_v25": float(base.mean()), "mean_new": float(new.mean()), "median_v25": float(np.median(base)), "median_new": float(np.median(new))}
report["all_checks_pass"] = ok
out_path = HERE / "build" / f"candidate_v28_w{W_MIX:g}.parquet"
if ok:
    out.to_parquet(out_path, index=False)
    print(f"\nALL CHECKS PASS -> wrote {out_path}")
else:
    print("\nCHECKS FAILED -> parquet NOT written")
(HERE / "build" / f"candidate_v28_w{W_MIX:g}.json").write_text(json.dumps(report, indent=2))
sys.exit(0 if ok else 1)
