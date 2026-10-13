#!/usr/bin/env python3
"""Arm per_airport_tuned: a hyperparameter configuration chosen per airport.

The hypothesis
--------------
`per_airport` already gives each airport its own **stopping point** — measured
best iterations run 72 (LSZH) to 195 (EDDF) where the pooled model compromised at
259 — and that alone was worth 19.8 s of val RMSE. If the airports differ that
much in how long they need to train, they may also differ in how much capacity
they want: a 90k-row airport with a simple layout and a 180k-row airport with a
complex one are not obviously served by the same `num_leaves`.

The control this arm exists to provide
--------------------------------------
Picking a configuration per airport is **ten selections on the validation labels
instead of one**, so a naive comparison against the single-config baseline
conflates two things: whether per-airport tuning genuinely helps, and whether ten
bites at the val apple flatter it. This arm therefore reports three numbers from
the same sweep:

  A. **baseline** — the pooled arm's default config, applied to every airport.
     This is the existing `per_airport` arm.
  B. **best single global config** — the config with the lowest *summed* val SSE
     across airports, applied to every airport. One selection on val.
  C. **per-airport config** — each airport takes its own best config. Ten
     selections on val.

**C minus B is the question.** If per-airport tuning only matches B, the honest
conclusion is "tune once globally"; the extra nine selections bought nothing but
optimism. B minus A says whether the default config was simply a poor choice,
which is a different and cheaper finding.

Selection surface: VAL, declared. The test set is not read here.
"""
from __future__ import annotations

import sys as _s, pathlib as _p; _r = _p.Path(__file__).resolve().parents[2]; exec(open(_r / "tools" / "repo_layout.py").read()); dual(_r / "exploration")
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common import dataset as D
from common import features as F
from common.metrics import rmse
from common.bootstrap import (declare_selection_set, run_dir, seed_everything,
                              write_json_guarded)
from pipeline_per_airport.model import PerAirportLGBM, AIRPORT_COL
from pipeline_lightgbm.train import PARAMS as BASE

# Same grid as stage7_tune_lightgbm.py, so a per-airport result is comparable
# with the pooled tuning result (which bought 6.51 s).
GRID = {
    "base_lr05_lv255_md100": {},
    "lr02_lv255_md100": {"learning_rate": 0.02},
    "lr02_lv511_md50": {"learning_rate": 0.02, "num_leaves": 511, "min_data_in_leaf": 50},
    "lr05_lv511_md50": {"num_leaves": 511, "min_data_in_leaf": 50},
    "lr05_lv127_md200": {"num_leaves": 127, "min_data_in_leaf": 200},
    "lr01_lv255_md100": {"learning_rate": 0.01},
}

ARM = "per_airport_tuned"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--num-boost-round", type=int, default=5000)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu", "cuda"])
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--freeze-from", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    import lightgbm as lgb

    seed_everything(args.seed)
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed,
                  extra=({"tag": args.tag} if args.tag else None))

    Xtr, ytr, pool_meta = D.load_part(args.split, "train", args.featureset,
                                      filter_target=True,
                                      outlier_policy=args.outlier_policy)
    cats = [c for c in F.CATEGORICAL if c in Xtr.columns]
    ytr_a = np.asarray(ytr, dtype="float64")

    refit_only = not config.SPLITS[args.split]["val"]
    frozen = None
    if refit_only:
        if not args.freeze_from:
            raise SystemExit(
                "split '{}' has no validation set, so no configuration can be "
                "selected on it. Pass --freeze-from.".format(args.split))
        frozen = json.loads(Path(args.freeze_from).read_text())
        if frozen.get("split") in config.SELECTION_FORBIDDEN_SPLITS:
            raise SystemExit("--freeze-from points at a non-selection surface.")
    else:
        Xva, yva, meta_va = D.load_part(args.split, "val", args.featureset,
                                        filter_target=False)
        Xva = D.align_columns(Xtr, Xva)
        yva_a = np.asarray(yva, dtype="float64")

    def base_params(overrides):
        p = dict(BASE); p.update(overrides)
        p.update({"seed": args.seed, "bagging_seed": args.seed,
                  "feature_fraction_seed": args.seed, "device_type": args.device})
        return p

    ap_tr = pool_meta[AIRPORT_COL].astype("object").to_numpy()
    airports = sorted(pd.unique(ap_tr))

    # ---------------- frozen path: configs arrive from an honest run -----------
    if refit_only:
        chosen = frozen["config_per_airport"]
        rounds = frozen["best_iteration_per_airport"]
        model = PerAirportLGBM(base_params({}), args.num_boost_round, args.patience, cats)
        for a in airports:
            m = ap_tr == a
            p = base_params(GRID[chosen[str(a)]])
            n_r = max(1, int(round(rounds[str(a)] * (m.sum() / frozen["n_train_per_airport"][str(a)]))))
            d = lgb.Dataset(Xtr[m], label=ytr_a[m], categorical_feature=cats,
                            free_raw_data=False)
            b = lgb.train(p, d, num_boost_round=n_r, callbacks=[lgb.log_evaluation(0)])
            model.boosters_[str(a)] = b
            model.best_iter_[str(a)] = n_r
            model.n_train_[str(a)] = int(m.sum())
            model.convergence_[str(a)] = {"refit_only": True, "rounds": n_r,
                                          "hit_max_rounds": False}
            print("  {:6s} frozen config {:22s} -> {} rounds".format(
                str(a), chosen[str(a)], n_r))
        sweep, chosen_cfg, global_cfg = frozen.get("sweep"), chosen, frozen.get("best_global_config")
        val_rmse = None
    else:
        ap_va = meta_va[AIRPORT_COL].astype("object").to_numpy()
        # sse[config][airport] so the global-config control is computable after
        sse = {name: {} for name in GRID}
        best_it = {name: {} for name in GRID}
        preds = {name: {} for name in GRID}
        print("[{}] sweeping {} configs x {} airports".format(ARM, len(GRID), len(airports)))
        for a in airports:
            mt, mv = ap_tr == a, ap_va == a
            dtr = lgb.Dataset(Xtr[mt], label=ytr_a[mt], categorical_feature=cats,
                              free_raw_data=False)
            line = []
            for name, ov in GRID.items():
                p = base_params(ov)
                dva = lgb.Dataset(Xva[mv], label=yva_a[mv], categorical_feature=cats,
                                  reference=dtr, free_raw_data=False)
                b = lgb.train(p, dtr, num_boost_round=args.num_boost_round,
                              valid_sets=[dva], valid_names=["val"],
                              callbacks=[lgb.early_stopping(args.patience, verbose=False),
                                         lgb.log_evaluation(0)])
                pr = np.clip(b.predict(Xva[mv], num_iteration=b.best_iteration),
                             config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
                sse[name][str(a)] = float(((yva_a[mv] - pr) ** 2).sum())
                best_it[name][str(a)] = int(b.best_iteration)
                preds[name][str(a)] = pr
                line.append("{}={:.1f}".format(name.split("_")[0], rmse(yva_a[mv], pr)))
            print("   {:6s} n_tr={:7d} n_va={:6d}  {}".format(
                str(a), int(mt.sum()), int(mv.sum()), "  ".join(line)))

        n_val = len(yva_a)
        # A: the default config everywhere
        rmse_A = float(np.sqrt(sum(sse["base_lr05_lv255_md100"].values()) / n_val))
        # B: one global config, chosen on summed val SSE
        totals = {name: sum(d.values()) for name, d in sse.items()}
        global_cfg = min(totals, key=totals.get)
        rmse_B = float(np.sqrt(totals[global_cfg] / n_val))
        # C: per-airport config
        chosen_cfg = {a: min(GRID, key=lambda nm: sse[nm][a]) for a in map(str, airports)}
        rmse_C = float(np.sqrt(sum(sse[chosen_cfg[a]][a] for a in chosen_cfg) / n_val))

        print("\n  A  default config everywhere        : val {:.3f}".format(rmse_A))
        print("  B  best SINGLE global config ({:22s}): val {:.3f}  ({:+.3f} vs A)".format(
            global_cfg, rmse_B, rmse_B - rmse_A))
        print("  C  per-airport config               : val {:.3f}  ({:+.3f} vs B)".format(
            rmse_C, rmse_C - rmse_B))
        print("     -> C-B is the per-airport-tuning question; A-B is just "
              "'was the default a bad global choice'")
        print("     per_airport seed spread on val is 0.29 s; a gap under that is noise")
        for a in sorted(chosen_cfg):
            mark = "" if chosen_cfg[a] == global_cfg else "   <-- differs from global"
            print("     {:6s} {:22s}{}".format(a, chosen_cfg[a], mark))

        model = PerAirportLGBM(base_params({}), args.num_boost_round, args.patience, cats)
        for a in map(str, airports):
            mt = ap_tr == a
            p = base_params(GRID[chosen_cfg[a]])
            dtr = lgb.Dataset(Xtr[mt], label=ytr_a[mt], categorical_feature=cats,
                              free_raw_data=False)
            b = lgb.train(p, dtr, num_boost_round=best_it[chosen_cfg[a]][a],
                          callbacks=[lgb.log_evaluation(0)])
            b.best_iteration = best_it[chosen_cfg[a]][a]
            model.boosters_[a] = b
            model.best_iter_[a] = best_it[chosen_cfg[a]][a]
            model.n_train_[a] = int(mt.sum())
            model.convergence_[a] = {"best_iteration": best_it[chosen_cfg[a]][a],
                                     "max_rounds": args.num_boost_round,
                                     "hit_max_rounds": False,
                                     "early_stop_fired": True}
        vp = np.clip(model.predict(Xva, meta_va), config.PREDICTION_CLIP_MIN_SEC,
                     config.PREDICTION_CLIP_MAX_SEC)
        val_rmse = rmse(yva_a, vp)
        print("\n[{}] val RMSE (clipped, refit at chosen configs) = {:.3f}".format(
            ARM, val_rmse))
        sweep = {"val_sse_by_config_and_airport": sse,
                 "best_iteration_by_config_and_airport": best_it,
                 "rmse_A_default_everywhere": rmse_A,
                 "rmse_B_best_single_global": rmse_B,
                 "rmse_C_per_airport_config": rmse_C,
                 "C_minus_B": rmse_C - rmse_B, "B_minus_A": rmse_B - rmse_A}

    consumed = np.concatenate([Xtr.index.to_numpy()[ap_tr == a] for a in airports])
    if len(consumed) != len(Xtr) or set(consumed) != set(Xtr.index.to_numpy()):
        raise SystemExit("per-airport routing consumed {} of {} rows".format(
            len(consumed), len(Xtr)))

    sel = declare_selection_set(
        "per_airport_tuned config + best_iteration (one per airport)",
        selected_on="frozen" if refit_only else "val",
        value={"config_per_airport": chosen_cfg, "best_global_config": global_cfg})

    with open(out / "model.pkl", "wb") as fh:
        pickle.dump(model, fh)
    write_json_guarded(out / "model.json", {
        "arm": ARM, "split": args.split, "featureset": args.featureset,
        "seed": args.seed, "grid": {k: v for k, v in GRID.items()},
        "config_per_airport": chosen_cfg, "best_global_config": global_cfg,
        "best_iteration_per_airport": {k: int(v) for k, v in model.best_iter_.items()},
        "n_train_per_airport": dict(model.n_train_),
        "convergence_per_airport": model.convergence_,
        "sweep": sweep,
        "selection_note": ("a config per airport is TEN selections on val, not one. "
                           "`sweep.C_minus_B` isolates whether that beats a single "
                           "globally-chosen config; `B_minus_A` is the separate, "
                           "cheaper question of whether the default was a poor choice."),
        "val_rmse_clipped": val_rmse, "refit_only": bool(refit_only),
        "selection": dict(sel),
        "n_train": int(len(Xtr)), "n_features": int(Xtr.shape[1]),
        "train_pool_ids_sha": D.pool_fingerprint(consumed),
        "outlier_policy": dict(pool_meta.attrs.get("outlier_policy", {})),
        "categorical_features": cats,
    }, force=args.force)
    print("[{}] saved -> {}".format(ARM, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
