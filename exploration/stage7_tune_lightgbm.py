#!/usr/bin/env python3
"""Stage 7: a small, declared hyperparameter sweep for the LightGBM arm.

Scope and honesty
-----------------
This tunes ONE arm. It is not a fair-fight retune of every candidate, so its
output may NOT be used to re-crown a winner over an untuned rival -- that would
be comparing a tuned model against an untuned one and calling the gap "method".
Its output is used for exactly two things: (1) the configuration the submission
model is refit with, and (2) an honest statement of how much headroom simple
tuning buys on this problem.

Every configuration gets its OWN artifact path via `train.py --tag`, because the
protocol key (split-featureset-policy, seed) does not name hyperparameters, and a
sweep that varies something the key does not name overwrites the run it is being
compared against (MISTAKES.md #35/#50).

Selection surface: VAL. The test set is never read here.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from common.bootstrap import declare_selection_set, run_dir, write_json_guarded

ROOT = Path(__file__).resolve().parent

# name -> flags. The baseline is included deliberately: a sweep that does not
# re-measure its own starting point cannot tell a real gain from a changed
# environment (a different feature revision, a different outlier policy).
GRID = {
    "base_lr05_lv255_md100": [],
    "lr02_lv255_md100": ["--learning-rate", "0.02"],
    "lr02_lv511_md50": ["--learning-rate", "0.02", "--num-leaves", "511",
                        "--min-data-in-leaf", "50"],
    "lr05_lv511_md50": ["--num-leaves", "511", "--min-data-in-leaf", "50"],
    "lr05_lv127_md200": ["--num-leaves", "127", "--min-data-in-leaf", "200"],
    "lr01_lv255_md100": ["--learning-rate", "0.01"],
}


def run(cmd: "list[str]") -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.call([sys.executable] + cmd, cwd=str(ROOT))
    print("[{:.1f}s rc={}]".format(time.time() - t0, rc), flush=True)
    if rc != 0:
        raise SystemExit("HALT: {} failed rc={}".format(" ".join(cmd), rc))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--only", default="", help="comma-separated subset of GRID keys")
    ap.add_argument("--skip-train", action="store_true",
                    help="collect results from runs already on disk")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if args.split in config.SELECTION_FORBIDDEN_SPLITS:
        raise SystemExit("split {!r} is not a selection surface".format(args.split))

    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(GRID)
    unknown = [n for n in names if n not in GRID]
    if unknown:
        raise SystemExit("unknown config(s) {}; known: {}".format(unknown, list(GRID)))

    base = ["--split", args.split, "--featureset", args.featureset,
            "--outlier-policy", args.outlier_policy, "--seed", str(args.seed)]
    f = ["--force"] if args.force else []

    rows = []
    for name in names:
        tag = ["--tag", name]
        if not args.skip_train:
            run(["pipeline_lightgbm/train.py"] + base + GRID[name] + tag + f)
            run(["-m", "pipeline_lightgbm.evaluate"] + base + tag
                + ["--part", "val"] + f)
        d = run_dir(config.ARTIFACTS / "runs", "lightgbm",
                    protocol=config.protocol_key(args.split, args.featureset,
                                                 args.outlier_policy),
                    seed=args.seed, extra={"tag": name}, create=False)
        m = json.loads((d / "model.json").read_text())
        v = json.loads((d / "eval_val.json").read_text())
        rows.append({
            "config": name, "flags": GRID[name],
            "val_rmse": v["rmse"], "val_rmse_in_band": v.get("rmse_in_band"),
            "val_lb_weighted_rmse": v.get("leaderboard_weighted_rmse"),
            "best_iteration": m.get("best_iteration"),
            "early_stop_fired": m.get("convergence", {}).get("early_stop_fired"),
            "hit_max_epochs": m.get("convergence", {}).get("hit_max_epochs"),
            "learning_rate": m["params"]["learning_rate"],
            "num_leaves": m["params"]["num_leaves"],
            "min_data_in_leaf": m["params"]["min_data_in_leaf"],
            "feature_fingerprint": m.get("feature_fingerprint"),
            "train_pool_ids_sha": m.get("train_pool_ids_sha"),
            "run_dir": str(d.relative_to(ROOT)),
        })

    fps = {r["feature_fingerprint"] for r in rows}
    pools = {r["train_pool_ids_sha"] for r in rows}
    if len(fps) > 1 or len(pools) > 1:
        raise SystemExit(
            "sweep integrity: configurations differ in feature code {} or training "
            "pool {} -- the gap would not be a hyperparameter effect.".format(fps, pools))

    rows.sort(key=lambda r: r["val_rmse"])
    print("\nranked on VAL RMSE (lower is better; test never read here):")
    print("  {:26s} {:>9s} {:>9s} {:>7s} {:>6s}".format(
        "config", "val", "in-band", "iters", "ES"))
    for r in rows:
        print("  {:26s} {:9.2f} {:9.2f} {:7d} {:>6s}".format(
            r["config"], r["val_rmse"], r["val_rmse_in_band"] or float("nan"),
            r["best_iteration"] or -1,
            "yes" if r["early_stop_fired"] else "NO(cap)"))
    best = rows[0]
    baseline = next((r for r in rows if r["config"].startswith("base")), None)
    if baseline:
        print("  best {} beats the baseline by {:.2f} s of val RMSE".format(
            best["config"], baseline["val_rmse"] - best["val_rmse"]))
    capped = [r["config"] for r in rows if r["hit_max_epochs"]]
    if capped:
        print("  WARNING: {} hit the round cap -- raise it; those are not "
              "converged results.".format(capped))

    sel = declare_selection_set("lightgbm hyperparameters", selected_on="val",
                                value=best["config"])
    out_dir = config.ARTIFACTS / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_guarded(
        out_dir / "lightgbm__{}__seed{}.json".format(
            config.protocol_key(args.split, args.featureset, args.outlier_policy),
            args.seed),
        {"split": args.split, "featureset": args.featureset,
         "outlier_policy": args.outlier_policy, "seed": args.seed,
         "selection": dict(sel), "results": rows, "best": best,
         "baseline": baseline,
         "gain_vs_baseline_sec": (None if not baseline else
                                  baseline["val_rmse"] - best["val_rmse"]),
         "scope_caveat": ("only the LightGBM arm was tuned. This result may not be "
                          "used to re-rank it against an untuned arm; it selects the "
                          "configuration the submission is refit with.")},
        force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
