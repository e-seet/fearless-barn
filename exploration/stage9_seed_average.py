#!/usr/bin/env python3
"""Stage 9: what does averaging an arm's predictions across seeds buy?

Why it is worth a stage of its own
----------------------------------
This build has measured, repeatedly, that its neural arms are seed-unstable:
`mlp_antigravity` swings 65.8 s of test RMSE across seeds 42/43/44 and
`ensemble_antigravity` 35.6 s, where the LightGBM family swings ~2 s. A model
whose predictions move that much between seeds has variance that averaging
removes almost for free -- and every seed's model is already trained and on disk,
so this costs prediction passes only.

It is also the honest form of the one Mode 0 lever this build identified and
never used: `kaggle_solutions/facial-keypoints-detection/sol1_four_stage_pipeline/`
reports its best number from an ensemble of 7 and 5 models *within* each group,
not from one model per group. Averaging seeds is that idea in its simplest form.

What it does NOT do: reduce bias. If an arm is systematically wrong in a band,
three seeds of it are systematically wrong in the same band. So the expected gain
is concentrated in the high-variance arms, and a flat result for the GBM family
would be the correct outcome rather than a disappointment.

Reports single-seed and seed-averaged RMSE side by side on val and test, per arm.
Selection surface: none -- this stage measures, it does not choose.
"""
from __future__ import annotations

import sys as _s, pathlib as _p; _r = _p.Path(__file__).resolve().parents[1]; exec(open(_r / "tools" / "repo_layout.py").read()); dual(_r / "exploration")
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from common import dataset as D
from common.metrics import rmse
from common.predict import predict_arm
from common.bootstrap import run_dir, write_json_guarded
from stage6_clip_sweep import band_sse


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--arms", default="per_airport,per_airport_stack,lightgbm,"
                                      "ensemble_antigravity,mlp_antigravity",
                    help="comma-separated; an arm is skipped unless >=2 seeds exist")
    ap.add_argument("--tags", default="per_airport_stack=inner2",
                    help="comma-separated arm=tag pairs for arms whose run is tagged")
    ap.add_argument("--seeds", default=",".join(str(s) for s in config.SWEEP_SEEDS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    tags = dict(kv.split("=", 1) for kv in args.tags.split(",") if "=" in kv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    Xtr, _, _ = D.load_part(args.split, "train", args.featureset, filter_target=True,
                            outlier_policy=args.outlier_policy)

    report = {"split": args.split, "featureset": args.featureset,
              "outlier_policy": args.outlier_policy, "seeds": seeds, "parts": {}}
    for part in ("val", "test"):
        X, y, meta = D.load_part(args.split, part, args.featureset, filter_target=False)
        y = np.asarray(y, dtype="float64")
        rows = []
        print("\n=== {} / {} (n={}) ===".format(args.split, part, y.size))
        print("  {:22s} {:>10s} {:>10s} {:>10s} {:>9s} {:>8s}".format(
            "arm", "best seed", "worst seed", "AVERAGED", "gain", "n_seeds"))
        for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
            preds, used = [], []
            for s in seeds:
                d = run_dir(config.ARTIFACTS / "runs", arm, protocol=proto, seed=s,
                            extra=({"tag": tags[arm]} if arm in tags else None),
                            create=False)
                if not d.exists():
                    continue
                try:
                    preds.append(predict_arm(arm, args.split, part, args.featureset,
                                             s, args.outlier_policy, X=X, meta=meta,
                                             Xtr=Xtr, tag=tags.get(arm)))
                    used.append(s)
                except Exception as exc:                      # noqa: BLE001
                    print("  (skip {} seed {}: {})".format(arm, s, exc))
            if len(preds) < 2:
                print("  (skip {}: {} seed(s) available, need >=2)".format(arm, len(preds)))
                continue
            singles = [rmse(y, p) for p in preds]
            avg = np.mean(np.vstack(preds), axis=0)
            r_avg = rmse(y, avg)
            print("  {:22s} {:10.2f} {:10.2f} {:10.2f} {:+9.2f} {:8d}".format(
                arm, min(singles), max(singles), r_avg,
                r_avg - float(np.mean(singles)), len(preds)))
            rows.append({"arm": arm, "seeds": used, "per_seed_rmse": singles,
                         "mean_single_rmse": float(np.mean(singles)),
                         "averaged_rmse": r_avg,
                         "gain_vs_mean_single": float(np.mean(singles)) - r_avg,
                         "gain_vs_best_single": min(singles) - r_avg,
                         "bands_averaged": band_sse(y, avg)})
        report["parts"][part] = rows

    write_json_guarded(
        config.ARTIFACTS / "seed_average" / "report__{}.json".format(proto),
        report, force=args.force)
    print("\n  note: a gain here is variance reduction, not a better model. It is "
          "expected to be large for the neural arms and near zero for the GBM "
          "family, whose seed spread is ~2 s to begin with.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
