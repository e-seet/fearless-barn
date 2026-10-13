#!/usr/bin/env python3
"""Orchestrator: the whole build, in dependency order, as one command.

Stage N+1 does not start until stage N's gate passes -- each step's non-zero exit
halts the run. This is the file the notebook mirrors; keep them in step.

Usage
  python3 run_all.py --stage audit          # stages 1 + 1b only
  python3 run_all.py --featureset with_aobt # one full protocol
  python3 run_all.py --all-protocols        # both splits x both featuresets
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

ROOT = Path(__file__).resolve().parent
ARMS = list(config.ARMS)   # one list, in config (see config.ARMS)


def run(cmd: "list[str]") -> None:
    print("\n" + "=" * 78)
    print("$ " + " ".join(cmd))
    print("=" * 78, flush=True)
    t0 = time.time()
    rc = subprocess.call([sys.executable] + cmd, cwd=str(ROOT))
    print("[{:.1f}s, rc={}]".format(time.time() - t0, rc))
    if rc != 0:
        raise SystemExit(
            "HALT: `{}` failed with rc={}. A stage gate did not pass; the next "
            "stage must not run on its output.".format(" ".join(cmd), rc))


def protocol(split: str, featureset: str, policy: str, seed: int, force: bool) -> None:
    f = ["--force"] if force else []
    base = ["--split", split, "--featureset", featureset,
            "--outlier-policy", policy, "--seed", str(seed)]
    run(["stage4_controls.py"] + base + f)
    run(["stage5_ablations.py"] + base + f)
    for arm in ARMS:
        run(["pipeline_{}/train.py".format(arm)] + base + f)
        for part in ("val", "test"):
            run(["-m", "pipeline_{}.evaluate".format(arm)] + base + ["--part", part] + f)
    run(["compare.py"] + base + f)


def sweep(split: str, featureset: str, policy: str, arms: "list[str]",
          force: bool) -> None:
    """Repeat-seed sweep over the TOP-TWO arms (design-gate finding S7).

    Root CLAUDE.md requires the winner's margin to survive a seed sweep -- and a
    margin has two sides, so sweeping only the winner cannot check it. Sweeping
    one side is also the shape of MISTAKES.md #73: an orchestrator that silently
    omitted the seed invocations behind a registered range.
    """
    f = ["--force"] if force else []
    for seed in config.SWEEP_SEEDS:
        for arm in arms:
            base = ["--split", split, "--featureset", featureset,
                    "--outlier-policy", policy, "--seed", str(seed)]
            run(["pipeline_{}/train.py".format(arm)] + base + f)
            for part in ("val", "test"):
                run(["-m", "pipeline_{}.evaluate".format(arm)] + base
                    + ["--part", part] + f)
        run(["compare.py", "--split", split, "--featureset", featureset,
             "--outlier-policy", policy, "--seed", str(seed)] + f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=["audit", "split", "all", "sweep"])
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--sweep-policies", action="store_true",
                    help="run every outlier policy (design-gate B1: the training "
                         "tail filter is an axis to sweep, not a constant)")
    ap.add_argument("--sweep-arms", default="ensemble_antigravity,mlp_antigravity,lightgbm",
                    help="arms carried through the repeat-seed sweep; must be at "
                         "least the top TWO, since a margin has two sides. The "
                         "default was 'lightgbm,ridge' -- neither of which is in "
                         "the top two under the corrected protocol, so the sweep "
                         "it ran could not have checked the winner's margin at "
                         "all. It now names the measured top two plus this "
                         "build's own best arm. NOTE the blend arm needs its BASE "
                         "arms present at the same seed, so a sweep including it "
                         "should pass all six.")
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--all-protocols", action="store_true",
                    help="every split x featureset combination")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    f = ["--force"] if args.force else []

    if args.stage not in ("sweep",):
        run(["stage1_audit.py"] + f)
        run(["stage1b_label_quality.py"] + f)
        if args.stage == "audit":
            return 0

        # 'full' is the refit-only protocol; it is built here but trained separately
        # (submit_refit), because it has no val and nothing may be selected on it.
        splits = ([s for s in sorted(config.SPLITS)] if args.all_protocols
                  else [args.split])
        for s in splits:
            run(["stage2_split.py", "--split", s] + f)
        if args.stage == "split":
            return 0
    else:
        splits = [args.split]

    fsets = list(config.FEATURESETS) if args.all_protocols else [args.featureset]
    policies = (sorted(config.OUTLIER_POLICIES) if args.sweep_policies
                else [args.outlier_policy])
    trainable = [s for s in splits if config.SPLITS[s]["val"]]
    if args.stage == "sweep":
        arms = [a.strip() for a in args.sweep_arms.split(",") if a.strip()]
        if len(arms) < 2:
            raise SystemExit("--sweep-arms needs at least two arms: a margin has "
                             "two sides, and sweeping one cannot confirm a gap")
        for s in trainable:
            for fs in fsets:
                for pol in policies:
                    sweep(s, fs, pol, arms, args.force)
        return 0

    for s in trainable:
        for fs in fsets:
            for pol in policies:
                protocol(s, fs, pol, args.seed, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
