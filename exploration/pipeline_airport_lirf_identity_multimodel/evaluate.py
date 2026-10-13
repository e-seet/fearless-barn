#!/usr/bin/env python3
"""Evaluate pipeline_airport_lirf_identity_multimodel arm."""
from __future__ import annotations

import sys as _s, pathlib as _p; _r = _p.Path(__file__).resolve().parents[2]; exec(open(_r / "tools" / "repo_layout.py").read()); dual(_r / "exploration")
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common import dataset as D
from common import evaluate as E
from common.predict import predict_arm
from common.bootstrap import run_dir, seed_everything, write_json_guarded
from pipeline_airport_lirf_identity_multimodel.train import ARM, apply_lirf_identity_override


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--part", default="val", choices=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--tag", default="idband",
                    help="Which trained variant to evaluate (matches train.py's "
                         "--band / --tag). Default idband, the verified identity.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed,
                  extra=({"tag": args.tag} if args.tag else None))
    # Read the band and base arm back from the RUN's own record, not from the
    # module constants: evaluate must reproduce what that run actually did, and
    # the constants are defaults that can change (they did, in the 2026-09
    # review). A run evaluated under different settings than it was trained on
    # is a silently wrong number.
    meta_json = json.loads((out / "model.json").read_text())
    threshold = meta_json["threshold_sec"]
    band_hi = meta_json.get("identity_band_hi_sec")
    base_arm = meta_json.get("base_arm")
    base_tag = meta_json.get("base_tag")
    if base_arm is None:
        raise SystemExit("model.json has no base_arm; re-run train.py for this run dir")

    Xtr, ytr, _ = D.load_part(args.split, "train", args.featureset,
                              filter_target=True, outlier_policy=args.outlier_policy)
    X, y, meta = D.load_part(args.split, args.part, args.featureset, filter_target=False)

    p_base = predict_arm(base_arm, args.split, args.part, args.featureset, args.seed,
                         args.outlier_policy, X=X, meta=meta, Xtr=Xtr, tag=base_tag)
    ap_arr = meta["ADEP_mvt"].astype("object").to_numpy()
    pred, n_overridden = apply_lirf_identity_override(p_base, X, ap_arr, threshold,
                                                      band_hi=band_hi)

    res = E.evaluate(y, pred, meta, ytr, part=args.part, arm=ARM, split=args.split,
                     featureset=args.featureset, seed=args.seed,
                     extra={"threshold_sec": threshold, "n_overridden": n_overridden,
                            "band": meta_json.get("band"), "base_arm": base_arm})
    E.print_result(res)
    write_json_guarded(out / "eval_{}.json".format(args.part), res, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
