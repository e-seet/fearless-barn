#!/usr/bin/env python3
"""TabNet-at-LIRF override on top of a base blend -- registered as a real arm.

    pred[LIRF] = (1 - w) * base[LIRF] + w * mean(healthy TabNet seeds)[LIRF]
    every other row: the base arm's prediction, unchanged.

This is the recipe behind **v29 = 293.6685, the best board score**. Until now it
existed only as `tests_tabnet/{train_tabnet_pred,blend_variants,build_candidate}.py`
operating on `submissions/fearless_barn_v25.parquet` as a file, with
`"not_reproducible_via_submit_py": true` written into its own meta. The Sep 4
data refresh had already forced a rebuild of all 17 submissions once and got v17
rejected for staleness; if `ranking.parquet` grows again, v25 rebuilds through
`submit.py` and v29 would have to be hand-reassembled. See
`../CONVERGENCE_AUDIT.md` sec.6a.

This arm does NOT retrain TabNet. It records the selection -- base arm + tag,
mix weight, and which seeds passed the health gate -- as a run like any other,
so `submit.py --arm tabnet_lirf_override` can rebuild the file and the
feature-fingerprint gate can see the whole chain. The TabNet predictions
themselves come from `tests_tabnet/build/final_seed<N>.npz`.

**A rebuild is NOT byte-identical to the uploaded v28/v29.** `submit.py` writes
`np.rint(pred).astype("int32")`; `build_candidate.py` cast with `.astype(int32)`,
which truncates. Rebuilding through this arm therefore shifts the 26,899 LIRF
rows by 0 or 1 second (~+0.5 s mean) -- non-LIRF rows are untouched and the
float arithmetic before the cast matches the uploaded files exactly (verified).
`rint` is kept because it is what every other submission in this build used;
the board's 293.6685 belongs to the truncated artifact.

**Known limit, stated rather than hidden:** those npz files hold predictions for
the ranking rows that existed when they were written, not TabNet model binaries
(`train_tabnet_pred.py` saved predictions only). If `ranking.parquet` grows, the
saved predictions no longer cover every scored row and `submit.py` will fail
loudly on the id check rather than shipping a partial override. Recovering from
that means re-running `train_tabnet_pred.py final <seed>` for the recorded
seeds -- deterministic, since the 2025 training data is unchanged by a ranking
refresh, but ~80 min/seed. Runs from now on also save the model itself.

usage:
  python pipeline_tabnet_lirf_override/train.py --split full --seed 42 \
      --outlier-policy keep_all --w 0.55 --base-tag msmlp18fi --tag lirf_w0.55
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common import dataset as D
from common.bootstrap import (declare_selection_set, run_dir, seed_everything,
                              write_json_guarded)
from pipeline_tabnet_lirf_override.gate import (GATE_MIN_BEST_EPOCH,
                                                GATE_VAL_RMSE, seed_health)

ARM = "tabnet_lirf_override"
TABNET_BUILD = config.SOLUTION_DIR / "tests_tabnet" / "build"
OVERRIDE_AIRPORT = "LIRF"


def seed_preds_path(seed: int) -> Path:
    return TABNET_BUILD / f"final_seed{seed}.npz"


def load_gated_seeds(seeds, min_seeds=5):
    """(accepted, rejected, arrays) after the health gate. Never drops silently."""
    accepted, rejected, arrays = [], {}, {}
    for s in seeds:
        p = seed_preds_path(s)
        if not p.exists():
            rejected[str(s)] = f"no predictions at {p.name}"
            continue
        z = np.load(p, allow_pickle=True)
        ok, reason = seed_health(z)
        if ok:
            accepted.append(s)
            arrays[s] = z
        else:
            rejected[str(s)] = reason
    for s, why in sorted(rejected.items()):
        print(f"[{ARM}] REJECTED seed {s}: {why}")
    if len(accepted) < min_seeds:
        raise SystemExit(
            f"only {len(accepted)} healthy TabNet seeds ({accepted}), need >= {min_seeds}. "
            f"Rejected: {rejected}")
    return accepted, rejected, arrays


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def apply_override(p_base, tabnet_mean, airports, w: float):
    """The override itself. Returns (pred, n_changed)."""
    mask = (np.asarray(airports).astype(str) == OVERRIDE_AIRPORT)
    pred = np.asarray(p_base, dtype="float64").copy()
    pred[mask] = np.clip((1.0 - w) * pred[mask] + w * np.asarray(tabnet_mean)[mask],
                         config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
    return pred, int(mask.sum())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="full", choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--base-arm", default="airport_operator_multimodel")
    ap.add_argument("--base-tag", default="msmlp18fi",
                    help="the TAGGED base run the override sits on (v29's base "
                         "is the 18-seed msmlp18fi blend, not the untagged one)")
    ap.add_argument("--w", type=float, required=True,
                    help="LIRF mix weight on the TabNet ensemble, in [0, 1]")
    ap.add_argument("--w-source", default="",
                    help="how w was chosen -- recorded verbatim so a fitted "
                         "weight can never be mistaken for a chosen constant")
    ap.add_argument("--tabnet-seeds", default="",
                    help="comma-separated; default = every final_seed*.npz on disk, gated")
    ap.add_argument("--min-seeds", type=int, default=5)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if not 0.0 <= args.w <= 1.0:
        raise SystemExit(f"--w must be in [0, 1], got {args.w}")
    seed_everything(args.seed)

    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed,
                  extra=({"tag": args.tag} if args.tag else None))

    if args.tabnet_seeds:
        cand = [int(x) for x in args.tabnet_seeds.split(",") if x.strip()]
    else:
        cand = sorted(int(p.stem.split("seed")[1])
                      for p in TABNET_BUILD.glob("final_seed*.npz"))
    print(f"[{ARM}] candidate TabNet seeds: {cand}")
    accepted, rejected, arrays = load_gated_seeds(cand, min_seeds=args.min_seeds)
    print(f"[{ARM}] healthy seeds used: {accepted}")

    # The selection is frozen HERE, at train time, and predict_ranking replays the
    # recorded list. Re-gating at predict time would let a later seed silently
    # join or leave the ensemble behind a submission that was already scored.
    declare_selection_set(f"{ARM} healthy TabNet seeds", selected_on="dec_val_gate",
                          value=accepted)

    # Coverage check against the CURRENT scored set. This is the staleness gate:
    # the saved predictions are keyed on the ranking ids that existed when the
    # seeds were trained, so a refreshed ranking.parquet must fail here rather
    # than at upload (v17 was rejected by the scorer for exactly this).
    _, meta = D.load_ranking_features(args.featureset)
    rank_ids = meta.index.to_numpy()
    for s in accepted:
        got = arrays[s]["rank_ids"]
        if not np.array_equal(got, rank_ids):
            raise SystemExit(
                f"HALT: TabNet seed {s}'s saved predictions cover {len(got)} ranking rows, "
                f"the current scored set has {len(rank_ids)} (and/or a different order). "
                f"These predictions predate the current ranking.parquet. Re-run "
                f"`python tests_tabnet/train_tabnet_pred.py final {s}` for every "
                f"recorded seed before rebuilding this arm.")

    airports = meta["ADEP_mvt"].astype(str).to_numpy()
    n_lirf = int((airports == OVERRIDE_AIRPORT).sum())
    tab = np.mean([arrays[s]["pr"] for s in accepted], axis=0)
    print(f"[{ARM}] {OVERRIDE_AIRPORT} rows in the scored set: {n_lirf} of {len(airports)} "
          f"({n_lirf / len(airports):.2%})")
    if n_lirf == 0:
        raise SystemExit(f"no {OVERRIDE_AIRPORT} rows in the scored set -- the override would be a no-op")

    write_json_guarded(out / "model.json", {
        "arm": ARM,
        "split": args.split,
        "featureset": args.featureset,
        "outlier_policy": args.outlier_policy,
        "seed": args.seed,
        "base_arm": args.base_arm,
        "base_tag": args.base_tag,
        "override_airport": OVERRIDE_AIRPORT,
        "w": float(args.w),
        "w_source": args.w_source or None,
        "tabnet_seeds": accepted,
        "tabnet_seeds_rejected": rejected,
        "tabnet_pred_dir": str(TABNET_BUILD.relative_to(config.SOLUTION_DIR)),
        "tabnet_pred_sha256": {str(s): sha256_file(seed_preds_path(s)) for s in accepted},
        "gate": {"dec_val_rmse_max": GATE_VAL_RMSE,
                 "best_epoch_min": GATE_MIN_BEST_EPOCH,
                 "must_early_stop": True,
                 "must_have_labels_aligned_marker": True},
        "n_ranking_rows": int(len(rank_ids)),
        "n_override_rows": n_lirf,
        "tabnet_mean_at_override": float(tab[airports == OVERRIDE_AIRPORT].mean()),
        "feature_fingerprint": config.feature_code_fingerprint(),
        "selection": {"what": f"{ARM} healthy TabNet seeds",
                      "selected_on": "dec_val_gate", "value": accepted},
        "note": "Prediction-only arm: it does not train TabNet. predict_ranking "
                "replays the recorded seed list against tests_tabnet/build/.",
    }, force=args.force)
    print(f"[{ARM}] wrote {out / 'model.json'}")
    print(f"[{ARM}] submit with: python submit.py --arm {ARM} --split {args.split} "
          f"--featureset {args.featureset} --outlier-policy {args.outlier_policy} "
          f"--seed {args.seed}" + (f" --tag {args.tag}" if args.tag else "") +
          " --version <N>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
