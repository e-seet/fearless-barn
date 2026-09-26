#!/usr/bin/env python3
"""FT-Transformer 3-member per-airport blend -- registered arm.

    pred[airport] = w_ftt*FTT + w_tree*airport_operator_blend + w_mlp*mlp_antigravity

Same construction as `airport_operator_catboost_multimodel` with FTT in the slot
CatBoost occupies there, and the same per-airport simplex solver (w >= 0,
sum = 1). Weights are solved on `forward_inner` val (Jul-Aug, a holdout carved
from TRAIN) and applied frozen; `full` is in SELECTION_FORBIDDEN_SPLITS and
nothing is fitted on it.

Evidence: `tests_ftt/RESULT.md`. Confirmed on three independent periods with a
bar fixed in advance (`tests_ftt/PREREGISTERED.md`,
`tests_ftt/PREREGISTERED_seasonal.md`). Honest effect size ~1.5 s in-band; the
one -3.6 s reading was the optimistic outlier of four.

**This arm does not train FTT.** Like `tabnet_lirf_override`, it records the
selection -- seed list, weights, gates -- and applies it. FTT predictions come
from `tests_ftt/preds_seed<N>_full_lr1e-3.npz`, which now also carry the model
binary alongside (`model_seed<N>_full_lr1e-3.pt`), so a refreshed
ranking.parquet can be re-predicted rather than retrained -- the gap that made
v29 unrebuildable (../CONVERGENCE_AUDIT.md sec.6a).

usage:
  python pipeline_ftt_multimodel/train.py --split full --seed 42 \
      --outlier-policy keep_all --tag ftt3_mlp18
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

ARM = "ftt_multimodel"
FTT_DIR = config.SOLUTION_DIR / "tests_ftt"
WEIGHT_ORDER = ["ftt", "tree", "mlp"]


def ftt_preds_path(seed: int, split: str = "full", tag: str = "lr1e-3") -> Path:
    sp = "" if split == "forward" else f"_{split}"
    return FTT_DIR / f"preds_seed{seed}{sp}_{tag}.npz"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def blend_per_airport(p_ftt, p_tree, p_mlp, airports, weights):
    """Apply the frozen per-airport simplex weights. Returns the blend."""
    ap = np.asarray(airports).astype(str)
    out = np.zeros(len(ap), dtype="float64")
    for a in np.unique(ap):
        m = ap == a
        w = weights.get(str(a), [1 / 3, 1 / 3, 1 / 3])
        out[m] = w[0] * np.asarray(p_ftt)[m] + w[1] * np.asarray(p_tree)[m] \
            + w[2] * np.asarray(p_mlp)[m]
    return np.clip(out, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="full", choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--weights-file", default=str(FTT_DIR / "final_weights.json"),
                    help="solved on forward_inner by tests_ftt/solve_final_weights.py")
    ap.add_argument("--ftt-seeds", default="42,43,44")
    ap.add_argument("--ftt-tag", default="lr1e-3")
    ap.add_argument("--ftt-members", default="",
                    help="optional seed:tag list mixing FTT variants, e.g. 42:lr1e-3,42:d192lr5; "
                         "overrides --ftt-seeds/--ftt-tag (equal weight per member)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    seed_everything(a.seed)

    wf = json.loads(Path(a.weights_file).read_text())
    W = wf["weights_per_airport"]
    mlp_seeds = wf["mlp_seeds"]
    ftt_seeds = [int(x) for x in a.ftt_seeds.split(",") if x.strip()]
    members = ([(int(m.split(":")[0]), m.split(":")[1]) for m in a.ftt_members.split(",") if m.strip()]
               if a.ftt_members else [(s, a.ftt_tag) for s in ftt_seeds])
    if a.ftt_members:
        ftt_seeds = [s for s, _ in members]

    if wf.get("feature_fingerprint") != config.feature_code_fingerprint():
        raise SystemExit(
            "HALT: the weights were solved against features.py fingerprint {}, "
            "current is {}. Re-solve before building.".format(
                wf.get("feature_fingerprint"), config.feature_code_fingerprint()))

    # Every prior solve put FTT at exactly 0 at LIRF, across 3/6/12/18-seed MLP
    # partners and six multistart points. A non-zero here means the solve moved,
    # not that LIRF changed its mind.
    lirf_w = float(W["LIRF"][0])
    if lirf_w > 1e-6:
        raise SystemExit(
            f"HALT: FTT weight at LIRF is {lirf_w:.4f}, expected 0. Every "
            f"independent solve so far gave exactly 0 there. Investigate the "
            f"solve before shipping it.")

    _, meta = D.load_ranking_features(a.featureset)
    rank_ids = meta.index.to_numpy()
    shas = {}
    for s, tg in members:
        p = ftt_preds_path(s, a.split, tg)
        if not p.exists():
            raise SystemExit(f"missing FTT predictions at {p}. Train with: python "
                             f"tests_ftt/run_ftt.py {s} --split {a.split} --lr 1e-3 "
                             f"--freeze-epochs 112 --predict-ranking --tag {a.ftt_tag}")
        z = np.load(p, allow_pickle=True)
        if "pr" not in z.files:
            raise SystemExit(f"{p.name} has no ranking predictions -- re-run with "
                             f"--predict-ranking.")
        if not np.array_equal(z["rank_ids"], rank_ids):
            raise SystemExit(
                f"HALT: FTT seed {s} covers {len(z['rank_ids'])} ranking rows, the "
                f"current scored set has {len(rank_ids)}. Those predictions predate "
                f"the current ranking.parquet -- re-predict before building.")
        shas[f"{s}:{tg}" if a.ftt_members else str(s)] = sha256_file(p)

    declare_selection_set(f"{ARM} per-airport weights", selected_on="forward_inner",
                          value=W)

    proto = config.protocol_key(a.split, a.featureset, a.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=a.seed,
                  extra=({"tag": a.tag} if a.tag else None))
    write_json_guarded(out / "model.json", {
        "arm": ARM,
        "split": a.split,
        "featureset": a.featureset,
        "outlier_policy": a.outlier_policy,
        "seed": a.seed,
        "weights_per_airport": W,
        "weight_order": WEIGHT_ORDER,
        "weight_split": wf["weight_split"],
        "solver": wf["solver"],
        "mlp_seeds": mlp_seeds,
        "ftt_seeds": ftt_seeds,
        "ftt_tag": a.ftt_tag,
        **({"ftt_members": [[s, tg] for s, tg in members]} if a.ftt_members else {}),
        "ftt_pred_dir": str(FTT_DIR.relative_to(config.SOLUTION_DIR)),
        "ftt_pred_sha256": shas,
        "ftt_freeze_epochs": 112,
        "ftt_freeze_rationale": "median best epoch of the three forward seeds "
                                "(102/112/117); full has no val so the count is "
                                "frozen, and the median is used for every member "
                                "rather than one seed's (CONVERGENCE_AUDIT.md sec.2)",
        "n_ranking_rows": int(len(rank_ids)),
        "feature_fingerprint": config.feature_code_fingerprint(),
        "evidence": "tests_ftt/RESULT.md -- confirmed on 3 independent periods; "
                    "honest effect ~1.5s in-band",
        "note": "prediction-only arm: it does not train FTT.",
    }, force=a.force)
    print(f"[{ARM}] wrote {out / 'model.json'}")
    print(f"[{ARM}] FTT seeds {ftt_seeds} | MLP seeds {mlp_seeds} ({len(mlp_seeds)})")
    print(f"[{ARM}] LIRF weights {W['LIRF']} (FTT must be 0)")
    print(f"[{ARM}] submit with: python submit.py --arm {ARM} --split {a.split} "
          f"--featureset {a.featureset} --outlier-policy {a.outlier_policy} "
          f"--seed {a.seed}" + (f" --tag {a.tag}" if a.tag else "") + " --version <N>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
