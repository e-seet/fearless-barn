#!/usr/bin/env python3
"""Assemble a base-model submission (stage 1) exactly as submit.py does for tabnet_lirf_override(ftt_multimodel),
but with an optional fourth member (CatBoost cb1, seeds 42-44) in the per-airport blend.

  --weights tests_ftt/final_weights_c3d6.json --check submissions/fearless_barn_v42.parquet   (CONTROL: must be identical)
  --weights tests_ftt_wide/final_weights_v58_s3.json --out tests_ftt_wide/v58_base_s3.parquet
  --weights tests_ftt_wide/final_weights_v58_s4.json --out tests_ftt_wide/v58_base_s4.parquet

Member predictions on the ranking set: tree / MLP-18 / CatBoost via submit.predict_ranking (cached), FTT members
from tests_ftt/preds_seed{s}_full_{tag}.npz["pr"], TabNet LIRF override with v42's recorded seeds and w = 0.459."""
import argparse, filecmp, json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT))
import config
import submit
from pipeline_tabnet_lirf_override.train import apply_override, seed_preds_path

FS, POL, SEED = "with_aobt", "keep_all", 42
CACHE = HERE / "rank_cache"; CACHE.mkdir(exist_ok=True)
OVR = json.loads((ROOT / "artifacts" / "runs" / "tabnet_lirf_override__full-with_aobt-keep_all__seed42__tag=lirf_w0.459_c3d6" / "model.json").read_text())


def cached(name, fn):
    f = CACHE / f"{name}.npy"
    if f.exists():
        return np.load(f)
    v = fn(); np.save(f, v); return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--check", type=Path, default=None)
    a = ap.parse_args()
    wf = json.loads(a.weights.read_text()); W = wf["weights_per_airport"]
    _, meta = submit.predict_ranking("airport_operator_blend", "full", FS, SEED, POL)
    rank_ids = meta.index.to_numpy(); aps = meta["ADEP_mvt"].astype(str).to_numpy()
    tree = cached("tree", lambda: submit.predict_ranking("airport_operator_blend", "full", FS, SEED, POL)[0])
    mlp = cached("mlp18", lambda: np.mean([submit.predict_ranking("mlp_antigravity", "full", FS, s, POL)[0] for s in wf["mlp_seeds"]], axis=0))
    ftt = []
    for s, tg in wf["ftt_members"]:
        z = np.load(ROOT / "tests_ftt" / f"preds_seed{int(s)}_full_{tg}.npz", allow_pickle=True)
        assert np.array_equal(z["rank_ids"], rank_ids), f"FTT {s}:{tg} ranking ids differ"
        ftt.append(z["pr"])
    p_ftt = np.average(ftt, axis=0, weights=wf.get("ftt_member_weights"))
    members = [p_ftt, tree, mlp]
    if len(next(iter(W.values()))) == 4:
        members.append(cached("catboost_cb1", lambda: np.mean([submit.predict_ranking("catboost_antigravity", "full", FS, s, POL, tag="cb1")[0] for s in (42, 43, 44)], axis=0)))
    base = np.zeros(len(rank_ids))
    for k in np.unique(aps):
        m = aps == k
        w = W.get(str(k), [1 / len(members)] * len(members))
        base[m] = sum(wi * p[m] for wi, p in zip(w, members))
    tab = []
    for s in OVR["tabnet_seeds"]:
        z = np.load(seed_preds_path(int(s)), allow_pickle=True)
        assert np.array_equal(z["rank_ids"], rank_ids)
        tab.append(z["pr"])
    pred, n_over = apply_override(base, np.mean(tab, axis=0), aps, float(OVR["w"]))
    tmpl = pd.read_parquet(config.SUBMITTING_PARQUET)
    sub = pd.DataFrame({config.ID: tmpl[config.ID].to_numpy(), config.TARGET: np.rint(pred).astype("int32")})
    errs = submit.validate(sub)
    print(f"{len(members)} members | LIRF override rows {n_over} | validation {errs or 'PASS'}")
    out = a.out or (HERE / "_check.parquet")
    sub.to_parquet(out, index=False)
    if a.check:
        ref = pd.read_parquet(a.check)
        print(f"CHECK vs {a.check.name}: identical rows {int((ref[config.TARGET].to_numpy() == sub[config.TARGET].to_numpy()).sum())} / {len(ref)}; "
              f"bytes identical {filecmp.cmp(a.check, out, shallow=False)}")


if __name__ == "__main__":
    main()
