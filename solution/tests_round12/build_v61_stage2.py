#!/usr/bin/env python3
"""v61 stage 2 (round 12 PASS): the K2 corrector (3000 trees, 127 leaves, lr 0.015; seeds 42-46) trained on residuals of the
REALISTIC held-out base (v42's per-airport weights on the held-out tree / MLP / FTT members) over all 7 months.
Writes tests_adsb_full/model_v61 (same layout as model_v60), and checks the saved boosters reload identically."""
import json, sys
from pathlib import Path
import numpy as np
import lightgbm as lgb
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
for p in (ROOT, ROOT / "tests_adsb_full", ROOT / "tests_round10"):
    sys.path.insert(0, str(p))
import round10 as R10

K2 = {**R10.PARAMS_K, "n_estimators": 3000, "num_leaves": 127, "learning_rate": 0.015}
W = json.loads((ROOT / "tests_ftt" / "final_weights_c3d6.json").read_text())["weights_per_airport"]
df = R10.frame()
w = np.array([W.get(a, [1 / 3] * 3) for a in df.ap])
df["base"] = w[:, 0] * (df.base + df.d_ftt) + w[:, 1] * (df.base + df.d_tree) + w[:, 2] * (df.base + df.d_mlp)
trc = df[df["cov"]]
MD = ROOT / "tests_adsb_full" / "model_v61"; MD.mkdir(exist_ok=True)
preds = []
for s in R10.SEEDS:
    m = lgb.LGBMRegressor(**{**K2, "random_state": s}).fit(trc[R10.FEAT_A], trc.y - trc.base)
    m.booster_.save_model(str(MD / f"adsb_corrector_seed{s}.txt"))
    preds.append(m.predict(trc[R10.FEAT_A].head(5000)))
cfg = json.loads((ROOT / "tests_adsb_full" / "model_v60" / "config.json").read_text())
cfg.update({"params": K2, "seeds": list(R10.SEEDS), "training_base": "realistic: v42 per-airport weights (final_weights_c3d6) on held-out tree/MLP/FTT",
            "n_train": int(len(trc)), "evidence": "tests_round12/PREREGISTERED.md, round12.json"})
(MD / "config.json").write_text(json.dumps(cfg, indent=1))
re = [lgb.Booster(model_file=str(MD / f"adsb_corrector_seed{s}.txt")).predict(trc[R10.FEAT_A].head(5000)) for s in R10.SEEDS]
print(f"trained on {len(trc)} covered rows (realistic base); reload identical: {all(np.array_equal(a, b) for a, b in zip(preds, re))}")
