#!/usr/bin/env python3
"""Save an mlp_antigravity run's predictions on a split part (same code as pipeline_mlp_antigravity/evaluate.py).
usage: dump_mlp.py <split> <part> <seed> [<seed> ...]  -> tests_febjun/preds/mlp_<split>_<part>_seed<s>.npy"""
import json, pickle, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT))
import config
from common import dataset as D
from common.bootstrap import run_dir
from pipeline_mlp_antigravity.model import TabularMLP
from pipeline_mlp_antigravity.train import TabularDataset, encode_data

split, part, seeds = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3:]]
OUT = HERE / "preds"; OUT.mkdir(exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X, y, meta = D.load_part(split, part, "with_aobt", filter_target=False)
for s in seeds:
    rd = run_dir(config.ARTIFACTS / "runs", "mlp_antigravity", protocol=config.protocol_key(split, "with_aobt", "keep_all"), seed=s)
    prep = pickle.load(open(rd / "preprocessor.pkl", "rb"))
    assert not json.loads((rd / "model.json").read_text()).get("congest", False)
    cat_t, num_t = encode_data(X, prep["cat_cols"], prep["num_cols"], prep["cat_maps"], prep["scaler"], prep["medians"])
    loader = DataLoader(TabularDataset(cat_t, num_t), batch_size=4096, shuffle=False, num_workers=0)
    model = TabularMLP(prep["cat_cardinalities"], num_features=len(prep["num_cols"]),
                       hidden_dims=tuple(prep.get("hidden_dims", (256, 128, 64))), dropout=prep.get("dropout", 0.1),
                       prior_mean=prep.get("prior_mean", 990.0), residual=prep.get("residual", False)).to(dev)
    model.load_state_dict(torch.load(rd / "model.pt", map_location=dev)); model.eval()
    ps = []
    with torch.no_grad():
        for xc, xn in loader:
            ps.append(model({c: xc[c].to(dev) for c in xc}, xn.to(dev)).cpu().numpy())
    p = np.clip(np.concatenate(ps), config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
    np.save(OUT / f"mlp_{split}_{part}_seed{s}.npy", p)
    print(f"mlp {split} {part} seed {s}: n={len(p)}", flush=True)
