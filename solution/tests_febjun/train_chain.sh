#!/bin/sh
# tests_febjun/PREREGISTERED.md: held-out base for Feb-Jun 2025 on split holdout_febjun (sequential on one GPU).
cd /c/Users/Admin/Desktop/eddie/fearless-barn/exploration || exit 1
L=tests_febjun/train_chain.log
S=holdout_febjun
echo "[$(date)] tree" >> $L
python pipeline_airport_operator_blend/train.py --split $S --featureset with_aobt --outlier-policy keep_all --seed 42 > tests_febjun/train_tree.log 2>&1 || { echo "[$(date)] TREE FAILED" >> $L; tail -5 tests_febjun/train_tree.log >> $L; exit 1; }
grep -E "val RMSE" tests_febjun/train_tree.log >> $L
for s in 42 43 44 45 46 47; do
  [ -f tests_febjun/STOP ] && exit 0
  echo "[$(date)] mlp seed $s" >> $L
  python pipeline_mlp_antigravity/train.py --split $S --featureset with_aobt --outlier-policy keep_all --seed $s > tests_febjun/train_mlp_$s.log 2>&1 || { echo "[$(date)] MLP FAILED $s" >> $L; tail -5 tests_febjun/train_mlp_$s.log >> $L; exit 1; }
  grep -iE "val RMSE|best_epoch|HIT|cap" tests_febjun/train_mlp_$s.log | tail -2 >> $L
done
[ -f tests_febjun/STOP ] && exit 0
echo "[$(date)] ftt seeds 42 43 44" >> $L
python tests_ftt/run_ftt.py 42 43 44 --split $S --lr 1e-3 --epochs 180 --tag lr1e-3 > tests_febjun/train_ftt.log 2>&1 || { echo "[$(date)] FTT FAILED" >> $L; tail -5 tests_febjun/train_ftt.log >> $L; exit 1; }
echo "[$(date)] dump predictions" >> $L
python tests_febjun/dump_mlp.py $S test 42 43 44 45 46 47 >> $L 2>&1
python - >> $L 2>&1 <<'PY'
import pickle, sys, numpy as np
sys.path.insert(0, ".")
import config
from common import dataset as D
from pipeline_airport_operator_blend.evaluate import blend
rd = config.ARTIFACTS / "runs" / "airport_operator_blend__holdout_febjun-with_aobt-keep_all__seed42"
b = pickle.load(open(rd / "model.pkl", "rb"))
Xtr, _, _ = D.load_part("holdout_febjun", "train", "with_aobt", filter_target=True, outlier_policy="keep_all")
X, y, meta = D.load_part("holdout_febjun", "test", "with_aobt", filter_target=False)
p = np.clip(blend(b, D.align_columns(Xtr, X), meta), config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
np.save("tests_febjun/preds/tree_holdout_febjun_test.npy", p); print("tree preds", len(p))
for s in (42, 43, 44):
    z = np.load(f"tests_ftt/preds_seed{s}_holdout_febjun_lr1e-3.npz"); print("ftt", s, "hit_cap", bool(z["hit_cap"]), "best_epoch", int(z["best_epoch"]), "epochs_run", int(z["epochs_run"]))
PY
echo "[$(date)] ALL DONE" >> $L
