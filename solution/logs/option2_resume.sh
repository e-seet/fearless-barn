set -e
cd "$(dirname "$0")/.."
IB="--split forward_inner --featureset with_aobt --outlier-policy keep_all --seed 42 --force"
B="--split forward --featureset with_aobt --outlier-policy keep_all --seed 42 --force"

echo "########## 1/4 MLP re-run, cap 45 (previous run hit 25/25, patience=6) ##########"
python pipeline_mlp_antigravity/train.py $IB --epochs 45
python -m pipeline_mlp_antigravity.evaluate $IB --part val

echo "########## 2/4 assert it converged before anything depends on it ##########"
python - <<'PY'
import json, sys
c = json.load(open("artifacts/runs/mlp_antigravity__forward_inner-with_aobt-keep_all__seed42/model.json"))["convergence"]
print("  best_epoch=%s stop=%s max=%s early_stop_fired=%s hit_cap=%s"
      % (c.get("best_epoch"), c.get("stop_epoch"), c.get("max_epochs"),
         c.get("early_stop_fired"), c.get("hit_max_epochs")))
if c.get("hit_max_epochs"):
    sys.exit("HALT: still hit the cap -- raise --epochs again before fitting weights on it")
print("  OK: converged")
PY

echo "########## 3/4 ensemble rebuilt on the converged MLP ##########"
python pipeline_ensemble_antigravity/train.py $IB
python -m pipeline_ensemble_antigravity.evaluate $IB --part val

echo "########## 4/4 OPTION 2: cross_blend, weights fitted on forward_inner ##########"
python pipeline_cross_blend/train.py $B --weight-mode per_airport --weight-split forward_inner --tag w_inner
python -m pipeline_cross_blend.evaluate $B --tag w_inner --part val
python -m pipeline_cross_blend.evaluate $B --tag w_inner --part test
echo "OPTION2 RESUME DONE"
