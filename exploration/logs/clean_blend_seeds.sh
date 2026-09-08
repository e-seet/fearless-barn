set -e
cd "$(dirname "$0")/.."
# One sequential script; no cross-script process waits (MISTAKES #83).
for S in 43 44; do
  IB="--split forward_inner --featureset with_aobt --outlier-policy keep_all --seed $S --force"
  B="--split forward --featureset with_aobt --outlier-policy keep_all --seed $S --force"
  echo "########## seed $S: inner arms ##########"
  for A in lightgbm xgboost_antigravity ridge naive_offblock; do
    python pipeline_$A/train.py $IB
    python -m pipeline_$A.evaluate $IB --part val
  done
  python pipeline_mlp_antigravity/train.py $IB --epochs 45
  python -m pipeline_mlp_antigravity.evaluate $IB --part val
  python pipeline_per_airport/train.py $IB
  python -m pipeline_per_airport.evaluate $IB --part val
  python pipeline_per_airport_stack/train.py $IB --inner-val-months 1 --tag inner2
  python -m pipeline_per_airport_stack.evaluate $IB --tag inner2 --part val
  python pipeline_ensemble_antigravity/train.py $IB
  python -m pipeline_ensemble_antigravity.evaluate $IB --part val
  echo "########## seed $S: clean blend ##########"
  python pipeline_cross_blend/train.py $B --weight-mode per_airport --weight-split forward_inner --tag w_inner
  python -m pipeline_cross_blend.evaluate $B --tag w_inner --part val
  python -m pipeline_cross_blend.evaluate $B --tag w_inner --part test
done
echo "CLEAN BLEND SEEDS DONE"
