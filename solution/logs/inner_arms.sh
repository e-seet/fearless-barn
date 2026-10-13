set -e
cd "$(dirname "$0")/.."
B="--split forward_inner --featureset with_aobt --outlier-policy keep_all --seed 42 --force"
for ARM in naive_offblock ridge lightgbm xgboost_antigravity mlp_antigravity; do
  echo "########## $ARM on forward_inner ##########"
  python pipeline_$ARM/train.py $B
  python -m pipeline_$ARM.evaluate $B --part val
done
echo "########## per_airport on forward_inner ##########"
python pipeline_per_airport/train.py $B
python -m pipeline_per_airport.evaluate $B --part val
echo "########## ensemble on forward_inner ##########"
python pipeline_ensemble_antigravity/train.py $B
python -m pipeline_ensemble_antigravity.evaluate $B --part val
echo "INNER ARMS DONE"
