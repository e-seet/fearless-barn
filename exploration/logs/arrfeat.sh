set -e
cd "$(dirname "$0")/.."
B="--split forward --featureset with_aobt --outlier-policy keep_all --seed 42 --force"
R=artifacts/runs/lightgbm_arrfeat__forward-with_aobt-keep_all__seed42
if [ ! -e "$R/eval_test.json" ]; then
  echo "########## lightgbm_arrfeat (11 extra cols incl. per-stand taxi-in) ##########"
  python pipeline_lightgbm_arrfeat/train.py $B
  python -m pipeline_lightgbm_arrfeat.evaluate $B --part val
  python -m pipeline_lightgbm_arrfeat.evaluate $B --part test
else
  echo "  [skip] arrfeat already done"
fi
echo "ARRFEAT DONE"
