set -e
cd "$(dirname "$0")/.."
B="--split forward --featureset with_aobt --outlier-policy keep_all --seed 42 --force"
R=artifacts/runs
if [ ! -e "$R/per_airport_arrfeat__forward-with_aobt-keep_all__seed42/eval_test.json" ]; then
  echo "########## per_airport_arrfeat -- the test the hypothesis actually predicts ##########"
  python pipeline_per_airport_arrfeat/train.py $B
  python -m pipeline_per_airport_arrfeat.evaluate $B --part val
  python -m pipeline_per_airport_arrfeat.evaluate $B --part test
else echo "  [skip] per_airport_arrfeat done"; fi
if [ ! -e "$R/../ablations/feature_groups__forward-with_aobt-keep_all__seed42.json" ]; then
  echo "########## feature-group ablation (idea 3) ##########"
  python stage10_feature_ablation.py $B
else echo "  [skip] ablation done"; fi
echo "ROUND4 DONE"
