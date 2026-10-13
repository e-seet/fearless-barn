set -e
cd "$(dirname "$0")/.."
B="--split forward --featureset with_aobt --outlier-policy keep_all --seed 42 --force"
R=artifacts/runs/per_airport_tuned__forward-with_aobt-keep_all__seed42
if [ ! -e "$R/eval_test.json" ]; then
  python pipeline_per_airport_tuned/train.py $B
  python -m pipeline_per_airport_tuned.evaluate $B --part val
  python -m pipeline_per_airport_tuned.evaluate $B --part test
else echo "  [skip] already done"; fi
echo "PER_AIRPORT_TUNED DONE"
