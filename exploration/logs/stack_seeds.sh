set -e
cd "$(dirname "$0")/.."
for S in 43 44; do
  echo "########## per_airport_stack inner2 seed $S ##########"
  B="--split forward --featureset with_aobt --outlier-policy keep_all --seed $S --force"
  python pipeline_per_airport_stack/train.py $B --inner-val-months 2 --tag inner2
  python -m pipeline_per_airport_stack.evaluate $B --tag inner2 --part val
  python -m pipeline_per_airport_stack.evaluate $B --tag inner2 --part test
done
echo "STACK SEEDS DONE"
