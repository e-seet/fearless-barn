set -e
cd "$(dirname "$0")/.."
# Wait for the round-2 script to finish so the two do not contend for cores.
while pgrep -f "round2_arms.sh" >/dev/null 2>&1; do sleep 20; done
B="--split forward --featureset with_aobt --outlier-policy keep_all --seed 42 --force"
for M in 2 3; do
  echo "########## per_airport_stack, inner-val-months=$M ##########"
  python pipeline_per_airport_stack/train.py $B --inner-val-months $M --tag "inner${M}"
  python -m pipeline_per_airport_stack.evaluate $B --tag "inner${M}" --part val
  python -m pipeline_per_airport_stack.evaluate $B --tag "inner${M}" --part test
done
echo "ROUND3 DONE"
