set -e
cd "$(dirname "$0")/.."
while pgrep -f "stage8_triage.py" >/dev/null 2>&1 || pgrep -f "round3_stack.sh" >/dev/null 2>&1; do sleep 20; done
for S in 43 44; do
  echo "########## per_airport seed $S ##########"
  B="--split forward --featureset with_aobt --outlier-policy keep_all --seed $S --force"
  python pipeline_per_airport/train.py $B
  python -m pipeline_per_airport.evaluate $B --part val
  python -m pipeline_per_airport.evaluate $B --part test
done
echo "PER_AIRPORT SEEDS DONE"
