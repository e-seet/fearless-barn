set -e
cd "$(dirname "$0")/.."
B="--split forward --featureset with_aobt --outlier-policy keep_all --seed 42 --force"
echo "########## per_airport ##########"
python pipeline_per_airport/train.py $B
python -m pipeline_per_airport.evaluate $B --part val
python -m pipeline_per_airport.evaluate $B --part test
echo "########## lightgbm_recency ##########"
python pipeline_lightgbm_recency/train.py $B
python -m pipeline_lightgbm_recency.evaluate $B --part val
python -m pipeline_lightgbm_recency.evaluate $B --part test
echo "########## compare ##########"
python compare.py $B
echo "NEW ARMS DONE"
