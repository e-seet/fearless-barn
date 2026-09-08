set -e
cd "$(dirname "$0")/.."
B="--split forward --featureset with_aobt --outlier-policy keep_all --seed 42 --force"
echo "########## per_airport_stack ##########"
python pipeline_per_airport_stack/train.py $B
python -m pipeline_per_airport_stack.evaluate $B --part val
python -m pipeline_per_airport_stack.evaluate $B --part test
echo "########## lightgbm_unsup ##########"
python pipeline_lightgbm_unsup/train.py $B
python -m pipeline_lightgbm_unsup.evaluate $B --part val
python -m pipeline_lightgbm_unsup.evaluate $B --part test
echo "########## compare (all arms) ##########"
python compare.py $B
echo "########## band triage ##########"
python stage8_triage.py $B
echo "ROUND2 DONE"
