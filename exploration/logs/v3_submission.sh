set -e
cd "$(dirname "$0")/.."
FW=artifacts/runs
P="--split full --featureset with_aobt --outlier-policy keep_all --seed 42 --force"
echo "##### per_airport_stack 12-month refit (weights + rounds frozen from forward) #####"
python pipeline_per_airport_stack/train.py $P --tag inner2 \
  --freeze-from $FW/per_airport_stack__forward-with_aobt-keep_all__seed42__tag=inner2/model.json
echo "##### build submission v3 #####"
python submit.py --arm per_airport_stack --tag inner2 --split full --featureset with_aobt \
  --outlier-policy keep_all --seed 42 --team PLACEHOLDER_TEAM --version 3
echo "V3 DONE"
