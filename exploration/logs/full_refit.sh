set -e
cd "$(dirname "$0")/.."
FW=artifacts/runs
P="--split full --featureset with_aobt --outlier-policy keep_all --seed 42 --force"
echo "##### lightgbm full refit #####"
python pipeline_lightgbm/train.py $P --freeze-from $FW/lightgbm__forward-with_aobt-keep_all__seed42/model.json
echo "##### mlp full refit #####"
python pipeline_mlp_antigravity/train.py $P --freeze-from $FW/mlp_antigravity__forward-with_aobt-keep_all__seed42/model.json
echo "##### ensemble full refit (weights frozen) #####"
python pipeline_ensemble_antigravity/train.py $P --freeze-from $FW/ensemble_antigravity__forward-with_aobt-keep_all__seed42/model.json
echo "FULL REFIT DONE"
