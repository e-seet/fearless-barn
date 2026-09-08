set -e
B="--split forward --featureset with_aobt --outlier-policy winsor_p9999 --seed 42 --force"
python -u pipeline_ridge/train.py $B
python -u -m pipeline_ridge.evaluate $B --part val
python -u -m pipeline_ridge.evaluate $B --part test
python -u pipeline_lightgbm/train.py $B
python -u -m pipeline_lightgbm.evaluate $B --part val
python -u -m pipeline_lightgbm.evaluate $B --part test
python -u compare.py $B
