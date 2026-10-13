set -e
cd "$(dirname "$0")/.."
python pipeline_lightgbm/train.py --split forward --featureset with_aobt --outlier-policy winsor_p9999 --seed 42 --force
python -m pipeline_lightgbm.evaluate --split forward --featureset with_aobt --outlier-policy winsor_p9999 --seed 42 --part val --force
python -m pipeline_lightgbm.evaluate --split forward --featureset with_aobt --outlier-policy winsor_p9999 --seed 42 --part test --force
echo "P9999 RERUN DONE"
