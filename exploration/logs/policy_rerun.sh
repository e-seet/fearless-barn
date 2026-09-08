set -e
cd "$(dirname "$0")/.."
for POL in keep_all winsor_7200 winsor_21600 drop_7200; do
  python pipeline_lightgbm/train.py --split forward --featureset with_aobt --outlier-policy $POL --seed 42 --force
  python -m pipeline_lightgbm.evaluate --split forward --featureset with_aobt --outlier-policy $POL --seed 42 --part val --force
  python -m pipeline_lightgbm.evaluate --split forward --featureset with_aobt --outlier-policy $POL --seed 42 --part test --force
done
echo "POLICY RERUN DONE"
