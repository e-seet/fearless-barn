set -e
cd "$(dirname "$0")/.."
POL=drop_date_artifacts

for S in 42 43 44; do
  echo "##### mlp_antigravity forward seed=$S #####"
  python pipeline_mlp_antigravity/train.py --split forward --featureset with_aobt --outlier-policy $POL --seed $S --force

  echo "##### mlp_antigravity forward_inner seed=$S #####"
  python pipeline_mlp_antigravity/train.py --split forward_inner --featureset with_aobt --outlier-policy $POL --seed $S --force

  echo "##### airport_operator_blend forward seed=$S (congest) #####"
  python pipeline_airport_operator_blend/train.py --split forward --featureset with_aobt --outlier-policy $POL --seed $S --congest --tag congest --force

  echo "##### airport_operator_blend forward_inner seed=$S (congest) #####"
  python pipeline_airport_operator_blend/train.py --split forward_inner --featureset with_aobt --outlier-policy $POL --seed $S --congest --tag congest --force

  echo "##### airport_operator_multimodel forward seed=$S #####"
  python pipeline_airport_operator_multimodel/train.py --split forward --featureset with_aobt --outlier-policy $POL --seed $S \
      --tree-tag congest --mlp-seeds "42,43,44" --tag congest_msmlp --force

  echo "##### eval val/test seed=$S #####"
  python -m pipeline_airport_operator_multimodel.evaluate --split forward --featureset with_aobt --outlier-policy $POL --seed $S --tag congest_msmlp --part val --force
  python -m pipeline_airport_operator_multimodel.evaluate --split forward --featureset with_aobt --outlier-policy $POL --seed $S --tag congest_msmlp --part test --force
done
echo "DROPART TEST DONE"
