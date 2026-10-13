set -e
cd "$(dirname "$0")/.."
POL=drop_date_artifacts

run_if_missing() {
  local rundir="$1"; shift
  if [ -f "artifacts/runs/${rundir}/model.json" ]; then
    echo "##### SKIP (already complete): ${rundir} #####"
  else
    echo "##### RUN: ${rundir} #####"
    "$@"
  fi
}

# Phase 1: mlp_antigravity, all seeds, both splits (prereq for every blend below)
for S in 42 43 44; do
  run_if_missing "mlp_antigravity__forward-with_aobt-${POL}__seed${S}" \
    python pipeline_mlp_antigravity/train.py --split forward --featureset with_aobt --outlier-policy $POL --seed $S
  run_if_missing "mlp_antigravity__forward_inner-with_aobt-${POL}__seed${S}" \
    python pipeline_mlp_antigravity/train.py --split forward_inner --featureset with_aobt --outlier-policy $POL --seed $S
done

# Phase 2: airport_operator_blend (congest), all seeds, both splits
for S in 42 43 44; do
  run_if_missing "airport_operator_blend__forward-with_aobt-${POL}__seed${S}__tag=congest" \
    python pipeline_airport_operator_blend/train.py --split forward --featureset with_aobt --outlier-policy $POL --seed $S --congest --tag congest
  run_if_missing "airport_operator_blend__forward_inner-with_aobt-${POL}__seed${S}__tag=congest" \
    python pipeline_airport_operator_blend/train.py --split forward_inner --featureset with_aobt --outlier-policy $POL --seed $S --congest --tag congest
done

# Phase 3: airport_operator_multimodel blend + eval, all seeds
for S in 42 43 44; do
  run_if_missing "airport_operator_multimodel__forward-with_aobt-${POL}__seed${S}__tag=congest_msmlp" \
    python pipeline_airport_operator_multimodel/train.py --split forward --featureset with_aobt --outlier-policy $POL --seed $S \
        --tree-tag congest --mlp-seeds "42,43,44" --tag congest_msmlp

  echo "##### eval val/test seed=$S #####"
  python -m pipeline_airport_operator_multimodel.evaluate --split forward --featureset with_aobt --outlier-policy $POL --seed $S --tag congest_msmlp --part val --force
  python -m pipeline_airport_operator_multimodel.evaluate --split forward --featureset with_aobt --outlier-policy $POL --seed $S --tag congest_msmlp --part test --force
done
echo "DROPART TEST DONE"
