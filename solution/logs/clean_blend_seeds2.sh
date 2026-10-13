set -e
cd "$(dirname "$0")/.."
# RESUMABLE: background jobs in this environment get killed mid-flight, so every
# step checks for its own output and skips if present. Repeated kills then still
# make forward progress instead of restarting from zero.
run_if_missing () {
  target="$1"; shift
  if [ -e "$target" ]; then echo "  [skip] $(basename $(dirname $target))"; return 0; fi
  echo "  [run ] $(basename $(dirname $target))"
  bash -c "$*"
}
R=artifacts/runs
for S in 43 44; do
  IB="--split forward_inner --featureset with_aobt --outlier-policy keep_all --seed $S --force"
  B="--split forward --featureset with_aobt --outlier-policy keep_all --seed $S --force"
  echo "########## seed $S ##########"
  for A in lightgbm xgboost_antigravity ridge naive_offblock per_airport; do
    run_if_missing "$R/${A}__forward_inner-with_aobt-keep_all__seed$S/eval_val.json" \
      "python pipeline_$A/train.py $IB && python -m pipeline_$A.evaluate $IB --part val"
  done
  run_if_missing "$R/mlp_antigravity__forward_inner-with_aobt-keep_all__seed$S/eval_val.json" \
    "python pipeline_mlp_antigravity/train.py $IB --epochs 45 && python -m pipeline_mlp_antigravity.evaluate $IB --part val"
  run_if_missing "$R/per_airport_stack__forward_inner-with_aobt-keep_all__seed${S}__tag=inner2/eval_val.json" \
    "python pipeline_per_airport_stack/train.py $IB --inner-val-months 1 --tag inner2 && python -m pipeline_per_airport_stack.evaluate $IB --tag inner2 --part val"
  run_if_missing "$R/ensemble_antigravity__forward_inner-with_aobt-keep_all__seed$S/eval_val.json" \
    "python pipeline_ensemble_antigravity/train.py $IB && python -m pipeline_ensemble_antigravity.evaluate $IB --part val"
  run_if_missing "$R/cross_blend__forward-with_aobt-keep_all__seed${S}__tag=w_inner/eval_test.json" \
    "python pipeline_cross_blend/train.py $B --weight-mode per_airport --weight-split forward_inner --tag w_inner && python -m pipeline_cross_blend.evaluate $B --tag w_inner --part val && python -m pipeline_cross_blend.evaluate $B --tag w_inner --part test"
done
echo "CLEAN BLEND SEEDS DONE"
