#!/usr/bin/env bash
# Finish seed 47 (the first resume died mid forward-MLP after a ~13h machine gap).
# per_airport + airport_operator_blend already on disk for seed 47; redo the MLPs
# and the family. Standalone so a second death doesn't lose seeds 45/46.
set -e
cd "$(dirname "$0")/.."
PY="python -u"
S=47
IB="--split forward_inner --featureset with_aobt --outlier-policy keep_all --seed $S --force"
B="--split forward --featureset with_aobt --outlier-policy keep_all --seed $S --force"

$PY pipeline_mlp_antigravity/train.py $B  --epochs 25
$PY pipeline_mlp_antigravity/train.py $IB --epochs 45
for A in airport_operator_multimodel airport_wtc_multimodel airport_runway_multimodel \
         airport_wtc_rec_multimodel airport_unsup_multimodel multimodel_antigravity; do
  echo "---- seed $S : $A ----"
  $PY pipeline_$A/train.py $B
  $PY -m pipeline_$A.evaluate $B --part val
  $PY -m pipeline_$A.evaluate $B --part test
done
echo "############ SEED 47 DONE ############ $(date)"
$PY tests_seed_robustness/analyze.py | tee tests_seed_robustness/result_seeds_42-47.txt
