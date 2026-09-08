#!/usr/bin/env bash
# Resume the seed 45-47 robustness sweep after the first attempt's seed-45
# MLP-inner step wedged under CPU contention (ran alongside the log1p experiment).
# Seed 45 keeps what already finished (per_airport, airport_operator_blend, mlp
# forward -- all on disk); seeds 46/47 run from scratch. Nothing else competes now.
set -e
cd "$(dirname "$0")/.."
PY="python -u"
ARMS="airport_operator_multimodel airport_wtc_multimodel airport_runway_multimodel airport_wtc_rec_multimodel airport_unsup_multimodel multimodel_antigravity"

run_family () {   # $1 = seed
  local S=$1
  local B="--split forward --featureset with_aobt --outlier-policy keep_all --seed $S --force"
  for A in $ARMS; do
    echo "---- seed $S : $A ----"
    $PY pipeline_$A/train.py $B
    $PY -m pipeline_$A.evaluate $B --part val
    $PY -m pipeline_$A.evaluate $B --part test
  done
}

echo "############ SEED 45 (resume: mlp-inner + family) ############ $(date)"
$PY pipeline_mlp_antigravity/train.py --split forward_inner --featureset with_aobt \
    --outlier-policy keep_all --seed 45 --force --epochs 45
run_family 45

for S in 46 47; do
  echo "############ SEED $S (full) ############ $(date)"
  IB="--split forward_inner --featureset with_aobt --outlier-policy keep_all --seed $S --force"
  B="--split forward --featureset with_aobt --outlier-policy keep_all --seed $S --force"
  $PY pipeline_per_airport/train.py $B
  $PY pipeline_per_airport/train.py $IB
  $PY pipeline_airport_operator_blend/train.py $B
  $PY pipeline_airport_operator_blend/train.py $IB
  $PY pipeline_mlp_antigravity/train.py $B  --epochs 25
  $PY pipeline_mlp_antigravity/train.py $IB --epochs 45
  run_family $S
done

echo "############ ALL SEEDS 45-47 DONE ############ $(date)"
$PY tests_seed_robustness/analyze.py | tee tests_seed_robustness/result_seeds_42-47.txt
