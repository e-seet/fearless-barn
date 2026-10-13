#!/usr/bin/env bash
# Seed-robustness test: rerun the airport_*_multimodel family at seeds 45/46/47
# to check whether the ~0.3 s win margins in the 42/43/44 "champion" ranking hold
# or wander once more seeds are added. One sequential script (MISTAKES #83: no
# cross-process waits). Logs to tests_seed_robustness/run.log.
set -e
cd "$(dirname "$0")/.."

ARMS="airport_operator_multimodel airport_wtc_multimodel airport_runway_multimodel airport_wtc_rec_multimodel airport_unsup_multimodel multimodel_antigravity"

for S in 45 46 47; do
  echo "############################## SEED $S ##############################"
  IB="--split forward_inner --featureset with_aobt --outlier-policy keep_all --seed $S --force"
  B="--split forward --featureset with_aobt --outlier-policy keep_all --seed $S --force"

  echo "===== seed $S: base parents (per-airport, tree blend, MLP), both surfaces ====="
  # per_airport: bare tree parent for multimodel_antigravity (predict.py blends
  # per_airport + mlp_antigravity for that arm, not airport_operator_blend).
  python pipeline_per_airport/train.py $B
  python pipeline_per_airport/train.py $IB
  python pipeline_airport_operator_blend/train.py $B
  python pipeline_airport_operator_blend/train.py $IB
  python pipeline_mlp_antigravity/train.py $B  --epochs 25
  python pipeline_mlp_antigravity/train.py $IB --epochs 45

  echo "===== seed $S: multimodel family (solve alphas on forward_inner, eval on forward) ====="
  for A in $ARMS; do
    python pipeline_$A/train.py $B
    python -m pipeline_$A.evaluate $B --part val
    python -m pipeline_$A.evaluate $B --part test
  done
done

echo "ALL SEEDS 45-47 DONE"
python tests_seed_robustness/analyze.py | tee tests_seed_robustness/result_seeds_42-47.txt
