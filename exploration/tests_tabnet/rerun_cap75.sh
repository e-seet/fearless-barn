#!/usr/bin/env bash
# Rec 1: re-run the two final TabNet seeds the health gate excluded for hitting
# the 60-epoch cap (47: best 50/ran 60; 51: best 47/ran 60), at a cap set by the
# repo's own rule -- "set the cap from an OBSERVED natural best-epoch with at
# least 1.5x headroom" (tools/ml_common/convergence.py note, MISTAKES.md #70).
# Observed best = 50 -> cap 75. The truncated runs are preserved as
# build/TRUNCATED60_final_seed{47,51}.{npz,log} for the before/after comparison.
set -u
cd "$(dirname "$0")/.."
export TABNET_MAX_EPOCHS=75
for s in 47 51; do
  echo "=== final seed $s at cap 75 ($(date)) ==="
  python tests_tabnet/train_tabnet_pred.py final "$s" 2>&1 | tee "tests_tabnet/build/final_seed${s}.log"
done
echo "=== all done ($(date)) ==="
