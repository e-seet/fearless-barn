#!/usr/bin/env bash
# Wait for all 12 retrained seeds, then run the PAIRED old-vs-new comparison.
# Writes to alpha_paired_old_vs_new.txt so the result survives the session.
set -u
cd "$(dirname "$0")/.."
echo "waiting for the retrain to finish ($(date))"
until grep -q "=== all done" tests_mlp_convergence/retrain.log 2>/dev/null; do sleep 300; done
echo "retrain done ($(date)); running the paired comparison"
python tests_mlp_convergence/alpha_paired_old_vs_new.py 2>&1 \
  | tee tests_mlp_convergence/alpha_paired_old_vs_new.txt
echo "=== comparison done ($(date)) ==="
