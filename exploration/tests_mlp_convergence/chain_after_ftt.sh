#!/usr/bin/env bash
# Wait for the FT-Transformer cap-90 run to finish, then start the rec-3 retrain.
# Sequential on purpose: one RTX 3080 Ti, two CUDA jobs just contend.
set -u
cd "$(dirname "$0")/.."
echo "waiting for tests_ftt cap-90 seed 42 to finish ($(date))"
until grep -q '^seed 42:' tests_ftt/run_lr1e-3_e90.log 2>/dev/null; do sleep 60; done
echo "FTT done ($(date)); starting the forward_inner retrain"
bash tests_mlp_convergence/retrain_capped_inner.sh
