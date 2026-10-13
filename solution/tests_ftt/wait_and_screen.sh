#!/usr/bin/env bash
# Wait for FTT seeds 43 and 44 at cap 132, then run the pre-registered 3-seed
# screen. Writes the result to screen_seeds424344_e132.txt so it survives
# regardless of what happens to the session that launched it.
set -u
cd "$(dirname "$0")/.."
echo "waiting for seeds 43 and 44 ($(date))"
until [ -f tests_ftt/preds_seed43_lr1e-3_e132.npz ] && [ -f tests_ftt/preds_seed44_lr1e-3_e132.npz ]; do
  sleep 60
done
sleep 10   # let the last npz finish flushing
echo "both seeds present ($(date)); running the 3-seed screen"
python tests_ftt/decorrelation.py --tag lr1e-3_e132 --seeds 42 43 44 \
    2>&1 | tee tests_ftt/screen_seeds424344_e132.txt
echo "=== screen done ($(date)) ==="
