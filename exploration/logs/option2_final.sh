set -e
cd "$(dirname "$0")/.."
# No pgrep on this platform, so wait on a real condition instead of a process:
# every prerequisite writes a marker line to its own log when it completes.
wait_for () {  # $1 = log, $2 = marker
  while ! grep -q "$2" "$1" 2>/dev/null; do
    grep -q "Traceback" "$1" 2>/dev/null && { echo "PREREQ FAILED: $1"; exit 1; }
    sleep 20
  done
}
wait_for logs/inner_mlp_fix.log "INNER MLP FIX DONE"
wait_for logs/option2.log "OPTION2 DONE"

echo "##### verifying the inner MLP actually converged before trusting anything #####"
python - <<'PY'
import json, sys
d = json.load(open("artifacts/runs/mlp_antigravity__forward_inner-with_aobt-keep_all__seed42/model.json"))
c = d.get("convergence", {})
print("  inner MLP: best_epoch=%s stop=%s max=%s early_stop_fired=%s hit_cap=%s"
      % (c.get("best_epoch"), c.get("stop_epoch"), c.get("max_epochs"),
         c.get("early_stop_fired"), c.get("hit_max_epochs")))
if c.get("hit_max_epochs"):
    sys.exit("HALT: inner MLP still hit its cap; raise --epochs again before fitting weights on it")
PY

B="--split forward --featureset with_aobt --outlier-policy keep_all --seed 42 --force"
echo "##### OPTION 2 (re-run last, so it cannot use a stale ensemble) #####"
python pipeline_cross_blend/train.py $B --weight-mode per_airport --weight-split forward_inner --tag w_inner
python -m pipeline_cross_blend.evaluate $B --tag w_inner --part val
python -m pipeline_cross_blend.evaluate $B --tag w_inner --part test
echo "OPTION2 FINAL DONE"
