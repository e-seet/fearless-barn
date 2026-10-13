#!/usr/bin/env bash
# Rec 3: retrain the forward_inner MLP seeds whose early stopping never fired.
#
# At --epochs 25, twelve of the eighteen forward_inner seeds behind v25's blend
# weights ended AT the cap with best_epoch 19-25 -- `hit_max_epochs: true`,
# `early_stop_fired: false`, `converged: false` in each model.json. For those
# runs `best_epoch` is the last epoch paid for, not a selected one.
#
# This is a CORRECTNESS fix, not a score lever, and the distinction is measured
# rather than assumed: ../CONVERGENCE_AUDIT.md sec.1 re-solved the per-airport
# blend alphas from capped vs converged seeds and got a mean shift of +0.003
# with mixed signs, and the seeds that already had a longer leash do not beat
# the rest on val. Expect the weights to move by ~nothing. The point is that
# the declared stopping criterion actually fires.
#
# The default cap is now 45 (pipeline_mlp_antigravity/train.py), so --epochs is
# not passed; --force is required because each run directory already exists.
#
# Restartable: a seed whose run already early-stopped is skipped, so the script
# can be re-run after an interruption without redoing finished work.
#
# ~1 h/seed on the RTX 3080 Ti => ~12 h for all twelve. Runs sequentially: two
# concurrent CUDA jobs on one card just contend.
set -u
cd "$(dirname "$0")/.."

CAPPED_SEEDS="42 43 44 48 49 50 51 53 54 55 57 58"
BACKUP="tests_mlp_convergence/TRUNCATED25"

# Clean-stop plumbing. Learned the hard way on 2026-09-21: killing the python
# child with `taskkill /IM python.exe` left THIS loop alive, so it advanced to
# the next seed and started overwriting a run directory seconds after the
# previous one had been restored from backup. A loop that outlives its kill is
# a loop that corrupts artifacts.
#
#   stop politely : touch tests_mlp_convergence/STOP
#                   (exits cleanly before the next seed, current one finishes)
#   stop now      : kill $(cat tests_mlp_convergence/retrain.pid)
#                   (kills the LOOP; the trap takes the running child with it)
PIDFILE="tests_mlp_convergence/retrain.pid"
STOPFILE="tests_mlp_convergence/STOP"
CHILD=""
echo $$ > "$PIDFILE"
cleanup() {
  echo "=== interrupted -- stopping child and exiting ==="
  [ -n "$CHILD" ] && kill "$CHILD" 2>/dev/null
  rm -f "$PIDFILE"
  exit 130
}
trap cleanup INT TERM
rm -f "$STOPFILE"

# Preserve the truncated runs BEFORE --force overwrites them. This repo's own
# cap rule says so explicitly (tools/ml_common/convergence.py, MISTAKES.md #70):
# "Raise the cap and re-run into a NEW output directory; do not reuse the
# truncated one." The truncated model.pt files are what CONVERGENCE_AUDIT.md
# sec.1's capped-vs-converged alpha table was computed from, so overwriting them
# would make that measurement unreproducible. 45 MB for all twelve.
mkdir -p "$BACKUP"
for s in $CAPPED_SEEDS; do
  RUN="artifacts/runs/mlp_antigravity__forward_inner-with_aobt-keep_all__seed${s}"
  DST="$BACKUP/seed${s}"
  if [ -d "$RUN" ] && [ ! -d "$DST" ]; then
    cp -r "$RUN" "$DST"
    echo "backed up seed $s -> $DST"
  fi
done

for s in $CAPPED_SEEDS; do
  RUN="artifacts/runs/mlp_antigravity__forward_inner-with_aobt-keep_all__seed${s}"
  if [ -f "$RUN/model.json" ] && python -c "
import json,sys
c = json.load(open(r'$RUN/model.json')).get('convergence', {})
sys.exit(0 if c.get('early_stop_fired') and not c.get('hit_max_epochs') else 1)
" 2>/dev/null; then
    echo "=== seed $s already early-stopped, skip ==="
    continue
  fi
  if [ -f "$STOPFILE" ]; then
    echo "=== STOP file present -- exiting cleanly before seed $s ==="
    rm -f "$PIDFILE"
    exit 0
  fi
  echo "=== forward_inner seed $s at the new default cap ($(date)) ==="
  python pipeline_mlp_antigravity/train.py \
      --split forward_inner --seed "$s" \
      --featureset with_aobt --outlier-policy keep_all --force &
  CHILD=$!
  wait "$CHILD"
done
rm -f "$PIDFILE"

echo "=== all done ($(date)) ==="
echo "Next: re-solve the blend weights and compare against the shipped alphas --"
echo "  python tests_mlp_convergence/alpha_cap_vs_converged.py"
