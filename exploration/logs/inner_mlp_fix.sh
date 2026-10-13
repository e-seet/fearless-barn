set -e
cd "$(dirname "$0")/.."
# Wait for the first inner pass to finish so we do not contend.
while pgrep -f "inner_arms.sh" >/dev/null 2>&1; do sleep 20; done
B="--split forward_inner --featureset with_aobt --outlier-policy keep_all --seed 42 --force"
echo "########## MLP re-run with a raised cap (previous run hit 25/25) ##########"
python pipeline_mlp_antigravity/train.py $B --epochs 60
python -m pipeline_mlp_antigravity.evaluate $B --part val
echo "########## ensemble rebuilt on the converged MLP ##########"
python pipeline_ensemble_antigravity/train.py $B
python -m pipeline_ensemble_antigravity.evaluate $B --part val
echo "INNER MLP FIX DONE"
