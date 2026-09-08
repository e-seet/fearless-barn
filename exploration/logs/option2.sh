set -e
cd "$(dirname "$0")/.."
while pgrep -f "inner_mlp_fix.sh" >/dev/null 2>&1 || pgrep -f "inner_arms.sh" >/dev/null 2>&1; do sleep 20; done
IB="--split forward_inner --featureset with_aobt --outlier-policy keep_all --seed 42 --force"
echo "########## per_airport_stack on forward_inner (arm A must exist there) ##########"
python pipeline_per_airport_stack/train.py $IB --inner-val-months 1 --tag inner2
python -m pipeline_per_airport_stack.evaluate $IB --tag inner2 --part val
B="--split forward --featureset with_aobt --outlier-policy keep_all --seed 42 --force"
echo "########## OPTION 2: cross_blend, weights from forward_inner (clean) ##########"
python pipeline_cross_blend/train.py $B --weight-mode per_airport --weight-split forward_inner --tag w_inner
python -m pipeline_cross_blend.evaluate $B --tag w_inner --part val
python -m pipeline_cross_blend.evaluate $B --tag w_inner --part test
echo "OPTION2 DONE"
