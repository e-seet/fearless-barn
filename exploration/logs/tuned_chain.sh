set -e
cd "$(dirname "$0")/.."
R=artifacts/runs
CFG=lr02_lv511_md50
TAG=inner2_tuned
FI="--split forward_inner --featureset with_aobt --outlier-policy keep_all --seed 42 --force"
FW="--split forward --featureset with_aobt --outlier-policy keep_all --seed 42 --force"

# resumable: each step skips if its own output exists
step () { if [ -e "$1" ]; then echo "  [skip] $2"; else echo "  [run ] $2"; shift 2; bash -c "$*"; fi; }

echo "########## 1/3 per_airport_stack @ $CFG on forward_inner (weight surface) ##########"
step "$R/per_airport_stack__forward_inner-with_aobt-keep_all__seed42__tag=$TAG/eval_val.json" \
     "stack tuned on forward_inner" \
     "python pipeline_per_airport_stack/train.py $FI --lgb-config $CFG --inner-val-months 1 --tag $TAG && python -m pipeline_per_airport_stack.evaluate $FI --tag $TAG --part val"

echo "########## 2/3 per_airport_stack @ $CFG on forward ##########"
step "$R/per_airport_stack__forward-with_aobt-keep_all__seed42__tag=$TAG/eval_test.json" \
     "stack tuned on forward" \
     "python pipeline_per_airport_stack/train.py $FW --lgb-config $CFG --inner-val-months 2 --tag $TAG && python -m pipeline_per_airport_stack.evaluate $FW --tag $TAG --part val && python -m pipeline_per_airport_stack.evaluate $FW --tag $TAG --part test"

echo "########## 3/3 cross_blend on the tuned stack, weights from forward_inner ##########"
step "$R/cross_blend__forward-with_aobt-keep_all__seed42__tag=w_inner_tuned/eval_test.json" \
     "cross_blend tuned" \
     "python pipeline_cross_blend/train.py $FW --weight-mode per_airport --weight-split forward_inner --arm-a per_airport_stack --arm-a-tag $TAG --tag w_inner_tuned && python -m pipeline_cross_blend.evaluate $FW --tag w_inner_tuned --part val && python -m pipeline_cross_blend.evaluate $FW --tag w_inner_tuned --part test"
echo "TUNED CHAIN DONE"
