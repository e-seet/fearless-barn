set -e
cd "$(dirname "$0")/.."
B="--split forward --featureset with_aobt --outlier-policy keep_all --seed 42 --force"
for MODE in global per_airport; do
  echo "########## cross_blend weight-mode=$MODE ##########"
  python pipeline_cross_blend/train.py $B --weight-mode $MODE --tag "w_$MODE"
  python -m pipeline_cross_blend.evaluate $B --tag "w_$MODE" --part val
  python -m pipeline_cross_blend.evaluate $B --tag "w_$MODE" --part test
done
echo "CROSS BLEND DONE"
