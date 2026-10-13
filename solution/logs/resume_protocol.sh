set -e
B="--split forward --featureset with_aobt --outlier-policy winsor_p9999 --seed 42 --force"
python stage5_ablations.py $B
for arm in naive_offblock ridge lightgbm; do
  python pipeline_${arm}/train.py $B
  python -m pipeline_${arm}.evaluate $B --part val
  python -m pipeline_${arm}.evaluate $B --part test
done
python compare.py $B
