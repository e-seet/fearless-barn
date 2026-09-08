set -e
for POL in keep_all winsor_7200 winsor_21600 drop_7200; do
  B="--split forward --featureset with_aobt --outlier-policy $POL --seed 42 --force"
  echo "######## POLICY $POL ########"
  python -u stage4_controls.py $B
  python -u stage5_ablations.py $B
  for arm in naive_offblock ridge lightgbm; do
    python -u pipeline_${arm}/train.py $B
    python -u -m pipeline_${arm}.evaluate $B --part val
    python -u -m pipeline_${arm}.evaluate $B --part test
  done
  python -u compare.py $B
done
