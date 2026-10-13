set -e
cd "$(dirname "$0")/.."
echo "########## 1. no_aobt featureset (the ablation PLAN.md promised) ##########"
python run_all.py --stage all --split forward --featureset no_aobt --outlier-policy keep_all --seed 42 --force
echo "########## 2. repeat-seed sweep, ALL six arms (the ensemble needs its bases at the same seed) ##########"
python run_all.py --stage sweep --split forward --featureset with_aobt --outlier-policy keep_all \
  --sweep-arms naive_offblock,ridge,lightgbm,xgboost_antigravity,mlp_antigravity,ensemble_antigravity --force
echo "########## 3. seasonal composition probe (selection-forbidden) ##########"
python run_all.py --stage all --split seasonal --featureset with_aobt --outlier-policy keep_all --seed 42 --force
echo "ALL RESUME STAGES DONE"
