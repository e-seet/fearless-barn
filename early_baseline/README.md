# `early_baseline/` — the first-week baseline (superseded)

This folder is **not** the submitted solution. The solution is in [`../solution/`](../solution/);
start with [`../solution/WRITEUP.md`](../solution/WRITEUP.md).

It is the first build of the project (5–16 September 2026), kept because the later work was built on
its checks:

| path | what it is |
|---|---|
| `stage1_audit.py` | data audit, including whether any provided timestamp reconstructs the target |
| `stage2_split.py` | the first time-ordered train / validation / test split |
| `stage6_clip_sweep.py` | a check of prediction clipping |
| `pipeline_naive_offblock/` | a baseline that uses the recorded off-block proxy directly |
| `pipeline_lightgbm/` | one pooled LightGBM model |
| `common/`, `config.py` | data loading and features, as they were then |
| `artifacts/` | the outputs of the runs above |
| `notebook.ipynb`, `eda_notebook/` | early notebooks |

Nothing in `solution/` imports from this folder.
