# Tree-side seed averaging (airport_operator_blend 1 → 6 seeds) — pre-registration

Written 2026-09-25, **before any seasonal_full or full tree seed beyond 42 was
trained**. Plan 15 follow-up; v32 = 287.9898 is the base.

## Why

The tree member (`airport_operator_blend`: per-airport + per-operator LightGBM
with feature and bagging fractions of 0.8) is seed-dependent, yet v32 ships **one**
seed. The board has twice paid far more for seed averaging than local surfaces
showed (MLP 1 → 3: −0.9 local, −18 board; FTT 3 → 6: −0.2 local, −1.79 board).
FTT 6 → 7 tied (v34), so seed-averaging gains do saturate.

**Descriptive sizing already seen** (`size.txt`; forward seeds 42–47 existed): in
the 3-member blend with weights re-solved on forward_inner, 1 → 6 tree seeds gives
in-band **−0.091 (val) / −0.184 (test)**, and moves the blend by an RMS of 7.3–7.7
against 11.5–12.7 for FTT 3 → 6. Because these forward numbers were seen first,
**the decisive condition is the seasonal_full one below, which has not been seen.**

## Training (identical recipe to seed 42)

- `seasonal_full`, seeds 43–47: `pipeline_airport_operator_blend/train.py --split seasonal_full --seed s`.
- `full`, seeds 43–47: `--split full --seed s --freeze-from <forward seed s model.json>`,
  exactly as seed 42's full run was frozen from forward seed 42.

## Bar (all must hold; in-band = y ≤ 7200)

Weights are solved on forward_inner for each tree side separately (tree seed 42
vs the mean of seeds 42–47), frozen, and applied with the matching tree side.
FTT = 6-seed mean.

1. **seasonal_full test (Jan + Jul):** 6-seed-tree blend in-band **lower** than
   1-seed (with the 6-seed MLP and forward_inner-solved weights, as in
   `tests_ftt/confirm_seasonal_full.py`);
2. same sign on forward val and forward test (already seen: −0.09 / −0.18);
3. **tail guard:** Jan + Jul pooled not worse by more than 2.0 s;
4. LIRF unchanged (the tree weight at LIRF is 0; a non-zero value halts the build).

**PASS →** extend the `ftt_multimodel` arm to average a recorded tree-seed list,
re-solve the `full` weights with it, register a new tag with the TabNet override
on top, check against v32 in scratch (LIRF identical), and show the user before
`submit.py`. **FAIL →** closed; v32 stays.

**Expected board effect:** −0.3 to −1.0, genuinely uncertain. v34 moved
predictions by a similar RMS and tied.
