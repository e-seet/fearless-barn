# Large-FTT 3 → 6 seeds in the mixed FTT slot (board probe) — written before training

2026-09-27. Base: **v36 = 287.2876** (FTT slot = 6 small + 3 d192, equal weight per
model; the large share is 1/3). v38 is the byte-identical re-upload of v36.

**Why.** The board has twice paid more for FTT-slot variety than the local surfaces
showed (v32 −1.79; v36 −0.64 on a local Jan + Jul tie). The d192 seeds disagree far
more than the small ones (forward val 268.6–303.0 against 263.3–267.2), so averaging
more of them has room to reduce variance. The 7th *small* seed tied (v34), so a gain
is not automatic.

**Change, and only this change:** d192 seeds 45, 46, 47 added. The large model's
**share stays 1/3**: 6 small at weight 1 and 6 large at weight 0.5 (`seed:tag:weight`
members). Everything else is as v36: forward_inner weight solve (default solver),
tree seed 42, 18-seed MLP, TabNet LIRF w = 0.459.

**Training:** `run_ftt.py --d-token 192 --amp --lr 5e-4 --tag d192lr5`:
forward_inner seeds 45–47 (cap 260, patience 10) and full seeds 45–47 (fixed 99 epochs,
as seeds 42–44). No local screen: the local check did not predict v36's gain, so this is
**a board probe**.

**Expectation:** between a tie and about −0.5 vs v36. The fallback is v36/v38
re-uploaded last.
