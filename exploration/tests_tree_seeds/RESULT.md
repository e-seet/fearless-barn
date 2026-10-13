# Tree-side seed averaging (airport_operator_blend 1 → 6 seeds) — result: FAIL

Bar: `PREREGISTERED.md`, fixed before seeds 43–47 were trained on seasonal_full
or full. `check.py` → `check.txt` / `check.json`. All seeds trained with seed 42's
recipe (seasonal_full ~9.5 min/seed; full ~2.5 min/seed, frozen from each forward
seed).

In-band RMSE of the 3-member blend [FTT6, tree, MLP], tree = seed 42 vs the mean of
42–47, weights re-solved on forward_inner for each:

| surface | 1 seed | 6 seeds | Δ |
|---|---:|---:|---:|
| forward val (Sep–Oct) | 223.875 | 223.784 | −0.091 |
| forward test (Nov–Dec) | 215.484 | 215.300 | −0.184 |
| seasonal_full val (Dec) | 211.495 | 211.492 | −0.003 |
| **seasonal_full test (Jan + Jul)** | **250.971** | **251.134** | **+0.163** |

**FAIL on the decisive condition** (Jan + Jul in-band). The pooled guard and LIRF
conditions hold (LIRF identical; tree weight 0 there). This is the fourth lead
this week to improve Nov–Dec but not Jan + Jul.

**Why this check is trusted:** for FTT seed averaging the Jan + Jul check
predicted the board direction both times. 3 → 6 passed (−0.128) and the board paid
(v32, −1.79); 6 → 9 failed (+0.062) and the 7-seed board probe tied (v34, +0.04).

**Closed.** The full-data tree seeds 43–47 are kept on disk but unused.
