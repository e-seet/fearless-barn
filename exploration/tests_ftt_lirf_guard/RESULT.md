# FTT at LIRF with a downward-only guard — result: FAIL (w = 0)

Pre-registered in `PREREGISTERED.md`. `check.py` → `result.txt` / `result.json`.

The least-squares weight on forward_inner LIRF rows came out **negative**, so it
clamps to **w = 0**, and every surface is unchanged. FTT sits below the shipped
LIRF prediction on 66 % of LIRF rows, yet pulling those rows down *increases*
pooled squared error. LIRF predictions are already biased low where the error
mass is, in the tail (`tests_tail_diagnosis`: v31 bias −2,600 to −24,000 s in
the tail bands). Any downward move costs more there than it gains in-band.

**Closed.** This also confirms the six prior solves' verdict (FTT weight exactly
0 at LIRF) from a direction that removed FTT's upward tail errors: even
downward-only, FTT adds nothing at LIRF.
