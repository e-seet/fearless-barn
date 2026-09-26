# C2 board probe (6 small + 3 d192 FTT) — decided after the local verdict

Written 2026-09-26, **before the build**. This is **a board probe of a candidate that
FAILED its pre-registered local bar**, not a pass. Forward val −0.28 and test
−0.56, but Jan + Jul **+0.03** (a tie). It is the only near-miss this week with a
sizeable forward gain. On two FTT seed questions the Jan + Jul check matched the
board, so **the expectation is a tie with v35 (287.9247).**

- Needs d192 `full` seeds 43 and 44 (seed 42's `full` run finished at 99 epochs,
  the median forward best).
- Build: FTT slot = mean of the 6 small + 3 d192 `full` models; per-airport weights
  re-solved on forward_inner exactly as `check.py` did for C2; TabNet LIRF
  override at w = 0.459 on top, as v35.
- **Ranking uses the last submission.** If the probe scores worse than v35, v35 is
  rebuilt byte-identically under a new version number and uploaded last.

## Result (2026-09-27)

**v36 = 287.2876, −0.637 vs v35.** A candidate that tied locally on Jan + Jul (+0.03) and helped the forward months (−0.28 / −0.56) *gained* on the board. This is the second time the board has rewarded more FTT diversity beyond what Jan + Jul showed (v32 paid ~4× its local estimate). The 6→7-seed tie (v34) shows it is not automatic.
