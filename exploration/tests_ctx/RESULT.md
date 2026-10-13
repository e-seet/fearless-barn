# Context corrector on every row — result: FAIL (2026-10-09)

Pre-registration: `PREREGISTERED.md`. Numbers: `ctx.json`.

| check | 7 months (Jan, Jul–Dec 2025) | held-out Feb–Jun 2025 |
|---|---|---|
| pooled | 265.677 → 264.498 (−1.18) | 274.018 → 269.773 (−4.25) |
| board-month pair | Jan + Jul **+0.44** | Feb + Jun −7.39 |
| months better | 6/7 (Jan **+1.29**) | 5/5 (Feb −14.4) |
| best 20 rows removed | −0.75 | −3.71 |
| gain from rows with y ≤ 3,600 s | positive, but ~2 % of the total | positive, ~8 % of the total |
| surface verdict | **FAIL** (Jan + Jul worse) | PASS |

**Drift gate: PASS.** The largest standardized shift is 0.09.

**Verdict: FAIL.** The pre-registration requires both surfaces. The corrector is worse on Jan + Jul, two of
the four final months, and it gets nearly all of its gain from the tail.

**Why the Feb–Jun gain is likely overstated.** The Feb–Jun reference base comes from models trained on six
months, mostly after the months they predict (`tests_febjun`; base RMSE 285 against the 7-month surface's
recipe). A weaker base leaves more residual for any corrector to take. The 7-month surface is closer to the
base actually shipped.

**Closed. The final file stays as built.**
