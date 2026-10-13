# LIRF no-record specialist — result: FAIL (2026-10-08)

Pre-registration: `PREREGISTERED.md`. Reference: F clipped to the LOBT window; the candidate is clipped too.

| month | scope rows | scope RMSE F → S | month RMSE change |
|---|---:|---:|---:|
| Jan | 60 | 4,253 → 3,408 | −3.76 |
| Jul | 337 | 5,470 → 3,748 | −44.58 |
| Aug | 197 | 4,113 → 3,654 | −7.36 |
| Sep | 168 | 4,651 → 4,659 | +0.14 |
| Oct | 115 | 2,907 → 2,630 | −2.23 |
| Nov | 52 | 7,983 → 5,256 | −25.96 |
| Dec | 88 | 3,116 → 2,136 | −6.79 |

| | F | with S |
|---|---:|---:|
| pooled, 7 months | 265.677 | 251.675 (−14.00) |
| Jan + Jul | 337.560 | 311.912 (−25.65) |
| best 5 rows removed | 252.499 | 249.040 |
| **best 20 rows removed** | **243.741** | **246.270 (+2.53)** |
| best 50 rows removed | 240.268 | 245.085 |

**FAIL** on bar 4: the gain is in a few dozen multi-hour rows. On the rest of the scope the specialist is
worse than F, increasingly so as more top rows are removed.

**Context, not changing the verdict.** quirky-honey report −24.1 on the board from the same idea (302.01 →
277.91), close to our Jan + Jul −25.6; their tail rows transferred to 2026. outgoing-boat report that
tail-only gains are anti-correlated with the board.

**Follow-up.** The same specialist is scored on held-out Feb–Jun 2025 in `tests_final/` (a surface this
test did not use). Any hybrid (specialist only on rows far behind schedule) needs its own pre-registration.
