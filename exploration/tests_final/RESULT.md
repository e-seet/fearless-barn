# Final round: month-scoped stages re-checked on held-out Feb–Jun 2025 — result

Pre-registration: `PREREGISTERED.md` (and `../tests_lirf_spec/PREREGISTERED_hybrid.md` for the hybrid).
Script: `months_of_concern.py`; numbers in `months_of_concern.json`.

**RMSE on held-out Feb–Jun 2025** (851,813 departures):

| arm | Feb | Mar | Apr | May | Jun | Feb + Jun | Feb–Jun |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 319.72 | 225.49 | 215.91 | 308.26 | 333.90 | 327.74 | 285.11 |
| + stage 2 (`model_v61`) | 309.93 | 210.86 | 205.97 | 296.98 | 323.12 | 317.39 | 274.13 |
| shipped (+ LFPG corrector) | 309.93 | 210.86 | 205.97 | 296.98 | 323.11 | 317.39 | 274.13 |

**The design transfers to the new months:** stage 2 gives −11.0 (5/5 months, −11.4 with the best 20 rows
removed). The LFPG corrector is neutral there.

**Each change against the shipped design** (bars: Feb + Jun better, Feb–Jun better, ≥ 2/3 of changed months
better, better with the best 20 rows removed):

| change | rows | Feb–Jun | Feb + Jun | months | best 20 removed | verdict |
|---|---:|---:|---:|---:|---:|---|
| LIRF ADS-B corrector in every month (not only Jan) | 13,009 | −0.30 | −0.47 | 2/2 | −0.19 | **PASS** |
| LOBT-window clip | 86 | −0.11 | −0.11 | 5/5 | −0.005 | **PASS** |
| frost +30 s in every month | 3,305 | −0.14 | −0.31 | 1/3 | −0.12 | FAIL (months) |
| thunderstorm +200 s in every month | 827 | −0.06 | −0.15 | 3/5 | −0.005 | FAIL (months) |
| LIRF no-record specialist | 470 | −15.81 | −32.0 | 4/5 | +3.29 | FAIL (best 20) |
| hybrid specialist (> 6 h behind schedule) | 14 | −14.34 | −24.5 | 4/5 | +0.009 | FAIL (best 20) |

**The hybrid, across both surfaces:** better in 11 of the 12 months of 2025 (7/7 and 4/5, May unchanged),
−15.6 and −14.3 pooled, touching 49 and 14 rows. It fails the best-20 bar on both. On Feb–Jun only 14 rows
change, so the best-20 bar removes every changed row: it cannot be passed there by construction. Its gain
is entirely in ~20–30 LIRF no-record departures per year that took off many hours behind schedule.
