# Carrier-echo hedge for LIRF no-record departures — pre-registration

Written 2026-09-30, after the exploratory step (`explore.txt`, `lirf.txt`) and **before any
comparison with model predictions**. Base: v42 = 286.5256 (best and last). Source of the idea:
the competition forum ("the echo of the planned time in the off-block is mostly a carrier thing
when there's no NM match, and pretty stable month to month"; "hedge rather than commit").

## What the exploration showed (2025, read-only)

- "Echo" = the recorded off-block equals the scheduled time (±60 s), so the label equals the
  scheduled delay `ds`. On no-record departures it happens almost only at **LIRF** (49 % vs
  0–4 % elsewhere).
- Within LIRF no-record, **the carrier** (prefix of `FLIGHT_mvt`) separates echo from normal:
  several carriers ≈ 1.0, several ≈ 0.0, mostly stable across quarters. The leave-one-month-out
  carrier rate beats the LIRF constant on Brier score (0.255 → 0.178; Jan 0.248 → 0.170,
  Jul 0.262 → 0.185). Echo rows have median y ≈ 7,000 s against ≈ 1,000 s otherwise.
- The earlier airport-level rule (`idband`: *commit* to y = ds for LIRF no-record, ds > 25.2 k)
  **regressed on the 2026 board twice (+0.86)**. This must therefore be a *hedge*, not a commitment.

## The rule (H1), fixed now

For every LIRF departure without an NM record (`FLIGHT_ID_mvt` missing):

    pred = p · ds + (1 − p) · b

- p = the carrier's echo rate, smoothed toward the LIRF rate with weight k = 10, **computed only
  from months other than the one being predicted** (for the board: all 12 months of 2025).
- b = the median true taxi-out of non-echo LIRF no-record departures in those same other months.
- p · ds uses ds clipped at [0, 86,400]. Every other row keeps the model's prediction.

Also reported descriptively (not the bar): H2 = the average of H1 and the model's prediction.

## Evaluation surfaces and bar

The model prediction at LIRF is the shipped LIRF recipe on each surface: 0.541 × MLP + 0.459 × TabNet.

| surface | months | role |
|---|---|---|
| seasonal_full test | **Jan + Jul 2025** | **decisive** |
| forward val + test | Sep–Dec 2025 | corroborating |
| forward_inner val | Jul–Aug 2025 | corroborating |

**PASS requires all of:**
1. H1 lowers **pooled RMSE over all rows** on Jan + Jul (the rule targets the tail, so pooled is
   the relevant metric) **and** lowers LIRF no-record RMSE on Jan and on Jul separately;
2. the same direction on Sep–Dec and on Jul–Aug;
3. the gain is not carried by a single row: removing the one row with the largest improvement
   still leaves an improvement on Jan + Jul.

**PASS →** one board probe (v43 = v42 with H1 applied to the scored LIRF no-record rows, p and b
from all of 2025), shown to the user before `submit.py`. **If it loses, v42 is re-uploaded last
as a byte copy.** Given the `idband` history, the expectation is written before the upload.
**FAIL →** closed.
