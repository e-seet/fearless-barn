# Weather (METAR) value check — CLOSED, do not rebuild

**Date:** 2026-09-11
**Question:** does external METAR data explain forward/test residual variance that
the current (v22-class `airport_operator_multimodel` msmlp) model does not already
capture — enough to justify a fingerprint-breaking full stack retrain?

**Answer: no.** Weather's out-of-sample incremental signal over the current
feature set is inside the noise floor of the test. Not worth a retrain, and a
hand-tuned post-hoc correction is a negative-EV bet given board history.

## What was done (no fingerprint change, no GPU)

- `fetch_metar.py` — pulled half-hourly METAR for all 10 DC2026 departure
  airports (LTFM LFPG EGLL EHAM LEMD EDDF LEBL LIRF EDDM LSZH) from Iowa State
  mesonet ASOS, 2025-01 → 2026-07. Written to `prc-2026-datasets-wx/`.
  **Coverage is uniform and complete — including LTFM** (the 14%-of-scored-set
  airport that was the main coverage risk). ~17.5k obs/station in 2025, ~10k in
  the Jan–Jul 2026 scored window, ~30 min median gap. See `metar_coverage.csv`.
- `wx_value_check.py` — baseline = `airport_operator_multimodel` tag=msmlp on
  `forward/test` (Nov–Dec 2025), RMSE 239.87, 328,009 rows, **100% METAR-joined**
  (nearest obs at-or-before movement time, ≤90 min). Then three tests on the
  residual `resid = y − pred`.

## Test 5a — residual RMSE by weather bucket (LOOKS compelling, is misleading alone)

| bucket | n | RMSE in | RMSE elsewhere | resid bias |
|---|---|---|---|---|
| thunderstorm | 491 | 468.3 | 239.4 | **+224.8** |
| snow / ice pellets | 3,037 | 431.2 | 237.4 | **+168.4** |
| freezing (FZ or ≤0 °C) | 20,673 | 322.2 | 233.3 | +79.0 |
| visibility < 3 mi | 18,115 | 325.9 | 233.9 | +83.2 |
| fog / mist | 18,300 | 312.3 | 234.9 | +73.2 |
| LIFR | 18,892 | 304.0 | 235.4 | +71.5 |
| gust ≥ 25 kt | 4,133 | 304.1 | 238.9 | +33.5 |
| any precip | 37,825 | 255.4 | 237.8 | +32.5 |

Real, directionally consistent: the model **under-predicts** taxi-out in adverse
weather by a conditional mean shift of ~30–225 s. **But this table must always be
read next to 5b** — on its own it re-opens this question every time.

## Test 5b — out-of-sample residual-explaining power (THE DECIDING TEST)

LightGBM on the residual, train Nov → test Dec and vice versa:

| feature set | OOS R² on residual (fold1 / fold2) | note |
|---|---|---|
| **congestion only** | +0.0083 / +0.0058 → mean **+0.0070** | ≈0 BY CONSTRUCTION — these features are already in the model. This is the noise floor of the test. |
| **weather only** | −0.0196 / +0.0078 → mean **−0.0059** | *below* the congestion floor, negative on one fold |
| weather + congestion | +0.0026 / +0.0183 → mean +0.0104 | |
| **weather Δ over congestion** | **+0.0034** | smaller than the instrument's own error bar |

**Fold reversal:** adding the weather residual-model takes y-RMSE 239.9 → 232.3
on Nov→Dec but 239.9 → **248.2** (worse) on Dec→Nov. Congestion-only does the
same (229 / 248). A real signal does not flip sign when you swap train/test
months — the ±8-point swing is fold artifact, not weather.

## SSE arithmetic (best case, in-sample)

~18k rows × (83 s)² ≈ 1.24e8 correctable SSE against total ~1.89e10
→ ~0.65 % SSE → **< 1 RMSE point**, and that is the optimistic version.
Two further discounts:
1. Nov–Dec is the *most* weather-exposed window. The scored set is Jan–Jul 2026;
   Feb–Jul carry almost none of the freezing/snow mass driving 5a. Board effect
   is smaller than 1 point.
2. 5a is a conditional mean shift on 5.5 % of rows; 5b says it cannot be
   localised row-by-row out of sample.

## Test 5c — tail association (rows with |resid| > 600 s)

Weather flags are enriched 2–12× in big-residual rows (thunder ×12.2, snow ×7.4,
low-vis ×3.0, freezing ×3.0, fog ×2.7). Weather *co-occurs* with the tail but
does not predict its magnitude — and the tail is already known to be
LIRF-dominated and ~irreducible (see plan 13 § outliers).

## Why not a post-hoc correction (no fingerprint change needed)?

Same shape of bet as `congest` (+0.01 board), `idband` (3/3 val → +0.86 board),
candidate 9 (−1.27 val → +3.15 test): tuned on a 2025 holdout, applied to a 2026
scored set with different weather composition, for a sub-1-point gain that 5b
says is fold-unstable. Expected value is negative. Not submitted.

## Part 2 — the January / LFPG / LIRF check (2026-09-11)

Prompted by the `seasonal` split (holds out Jan+Jul): every arm scores 240 on
July but **400–470 on January**, and within January **LFPG ~800 / LIRF ~790
RMSE** vs ~200 elsewhere. Looked like winter de-icing. It is not.
Script: `jan_lfpg_lirf_decomp.py`.

**The January blow-up is ~20–40 data-artifact rows, not weather:**
- LFPG+LIRF Jan-2025 rows with baseline residual > 1800 s: **187 rows (0.64 %)
  carrying 88.5 % of SSE**. > 3600 s: 39 rows, 85 % of SSE. > 5400 s: 20 rows.
- AOBT_3 **missing** on them: 23 % at the 1800 s cut, **69 % at 3600 s, 85 % at
  5400 s** (vs 0.9 % of all Jan focus rows). The proxy is the model's #1 feature;
  these are the rows where it is gone.
- On the AOBT-present subset, `|true − proxy|` is median 2400–6500 s — the
  recorded off-block is inconsistent with the taxi-out by 40 min to 1.8 h.
- Max taxi-out among them: **84,240 s (LFPG) / 87,177 s (LIRF)** — i.e. ~24 h.
  Pure artifacts.
- **Not date-clustered:** residual > 1800 s is spread over all 31 days; top-5
  day×airport buckets hold only 38 %.
- **METAR on blow-up days ≈ normal January days**: snow 0.13 vs 0.18, freezing
  5.8 vs 5.2, low-vis 5.0 vs 5.4 obs/day. LIRF's worst days had **zero** snow and
  **zero** freezing in the METAR.

**LIRF's long-taxi tail is year-round and heaviest in summer**, not a January
thing: rows > 3600 s per month in 2025 — Jan 70, Feb 88, … **Jun 250, Jul 390**,
Aug 250. Fiumicino just has long/variable ground ops (remote apron bussing, tows).

**It does not transfer to the 2026 scored set:**
- LFPG/LIRF 2026 January: **AOBT-missing is 1.0–1.8 %** (vs 23–85 % in the
  2025-Jan tail). The artifact pathology is a 2025-data problem.
- v24 already predicts LFPG 2026-Jan at median 925 / p99 2906 (68 rows > 3600 s
  of 18,129) and LIRF 2026-Jul at p99 3705 / 171 rows > 3600 s — the model *is*
  attempting the LIRF tail where it exists in the scored data.

**This is why the real board score (295.75) beats what the seasonal split
threatened (360–417):** the 2026 inference data is clean on exactly the axis
(missing AOBT + implausible taxi-out) that inflates the 2025 seasonal January.

## Conclusions (both parts)

1. **Weather is closed.** Neither the Nov–Dec bulk test nor the January tail
   decomposition finds a feature-able weather signal. The January airport/month
   pattern that briefly reopened it is data artifacts.
2. **No cheap lever remains in the tail.** ~30 missing-AOBT / implausible rows
   per winter month cannot be predicted from schedule features, and they are
   mostly absent from the 2026 scored data anyway.
3. **The gap to the leaders (youthful-giraffe 245, field 245–267) is bulk
   in-band accuracy across all months**, not a tail/weather trick. Seasonal
   lightgbm in-band RMSE 236 vs forward 219 — January is only ~17 in-band RMSE
   harder, the rest of the seasonal horror is the artifacts. Closing 40+ points
   means a stronger base model, i.e. a from-scratch modelling effort, not a lever
   on this pipeline.

## Kept for a possible future revisit

`prc-2026-datasets-wx/` (raw METAR, parsed + 100 % joined) and all three scripts
(`fetch_metar.py`, `wx_value_check.py`, `jan_lfpg_lirf_decomp.py`).
