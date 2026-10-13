# FT-Transformer on seasonal_full — bar registered BEFORE any number

Written 2026-09-22, while the seasonal_full FTT seeds were still training and
before any seasonal_full FTT number existed. Not to be edited afterwards;
results go in `RESULT.md`.

## Why a third surface

`forward` gave two readings that **disagree by ~4x** on the size of FTT's
marginal contribution over a 12-seed MLP:

| surface | in-band | non-LIRF |
|---|---:|---:|
| forward val (Sep-Oct) | **-0.86** | -1.47 |
| forward test (Nov-Dec) | **-3.63** | -4.22 |

Two surfaces, one strong and one weak, is not enough to justify a submission
slot -- and averaging them would hide the disagreement rather than resolve it.
`seasonal_full` (train Feb-Jun + Aug-Nov, val Dec, **test Jan + Jul 2025**) is
a genuinely independent third period, and it is the surface TabNet's LIRF
override was confirmed on, so the precedent and the tooling both exist.

## Protocol (copied from `tests_tabnet/confirm_seasonal_full.py`)

`seasonal_full` is in `config.SELECTION_FORBIDDEN_SPLITS`. **Nothing may be
selected on it.** So:

* Per-airport weights are solved on **forward_inner** and applied **frozen**.
  Both the 2-member (tree + MLP) and 3-member (FTT + tree + MLP) weights come
  from that solve; neither is re-fitted here.
* MLP seeds: **42-47 (six)**, the seeds present as registered
  `mlp_antigravity__seasonal_full` runs. The forward_inner solve is re-run at
  the same six so the seed count matches between solve and apply. (The forward
  ablation used twelve, which is all that exists on `forward`.)
* Tree side: registered `airport_operator_blend__seasonal_full` seed 42.
* FTT: seeds 42/43/44 trained on seasonal_full, early-stopped on Dec val.

**Dec val is reported but is NOT a bar.** It is the surface FTT's own early
stopping selected on, so a gain there is partly self-confirming. Jan + Jul test
is the honest read.

## The bar

**CONFIRM** requires, on seasonal_full **TEST** (Jan + Jul):

1. in-band (y <= 7200) gain **>= 1.0 s**, and
2. non-LIRF gain **>= 1.0 s**, and
3. both in the same direction as forward test (i.e. FTT helps).

Rationale for 1.0 s: the two forward surfaces bracket -0.86 to -3.63 in-band.
A third surface at or above 1.0 s means two of three periods show a gain
materially above noise on 400k+ rows; a third surface below it means the
forward-test reading was the outlier and the effect is period-specific.

**Sanity check, not a bar:** FTT's solved LIRF weight should remain ~0. It has
been exactly 0.000 in all three solves so far; if the six-seed solve moves it
off zero, distrust the solve rather than reporting it.

**In-band is the primary metric, not pooled.** Pooled RMSE on this build is
decided by a handful of date-artifact rows (~3 rows carry ~52 % of pooled
forward-test squared error), and `seasonal_full`'s January is known to contain
20-40 rows with missing AOBT_3 and implausible taxi-out up to ~24 h
(`tests_weather_metar` Part 2). Pooled will be reported for completeness and
must not be used to decide.

## Disposition rules, fixed in advance

* **CONFIRM** -> the effect is real across three periods. The next step is a
  `full`-split refit and a registered arm, and the submission decision goes to
  the user with the three-surface numbers attached.
* **NOT CONFIRMED** -> the forward-test reading stays a single-surface lead.
  **Nothing is built and no slot is spent.** The write-up keeps the full
  convergence history, which is valuable regardless.
* **No re-running this check at other settings.** One surface, one bar, one
  answer. Re-running a failed confirmation at new settings is how the
  `overtuned/` ladder happened.
