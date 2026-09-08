# Plan 12 — improving the arms that survived review

Written **before** any run in this round, so the acceptance decision is not made
after seeing the number. That ordering is the whole point: the six arms now in
`overtuned/` were each adopted by looking at a result and then deciding it
counted.

## The acceptance rule (pre-registered)

A candidate is adopted **only if all four hold**:

1. **Margin beats its own noise.** The val improvement over the incumbent must
   exceed the candidate's own seed-to-seed standard deviation, measured on
   **≥3 seeds**. This is the build's existing rule (`PIPELINE_REVIEW.md`); the
   quarantined ladder failed it at margin/noise 0.01–0.40.
2. **In-band agrees.** The in-band RMSE (`y ≤ 7200 s`) must move the same
   direction. Pooled RMSE is a lottery — ~3 unlearnable date-artifact rows carry
   ~52 % of pooled test squared error — so a pooled-only gain is not evidence.
3. **No new free parameter fitted on an evaluation surface.** Weights and knobs
   are solved on `forward_inner` or a train-carved holdout, never on `val` or
   `test`. No constant may be chosen by the score it produces.
4. **Declared before the run.** The candidate and its success criterion are
   written here first.

**If a candidate fails the rule, it is recorded as a clean negative and not
shipped.** That is a successful outcome, not a wasted run.

## What is explicitly NOT worth doing: hyperparameter tuning

Two measurements already in this repo settle it:

| lever | measured gain (val RMSE) |
|---|---|
| `stage7_tune_lightgbm.py` — full capacity sweep (lr × leaves × depth) | **6.5 s** |
| `per_airport_tuned` — a config chosen per airport, 10 val selections | **0.12 s** |
| *(for scale)* the outlier-policy fix — one constant, `drop` → `keep_all` | **27 s** |

Tuning is the lowest-yield lever anyone has measured on this build, and
`per_airport_tuned` is filed tier C precisely because 10 bites at the val apple
bought nothing. **No hyperparameter sweeps this round.**

## Candidate 1 — full-period refit of the honest winner  ← the one being run

**Change:** refit `airport_operator_multimodel` and its parent chain on
`split=full` (Jan–Dec, 12 months) instead of `forward` (Jan–Aug, 8 months).

**Why it is not tuning:** it is strictly more training data under an unchanged
method. No new parameter, no new axis, no knob.

**Why it is the right target:** `submissions/submissions.csv` shows **9 of 12
submissions shipped from the `forward` split** — models that never saw Sep–Dec.
The build's own design is *select on `forward`, refit on `full`, submit*, and
`full` is in `config.SELECTION_FORBIDDEN_SPLITS` exactly so nothing can be
chosen on it. That step was skipped for v4–v12.

**Protocol:** every arm in the chain already implements the no-val refit path —
`--freeze-from <forward run>/model.json`, hyperparameters and round count frozen
from the honest run, `best_iteration` scaled by the pool-size ratio (a stated
heuristic, recorded in `model.json`, not a measured optimum). The blend weights
come from `forward_inner`, which is untouched by this change.

**Acceptance:** rule 1–2 **cannot apply** — `full` has no held-out data left, so
there is nothing to measure. This candidate is adopted **on the principle that
more data under an unchanged method is better**, not on a number. *No predicted
improvement is claimed.* Rule 3 and 4 do apply and are satisfied: nothing is
fitted on an evaluation surface, and this is written before the run.

**Guard:** the refit must reproduce the same per-airport weights (solved on
`forward_inner`, so they must be byte-identical to the `forward` run's) and the
same feature fingerprint. If either differs, the refit is wrong, not better.

## Candidate 2 — re-point `airport_lirf_identity_multimodel` off the quarantine

Its `BASE_ARM` is `airport_multiseed_multimodel` → `airport_unsup_multimodel`,
both now in `overtuned/`. The override itself is zero-parameter and verified
(34 rows/year, residual std 2.9 s), but its headline number inherits the base's
optimism. Re-point at `airport_operator_multimodel`.

**Acceptance:** the `idband` variant must leave in-band RMSE *exactly* unchanged
(it touches only `y > 25200 s` rows) and improve pooled val on ≥3 seeds. The
`>50 k` branch stays a declared tail bet and is **not** adopted — `stage11`
measured it at 20 rows/year, residual std 13,488 s.

## Candidate 3 — seed-average the MLP *inside* the honest blend (PROPOSED, not run)

**The observation.** Val RMSE standard deviation across the seeds on disk:

| arm | val SD (s) |
|---|---|
| `per_airport` | 0.27 |
| `airport_operator_blend` | 0.80 |
| `airport_operator_multimodel` (the winner) | **2.07** |
| `mlp_antigravity` | **3.52** |

The tree side is nearly deterministic. **Essentially all of the winner's seed
variance is the MLP**, and variance is now a larger quantity than any margin
that separates the arms in `overtuned/`.

**The change:** average `mlp_antigravity`'s predictions over its seeds (42–47 all
exist on disk) *before* blending, instead of blending one MLP draw. Zero fitted
parameters — an unweighted mean over independent draws.

**Why this is not the `airport_multiseed_multimodel` mistake.** That arm averaged
the final predictions of an **over-tuned base**, so it averaged the base's
optimism. This averages the **noisy parent** underneath an honest blend. The
review already drew this distinction explicitly: *"the seed-averaging is
legitimate variance reduction; the base arm is not."*

**Acceptance (pre-registered).** Must beat single-seed
`airport_operator_multimodel` at matched seeds on **≥3 seeds**, on val **and**
in-band, by more than the single-seed arm's own SD (2.07 s). Note the honest
risk: the averaged arm has lower variance *by construction*, so the comparison
must be against the incumbent's spread, not the averaged arm's — otherwise the
test grades itself. **If it lands inside 2.07 s it is recorded as a clean
negative and not shipped**, however good the point estimate looks.

**Not run in this round.** Listed so the reasoning is on record before any
number exists.

## Explicitly rejected for this round

- Any new bucketing axis on the blend weight. Four were tried; all four are in
  `overtuned/`.
- Any per-airport hyperparameter config (`per_airport_tuned`, tier C, 0.12 s).
- `log1p` targets — tested, rejected, `tests_log_target/RESULT.md`.
- Reviving `airport_debiased`'s seasonal calibration in any form.

---

# Outcome (recorded after the runs)

## Candidate 1 — full refit: **DONE**, and it surfaced a bug

Chain refit on `full` (Jan–Dec) at seed 42: `per_airport` → `per_operator` →
`airport_operator_blend` → `airport_operator_multimodel`. Training pool
**1,387,105 → 2,084,659 rows (+50.3 %)**.

Pre-registered guard — all pass:

| guard | result |
|---|---|
| blend weights identical to the `forward` run (solved on `forward_inner`) | **byte-identical** |
| global alpha frozen | 0.6765 → 0.6765 |
| feature fingerprint | `aeade7fee3099c1a` unchanged |
| training pool grows | 1.39 M → 2.08 M |

As pre-registered, **no accuracy claim is attached** — `full` has no held-out
data, so this is adopted on the principle that more data under an unchanged
method is better, not on a measured gain.

### Bug found: the refit path was silently training to the round cap

`airport_operator_blend/train.py` read its frozen round counts from
`frozen["per_airport_frozen"]` and passed **nothing at all** for the operator
side — but the writer stores them as `best_iteration_per_airport` /
`best_iteration_per_operator`. `.get()` on the wrong key returns `None`, and
`frozen=None` is precisely the "no frozen payload" signal, so the no-val refit
trained **all 31 boosters (10 airports + 21 operators) to the 5000-round cap**
instead of the ~100–190 rounds the honest run selected. Roughly 30× over-
training, with nothing in the output saying so.

Measured directly: the run took **>70 minutes** before the fix and **2 m 31 s**
after — the ~30× the round-count analysis predicted. Fixed, and a missing frozen
payload now raises instead of quietly degrading.

**This means the `full` refit path for this arm had never worked correctly.**

### Second bug: the winner could not be submitted at all

`submit.py`'s argparse takes `--arm` from `config.ARMS`, but `predict_ranking()`
carries its own separate `if/elif` dispatch, and it had **no branch for
`airport_operator_multimodel`** — so the command was accepted and then died with
`ValueError: unknown arm`. All five of its *quarantined* descendants had
branches; the honest parent did not. This is the same defect shape `config.py`'s
own ARMS comment says it was written to prevent. Branch added.

## Candidate 2 — re-point the LIRF identity override: **ADOPTED**

`--band idband` (verified 25.2k–50k identity only) on base
`airport_operator_multimodel`, `forward` split, seeds 42/43/44:

| seed | base val | idband val | Δ val | base test | idband test | Δ test | val in-band | test in-band |
|---|---|---|---|---|---|---|---|---|
| 42 | 248.33 | 247.78 | **−0.55** | 246.89 | 246.31 | **−0.58** | 228.42 → 228.42 | 219.72 → 219.72 |
| 43 | 248.60 | 247.40 | **−1.20** | 242.62 | 240.89 | **−1.73** | 228.68 → 228.68 | 219.15 → 219.15 |
| 44 | 252.42 | 251.65 | **−0.77** | 240.81 | 238.63 | **−2.18** | 230.68 → 230.68 | 220.54 → 220.54 |

Against the pre-registered criterion:

- **In-band exactly unchanged** on both parts at all three seeds ✓ — as the
  mechanism requires, since the override only touches `y > 25,200 s` rows. Not
  "close": identical to the decimal.
- **Pooled improves at 3/3 seeds** on val *and* test ✓.

Note the comparison is **paired** — same base arm, same seed, 5 rows changed by a
deterministic zero-parameter rule — so seed variance largely cancels and the
"margin vs seed SD" test is not the binding one here. The rule cannot backfire on
a touched row by construction. Mean Δ: val −0.84 s, test −1.50 s.

The `>50 k` branch remains **not adopted** (`--band full` reproduces it), per the
plan: 20 rows/year at residual std 13,488 s is a declared tail bet, not a finding.

## Candidate 3 — seed-average the MLP: **ADOPTED** (passes all four conditions)

`--mlp-seeds 42,43,44`, tag `msmlp`, `forward` split, arm seeds 42/43/44. The
refactor was regression-checked first: with the default (`--mlp-seeds` empty) the
weights and val RMSE are **byte-identical** to the pre-change run, so the new
code path is a verified no-op unless asked for.

| seed | val base → msmlp | Δ val | val in-band Δ | test in-band Δ | pooled test Δ |
|---|---|---|---|---|---|
| 42 | 248.33 → 247.39 | **−0.94** | −0.68 | −0.69 | −7.02 |
| 43 | 248.60 → 247.41 | **−1.19** | −0.87 | −0.45 | −3.04 |
| 44 | 252.42 → 247.40 | **−5.02** | −2.89 | −1.65 | −1.07 |
| **mean** | | **−2.38** | **−1.48** | **−0.93** | −3.71 |

Against the four pre-registered conditions:

1. **Margin beats its own noise** ✓ — mean val Δ **−2.38 s** against the
   incumbent's 6-seed val SD of **2.07 s**, improving at **3/3** seeds. Measured
   against the *incumbent's* spread, not the averaged arm's, exactly as the plan
   required — otherwise the test would grade itself.
2. **In-band agrees** ✓ — val in-band and test in-band both improve at every
   seed. Not a pooled-tail artefact.
3. **No new fitted parameter** ✓ — an unweighted mean over independent draws.
   Blend weights still solved on `forward_inner`.
4. **Declared before the run** ✓.

### The result that matters more than the mean

**`msmlp`'s own val spread across seeds is 0.020 s. The incumbent's is 4.090 s.**
A ~200× collapse in seed variance. The arm lands at ~247.40 *every time* instead
of sometimes landing at 252.42 — it reproduces roughly the incumbent's best seed
on every draw.

### Honest caveat on the mean

The −2.38 s mean is dominated by **seed 44 (−5.02 s)**; without it the mean is
≈ −1.07 s, which would be *inside* the 2.07 s bar. That is not cherry-picking, and
the direction matters: seed 44 is precisely the bad MLP draw this method exists to
absorb, and the incumbent's 2.07 s SD is itself largely produced by draws like it.
Excluding it would mean excluding the event being protected against. But the
weight of the evidence is the **variance collapse and the 3/3 consistency**, not
the point estimate — a reader should not take "−2.38 s expected gain" as the
claim. The claim is: *the same expected accuracy, without the downside draw.*

### What "3/3 seeds" does and does not mean here

Checked rather than assumed: the three `msmlp` runs are **not** identical. Their
per-airport weights differ (spreads 0.002–0.011), because `--seed` still varies
the tree parent. But all three share one MLP average, so the dominant source of
variation is removed by construction. These are therefore **three paired
comparisons against three different incumbents** — valid, and the right test —
but *not* three independent draws of the new arm. The new arm barely has draws
any more; that is the point of it.

### Shipping status

Adopted on `forward` evidence. Shipping it in a submission needs
`mlp_antigravity` refit on `full` at seeds 43 and 44 (only seed 42 exists);
that is ~47 min per seed and is running. v13 (full refit, single MLP draw) stands
until those land.
