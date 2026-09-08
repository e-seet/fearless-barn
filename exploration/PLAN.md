# PLAN — PRC Data Challenge 2026: taxi-out time prediction

**Status:** code complete, audit stages executed, design-gate advisor review
acted on in full, **training not yet run** (the run target is a separate RTX
3080 Ti box; see §8). Findings marked *corrected at the design gate* below were
wrong in the first draft — see `RETROSPECTIVE.md` §3 for the full disposition.

---

## 1. Problem statement

Predict `TAXITIME_SEC_mvt` — the interval between off-block and takeoff — for
**215,876 departures** at European airports in January and July 2026. Scored on
**pooled RMSE** against a blanked ground truth. Training data is every movement
at 10 airports across 2025 (4,167,797 rows; 2,085,047 of them departures).

Target is defined exactly as `MVT_TIME_UTC_mvt − BLOCK_TIME_UTC_mvt`
(verified: identity holds on 100.0000% of training DEP rows). Both terms of that
difference are blanked in the scored set, so the task is genuinely predictive.

## 2. What the audit found (before any modelling)

Run: `python3 stage1_audit.py`, `python3 stage1b_label_quality.py`.

| # | Finding | Consequence for the design |
|---|---|---|
| A | `AOBT_3_flt` (flown-trajectory off-block) survives into the scored set at **98.5%** coverage. `MVT_TIME − AOBT_3` alone scores **RMSE 384.9 s** on 2025 vs **546.4 s** for the global mean. | The proxy is legitimate but dominates. Every arm runs under two featuresets, `with_aobt` and `no_aobt`, so its contribution is measured rather than assumed. |
| B | July 2026 in `ranking.parquet` covers **only EDDF, EGLL, EHAM**; January covers all 10. The scored set is 71% Jan / 29% Jul. | Pooled RMSE on a locally-composed split is not a leaderboard estimate. Every evaluation additionally reports a **composition-reweighted RMSE** and a per-airport breakdown. A second `seasonal` split protocol reproduces the scored composition directly. |
| C | The two independent off-block sources **disagree substantially**: within 60 s on only 8.7–26.4% of rows, RMS disagreement **255–557 s** per airport. | **Corrected at the design gate — this is NOT a floor.** The disagreement is largely *systematic*, hence learnable: a per-`stand_runway` median correction fitted on Jan–Aug takes the raw proxy from 396.25 to **354.02** on Sep–Oct. `Var(proxy − label)` bounds nothing; only the component of the label unpredictable from available features does. Treat ~350 s as a soft reference, not a floor. The real bound is arithmetic — see §6. |
| D | **LIRF** is the worst offender: label p99 is **2.19×** the proxy p99, and it holds 480 of the 584 training rows above 2 h. | LIRF gets its own per-airport error line in every report. Its contribution to pooled RMSE is expected to dominate and must not be averaged away. |
| E | Target contains impossible values: 388 rows ≤ 0 s (min −12 s), 584 rows > 2 h, max **131,167 s (36 h)**. The worst **0.1%** of rows carry **42.9%** of total squared error. | Outlier policy is the single largest lever — and **filtering the tail out of training is the wrong move**. See §3a. **Evaluation sets are never filtered** under any policy — the leaderboard scores every row. |
| I | **82.4% of the training rows above 2 h have no flight record at all** — `P(y>7200 \| no record)` is **429×** `P(y>7200 \| record)`, and `E[y \| no record]` is 1,384 s vs 987 s. The LIRF × no-record cell (n=1,488) has mean **6,530 s**. | The extreme tail is **structured, not noise**, and the build already has a feature for it (`has_flight_record`). This is why the tail must stay in training. *Speculative:* a movement with no flight record may have a `BLOCK_TIME` from a different source, or be a towed/overnight aircraft — making `MVT_TIME − BLOCK_TIME` not a real taxi at all. Unconfirmed, and it changes nothing operationally: the leaderboard scores those rows regardless. |
| F | **LTFM drifts +20.2%** (median off-block→takeoff 1201 s → 1444 s) between 2025 and Jan 2026; EHAM +6.5%, EDDF +4.5%. Scored-share-weighted mean absolute drift: **43.2 s**. | A 2025-fitted model under-predicts LTFM in 2026 by ~4 min. This is the strongest argument for the `with_aobt` arm: the proxy absorbs the drift because it is measured in 2026. |
| G | Docs claim 11 airports; **10** exist in the data. ~1.5% of scored departures have **no flight record at all** (every `_flt` column null). | Every model needs a no-flight-record fallback path; `has_flight_record` is an explicit feature. |
| H | A hypothesis that the extreme values were **day-rollover artifacts was tested and rejected** — only 2.4% resolve to a plausible band after removing whole days, and `AOBT_3` does not corroborate the shift. | The extremes are genuine mismatches (or genuine 24 h ground events), not a correctable encoding bug. Recorded as a negative result. |

**Not observable, stated as such:** the scored set's *label* tail cannot be
inspected. Its *proxy* tail is clean (max 6,134 s), but the training proxy is
already far cleaner than the training label (0.0009% vs 0.0280% above 2 h), so
proxy cleanliness carries **no information** about the scored label tail.

## 3. Candidate pipelines (three arms, two featuresets each)

Root CLAUDE.md requires ≥2 candidates: one interpretable/classical floor, one
learned. This build has three.

| Arm | Folder | What it is | Why it is here |
|---|---|---|---|
| 1 | `pipeline_naive_offblock/` | **No learning.** `MVT_TIME − AOBT_3`, backing off through a stand+runway → stand → runway → airport → global median table. | The operational floor — the taxi-out the organisers could compute without a model. Cf. `subfolder63`, where the grid operator's own shipped forecast beat five deep models. |
| 2 | `pipeline_ridge/` | Ridge on numeric + one-hot (low-card) + **out-of-fold target-encoded** high-card categoricals. | The interpretable learned floor. Linear, inspectable coefficients, cannot memorise 1.4M rows. |
| 3 | `pipeline_lightgbm/` | LightGBM, native categorical splits, L2 objective, early stopping on val RMSE. | The arm expected to win: interaction-heavy target, 6,345 stand-runway levels. |

### 3a. Outlier policy — a swept axis, not a constant

The first draft of this plan filtered the training pool to `[60, 7200]` s and
called it costless. The design-gate review showed it is not, with a measurement:
a two-cell `ADEP × has_flight_record` group-mean model on forward/val scores
**445.85** fitted on the unfiltered pool, **448.30** winsorised, and **454.68**
filtered. Under L2 the optimum is the conditional mean of the *full*
distribution, so dropping the tail biases every affected cell low — and finding I
says that tail is exactly the subpopulation the features already identify.

Policy is therefore swept (`--outlier-policy`, part of the artifact key):
`keep_all`, `winsor_7200`, `winsor_p9999` (default), `winsor_21600`, `drop_7200`.
Impossible values (y ≤ 0) are dropped under every policy — corrupt, not tail.

The **prediction clip** is a separate quantity, decoupled from the training bound
and set at p99.99 = **10,712 s**, so a model that confidently predicts a long taxi
is not muzzled. Raising it from 7,200 lowers the arithmetic floor ~20 s per part.

**Featureset is a protocol axis, not a folder axis.** `with_aobt` / `no_aobt`
differ only in whether `AOBT_3_flt` and its derived features are present, so they
are keyed into the artifact path via `run_key(arm, protocol, seed)` rather than
duplicating three folders into six. Split protocol is keyed the same way:
`protocol = "<split>-<featureset>"`.

## 4. Split design

No 2026 labels exist, so the honest test set must come out of 2025.

- **`forward` (primary).** train `2025-01..08` → val `2025-09..10` → test
  `2025-11..12`. Strictly time-ordered, so val and test are in the future
  relative to train — the same shape as train-2025 → score-2026.
- **`seasonal` (secondary, composition probe ONLY — not a selection surface).**
  test = `2025-01` (all 10 airports) + `2025-07` (**EDDF/EGLL/EHAM only**),
  reproducing the scored composition to within 0.4 pp (70.4/29.6 vs 70.7/29.3);
  val = `2025-12`; train = the rest. **It is anti-causal**: its January test rows
  precede the *entire* training pool, so the model uses the future to predict the
  past, and July is seasonally interpolated between two training months. It is
  therefore optimistic on the temporal axis by construction. `compare.py` refuses
  to crown a winner on it (`selection_valid: false`).
- **`full` (refit only).** All 12 months, no val/test. Nothing may be *selected*
  on it; hyperparameters arrive frozen from a `forward` run via `--freeze-from`.
  This is what `submit.py` ships by default.

Both are written and re-audited by `stage2_split.py` itself (the stage that
creates a split audits its own output). Gates: pairwise ID disjointness, month
exclusivity, non-empty parts, airport filter actually applied. Both pass.

**Final submission model** is refit on all 12 months with hyperparameters frozen
from the honest split — standard, and not a leak, because nothing is selected on
2026 data. Which of the three built submission files to upload, and why:
[SUBMISSION.md](SUBMISSION.md).

## 5. The leakage surface, and how it is closed

The dominant risk is **training on a column the scored set does not have**.

1. `config.BLANKED_ON_RANKING_DEP` declares the blanked columns; `stage1_audit.py`
   gate **G2** asserts the declaration against the real file rather than trusting it.
2. `common/io.apply_ranking_contract()` blanks the same columns on training DEP
   rows **before** featurisation, so a feature physically cannot read them.
3. `common/io.assert_ranking_computable()` raises if a feature's source column
   list intersects the blanked set.
4. Congestion features are built **only** from movement times and from *arrival*
   taxi-in (not blanked). Counting movements in a window after the flight is
   legitimate here — `ranking.parquet` ships every movement in the scored months.
5. Target encoding for the ridge arm is **out-of-fold on train, full-train map on
   val/test/ranking** (`common/encoders.py`).
6. `compare.py` ranks on val only (`declare_selection_set` raises otherwise) and
   asserts training-pool set-equality across arms.

## 6. Metric and baselines

Headline: **RMSE in seconds**. Reported beside it, every time:

- **the arithmetic hard floor** — with predictions clipped at `cap`, every row
  above it contributes at least `(y − cap)²` regardless of the model, so
  `RMSE ≥ sqrt(mean((y − min(y,cap))²))`. This is a real bound, unlike finding C's
  retracted "label-noise floor". Measured per part at the 10,712 s cap:
  forward/val **211.9**, forward/test **258.1**, seasonal/val **232.7**,
  seasonal/test **284.9**. Note the ~46 s spread between forward/val and
  forward/test: a train/test gap must be read against the *per-part* floor, and
  `forward` and `seasonal` RMSEs are **not comparable numbers**;
- **in-band RMSE** (y ≤ 7200) as a low-variance companion — the pooled figure is
  driven by a few dozen rows, so a ranking that rests on it alone rests on luck;
- trivial floors — train-mean and train-median constants, fitted on train, scored
  on the evaluation part;
- the **naive `AOBT_3` proxy** as the real baseline to beat (arm 1);
- **composition-reweighted RMSE** at the measured leaderboard airport mix;
- per-airport and per-month RMSE;
- **share of total squared error carried by the worst 0.1% / 1% / 5% of rows** —
  because on this target that share is where the competition is won or lost.

## 7. Quality gates, in order

| Stage | Gate |
|---|---|
| 1 audit | G1 no train/rank ID overlap · G2 blanked-column set matches declaration · G3 submitting IDs == ranking DEP IDs · G4 target identity holds |
| 1b label quality | reports the label-noise floor; no pass/fail — it calibrates belief |
| 2 split | pairwise disjointness · month exclusivity · non-empty · airport filter applied |
| 4 controls | label permutation collapses to the measured floor, bounded on **both** sides, tolerances scaled to the metric (5 repeats) · overfit-one-batch < 0.20× the batch floor, unhandicapped · **target-encoding leak probe** on the *training* side, where an in-fold encoding leak is visible |
| 5 ablations | the arm ranking is **unchanged** when absolute-time features are dropped — otherwise the comparison is measuring extrapolation, not method |
| train | early stopping **fired** — hitting the round cap is a finding, not a result |
| compare | ranked on val · training-pool equality via each arm's **own recorded fingerprint** of the rows it consumed · refuses to select on `seasonal` |
| submit | column/row/ID-set/uniqueness/finiteness/sign validation before a file is written |

## 8. Execution plan (not yet run)

Target: **local RTX 3080 Ti box**, CPU-side LightGBM (this workload is
memory-bandwidth bound; see `requirements.txt`). Deployment scope: **competition
only** — no export artifact or latency budget, but the quality bar is unchanged.

```
python3 -m pip install -r Mode2Test/solution/requirements.txt
python3 tools/preflight.py Mode2Test/solution --heavy-import lightgbm sklearn

cd Mode2Test/solution
python3 run_all.py --stage audit --force              # 1 + 1b
python3 run_all.py --stage split --all-protocols --force
python3 run_all.py --sweep-policies --force           # B1: settle the outlier policy first
python3 run_all.py --all-protocols --force            # then the full matrix
python3 run_all.py --stage sweep --sweep-arms lightgbm,ridge --force

# submission: refit on 12 months with hyperparameters frozen from the honest run
python3 pipeline_lightgbm/train.py --split full --featureset with_aobt \
  --freeze-from artifacts/runs/lightgbm__forward-with_aobt-winsor_p9999__seed42/model.json
python3 submit.py --arm lightgbm --split full --featureset with_aobt \
  --team <registered-team-name> --version 1
```

**Note `--heavy-import lightgbm sklearn` is space-separated.** The repeated-flag
form silently discarded all but the last name until this build fixed it
(`MISTAKES.md` #76) — a preflight that reported `8/8 passed` having never checked
LightGBM at all.

Then, before anything is registered: the repeat-seed sweep (seeds 42/43/44), the
result triage, and the advisor closing gate.

## 8b. Amendments made during execution (run session, after the design gate)

The plan above is left as written; this section records where execution departed
from it and why. Each change is evidenced in an artifact, not just asserted.

| # | Change | Why | Evidence |
|---|---|---|---|
| A1 | **Runs are stamped with a feature-code fingerprint, and `compare.py` refuses a mixed-code comparison.** | S6 keyed the feature *cache* on the code hash but nothing keyed the *runs*, so the first six-arm comparison ranked a 131-feature old-code ridge and a 49-feature seed sweep against a 57-feature new-code LightGBM. Only an incidental `n_features` field made it visible. | `MISTAKES.md` #80; `feature_check` block in every `compare_summary__*.json` |
| A2 | **The training-pool gate is a real cross-arm check.** | B2 was recorded as fixed but only its recording half existed; `compare.py` still compared a list against itself. | `MISTAKES.md` #79; `pool_check.arm_shas` |
| A3 | **`DEFAULT_OUTLIER_POLICY`: `winsor_p9999` -> `keep_all`.** | Re-running the policy sweep at one code version reversed it. Every winsorising bound cuts into a *learnable* heavy tail to reach 15 unlearnable rows; `keep_all` halves mid-tail RMSE (15,014 -> 7,675 s) for an in-band cost about a quarter the size. Under a pooled squared-error metric the optimal prediction is the conditional mean, which every winsorising policy biases downward. | `artifacts/clip/policy_cap_decision.json` |
| A4 | **`PREDICTION_CLIP_MAX_SEC`: 10,712 -> 86,400 s.** | The cap was a hyperparameter treated as a fact. At 10,712 s it truncated predictions the model was right to make: val RMSE 337.82 / 326.27 / 305.29 / 302.45 at caps 7200 / 10712 / 21600 / none. **Open:** chosen at seed 42 only, decided by 37 val rows. | `stage6_clip_sweep.py`, `artifacts/clip/` |
| A5 | **New stage 6 (cap sweep + band decomposition), stage 7 (LightGBM tuning), stage 8 (ranking triage).** | The pooled metric decomposes into three regimes and the plan's gates could not see them. | `artifacts/clip/`, `artifacts/tuning/`, `artifacts/triage/` |
| A6 | **Two new outlier policies, `winsor_43200` / `winsor_80000`.** | The band analysis shows where the regimes actually divide; every pre-existing bound cut in the wrong place. | `config.OUTLIER_POLICIES` |
| A7 | **`config.VAL_FITTED_ARMS` + a caveat in `compare_summary`.** | `pipeline_ensemble_antigravity` solves `nnls(P, y_val)` for its blend weights, so its val score is in-sample for those parameters in a way early stopping is not. It is still ranked; the summary just refuses to let the crown pass unqualified. | `compare_summary__*.json` -> `val_fitted_caveat` |

### The finding that reframes the metric (audit finding C, corrected)

Exactly **15 DEP rows** in the 2,085,047-row training set have a target in
[80,000, 95,000) s. Every one is 86,400 s + a plausible taxi time, and for every
one the off-block timestamp is one calendar day before the takeoff timestamp
(`(BLOCK.date - MVT.date).days == -1`, 15/15). Thirteen are LIRF. Their
`y - 86400` residual quantiles are [668, 774, 941, 1170, 1521] against
[572, 719, 912, 1188, 1469] for the normal taxi distribution.

This refines rather than overturns audit finding 6 (the day-rollover hypothesis,
recorded as rejected). It was rejected across the whole `y > 7200` population,
where it is indeed false. In this narrow cluster the shape is exact. It is *not*
safely dismissible as an artifact either: the single cluster row that also has an
independent off-block source (`AOBT_3_flt`) shows the same 24 h gap rather than
contradicting it. Either way the rows are unpredictable and the organisers score
them, so they must be predicted -- but their consequences are:

* **Three of them land in `forward/test`, where they carry 52% of pooled squared
  error on their own**, and one lands in `forward/val` carrying 14%.
* **Every tail row in either part is the same cell** -- LIRF, no flight record,
  no `AOBT_3` -- whose conditional distribution runs from ~1,000 s to ~87,600 s.
* **The arm ranking on test is decided by them.** Pooled: ensemble > mlp >
  lightgbm. With the artifact band excluded: **lightgbm > ensemble > mlp**. On the
  other 328,006 rows LightGBM is the better model. On val the two orderings agree,
  so the val-selected winner is a valid selection -- but its margin is not a
  bulk-accuracy margin, and `stage8_triage.py` reports both rankings for exactly
  this reason.
* **The B3 hard floor is now vacuous.** With the cap at 86,400 s almost no row
  exceeds it, so `hard_floor_rmse` is ~0 on every part. The band decomposition
  replaces it as the instrument for reading a suspiciously good number.

## 9. Open levers, deliberately not taken in v1

- **External weather (METAR).** De-icing is the obvious driver of January taxi-out
  at European airports and no weather column is shipped. Competition rules on
  external data need checking before this is worth building.
- **Airport layout / stand coordinates.** Taxi distance is the physical driver;
  only categorical stand and runway IDs are given. `stand_runway` is a proxy.
- **Recency weighting or per-airport bias correction** for finding F's drift.
- **Quantile / Huber objective**, if triage shows the tail is hurting more than
  the outlier filter can fix.
