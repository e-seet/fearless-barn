# PRC Data Challenge 2026 — taxi-out time prediction

> **Current summary: [`WRITEUP.md`](WRITEUP.md)** — best board score v32 = 287.9898 (2026-09-25).


Predict how long a departing aircraft spends between leaving the stand and
lifting off, at ten large European airports.

> **SUPERSEDED — this file is a mid-build snapshot, not the current state.**
> The build finished at **v25, RMSE 295.21 on the live leaderboard**. For the
> current, trustworthy picture read **[`PIPELINE_REVIEW.md`](PIPELINE_REVIEW.md)**
> (which arms to trust, which are quarantined) and
> **[`plans/13_live_board_next_steps.md`](plans/13_live_board_next_steps.md)**
> (the real submission history). In particular, §7 below names
> `airport_unsup_multimodel` as "reigning champion" — that arm was later found
> to be over-tuned (ranking inside its own seed noise) and is now quarantined
> in `overtuned/`; it was never the shipped model. This file is kept for its
> still-accurate glossary (§2) and build layout (§3), not for §4/§7's numbers.

**Status: `STATUS: in progress`** — not registered in `INDEX_MODE2.md`. See
[§9 Open items](#9-open-items) for exactly what is and is not settled.

---

## 1. The task in one paragraph

For each departure in a scored set of **215,876 rows** (January and July 2026),
predict `TAXITIME_SEC_mvt` — seconds from off-block to takeoff. Scored by
**pooled RMSE** over all scored rows, lower is better. Training data is the whole
of 2025 at the same ten airports: **4,167,797 movement records**, of which
**2,085,047 are departures**.

The submission is a parquet of `(MVT_ID_mvt, TAXITIME_SEC_mvt)` named
`<team>_v<N>.parquet`. `submit.py` builds and validates it but **never uploads** —
sending a submission stays a deliberate human action.

---

## 2. Acronyms and columns you need to read anything here

### Aviation / dataset

| Term | Meaning |
|---|---|
| **PRC** | Performance Review Commission (EUROCONTROL), who run the challenge |
| **MVT** | *Movement* — one recorded aircraft event. Columns from this table end `_mvt` |
| **FLT** | *Flight* — the flight-plan record joined onto a movement. Columns end `_flt` |
| **DEP / ARR** | `PHASE_mvt`: this movement is a departure or an arrival |
| **ADEP / ADES** | Aerodrome of **DEP**arture / **DES**tination, as 4-letter ICAO codes |
| **AOBT** | **A**ctual **O**ff-**B**lock **T**ime — measured moment the aircraft left the stand (`AOBT_3_flt`, source 3) |
| **EOBT** | **E**stimated **O**ff-**B**lock **T**ime — the *planned* off-block from the flight plan (`EOBT_1_flt`) |
| **IOBT / LOBT** | **I**nitial / **L**ast off-block time — earlier and latest planned values |
| **ARVT** | Arrival time (`ARVT_1_flt`, `ARVT_3_flt`) |
| **WK_TBL_CAT** | Wake turbulence category (Light / Medium / Heavy / Super) — a proxy for aircraft size |
| **STAND / RUNWAY** | The parking stand the aircraft left, and the runway it used |
| **taxi-out** | The target: off-block → takeoff |
| **taxi-in** | The arrival equivalent: landing → on-block. **Not blanked in the scored set** — see §5 |

### The ten airports (ICAO)

`EDDF` Frankfurt · `EDDM` Munich · `EGLL` London Heathrow · `EHAM` Amsterdam
Schiphol · `LEBL` Barcelona · `LEMD` Madrid · `LFPG` Paris CDG · `LIRF` Rome
Fiumicino · `LSZH` Zurich · `LTFM` Istanbul

`LIRF` matters far more than its 5.1% share of the scored set suggests — §5.

### Modelling / this repo

| Term | Meaning |
|---|---|
| **RMSE** | Root mean squared error, in seconds. The competition metric |
| **in-band** | Rows with `y <= 7200 s`. 99.97% of rows; the "ordinary" traffic |
| **mid-tail** | `7200 < y < 80000 s`. Real, learnable heavy tail |
| **artifact band** | `y >= 80000 s`. Date-error rows — see §5. **Unpredictable** |
| **decidable RMSE** | RMSE over in-band + mid-tail, i.e. **excluding the artifact band**. How much of a gap is method rather than luck |
| **arm** | One candidate pipeline (`pipeline_*/`) |
| **protocol** | A `split-featureset-outlierpolicy` combination; part of every artifact path |
| **val-fitted** | An arm whose free parameters were fitted on the validation labels, so its val score is in-sample |
| **LB-weighted** | RMSE reweighted to the scored set's airport mix. An approximation — it corrects airport but not month |

---

## 3. How the build is organised

```
solution/
  config.py             every path, split, policy and constant. Read this first.
  common/               shared library: io, features, dataset, metrics, evaluate,
                        predict (one loader for every arm), bootstrap (ml_common)
  pipeline_<arm>/       one folder per candidate: train.py + evaluate.py
  stage1_audit.py       data audit + hard gates      -> artifacts/audit/
  stage2_split.py       builds a split, re-audits it -> artifacts/splits/
  stage4_controls.py    label-permutation, overfit-one-batch, leak probe
  stage5_ablations.py   absolute-time-feature confound check
  stage6_clip_sweep.py  prediction cap + band decomposition
  stage7_tune_lightgbm.py   hyperparameter sweep (one arm, deliberately)
  stage8_triage.py      pooled vs decidable ranking  -> artifacts/triage/
  stage9_seed_average.py    what seed-averaging buys (built; see §9)
  compare.py            ranks arms on VAL, runs the gates, names the winner
  submit.py             builds + validates a submission. Does not upload
  run_all.py            orchestrator; run_all.ipynb mirrors it
```

**Split protocols** (`config.SPLITS`):

| name | train | val | test | purpose |
|---|---|---|---|---|
| `forward` | 2025-01..08 | 09,10 | 11,12 | **the honest split.** Strictly time-ordered, mirrors train-2025 → score-2026 |
| `seasonal` | 9 months | 12 | 01, 07 | composition probe. **Anti-causal** — its Jan test rows precede the training pool, so `compare.py` refuses to crown a winner on it |
| `forward_inner` | 2025-01..06 | 07,08 | — | inner surface for fitting stacking weights without touching val |
| `full` | all 12 | — | — | refit-only, for the submission. Nothing may be *selected* on it |

---

## 4. What was tested

**For the current, complete inventory across both agents (Claude + Antigravity)
and every arm/config, see [`result.md`](result.md)** — it is regenerated from
`results.csv` by `harvest_results.py` and supersedes this table. The table below
is the state at the third round and is kept for narrative continuity.

Thirteen arms across three rounds. All on `forward / with_aobt / keep_all`, seed 42.

| arm | val RMSE | test RMSE | test in-band | what it tests |
|---|---|---|---|---|
| **cross_blend (inner weights)** | **250.76** | **243.49** | **222.40** | per-airport blend of the two families, weights fitted on `forward_inner` |
| cross_blend (val weights) | 250.34 * | 246.38 | 224.27 | same, weights fitted on val (*in-sample) |
| ensemble_antigravity | 254.40 * | 270.36 | 251.62 | NNLS blend of the other arms (third-party) |
| mlp_antigravity | 255.20 | 275.14 | 259.71 | neural tabular model (third-party) |
| per_airport_stack | 281.93 | 286.44 | 223.58 | per-airport blended with pooled |
| per_airport | 282.66 | 287.36 | 224.68 | **one LightGBM per airport** |
| lightgbm_unsup | 301.00 | 313.74 | 231.50 | + SVD stand/runway embeddings, k-means day regimes |
| lightgbm_recency | 302.13 | 311.61 | 233.06 | + exponential recency weighting |
| lightgbm | 302.45 | 310.63 | 231.14 | the pooled GBM baseline |
| xgboost_antigravity | 322.83 | 346.31 | 245.29 | GBM diversity (third-party) |
| ridge | 391.41 | 407.35 | 330.72 | linear floor arm |
| naive_offblock | 481.97 | 484.76 | 364.75 | trivial floor: `MVT_TIME − AOBT_3` |
| *trivial mean baseline* | *502.21* | *523.38* | — | predict the training mean |
| lightgbm_arrfeat | 300.45 | 307.84 | 230.58 | + arrival taxi-in / runway-queue features (now run) |

**Since this table** (see `result.md` for numbers): `multimodel_antigravity`
(Antigravity's per-airport blend of `per_airport` + `mlp_antigravity`, weights on
`forward_inner`) was swept 42/43/44 and **ties `cross_blend (inner)`** — 3-seed
mean test ~248 s; it is now the `compare.py` val-selected winner over a 17-arm
field. `per_airport_tuned` ran: per-airport HP tuning beats a single global config
by only 0.12 s (tune once globally). Also added: `per_airport_arrfeat`,
`feature_pruned_antigravity`, `delay_twostage_antigravity`, and `_tuned` tag
variants of `cross_blend` / `per_airport_stack` (seed 42 only).

Also swept, not shown: five outlier policies, ten prediction caps, six LightGBM
configurations, three inner-holdout lengths, two featuresets (`with_aobt` /
`no_aobt`), and seeds 42/43/44 for the arms that mattered.

---

## 5. What actually matters, and what doesn't

### 5.1 The metric is decided by a handful of rows — this is the key fact

Exactly **15 departures in 2,085,047** have a target in [80,000, 95,000) s. Every
one is 86,400 s (one day) plus a plausible taxi time, and for every one the
off-block timestamp is one calendar day before takeoff. Thirteen are at `LIRF`.

**Three of them land in `forward/test`, where they carry 52% of pooled squared
error on their own.** They are not predictable — nothing in the feature set
identifies them beforehand.

Consequences, each measured:

- **The pooled test ranking is not a method ranking.** Pooled says
  `ensemble > mlp > per_airport`; with the artifact band excluded it says
  `per_airport_stack > per_airport > lightgbm > ensemble`. The val-selected
  winner is **6th** on the rows the metric can decide.
- **Always read `test in-band` / decidable beside pooled.** `stage8_triage.py`
  prints both and says whether they agree. Where they disagree, the pooled gap is
  a coin flip on date errors.
- **It reversed the outlier policy.** Every winsorising bound cuts into a
  *learnable* tail (92 val rows carrying 16% of val error) to reach 15
  unlearnable ones. `keep_all` won, and every arm improved 40–100 s.

### 5.2 What matters

- **Segmenting by airport.** One LightGBM per airport beat the pooled model by
  **19.8 s val / 23.3 s test**, with in-band improving too, at ~68× its own seed
  range. Mechanism: per-airport round counts run **72–195** where the pooled model
  compromised at 259 — each airport gets its own stopping point, which carrying
  `ADEP` as a categorical does not give you.
- **Blending the two model families.** They fail in complementary places —
  per-airport owns the 327,907 in-band rows, the neural arms own the 61 tail rows.
  A per-airport-weighted blend captures ~90% of the oracle bound: **LIRF gets
  w=0.000 (fully neural), the other nine get 0.64–0.95 (per-airport).**
- **The tail policy, not the hyperparameters.** Fixing the outlier policy and
  prediction cap bought 27 s; tuning LightGBM bought 6.5 s.

### 5.3 What does *not* matter (measured, not assumed)

- **`AOBT_3_flt` is worth ~1.6% of val RMSE** (4–5 s), not the ~160 s its
  standalone score suggests. The floor arm is *better* without it, because
  `MVT_TIME − AOBT_3` faithfully reproduces the same date error the target carries.
- **Recency weighting: noise.** Best half-life beat its own unweighted control by
  0.31 s against a 1.67 s seed spread. *Caveat:* `forward`'s val is 1–2 months
  ahead while the real task is 13 months, so this cannot test the horizon that matters.
- **Unsupervised features: no effect.** 1.1% of gain, ranks 33–60 of 67, val −1.4 s
  but test **+3.1 s worse**.

### 5.4 Seed stability — read every number with its range

| arm | val range | test range |
|---|---|---|
| per_airport | 0.29 | 0.98 |
| lightgbm | 1.67 | 2.29 |
| cross_blend (inner) | 2.35 † | 8.82 † |
| ensemble_antigravity | 6.54 | 35.59 |
| mlp_antigravity | 7.78 | **65.82** |

† two seeds so far; the third is running. The neural arms' instability is the tail
again: a single artifact row moves their score by tens of seconds.

---

## 6. How the result is produced

```bash
python -m pip install -r requirements.txt
python ../../tools/preflight.py .  --heavy-import lightgbm sklearn xgboost torch
python run_all.py --stage audit  --force          # data gates
python run_all.py --stage split  --all-protocols --force
python run_all.py --stage all --split forward --featureset with_aobt \
                  --outlier-policy keep_all --seed 42 --force
python stage8_triage.py --split forward --featureset with_aobt \
                        --outlier-policy keep_all --seed 42 --force
```

`run_all.ipynb` is the same sequence as one notebook. **It was reconstructed from
the run logs after the fact, not appended as each variant ran** — the notebook's
own header says so. `logs/*.log` is the primary record.

### Gates that will stop you

`compare.py` refuses a comparison when arms disagree on the training pool, on the
`features.py` revision they were built from, or when any arm hit its epoch/round
cap without early stopping. Each of those exists because it failed silently once —
see `MISTAKES.md` #79, #80, #82. The pool gate reads `train_pool_ids_sha` from
each arm's `model.json`; three Antigravity arms wrote it under a different key and
were silently skipped until the field was complete (`MISTAKES.md` #85).

---

## 7. Best result, against the competition requirement

The competition scores **pooled RMSE on 215,876 rows of Jan + Jul 2026**.

**Reigning Champion: `airport_unsup_multimodel`** — hierarchical multi-stage shrinkage gating over unsupervised congestion regimes (K-Means K=6 on instantaneous traffic density vectors) combined with 4-way physical operational regimes (`Heavy+` wake-turbulence class vs `NoRecord` vs `Regional` vs `Standard` commercial passenger traffic) blending `airport_operator_blend` (specialized LightGBM tree blend) and `mlp_antigravity` (deep tabular neural network with entity embeddings) with Stein / Empirical-Bayes shrinkage ($\lambda=1500$) solved out-of-sample on `forward_inner`:

- **Test RMSE:** 244.35s (seed 42) / 241.67s (seed 43) / 239.88s (seed 44) — **3-seed mean 241.96s** (breaks the 240s barrier on Seed 44: 239.88s!).
- **Leaderboard-weighted Test RMSE:** 232.16s (42) / 230.06s (43) / 228.63s (44) — **3-seed mean 230.28s**.
- **Validation RMSE:** 247.53s (42) / 248.25s (43) / 251.28s (44) — **3-seed mean 249.02s** (selected **#1 unanimously across ALL 3 seeds** by `compare.py`).
- **Hard Gate Verification:** Passes all `compare.py` gates across all 3 seeds: training pool equality ($n=1,387,105$, hash `d16701419a0d1c8703890a73c854ce9c`), feature-code agreement (`aeade7fee3099c1a`), and 100% out-of-sample weight solving (`weights_are_val_fitted: false`).

Replaces prior champions `airport_wtc_rec_multimodel` (mean 242.28s), `airport_wtc_multimodel` (mean 242.97s), `airport_operator_multimodel` (mean 243.30s), and `multimodel_antigravity` (mean 249.94s).

### Submissions on disk (`submissions/`)

| file | arm | why |
|---|---|---|
| `..._v1.parquet` | ensemble_antigravity | the val-selected winner under the original protocol |
| `..._v2.parquet` | lightgbm (tuned) | the low-variance pooled alternative |
| `..._v3.parquet` | per_airport_stack | best bulk accuracy at the time |
| `..._v4.parquet` | airport_unsup_multimodel | **reigning champion**; breaks 240s barrier on test (239.87s), unanimous #1 on all 3 seeds |

Every file is named `PLACEHOLDER_TEAM_*` — **rename to your registered team name before uploading or the ranking system rejects it.** Each has a `.meta.json` recording every contributing run, its feature fingerprint and round count, so a submission can always be traced back to the models that produced it.

---

## 8. What to read, and in what order

| file | what it holds | read it when |
|---|---|---|
| **`RETROSPECTIVE.md`** | **The most important document.** Every result, every negative result, the audit findings, the advisor log, and §6's mistakes table | Before trusting any number here |
| `RETROSPECTIVE.md` §4a | The date-artifact finding and why it reframes the metric | First — it changes how every other number reads |
| `RETROSPECTIVE.md` §4b | What `AOBT_3` is actually worth | Deciding whether the feature matters |
| `RETROSPECTIVE.md` §4c | The 3-seed sweep and winner stability | Before believing any ranking |
| `RETROSPECTIVE.md` §4f/4g | The improvement round and the cross-family blend | Understanding the current best |
| `PLAN.md` | The original plan, plus §8b **amendments made during execution** | Understanding why the protocol changed |
| `../../MISTAKES.md` #79–#83 | Five defects found in this build, with fixes and scope | Before reusing any of this code |
| `artifacts/triage/` | Pooled vs decidable rankings, per part | Reading any comparison |
| `artifacts/clip/policy_cap_decision.json` | Why `keep_all` + an 86,400 s cap | Questioning the outlier policy |
| `logs/*.log` | The primary execution record | When the notebook and reality disagree |

**On the retrospective specifically:** it documents negative results and
mis-predictions as carefully as successes — including three predictions the author
made in advance, two of which were wrong. That is deliberate. The wrong ones are
the most informative part, because both failed the same way: reasoning from
aggregate statistics to per-segment behaviour.

---

## 9. Open items

- **Cross-agent inventory + results snapshot:** [`result.md`](result.md) (+
  `results.csv`, `harvest_results.py`) — the current source of truth for what
  each `pipeline_*` folder does, who owns it, and every arm's numbers.
- **Seed sweep for `cross_blend`** — done (42/43/44). **`multimodel_antigravity`
  seed sweep** — done (42/43/44); ties `cross_blend (inner)`.
- **`v4` submission** from the cross-family blend family — still pending a
  decision on which implementation (`cross_blend w_inner` vs
  `multimodel_antigravity`); they are a tie, see `result.md` §2.2 / §3a.
- **`lightgbm_arrfeat`** — run (val 300.45 / test 307.84 / in-band 230.58); a
  small loss vs `lightgbm`. `per_airport_arrfeat` also run (val 282.87).
- **`per_airport_tuned`** — run. Per-airport HP tuning beats one global config by
  0.12 s: tune once globally (`result.md` §4).
- **`compare.py` `collect()` has no `--tag`** — tagged arms (`cross_blend`
  entirely, the real `per_airport_stack`) are outside its adjudicated field.
  Adding a `--tag` argument would let the top of the field be compared directly.
- **Three Antigravity arms** wrote `train_pool_fingerprint` where the pool gate
  reads `train_pool_ids_sha` — fixed, arms re-run, numbers unchanged
  (`MISTAKES.md` #85).
- **Registration is blocked** — `python3 tools/check_docs.py` fails on four broken
  links to `kaggle_solutions/README.md`, a file that was never committed. It is
  **unrelated to this build** and was deliberately not invented to make a gate pass.
- **Untried levers:** per-airport hyperparameter tuning; within-airport ensembling
  (the Facial Keypoints lever); external weather (METAR) — de-icing is the obvious
  January driver and the scored set is 71% January, but competition rules on
  external data need checking first.
