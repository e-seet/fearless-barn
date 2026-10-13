# Cross-check against a second external write-up (2026-09-19)

The user pasted a strategy/analysis document for this competition (taxi-out
time prediction) — source and authorship unknown, unlike the first
cross-check (`EXTERNAL_PLAN_CROSSCHECK.md`), which was explicitly attributed.
This file maps each of its claims to this pipeline's actual evidence.
Section numbers follow the source document's own three-part structure.
Verdict is one of: **Done**, **Done — closed**, **Partial**, **Not done**,
**Contradicted**, **Unverifiable**.

---

## 1. Where 66%+ of the error lives

### 1a. Rome (LIRF) ~27% of total error; BLOCK_TIME defaults to SCHED_TIME
in ~15.7% of flights; remote-apron stands (700s/800s/900s) have higher
default rates than contact gates (100s–400s)

**Contradicted on the headline number.** `DATA_QUALITY_AUDIT.md:108-109`:
`BLOCK_TIME_UTC_mvt == SCHED_TIME_UTC_mvt` for 8,123 rows **pooled across
all 10 airports — 0.39%**, already characterized and organizer-confirmed
(`conversation.md`), not a fresh discovery. No per-airport breakdown exists,
but even the maximally charitable read — crediting *all* 8,123 default rows
to LIRF alone — is 8,123/160,704 ≈ **5.1%**, still ~3× short of 15.7%. The
remote-apron-stand claim was not separately checked (no artifact ties stand
number range to default-rate) — **unverifiable**, not confirmed.

### 1b. Paris CDG (LFPG) ~24% of error; runway-crossing holding between
north/south terminals; cargo (FedEx hub, stands I48/CFE) apron-holding

**Partial / unverifiable.** A runway-configuration diagnostic exists
(`tests_runway_config/`) and found the ceiling from knowing "which runway
config is live" is **small — pooled 0.38s, LFPG 0.52s** — not written up as
a RESULT.md because it didn't clear the bar for one. Stands `I48` and `CFE`
are real LFPG parking positions (`artifacts/external/osm/LFPG.json:2808,4747`,
OSM ref data) — the identifiers aren't invented — but no test in this
pipeline has measured cargo-stand apron-holding specifically. Real-looking
detail, unconfirmed claim.

### 1c. London Heathrow (EGLL) ~15% of error; Terminal 5 (5xx stands, BA
hub) has different pushback dynamics vs. T2/T3/T4

**Not done — no supporting evidence, but a real reason to expect it's
already absorbed.** No mention of T5, 5xx, or terminal-specific pushback
analysis anywhere in this project's docs. `STAND_mvt` is fed to every model
as a raw categorical (`common/features.py:284,288`), so any real T5-vs-other
pushback difference is already implicitly available to the trees/embeddings
— the same "the model already has this information in some form" pattern
that closed most comparable levers this project has tested (§6, §9 below).

**On the ranking itself:** the pasted ordering (LIRF > LFPG > EGLL) doesn't
match this pipeline's measured per-airport RMSE. `TAXI_OUT_DRIVERS.md:71-78`
has EGLL at **1364.2** (n=239,546) — higher than LIRF (1194.6, n=160,704)
and LFPG (1019.8, n=239,487). EGLL likely dominates squared-error
contribution, not LIRF as claimed.

---

## 2. High-ROI Feature Engineering

### 2a. Fix the stand-prefix regex bug (`r"^([A-Za-z]+)"` → `r"^([A-Za-z]|\d)"`)

**Contradicted — describes code this pipeline doesn't have.** No
letter-prefix regex exists anywhere in `common/features.py`; `STAND_mvt` is
used directly as a raw categorical string (line 284/288), never passed
through a pier-extraction regex. There is no "100% of stands at LIRF/EGLL/
EDDM/LEBL turn to Unknown" bug to fix here.

### 2b. Preceding wake-turbulence queue (`traffic_runway_heavy_prev_30m`,
"+0.22 correlation with taxi duration")

**Done — closed.** Already tested: `tests_wtc_sequencing/RESULT.md`. A real,
permutation-verified, uniformly-signed effect at all 10 airports
(rho(prev_heavier, **residual**) = +0.0299, collapsing to −0.0007 under
permutation control — not directly comparable to the pasted +0.22, which is
correlation against raw duration, not residual). Raw group means show a real
gap (+22.44s vs +2.47s residual). But the pooled in-sample OLS ceiling is
**−0.134s**, an order of magnitude under this project's ~5s board-visibility
floor — the existing feature set (own WTC, runway congestion counts,
stand_runway) already absorbs almost all of it. Closed, not a live lever.

### 2c. True aeronautical runway wind (heading-derived headwind/crosswind/
tailwind/gusts)

**Done — closed.** `tests_wind_runway/summary.csv`: ceiling reductions are
small everywhere — pooled ~0.27–0.39s across splits, EGLL as low as 0.06s,
best case EDDM ~4.8s (forward split only). Nowhere near "severely restrict
runway throughput."

### 2d. Runway Demand Ratio (departures on assigned runway ±30m / total
airport departures ±30m)

**Partial.** Not built at this exact ratio-of-ratios formulation, but this
is a close relative of the congestion-count family already built and
repeatedly tested (`build_congestion`, `dep_prev_Xm`, `same_rwy_prev_Xm`,
`dep_surge_ratio_15_60` — see `EXTERNAL_PLAN_CROSSCHECK.md` §3–4/§5). Given
how consistently that whole family has closed near-zero once the underlying
categorical is already in the model, a real-but-small effect is the
reasonable prior, not the "high-ROI" framing claimed — but the exact
formulation hasn't been directly measured.

---

## 3. Model Architecture & Strategies

### 3a. Dedicated per-airport models ("+5s RMSE drop immediately")

**Done, extensively — not a novel quick win.** `per_airport__*` and
`per_airport_stack__*` arms exist across many seeds and tags in
`artifacts/runs/`. This has been standard practice in this pipeline, not an
untried idea.

### 3b. Model diversity: CatBoost dominates LFPG/LIRF, LightGBM wins LSZH
by ~6s

**Not done — no supporting evidence found.** CatBoost has been extensively
run (`catboost_antigravity__*`, `airport_operator_catboost_multimodel__*`,
many seeds/tags), but no per-airport CatBoost-vs-LightGBM breakdown matching
this specific pattern appears anywhere in `TAXI_OUT_DRIVERS.md`,
`RETROSPECTIVE.md`, or `pipeline_review.csv`. What *is* on record
(`EXTERNAL_PLAN_CROSSCHECK.md` §10) is that CatBoost alone is a **weaker**
standalone model on this data (~305–311 val RMSE, untuned) than the MLP
(~254–261) — it earns its place only as blend diversity, not as an
airport-specialist winner.

### 3c. Two-stage hurdle model (Huber-loss nominal stage + tail classifier
+ blend; "−45s LIRF, −11s LFPG")

**Contradicted, on two independent tests.** This exact architecture exists:
`pipeline_delay_twostage_antigravity/`. Recorded result
(`pipeline_review.csv`): val RMSE **314.96**, test **348.64**, against a
baseline around ~253–260 — logged verdict *"LOST (correctly rejected)...
well behind. 79% of departures have >60s excess so 'delayed or not' isn't a
rare-event split."* Separately, the core mechanism of stage 1 (Huber loss to
avoid overfitting the tail) was tested in isolation and found **decisively
negative: +82.2s worse than plain MSE across all 3 seeds (~23× noise
floor)**, because RMSE as the scoring metric specifically rewards fitting
the real heavy tail, which Huber is designed to underweight
(`EXTERNAL_PLAN_CROSSCHECK.md` §11). Both the whole architecture and its
key ingredient have been tested and both failed, for a stated, coherent
reason — this is the opposite of the claimed outcome, not merely untested.

### 3d. Quick wins — multi-seed bagging (2-3 seeds, "~0.6s free"); physical
floor (`max(240s, y)`)

**Multi-seed: done, far beyond what's described.** Seeds 42 through 53+ are
already standard across most model families in `artifacts/runs/` — not a
remaining quick win.

**Physical floor: partial correction, not the claimed win.** A prediction
floor already exists and was deliberately set (not swept) —
`config.PREDICTION_CLIP_MIN_SEC = 60`. `stage6_clip_sweep.py:124`'s
`sweep_caps(y, pred, floor=config.PREDICTION_CLIP_MIN_SEC)` takes the floor
as a **fixed input**; only the upper cap was swept
(`artifacts/clip/policy_cap_decision.json`). So the floor is 60s, not the
claimed 240s, and raising it to 240s is genuinely untested — worth a cheap
check, but not the sweep-backed "already optimal" state the floor's
*existence* might imply either way.

---

## One-line summary

Of the document's ~10 distinct claims: **1 confirmed-but-already-closed**
with a real mechanism (2b, wake turbulence — closes at −0.134s), **2
contradicted by direct measurement** (2a's regex bug doesn't exist in this
codebase; 3c's hurdle model was tested twice and lost badly), **1
contradicted on its headline statistic** (1a's 15.7% vs. actual 0.39%
pooled / ≤5.1% even LIRF-charitable), **2 already-standard-practice
presented as novel** (3a per-airport models, 3d multi-seed), **1 partial
correction** (3d physical floor — right direction, wrong number, and not
swept), **2 closed-small** (2c wind, 2d congestion-ratio family), and **2
with no supporting evidence anywhere in this project** (1c Terminal 5, 3b
CatBoost-per-airport dominance) plus **2 unverifiable-but-real-sounding
detail** (1a remote-apron stand ranges, 1b FedEx/cargo stands — real
identifiers, untested claim). Net: essentially nothing in this document
survives contact with this pipeline's actual measurements: not a source of
new leverage.
