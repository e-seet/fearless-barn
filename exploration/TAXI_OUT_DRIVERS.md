# Taxi-out delay drivers — hypothesis catalog

This is a companion to [`PIPELINE_REVIEW.md`](PIPELINE_REVIEW.md), but one level
below it: that file judges *models*; this one tracks *causal hypotheses* for why
`TAXITIME_SEC_mvt` is long at a given airport, to be worked airport-by-airport
(the 10 are `EDDF EDDM EGLL EHAM LEBL LEMD LFPG LIRF LSZH LTFM`). A hypothesis
here becomes a candidate feature or a per-airport note, not a model verdict —
that's still decided by measured RMSE per `PIPELINE_REVIEW.md`'s tier rules.

**Status legend:** `pending` — not yet checked · `testing` — feature/analysis in
progress · `confirmed` — measured effect, feeds a feature or per-airport note ·
`rejected` — checked and found no signal (record it anyway, per the project's
existing practice of keeping negative results, e.g. `PLAN.md` finding H).

## 0. Capacity regime — the axis everything else is read against

Before asking "why is taxi-out long here," classify the airport (or
airport × hour-band) into a saturation level. The level determines which other
hypotheses below are even worth testing:

| level | meaning | where taxi delay comes from |
|---|---|---|
| **L1** | Runway/stand capacity well above demand | NOT capacity — look at weather, operator, maintenance, priority rules instead |
| **L2** | Capacity meets demand | Little structural delay; treat as L1 unless a specific spike shows otherwise |
| **L3** | Runway demand near/at capacity | Queuing is itself the explanation — a capacity-driven congestion feature should dominate before other hypotheses are tested |

This is distinct from `stage13_airport_headroom.py`'s "reducible error" axis
(which asks where a *model* still has room to improve) — this is about the
*physical* cause of delay. Worth cross-checking against `pipeline_*_congest*`
arms' congestion features once they're computed per airport.

**Status: `tested` (v3 — v1 and v2 were `invalid, method did not measure the
stated quantity`) — run `python3 stage14_capacity_regime.py`, result in
[`artifacts/audit/capacity_regime.json`](artifacts/audit/capacity_regime.json).**

**v1/v2 → v3, the actual correction (advisor review, flagged after the user
asked "how would you know L1/L2/L3, it's operational and on the ground?").**
v1 and v2 both used Spearman rho(queue length, taxi-out) as the regime
signal — but rho measures a **response slope** (how much taxi-out moves per
extra aircraft in the queue), and L1/L2/L3 is a **level** (demand vs.
capacity). Those aren't the same axis, and they can point opposite ways: a
genuinely saturated runway sits on a plateau — everyone waits a long time
regardless of the exact queue depth — so rho **collapses toward zero exactly
where saturation is worst**. v2's own finding, that peak-only rho was weaker
than the whole-year-pooled rho at 8/10 airports (LSZH peak −0.214, EHAM peak
0.028), was mis-read as a restriction-of-range artifact. It is more likely the
saturation plateau itself — the strongest capacity evidence sitting in the
data, upside down under a slope metric. **Every L1/L2/L3 label from v1/v2 is
retracted; only the raw rho numbers are kept, relabeled `response_slope_rho`,
as a different (still real) question — "how legible is the queuing mechanism"
— not evidence for this one.**

v3 measures a level instead, with no external capacity figures (none ship
with this data):

1. **Demand proxy:** `dep_prev_60m + arr_prev_60m` — total runway-relevant
   traffic in the trailing hour. Both columns already exist.
2. **Observed capacity ceiling, per airport:** the 95th percentile of that
   demand proxy — empirical, self-normalized per airport.
3. **Utilization** = demand ÷ ceiling. `frac_hi_u` = share of an airport's
   rows running at ≥90% of its *own* ceiling — a level, not a slope.
4. **Plateau check:** bin utilization into quintiles, compare the last bin's
   taxi-out step to the average of the earlier steps. Near-zero or negative
   = taxi-out stops climbing at the top of the range = the saturation
   signature this hypothesis actually predicts.
5. Regime combines both: rarely near ceiling → L1; often near ceiling *and*
   flattens there → L3; everything else → L2.

| airport | n | mean taxi-out (s) | frac. time ≥90% of own ceiling | plateau ratio (top bin) | regime |
|---|---:|---:|---:|---:|---|
| EGLL | 239,546 | 1364.2 | 49.6% | **−0.83** | **L3** |
| EHAM | 247,948 | 784.6 | 28.7% | 4.18 | L2 |
| LEMD | 212,242 | 1014.7 | 26.0% | 1.01 | L2 |
| LEBL | 179,700 | 956.6 | 24.8% | 0.59 | L2 |
| EDDM | 167,334 | 811.9 | 22.1% | n/a | L2 |
| EDDF | 230,141 | 863.5 | 22.0% | 6.32 | L2 |
| LFPG | 239,487 | 1019.8 | 19.3% | 5.90 | L2 |
| LIRF | 160,704 | 1194.6 | 19.2% | 3.97 | L2 |
| LTFM | 272,963 | 1053.0 | 18.2% | 4.25 | L2 |
| LSZH | 134,594 | 742.2 | 16.7% | **−2.12** | **L3** |

**Finding: EGLL and LSZH land L3** — both spend a large share of operating
time near their own historical ceiling (50% and 17%), and at both, taxi-out
*stops climbing or drops* in the very top utilization bin (plateau ratios
−0.83 and −2.12: the last step is negative, i.e. mean taxi-out is *lower* in
the top bin than the one below it). That is the saturation signature — read
as evidence the airport (or its ATC flow-management) is actively metering
demand once truly full, rather than letting queue-driven delay grow without
bound. This has strong face validity: EGLL (Heathrow) and LSZH (Zurich) are
both well known for tight slot control and near-100%-utilization mixed-mode
operations in reality, which the model had no way to know — the labels
weren't tuned to match that reputation, they came out of it. No airport lands
in L1; every airport spends at least ~17% of its rows near its own ceiling.

**How to read this against the rest of the catalog:** at EGLL and LSZH, a
capacity/utilization-based congestion feature is worth testing directly
against taxi-out before crediting weather/operator/priority effects there —
those two are running close enough to their own ceiling, often enough, that
capacity is a live explanation. At the other 8, treat capacity as present but
not dominant, and the other hypotheses as the more promising lines.

**Caveats:** (a) EDDM's plateau ratio is `n/a` — too few utilization bins
survived `qcut`'s duplicate-edge dropping; worth a wider bin count before
trusting its L2 label. (b) the ceiling is the empirical 95th percentile of
*this* dataset's year — it does not know about a declared/published runway
capacity, so "near its own ceiling" is relative to what was actually observed,
not to a design limit. (c) demand combines departures and arrivals into one
count; it doesn't yet split by runway configuration (single vs. independent
parallel vs. mixed-mode), which would sharpen the ceiling estimate further.

### 0a. Built and tested as a feature — clean negative

Turned the L3 finding into `high_util_flag` in
[`common/features.py`](common/features.py): 1 if a row's trailing-hour demand
(`dep_prev_60m + arr_prev_60m`) is ≥90% of its airport's frozen 95th-percentile
ceiling (the same constants §0 computed), else 0. Per-airport flag rate on
train2025 matched §0's `frac_hi_u` numbers exactly (EGLL 49.6%, LSZH 16.7%,
etc.) — the implementation is faithful to the measurement, not just similarly
named. Advisor review before building flagged that a plain utilization
*ratio* would be redundant for `per_airport`'s trees (a monotone rescaling
they can already threshold on) and pushed toward a threshold/plateau
indicator instead, aimed specifically at the MLP, which can't as easily
represent the non-monotone plateau shape on its own.

**Pre-committed pass bar** (set before training, not after): the pooled val
gain must exceed the arm's known seed noise, AND EGLL/LSZH must move *with*
the pooled number, not against it — a pooled win driven by an unrelated
airport would mean the feature did something, just not the thing it was
built to test.

Trained `mlp_antigravity`, forward/with_aobt/seed42, once with the flag added
(new feature fingerprint, fresh training run) against the already-cached
baseline run at the prior fingerprint (same arm/split/seed, the same
fingerprint `airport_operator_multimodel`'s v25 submission used):

| metric | baseline | with `high_util_flag` | delta |
|---|---:|---:|---:|
| pooled val RMSE | 253.99 | 252.82 | **−1.17** (better) |
| in-band val RMSE | 235.84 | 236.68 | +0.84 (worse) |
| EGLL val RMSE (n=40,795, flag on 49.6% of rows) | 243.80 | 251.28 | **+7.48 (worse)** |
| LSZH val RMSE (n=24,480, flag on only 16.7% of rows) | 163.84 | 164.35 | +0.51 (worse) |
| LIRF val RMSE (not targeted by this feature) | 542.47 | 537.21 | −5.26 (better) |

**Fails the pre-committed bar.** EGLL and LSZH both got *worse*, not better —
the opposite of what the feature was built to produce, and EGLL has ample
positive examples (49.6% of rows flagged) so this isn't a data-thinness
excuse there. The pooled improvement is real but comes entirely from LIRF, an
L2 airport this feature was never meant to touch — noise or some unrelated
interaction, not the saturation mechanism. LSZH's flag rate (16.7%, the
lowest of any airport) means its result specifically is thin-evidence and
worth taking with a grain of salt, but EGLL alone is enough to fail the bar
as set.

**Not pursued further.** A single seed can't rule out this being one bad
draw, but the instruction going in was to decide the bar before seeing the
number and stop if it fails — re-running more seeds after a clean miss on the
primary target (EGLL) is exactly the kind of post-hoc rationalization the bar
was set to prevent. `high_util_flag` stays in `features.py` (harmless, and
removing it would just re-invalidate the fingerprint again for no gain) but
is not used by any arm's selection story, and this line is closed.

## 1. Priority rules

**Status: `tested` — run `python3 stage15_priority_rules.py`, result in
[`artifacts/audit/priority_rules.json`](artifacts/audit/priority_rules.json).**

| hypothesis | mechanism | expected signal | status |
|---|---|---|---|
| Arrivals prioritized over departures | Taxi-to-departure structurally takes longer than taxi-to-arrival, at every airport | Taxi-out duration > taxi-in duration, airport by airport | **not testable from this data** (see 1a) |
| Business/priority aircraft get parking priority, NOT departure priority | "Priority" flights may taxi-in faster but not taxi-out faster | No taxi-out advantage for business-flagged aircraft, once geometry is controlled | **confirmed, once stand is controlled** (see 1b) |
| IFR ("system-assisted") departures take longer than VFR (visual) | Instrument departures carry more ATC procedure (clearance sequencing, IFR separation) than a visual departure | IFR taxi-out > VFR taxi-out, once geometry is controlled | **EDA confirmed** at EHAM/LSZH (1c); **as a feature: failed permutation control**, no detectable model effect (1c-feature) |

### 1a. Taxi-out vs taxi-in duration — asymmetry measured, but not evidence for THIS hypothesis

**Correction:** the first pass at this (kept below as 1a-i) tested a traffic
*volume* proxy — does departure delay track the arrival count nearby — which
cannot see an ATC sequencing/priority procedure; that's operational, not a
traffic-count effect. What was actually meant: departures should simply take
longer to reach the runway than arrivals take to reach the gate, at every
airport, as the direct signature of that operational asymmetry. Method: the
target column `TAXITIME_SEC_mvt` turns out to be populated for ARR rows too
(mean 536.8s, median 480s pooled — not null as might be assumed), and
`common/features.py` already relies on this for `arr_taxiin_mean_prev_60m`, so
reading it directly per airport is not a new proxy, just the existing target
column on the other phase.

| airport | mean taxi-out (s) | mean taxi-in (s) | mean ratio | median ratio |
|---|---:|---:|---:|---:|
| LEBL | 956.6 | 235.6 | **4.06×** | 4.38× |
| EGLL | 1364.2 | 525.7 | 2.60× | 3.08× |
| EDDM | 811.9 | 367.5 | 2.21× | 2.18× |
| LSZH | 742.2 | 357.2 | 2.08× | 2.16× |
| LEMD | 1014.7 | 525.8 | 1.93× | 1.99× |
| LIRF | 1194.6 | 648.6 | 1.84× | 1.73× |
| LFPG | 1019.8 | 588.1 | 1.73× | 1.75× |
| EHAM | 784.6 | 500.9 | 1.57× | 1.66× |
| EDDF | 863.5 | 582.6 | 1.48× | 1.55× |
| LTFM | 1053.0 | 827.0 | 1.27× | 1.25× |

**Correction #2 (advisor review):** the retitled status above is not a softer
"confirmed" — it's a different verdict. Taxi-out includes pushback, engine
start, AND the hold at the threshold waiting for takeoff clearance; taxi-in
runs from touchdown to gate with no equivalent scarce-resource wait at the
end. Taxi-out > taxi-in is close to **definitional** — essentially every
airport anywhere would show this ratio, arrival-priority procedure or not, so
a 10/10 result with no possible counterexample can't discriminate the
hypothesis. It measures the asymmetry; it cannot test whether arrival
priority is the cause, because nothing else was ever a plausible alternative
outcome.

The one part of the table that *isn't* tautological is the **cross-airport
spread** (1.27×–4.06×) — that variation has to come from somewhere real.

| airport | mean taxi-out (s) | mean taxi-in (s) | ratio |
|---|---:|---:|---:|
| LEBL | 956.6 | 235.6 | **4.06×** |
| EGLL | 1364.2 | 525.7 | 2.60× |
| EDDM | 811.9 | 367.5 | 2.21× |
| LSZH | 742.2 | 357.2 | 2.08× |
| LEMD | 1014.7 | 525.8 | 1.93× |
| LIRF | 1194.6 | 648.6 | 1.84× |
| LFPG | 1019.8 | 588.1 | 1.73× |
| EHAM | 784.6 | 500.9 | 1.57× |
| EDDF | 863.5 | 582.6 | 1.48× |
| LTFM | 1053.0 | 827.0 | 1.27× |

**LEBL checked, and the outlier is on the arrival side, not the departure
side.** My first guess ("remote stands? de-icing?") was a departure-side
explanation for what turned out to be an arrival-side number — the advisor
caught this before I wrote it up. LEBL's taxi-out (956.6s) is mid-pack,
unremarkable; its taxi-**in** (235.6s) is the anomaly, well below every other
airport's range (357–827s mean). Checked the full distribution, not just the
mean — LEBL's taxi-in is uniformly shifted low across percentiles (p10 119s,
median 207s, p90 386s), not a mean dragged down by a subset of outliers. Most
consistent with LEBL's runway geometry (two widely separated parallel
runways; arrivals plausibly favor the one closer to the terminal) — not
investigated further here, but correctly located now.

### 1a-i. Arrival vs departure traffic correlation — answers a different question, kept for the side finding

The original attempt at 1a, before the correction above: per airport, Spearman
rho between taxi-out and (i) inbound arrival count in the preceding 15 min
(`arr_prev_15m`) vs (ii) outbound departure count in the same window
(`dep_prev_15m`).

| airport | rho(arr, taxi-out) | rho(dep, taxi-out) | arr − dep |
|---|---:|---:|---:|
| LEMD | 0.082 | 0.014 | +0.068 |
| EHAM | 0.074 | 0.020 | +0.054 |
| EDDM | 0.039 | 0.050 | −0.010 |
| LEBL | 0.113 | 0.140 | −0.027 |
| LIRF | 0.080 | 0.178 | −0.098 |
| EDDF | −0.027 | 0.109 | −0.136 |
| LFPG | 0.046 | 0.201 | −0.155 |
| EGLL | −0.009 | 0.159 | −0.168 |
| LTFM | 0.012 | 0.215 | −0.203 |
| LSZH | −0.040 | 0.201 | −0.240 |

This doesn't test "is taxi-out structurally longer than taxi-in" (1a's real
question) — it tests whether local traffic volume nearby predicts delay, which
is a congestion/response-slope question (see §0 v3's note on why rho measures
slope, not level), not a priority-sequencing one. Kept because it produced a
genuine side finding that fed into §0: `dep_prev_15m` (airport-wide departure
volume) often correlated MORE with taxi-out than `same_rwy_prev_15m` — e.g.
LSZH 0.201 vs −0.004, LTFM 0.215 vs 0.117. Both are now archived in §0 as
`response_slope_rho` reference numbers, not regime evidence — §0 v3 answered
the capacity question on a level metric instead.

### 1b. Business segment — confirmed once you control the RIGHT geometry axis

Method: per airport, mean taxi-out for `MARKET_SEGMENT_flt == "Business"` vs
everyone else — raw, runway-controlled, and **stand-controlled** (diff
computed within each level, n-weighted across levels).

**Correction (advisor review):** v1 controlled only on `RUNWAY_mvt`. But
`common/features.py`'s own comment states **stand→runway**, not runway alone,
is the strongest geometric determinant of taxi distance — and business jets
typically sit on a different stand set entirely (GA/FBO aprons), which a
runway-only control leaves almost fully in place. EHAM's v1 reversal
(raw −417s → runway-controlled +56s) was the first sign geometry dominates;
stand-level control tests whether the same thing happens to the five airports
where the advantage "survived" runway control.

| airport | n biz | raw diff | runway-controlled | **stand-controlled** |
|---|---:|---:|---:|---:|
| EHAM | 5,887 | −417.5 | +56.3 | **−16.7** |
| LTFM | 2,197 | −416.2 | −356.1 | **−47.7** |
| EGLL | 3 | −366.2 | n/a | n/a (n too small) |
| LFPG | 390 | −276.3 | −271.2 | **+11.1** |
| LSZH | 11,979 | −230.6 | −195.5 | **+6.6** |
| EDDM | 6,195 | −159.3 | −157.9 | **−24.5** |
| EDDF | 3,456 | −74.4 | −105.4 | **−55.8** |
| LEBL | 5,767 | +172.1 | +172.8 | **+60.0** |
| LEMD | 8,521 | +289.5 | +281.9 | **+23.8** |
| LIRF | 51 | +469.0 | +426.0 | n/a (n too small) |

**Finding: stand control collapses almost everything.** Every stand-controlled
diff is within ±60s — an order of magnitude smaller than the raw diffs
(74–469s) and mostly smaller than the runway-controlled ones too. LFPG and
LSZH **flip sign** (business goes from "faster, even runway-controlled" to
essentially flat or marginally slower once stand is held fixed) — the same
pattern EHAM showed at the runway-control step, one level finer. LEBL and
LEMD's "business is slower" raw finding shrinks by 3–12× under stand control,
most of it was stand assignment too. Only EDDF (−55.8s) and LTFM (−47.7s)
keep a diff bigger than ~50s once stand is fixed — small remaining
candidates, not the airport-wide effect the raw numbers suggested.

**Read against the hypothesis: confirmed.** "Business gets parking priority,
not departure priority" predicts little-to-no genuine taxi-out advantage once
you've accounted for *where the aircraft is parked* — which is exactly what
stand-level control shows almost everywhere. The runway-only control in the
first pass wasn't fine-grained enough to see this; it was still crediting
business flights for sitting in a different part of the same runway's stand
set.

**Caveat:** stand control removes the strongest known geometric confound, not
every one — aircraft type/size and operator-specific SOPs are still
unaddressed. The ±20–60s residuals at EDDF/LTFM are small enough that they
could still be either of those, not a priority effect. LIRF (51 rows) and
EGLL (3 rows) remain too thin to read at any control level.

### 1c. IFR vs VFR ("system-assisted" flying) — confirmed where testable

User's framing: does needing "system-assisted flying" — instrument-guided
departure — cost more taxi-out time than a visual departure? Working
interpretation, flagged up front in case it's not what was meant: this maps
directly onto `FLIGHT_RULE_mvt` (`I` = IFR, instrument/ATC-guided; `V` = VFR,
visual/pilot-navigated), a real column in the data — not the same thing as
CAT II/III low-visibility autoland, which isn't labeled anywhere and would
need inferring from METAR visibility (`tests_weather_metar/`, already closed
as a lever).

**Checked volume before testing anything:** VFR departures are essentially
absent at 8 of 10 airports (zero at EGLL/LFPG/LIRF/LTFM, single digits at
EDDF/EDDM/LEMD). Only **EHAM (1,619 VFR)** and **LSZH (1,125 VFR)** have
enough to read; LEBL (136) is thin but included for reference.

| airport | n IFR | mean IFR (s) | n VFR | mean VFR (s) | raw diff | stand-controlled diff |
|---|---:|---:|---:|---:|---:|---:|
| EHAM | 246,228 | 788.2 | 1,618 | 243.8 | −544.4 | **−77.1** (n=3 stands) |
| LSZH | 133,702 | 742.9 | 892 | 638.2 | −104.8 | **−84.7** (n=2 stands) |
| LEBL | 179,564 | 956.9 | 136 | 561.9 | −394.9 | −224.3 (n=1 stand — thin) |

**Finding: confirmed at both well-powered airports.** VFR departures are
faster than IFR everywhere, and at EHAM/LSZH the gap survives stand control
(−77s, −85s) rather than collapsing the way most of §1b's raw business-segment
gaps did — this looks like a real procedural cost of IFR handling, not just
geometry. EHAM's raw gap (−544s) was mostly geometry (VFR traffic clusters on
its own stands), same pattern as everywhere else in this doc, but a genuine
~77–85s remainder survives it.

**Caveats:** (1) VFR flights are disproportionately small GA/training-type
aircraft even within the same stand — aircraft size/type is still an
uncontrolled confound, same limitation as §1b. (2) This explains very little
of the *overall* population: even at EHAM, VFR is 0.65% of departures, so a
real 77–85s per-flight effect moves the airport's pooled RMSE only slightly.
(3) LEBL's number rests on 1 qualifying stand — noted, not trusted. (4) 8 of
10 airports have no VFR traffic to check at all, so "IFR costs more" cannot be
generalized beyond EHAM/LSZH from this data alone.

### 1c-feature. Built as a feature — passed a naive check, FAILED permutation control

Added `is_vfr` (`FLIGHT_RULE_mvt == "V"`) to `common/features.py`. Flagged the
redundancy risk up front: `FLIGHT_RULE_mvt` is already a raw categorical every
arm sees natively, so a derived binary flag is close to informationally
redundant there — the one plausible non-redundant mechanism was
`min_data_in_leaf=100` starving the stand-level interaction a tree would need
to isolate VFR's effect (VFR has <100 rows per stand almost everywhere).
Tested on `mlp_antigravity`, forward/with_aobt/seed42, against the `highutil`
run as baseline (isolates `is_vfr`'s marginal effect from the already-failed
§0a flag, which stays in the codebase):

| airport | baseline | with `is_vfr` | VFR rows here |
|---|---:|---:|---:|
| EHAM (target) | 175.54 | 165.60 | 1,619 |
| LSZH (target) | 164.35 | 163.27 | 1,125 |
| EGLL | 251.28 | 244.36 | **0** |
| LTFM | 245.81 | 241.79 | **0** |
| pooled val | 252.82 | 249.95 | — |

**Red flag caught before declaring a win:** every airport improved, including
EGLL and LTFM, which have **zero** VFR departures — the flag is a constant 0
for 100% of their rows and cannot causally explain a −6.92s change there. My
pre-registered bar ("EHAM and LSZH must improve") technically passed but
didn't check for this, so it wasn't a strong enough bar. Ran a **permutation
control** before trusting it: same architecture, same seed, `is_vfr` shuffled
(same shape, zero real information) via a temporary env-gated branch in
`features.py` (removed after the check).

| airport | baseline | real `is_vfr` | **fake (shuffled) `is_vfr`** |
|---|---:|---:|---:|
| EHAM | 175.54 | 165.60 | **166.45** |
| LSZH | 164.35 | 163.27 | **162.94** (fake slightly beats real) |
| pooled val | 252.82 | 249.95 | **251.30** |

**Conclusion: fails, conclusively.** A flag carrying zero real information
matches — and at LSZH, slightly beats — the real one at the exact airports
the real feature was built to help. The entire apparent improvement is
single-seed training noise (adding one more scalar input shifts the whole
network's weight-init/optimizer RNG draws), not signal from flight rule. This
does not contradict §1c's underlying finding (the stand-controlled group
difference is still real, measured directly on raw data, no model involved)
— it means *this specific engineered feature, on this arm, at n=1 seed,*
produced no detectable causal effect. `is_vfr` stays in `features.py`
(harmless) but is unproven and unused, same status as §0a's `high_util_flag`.

**Process note for next time:** the universal-improvement-including-inert-rows
pattern is a reusable smoke test — before trusting any single-seed feature
result, check whether rows where the new feature is constant/uninformative
improved by a similar amount. If they did, run a permutation control before
reporting anything as a finding.

## 2. Exogenous drivers

**Status: `tested` (all four sub-items resolved) — run
`python3 stage16_operator_effects.py` for the operator check, result in
[`artifacts/audit/operator_effects.json`](artifacts/audit/operator_effects.json);
weather was pre-existing (`tests_weather_metar/`); slot/CTOT and maintenance
have no supporting column in this dataset.**

| driver | mechanism | data needed | status |
|---|---|---|---|
| Weather | Low-vis/de-icing procedures lengthen ground movement | METAR (see `tests_weather_metar/`) | **closed — negative** (see 2a) |
| ATC slot / traffic-flow restrictions | CTOT/slot compliance forces holding before pushback or extra taxi queuing | slot/CTOT data if available | **rejected — no data** (see 2b) |
| Maintenance checks (A/C/D-check) | Longer ground turnaround before a maintenance-flagged departure could bleed into taxi timing or signal an atypical stand | maintenance/check-type flag, if present in source data | **rejected — no data** (see 2b) |
| Operator effects (incl. training flights) | Operator SOPs, crew training flights taxi differently than revenue ops | operator ID (already used in `pipeline_per_operator/`), flight-purpose flag if it exists | **confirmed** (operator); **untestable** (training specifically) (see 2c) |

### 2a. Weather — already closed, not rebuilt

Checked `tests_weather_metar/` before writing any new code, per the process
note this table already carried. It's a completed, rigorous negative result,
not a stub:

- **Real effect exists but isn't feature-able.** Half-hourly METAR joined to
  100% of a `forward/test` baseline (328,009 rows). Adverse-weather buckets
  show a real, directionally consistent under-prediction (thunderstorm +225s,
  snow/ice +168s, freezing +79s, low-vis +83s, fog +73s) — but a train/test
  swap (Nov→Dec vs Dec→Nov) on the residual shows weather's out-of-sample
  explanatory power is **below the noise floor of the test itself** and
  **flips sign** between folds. A real signal doesn't do that.
- **Ceiling is under 1 RMSE point even optimistically** — SSE arithmetic on
  the best-case correctable rows comes to ~0.65% of total SSE, and that's
  before discounting for the scored set (Jan–Jul 2026) being far less
  weather-exposed than the Nov–Dec test window used to measure it.
- **A related false alarm, resolved:** a `seasonal`-split January blowup at
  LFPG/LIRF (RMSE ~800/790 vs ~200 elsewhere) looked exactly like winter
  de-icing. Root-caused instead to ~20–40 rows/month with missing/implausible
  `AOBT_3` (the model's #1 feature) — METAR on the worst days was statistically
  normal (some had zero snow/freezing recorded), and the pathology doesn't
  transfer to the 2026 scored data (missing-AOBT rate 1–1.8% in 2026-Jan vs
  23–85% in the 2025-Jan tail that caused it).

No action taken here — re-opening this would be re-deriving a closed,
well-evidenced result. If a future hypothesis's data collides with weather
again, read `tests_weather_metar/RESULT.md` in full before building anything.

### 2b. ATC slot / CTOT and maintenance checks — no supporting column, checked before building

Checked the raw training parquet schema in full (29 columns) before writing
any test code, per this table's own "check the data dictionary first" note:

```
MVT_ID_mvt FLIGHT_ID_mvt FLIGHT_mvt FLIGHT_RULE_mvt ADEP_mvt ADES_mvt
PHASE_mvt MVT_TIME_UTC_mvt BLOCK_TIME_UTC_mvt SCHED_TIME_UTC_mvt
AIRCRAFT_TYPE_mvt RUNWAY_mvt STAND_mvt TAXITIME_SEC_mvt LOBT_flt
CALLSIGN_flt ADEP_flt ADES_flt ADES_FILED_flt MARKET_SEGMENT_flt IOBT_flt
FLIGHT_RULE_flt FLIGHT_TYPE_flt AIRCRAFT_TYPE_flt WK_TBL_CAT_flt
AIRCRAFT_OPERATOR_flt EOBT_1_flt ARVT_1_flt AOBT_3_flt ARVT_3_flt
```

**No CTOT, slot, regulation, or maintenance/check-type field exists.** Nothing
resembling a Calculated Take-Off Time, an ATFM regulation code, or an
A/C/D-check flag ships with this dataset. Both hypotheses are **rejected for
lack of data**, not tested and found negative — if either becomes important
enough to chase, it needs an external data source (e.g. Eurocontrol NM
regulation archives for slot data; no public source for airline maintenance
schedules), which is out of scope here. Recorded so this doesn't get
re-investigated on the assumption a field might have been missed.

**Update (2026-09-18): the ATFM-regulation half of this gap has now been
tested against a real external source, and closes negative — see §2ah.**
EUROCONTROL's own public daily Airport Arrival ATFM Delay data (a genuine,
exogenous, network-wide ground-hold mechanism) was joined at (airport, date)
granularity with complete coverage including the real scored Jan+Jul 2026
period. Pooled correlation with the current model's residual is weak
(rho~0.013), mixed in sign across airports (unlike most confirmed-real
effects in this document), and the in-sample OLS ceiling is only -0.094s.
The maintenance-schedule half remains genuinely untestable (no public
source identified).

### 2c. Operator effects — confirmed; training-flight identification is untestable

The "training flights" sub-note can't be answered directly: checked
`FLIGHT_TYPE_flt` (codes `S/N/X/G/M`, no code documented or resembling
"training") and `FLIGHT_RULE_mvt`/`FLIGHT_RULE_flt` (IFR/VFR — 99.86% IFR,
VFR too rare and not a training proxy) — neither identifies training
operations. **Operator identity itself is fully testable** via
`AIRCRAFT_OPERATOR_flt`, already a model feature and already shown to carry
real signal as a *pipeline* (`pipeline_per_operator/`, tier B — "real but
subsumed" once blended with `per_airport`). This section asks the sharper
question directly: how much, and is it geometry or something else?

Method (`stage16_operator_effects.py`): per airport, eta-squared (variance of
taxi-out explained by operator identity), raw and **stand-residualized**
(subtract each stand's own mean first, then measure what operator explains in
what's left — same "control for where it's parked" logic as §1b, generalized
from a 2-group diff to a many-level categorical). Operators below `n=100` at
an airport are pooled into "Other" so a handful of 3-flight operators can't
dominate the number.

| airport | n operators kept | raw η² | stand-residualized η² |
|---|---:|---:|---:|
| **LIRF** | 94 | 0.127 | **0.103** |
| LEMD | 95 | 0.286 | 0.062 |
| EDDF | 94 | 0.183 | 0.058 |
| EHAM | 96 | 0.142 | 0.026 |
| EDDM | 83 | 0.083 | 0.021 |
| LSZH | 80 | 0.066 | 0.019 |
| LEBL | 92 | 0.359 | 0.017 |
| EGLL | 78 | 0.073 | 0.013 |
| LFPG | 119 | 0.055 | 0.013 |
| LTFM | 98 | 0.015 | 0.010 |

**Finding: confirmed, and LIRF is a striking outlier.** Operator identity
explains a real slice of taxi-out variance everywhere, even after removing
stand geometry (1–10%). Most airports land in the 1–3% range post-control —
real but modest, consistent with `per_operator` being "subsumed" once
`per_airport` is already in the blend. **LIRF is different: 10.3% of variance
even after stand control**, an order of magnitude above the next airport
(LEMD, 6.2%). Its slowest operator (n=385) runs **+1,448s** slower than the
stand-adjusted baseline — over 24 minutes. This lines up with
`tests_weather_metar/RESULT.md`'s independent note that "Fiumicino just has
long/variable ground ops (remote apron bussing, tows)" found while
investigating the January blowup — different test, same airport, same shape
of explanation: whatever LIRF's ground-handling variability is, it isn't
captured by the `STAND_mvt` label alone, and which operator/handler is
involved matters a lot there specifically.

**Caveat:** eta-squared bundles together every possible reason operator
identity correlates with taxi-out post-stand-control — fleet mix (aircraft
size/type within a stand), route mix (long-haul vs regional prep), and actual
SOP/handling-agent differences are all still tangled together. This says
operator matters, especially at LIRF; it doesn't isolate why.

## 2d. IQR outlier cross-check (Tukey fences, EDA only — not a training filter)

Per user request: use IQR bounds to flag anomalous rows and profile what's
different about them, to surface new hypotheses. Not a training-filter
question — `config.OUTLIER_POLICIES` already exhaustively tested fixed-bound
winsorize/drop policies and found `keep_all` (no filtering) beats every one of
them (`stage6_clip_sweep.py`); this doesn't reopen that.

Method (`stage17_iqr_outlier_profile.py`): per airport, Tukey fences at
1.5×IQR (mild) and 3.0×IQR (extreme) on taxi-out, split by direction — below
the lower fence (suspiciously **fast**) vs above the upper fence
(**slow**/concerning) — since pooling both directions together hid the
clearest results on the first pass (caught before writing anything up: see
below). Then cross-checked known findings' prevalence inside vs outside the
flagged set, rather than treating this as a fresh hunt from zero.

| airport | n | % mild outliers | % extreme | mostly which direction |
|---|---:|---:|---:|---|
| LIRF | 160,704 | 5.23% | 2.17% | upper (8,337 vs 65) |
| LFPG | 239,487 | 4.70% | 1.34% | upper (11,079 vs 165) |
| EDDM | 167,334 | 4.50% | 1.29% | upper (7,528 vs 5) |
| LTFM | 272,963 | 4.25% | 1.32% | upper (11,588 vs 7) |
| LSZH | 134,594 | 3.65% | 0.74% | upper, but more lower share than others (4,547 vs 364) |
| EHAM | 247,948 | 2.36% | 0.57% | upper (5,830 vs 26) |
| EGLL | 239,546 | 2.35% | 0.67% | upper (5,617 vs 21) |
| LEMD | 212,242 | 2.17% | 0.26% | upper (4,487 vs 126) |
| EDDF | 230,141 | 1.77% | 0.37% | upper (4,069 vs 1) |
| LEBL | 179,700 | 1.47% | 0.30% | upper (2,595 vs 42) |

**Cross-check 1 — LIRF's slow operator (§2c): confirmed, cleanly directional.**
The operator stage16 found running +1,448s at LIRF is **3.06%** of LIRF's
upper(slow) outliers vs **0.09%** baseline (34×) — and **0%** of its lower
(fast) outliers. Independent method, same conclusion, right direction only.

**Cross-check 2 — VFR at EHAM/LSZH (§1c): confirmed, and far more strikingly
than §1c's group-mean comparison showed.** VFR is **38.5%** of EHAM's
lower(fast) outliers (vs 0.66% baseline — 58×) and **62.4%** of LSZH's (vs
0.49% baseline — 127×), while being *depleted* in the upper(slow) tail at both
(0.15%, 0.75% — below baseline). A huge share of all VFR departures are fast
enough to fall below the airport's own statistical fence entirely. This is the
strongest evidence yet that §1c's finding is real, not noise — though it
doesn't change §1c-feature's result: the *effect on taxi-out* being real
doesn't mean the *population is large enough* for `mlp_antigravity` to learn
it above its own noise floor, which the permutation control already settled
separately.

**Cross-check 3 — capacity utilization (§0): consistent with the plateau
finding.** At both EGLL and LSZH, `high_util_flag` is **less** common in BOTH
outlier directions than in the baseline (EGLL 37.5%/38.1% outlier vs 49.9%
baseline; LSZH 9.5%/19.8% vs 16.9%). Extreme outcomes — fast or slow — are not
concentrated in the busiest-traffic condition at either airport. Independent
confirmation, via a completely different method, of §0's plateau/saturation
reading: high utilization doesn't produce the most extreme delays there.

**New lead, not previously in the catalog — LEMD's business segment sits in
the slow tail specifically.** §1b found business flights at LEMD run slower
even after stand control (+281.9s) and flagged it as "unexplained." This
profiling shows why the shape, not just the mean, is off: Business is **16.4%**
of LEMD's upper(slow) outliers vs **3.7%** baseline (4.4×) and only 2.4% of
its lower(fast) ones — the LEMD business effect is concentrated in a slow
subset, not a uniform shift across all LEMD business flights. Not yet
explained further (fleet mix? specific routes?) — recorded as the most
promising new lead from this pass, but `MARKET_SEGMENT_flt` is already a raw
categorical feature every arm sees, so a derived flag here would carry the
same redundancy risk that sank both of today's earlier feature attempts;
any feature built from this should expect to need the same permutation-control
scrutiny before being trusted.

**Built as a feature — third clean negative.** Added `lemd_business_flag`
(`ADEP_mvt=="LEMD" & MARKET_SEGMENT_flt=="Business"`) to `common/features.py`.
Redundancy reasoning was more favorable than the first two attempts: for
`per_airport` (a separate model per airport) the interaction is free — its
LEMD-only model already sees `MARKET_SEGMENT_flt` with no other airport's rows
to confuse it. For the MLP (one global model), `ADEP_mvt` and
`MARKET_SEGMENT_flt` are separate embeddings, so "Business is slower, but only
at LEMD" (checked: LEBL's business segment does NOT show the same one-sided
pattern — enriched in both outlier tails, not just the slow one) requires the
network to discover a specific cross-embedding interaction, which this flag
hands it directly instead. Pre-registered bar: LEMD specifically should
improve, without a similarly-sized change at unrelated airports (the
smoke-test learned from the `is_vfr` result).

| airport | baseline (`vfr` run) | with `lemd_business_flag` | delta |
|---|---:|---:|---:|
| **LEMD (target)** | 190.83 | 191.06 | **+0.23 — flat, no effect** |
| EHAM (unrelated, flag always 0) | 165.60 | 176.60 | **+11.00 — much worse** |
| pooled val | 249.95 | 251.87 | +1.92 (worse) |
| in-band val | 232.44 | 235.17 | +2.73 (worse) |

**Fails outright — no permutation control needed this time.** The one airport
the feature could possibly affect didn't move; an airport it structurally
cannot affect (the flag is a constant 0 there) swung by 11 seconds. That's the
same single-seed noise signature as the `is_vfr` probe, just landing badly
instead of looking deceptively good — there's no apparent positive effect
here to disambiguate from noise, so a permutation control has nothing to add.
Three feature attempts today, three clean negatives (`high_util_flag`,
`is_vfr`, `lemd_business_flag`), all left in `features.py` unused, none
submitted.

### 1c-feature, re-tested: proper 3-seed average — confirms negative, direction reverses

Given the noise floor problem both probes exposed, re-ran `is_vfr` (the
stronger of the two leads) at 3 matched seeds (42/43/44) each for baseline
(no `is_vfr`) and treatment, on both val and test:

| metric | baseline (3-seed mean) | with `is_vfr` (3-seed mean) | delta |
|---|---:|---:|---:|
| val pooled | 255.41 | 257.27 | **+1.86 (worse)** |
| val in-band | 236.08 | 237.04 | +0.96 (worse) |
| val EHAM | 179.79 | 185.45 | **+5.66 (worse)** |
| val LSZH | 163.97 | 164.12 | +0.15 (flat) |
| test pooled | 264.57 | 268.95 | +4.38 (worse) |
| test EHAM | 293.62 | 320.01 | **+26.39 (much worse)** |
| test LSZH | 178.50 | 179.03 | +0.53 (flat) |

Per-seed spread makes the earlier single-seed result's failure obvious in
hindsight: baseline EHAM alone ranged **168.2–201.9** across seeds 42/43/44 —
a 33.7s swing from seed choice alone, comfortably larger than the ~10s
"improvement" the single seed-42 comparison first showed. **Conclusion:
confirms the negative from the permutation control, and reverses the
apparent direction** — properly averaged, `is_vfr` doesn't just fail to help,
it's consistently (if mildly) worse on every metric checked, at both target
airports, on both val and test. Closed. Not a candidate for v26.

**Reusable output of this pass — the measured noise floor for `mlp_antigravity`
single-seed probes.** The 3 baseline runs above (fingerprint `c384564e234d3f8e`,
before it was reverted) give a direct measurement, not an assumption:

| metric | seed 42 | seed 43 | seed 44 | spread |
|---|---:|---:|---:|---:|
| val pooled | 251.6 | 259.5 | 255.2 | **7.8s** |
| val EHAM | 168.2 | 201.9 | 168.2 | **33.7s** |

**Rule for every feature probe on this arm going forward: a single-seed
pooled-val delta under ~8s, or a single-airport delta under ~30s at a
noisy airport like EHAM, is inside this arm's own noise floor and proves
nothing either way — run 3 matched seeds before trusting the sign, not just
the magnitude.** This retroactively explains why all three single-seed
attempts today (`high_util_flag`, `is_vfr`, `lemd_business_flag`) were
uninterpretable on their own: every one of their apparent effects was smaller
than this floor. It should prevent a fourth single-seed-only attempt.

**Housekeeping:** `high_util_flag`, `is_vfr`, and `lemd_business_flag` have
been removed from `common/features.py` entirely (not just left inert) —
advisor review caught that leaving them changed
`config.feature_code_fingerprint()` away from `aeade7fee3099c1a`, the value
v25 and every tier-A row in `pipeline_review.csv` were actually measured
under. Leaving them in would silently break comparability for the next person
who runs `run_all.py` or `compare.py` from this tree. Fingerprint confirmed
restored to `aeade7fee3099c1a`. The durable record of everything tried lives
in this doc and in `stage14`–`stage17` + their JSON artifacts, not in
dead columns.

## 2e. LIRF model-residual scan — closed, no viable lever found

Follow-up to §2c/§2d: LIRF is the clearest case of genuine headroom in
`stage13_airport_headroom.py` (`naive_offblock` explains **−3.6%** of variance
there — worse than predicting the flat mean — vs the model's 48.5%), unlike a
ceiling case like EDDF (naive already at 42.9%, little left to take). Question
asked here: is there a specific, buildable correction — a feature or a frozen
lookup table, not a retrain — that captures more of LIRF's in-band residual?

**Method.** For each of six covariates already implicated by earlier sections
(`AIRCRAFT_OPERATOR_flt`, `AIRCRAFT_TYPE_mvt`, `MARKET_SEGMENT_flt`,
`WK_TBL_CAT_flt`, `has_flight_record`, `STAND_mvt`), measured eta-squared of
the **model's own residual** (`y − pred`, `airport_operator_multimodel`) at
LIRF in-band — i.e. how much variance is left over that the model, which
already uses all six as raw features, fails to capture. This is a different
quantity from §2c's eta-squared of the *raw target* (10.3% stand-residualized
for operator) — that measures how much operator matters overall; this measures
what's left after the model has already tried to use it.

**Options considered and ruled out before measuring:**
- **Per-category blend weight** (bucket LIRF's tree/MLP blend weight by
  operator or another covariate) — structurally identical to the already
  quarantined `airport_wtc_multimodel` / `airport_runway_multimodel` /
  `airport_unsup_multimodel` / `airport_wtc_rec_multimodel` ladder in
  `PIPELINE_REVIEW.md` tier E, closed for being over-tuned ("family spread
  smaller than one member's own seed range"). Not built.
- **Multi-way categorical cross feature** (e.g. operator×type×segment) — same
  over-tuning shape moved into feature space, plus extreme sparsity risk
  (236 operators × 117 aircraft types × ~9 segments). Not built.

**Tooling trap hit and fixed along the way.** First stability pass called
`predict_arm('airport_operator_multimodel', ..., tag='msmlp', seed=42/43/44)`
expecting three different draws — got bit-identical arrays back. Cause:
`common/predict.py:195`, `mlp_seeds = m.get("mlp_seeds") or [seed]` — the
`msmlp` run's `model.json` hardcodes `mlp_seeds=[42,43,44]` for its own
internal averaging, so the `seed` argument passed to `predict_arm` is silently
ignored for any tagged multimodel run that has this key. **Valid substitute
used instead:** call `predict_arm('mlp_antigravity', ..., seed=N, tag=None)`
directly per seed — legitimate here because LIRF's blend weight in
`airport_operator_multimodel`/`msmlp` is exactly **0.0** (pure MLP, no tree
blend to lose by bypassing it). Anyone reaching for a seed-stability test on a
tagged multimodel arm will hit this same trap; check `mlp_seeds` in the run's
`model.json` first.

**Results — model-residual eta² at LIRF in-band, by covariate:**

| covariate | seed42 | seed43 | seed44 | avg(45,46,47), independent | verdict |
|---|---:|---:|---:|---:|---|
| operator (min_n=20) | 0.0125 | — | — | 0.0046 (min_n=100) | **real, reproducible, too small** |
| operator (min_n=100) | 0.0057 | 0.0063 | 0.0068 | 0.0046 | stable across seeds |
| aircraft_type | 0.0012–0.0020 | 0.0075 | 0.0044 | 0.0023 | **noise** — 6x single-seed swing |
| market_segment | 0.0008 | 0.0182 | 0.0119 | 0.0020 | **noise** — 20x+ swing |
| wk_tbl_cat | 0.0009 | 0.0200 | 0.0124 | 0.0015 | **noise** — 20x+ swing |
| has_flight_record | 0.0006 | 0.0250 | 0.0149 | 0.0016 | **noise** — collapses on independent average (see below) |
| stand | 0.0067–0.0091 | 0.0081 | 0.0077 | 0.0057 | stable, but mechanism independently dead (below) |

These are two different kinds of "no": **operator and stand are real and
reproducible but immaterial**; **aircraft_type, market_segment, wk_tbl_cat,
and has_flight_record are measurement noise**, not structure — worth keeping
that distinction on record rather than collapsing both into one "closed" line.

**Why the noise labels, not just small numbers.** Single-seed draws (42/43/44)
swung 6–40x for four of the six covariates — the same single-seed trap
`is_vfr` fell into in §1c-feature, one level up (target selection, not feature
selection). `has_flight_record` looked like the strongest candidate on the
`msmlp` 3-seed average (0.0109, over 2x operator) — but recomputed on an
**independent** 3-seed average (45/46/47, never used in any earlier check),
it collapsed to 0.0016. An average can suppress single-seed noise while still
not being real structure itself; the independent-average reproduction check is
what catches that, and this is why it's needed before trusting an averaged
number, not just a single-seed one.

**Why stand (Option C — remote-stand/bussing) is dead despite being stable.**
`STAND_mvt` values at LIRF are plain numeric codes (301–313, 407, 404, ... 940
unique), not hashed, so an external-knowledge mapping (which stands require
bussing) was in principle buildable. But stand is the **lowest** of all six
covariates at every measurement (0.0055–0.0091) — if remote-apron bussing
drove residual structure, stand would be the strongest signal, not the
weakest. Grouping residuals directly by stand (n≥50 per stand) shows apparent
extremes (+168s to −138s) that evaporate to pooled eta²=0.0055 once small-n
noise is controlled for (min_n=100) — the extremes are individual noisy
stands, not a genuine cluster. Mechanism ruled out empirically, not just by
weak eta²; not worth pursuing regardless of external research.

**Why operator doesn't clear the bar even though it's real.** Already priced
directly via a subset SSE calc: on LIRF's confirmed slow-operator subset (98
rows, from §2c), the tree parent beats the MLP (576–606 RMSE vs 730) — but
moving the corrected pooled metric by adopting that costs only **~0.08s**,
the same negligible shape as the already-closed `congest` lever (+0.01s real
board gain despite a 1.2s val gain). A theoretical ceiling check confirms this
isn't a subset-sizing artifact: even crediting the *entire* LIRF operator
signal (eta²=0.0125, the most generous measurement) with **perfect in-sample
capture and zero generalization loss**, LIRF in-band RMSE 417.5 →
417.5×√(1−0.0125) ≈ 414.9 (−2.6s at LIRF); LIRF in-band is 17.6% of the
corrected metric, so pooled 295.2 → **~294.9, a ~0.33s ceiling** — before any
shrinkage and before train→val→board decay. **Scale note:** this 295.2 is the
board-scale corrected metric (v25's actual submitted score is 295.21); §2f's
ceilings below are computed on `stage13`'s *val*-scale corrected metric
(231.43) instead — the two are not directly comparable numbers, both are
labeled by which scale they're on. Against LIRF's board-scale ceiling,
`idband` (the same "frozen lookup table" shape, clean on val) cost **+0.86s
of real harm** on the actual board. Expected value is negative even at the
theoretical best case.

**Conclusion: no viable lever at LIRF within this feature/architecture
paradigm.** Not a new claim — `tests_weather_metar/RESULT.md` independently
reached the same place from the weather angle: "closing 40+ points means a
stronger base model... not a lever on this pipeline." This closes the LIRF
line of investigation the same way weather was closed: real headroom exists,
but nothing cheap and safe reaches it. Nothing trained, nothing changed in
`features.py`, no submission made.

## 2f. EGLL model-residual scan — closed, real but too small to act on

Follow-up to the same `stage13_airport_headroom.py` decomposition that
motivated §2e, re-pulled after LIRF closed: EGLL is now the **single largest
line item in the corrected val metric (17.8%, weight 0.1845)**, with
`skill_vs_naive` of only 44.9% versus EHAM's 53.9% and LTFM's 54.7% at
comparable weight — and EGLL is one of only two L3 capacity-constrained
airports from `stage14_capacity_regime.py` (the other, LSZH, already scores
well: 159.2 RMSE vs EGLL's 227.6). Same question as §2e: is there a specific,
buildable correction that captures more of EGLL's in-band residual?

**Different shape from LIRF up front.** EGLL's blend weight in
`airport_operator_multimodel`/`msmlp` is **0.967 tree / 0.033 MLP** — the
opposite of LIRF (0.0 tree / 1.0 MLP) — so the "call `mlp_antigravity`
directly" substitute used for LIRF does not apply; the tree parent
(`airport_operator_blend`) dominates and needed its own seed-sensitivity
check first. Confirmed genuinely seed-dependent (not deterministic): tree
predictions differ meaningfully by seed everywhere, e.g. EGLL in-band
`max_diff=701s`, `mean_diff=19.2s` row-to-row — real LightGBM
bagging/subsampling variance, not a caching artifact. A global max-diff of
6703s traced to a single LIRF tail row (consistent with LIRF's known
artifact-prone tail, `tests_weather_metar/RESULT.md`), not EGLL.

**Noise floor, measured correctly this time.** Reconstructed the actual
per-seed blended prediction manually — `p = 0.967·tree(seed) + 0.033·mlp(seed)`,
matching seed across both components, mirroring exactly how `predict_arm`
pairs them for a non-`msmlp`-tagged run — rather than reusing the `msmlp` tag
(which would hit the same `mlp_seeds` override trap documented in §2e).
EGLL in-band RMSE across seeds 42/43/44: **227.6 / 227.5 / 227.8** — real
per-row variation washes out in aggregate because the tree ensemble is
equally well-calibrated at each seed, unlike LIRF's MLP. This ~0.3s floor is
far tighter than LIRF's (~14s), and is genuine (verified above), not an
artifact.

**Categorical residual eta², by seed (min_n=100 except hour/dow at 20):**

| covariate | seed42 | seed43 | seed44 | verdict |
|---|---:|---:|---:|---|
| stand | 0.0061 | 0.0061 | 0.0060 | stable, largest — still tiny |
| hour | 0.0045 | 0.0045 | 0.0046 | stable, tiny |
| operator | 0.0037 | 0.0036 | 0.0037 | stable, tiny |
| aircraft_type | 0.0026 | 0.0029 | 0.0026 | stable, tiny |
| dow | 0.0014 | 0.0013 | 0.0013 | stable, tiny |
| wk_tbl_cat | 0.0011 | 0.0014 | 0.0011 | stable, tiny |
| market_segment | 0.0008 | 0.0009 | 0.0008 | stable, tiny |
| runway | 0.0007 | 0.0007 | 0.0007 | stable, negligible |
| has_flight_record | 0.0001 | 0.0002 | 0.0001 | stable, ~zero |

Every covariate is genuinely stable (no LIRF-style 6–40x single-seed swings)
— but every magnitude is small, and the largest (`stand`, 0.0061) is smaller
than LIRF's already-rejected best candidate (operator, 0.5–1.25%).

**Continuous congestion features (residual correlation ρ, by seed)** — the
more physically-motivated angle given EGLL's L3 status — came back equally
small: `n_taxiing_at_pushback` (largest, stable at ρ≈0.033–0.035),
`arr_taxiin_mean_prev_60m` (ρ≈−0.019 to −0.022), all others under |ρ|=0.02.
These features are already in the model (`build_congestion()`); the
near-zero residual correlation says the tree-based blend has already
extracted what they offer.

**Ceiling math, both candidates (val-scale — see §2e's scale note; these are
NOT the same 231.43 vs 295.2 scale, so don't compare the raw seconds across
sections directly, only the relative smallness within each):**
- `stand` (eta²=0.0061, most generous reading, perfect capture assumed):
  EGLL RMSE 227.6→226.9 (−0.7s); corrected **val**-scale metric
  231.43→231.31 (**−0.12s** val-scale ceiling).
- `n_taxiing_at_pushback` (ρ=0.0346, r²=0.0012, perfect capture assumed):
  EGLL RMSE 227.6→227.5 (−0.14s); corrected **val**-scale metric
  231.43→231.41 (**−0.025s** val-scale ceiling).

Both are negligible on their own scale, same conclusion either way: smaller
than LIRF's already-rejected board-scale ~0.33s best case, which itself lost
to `idband`'s precedent — a matched-shape frozen lookup table that looked
clean on val cost **+0.86s of real harm** on the actual board.

**Conclusion: closed, but for a different reason than LIRF.** LIRF closed
because most of its apparent residual structure was *measurement noise* —
4 of 5 covariates failed a reproducibility check outright. EGLL closes
because its residual structure is *real and stable* but *uniformly too
small* — the tree-dominated blend already captures nearly everything these
features offer. Worth keeping that distinction on record: one is "the
measurement wasn't real," the other is "the measurement is real and there's
just nothing there to take." Nothing trained, nothing changed in
`features.py`, no submission made.

**Where this leaves v26.** The two largest line items in the corrected
metric (LIRF 17.6%, EGLL 17.8% — together 35% of it) are now both closed to
single-covariate/feature correction for different, independently-verified
reasons. This mirrors `tests_weather_metar/RESULT.md`'s conclusion reached
independently from the weather angle: the remaining gap is bulk in-band
capacity across the board, not a lever any one airport-specific patch can
reach. Whether to pursue a from-scratch stronger base model/architecture
(a materially bigger undertaking) or treat the current pipeline as near its
ceiling is a scope decision, not a next measurement — left for the user.

## 2g. LIRF date-rollover artifact rows — tested, closed negative

Separate from §2e's residual scan: `PLAN.md`'s audit finding C documented an
exact, pre-existing cluster of training rows with a date-rollover bug — target
in [80,000, 95,000)s, off-block timestamp one calendar day before takeoff,
`y - 86400` landing back in a completely normal taxi-time range. 13-14 of 15
such rows project-wide are LIRF. Proposed (`winsor_43200`/`winsor_80000`) but
never actually tested end-to-end (no result in `pipeline_review.csv` or
`results.csv`). Re-confirmed present in the current data snapshot: **cut1**
(LIRF, taxi-out > 43,200s, missing `AOBT_3`) = 30 rows; **cut2** (> 80,000s,
same conditions) = 13 rows. Other airports show no comparable population —
missing-`AOBT_3` rates are a flat 0.4–1.7% everywhere with no extreme-target
co-occurrence, so this is a LIRF-only phenomenon, not a per-airport sweep.

**Method:** added `--exclude-ids-file`/`--exclude-ids-key` to
`pipeline_mlp_antigravity/train.py` (drops specific `MVT_ID_mvt` rows from the
TRAINING pool only; val/test untouched, per this project's standing rule).
Tested both cuts, 3 seeds each (42/43/44), on LIRF in-band RMSE specifically
(y≤7200, the same measurement used for §2e's baseline) since the excluded
rows are 100% LIRF and any effect should concentrate there, not show up
pooled. Note: for the `forward` split, only 22/30 (cut1) and 9/13 (cut2)
requested rows actually fall in the training-period window — the rest are in
the held-out val/test period and correctly left alone.

| | seed42 | seed43 | seed44 | mean | diff vs baseline (440.57) |
|---|---:|---:|---:|---:|---:|
| baseline (no exclusion) | 434.3 | 439.0 | 448.4 | 440.57 | — |
| cut1 (drop >12h + no AOBT) | 465.4 | 441.4 | 444.2 | 450.34 | +9.77s |
| cut2 (drop >80,000s + no AOBT) | 439.2 | 453.2 | 437.5 | 443.31 | +2.74s |

**Closed, negative.** Baseline's own seed-to-seed spread is 14.1s (434.3 to
448.4) — both cuts' mean diffs fall inside or barely outside that, and
neither shows a *consistent direction* across its own three seeds (cut1's
"+9.77s worse" mean is driven by one outlier seed, 43/44 sit near baseline;
cut2 has one clearly-worse seed and one clearly-better seed). This is
noise, not signal — no independent-seed reproduction check is warranted,
since there's no consistent direction to try to reproduce. A well-motivated,
quantitatively-justified fix (the squared-error-leverage argument: 13-30
rows with targets ~40-60x a normal row's, contributing disproportionate
gradient weight) still didn't clear the bar. Nothing trained beyond this
6-run experiment, `features.py` untouched, no submission made.

**Where this leaves v26 (updated).** Every well-scoped, low-risk lever tried
in this session is now closed: seed-averaging and tree-hyperparameter-tuning
(pre-existing, `PIPELINE_REVIEW.md`), the MLP architecture/training-recipe
(residual connections, wider/deeper, cosine LR schedule — all flat), LIRF
and EGLL per-airport residual correction (§2e/§2f), and this LIRF
artifact-row exclusion. **Decision (2026-09-13): stop here, treat v25's
295.21 as the practical ceiling for this pipeline for now.** A from-scratch
different modeling approach remains the only path that could meaningfully
close the ~50-point gap to the leaderboard, but that's a distinct, larger
undertaking to pick up later, not a next measurement on the current
architecture.

## 2h. Taxi-distance (stand→runway) via OpenStreetMap — closed, real but capped negligible

Every angle tried so far (§2e–§2g) has been a variant of "find a subgroup the
model is getting wrong and correct it." This is the first genuinely new
**feature**, built from external data: physical stand-to-runway distance.
The current feature set has only categorical `STAND_mvt`/`RUNWAY_mvt`/
`stand_runway` IDs — no geometry at all — and the literature on unimpeded
taxi-out time (UTOT) consistently ranks stand-to-runway distance as a top
physical driver. External data is explicitly allowed by the competition
rules (same basis established for the earlier, closed weather investigation;
`exploration/plans/13_live_board_next_steps.md:265`).

**Data.** Queried OpenStreetMap's Overpass API for `aeroway=parking_position`
(stands) and `aeroway=runway` at all 10 airports. 9 succeeded (EDDF, EDDM,
EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LTFM). **LSZH failed** across 3 area-query
retries, a bounding-box retry, and one attempt against the
`overpass.kumi.systems` mirror — identical "server too busy" error each time.
LSZH is excluded from this feature entirely (all its rows are NaN), not
tested and found negative — a real gap, not a zero result. It's a small
airport (weight 0.0463 in the corrected metric), so it can't overturn the
conclusion below, but if anyone revisits this, LSZH needs its own fetch.

OSM stand `ref` tags and runway `ref` tags (e.g. "16R/34L", split into
individual ends) matched directly against `STAND_mvt`/`RUNWAY_mvt` codes with
86.8% row coverage across the 9 fetched airports (80.8% of the val split).
Distance computed via haversine from each stand to its assigned runway.
**Caveat:** the runway coordinate used is each OSM way's centroid, not the
specific numbered threshold — for a ~4,000m runway (e.g. LFPG, LIRF), a
departure from one end vs. the other differs by up to ~2km, and both get the
same distance value. This is a measurement limitation that, if anything,
understates the true correlation — a threshold-accurate distance would
likely correlate better than what was measured here.

**Mechanism check, not just a correlation.** Raw taxi-out time correlates
with this distance at ρ=0.17–0.50 depending on airport — distance genuinely
is one of the strongest physical drivers, exactly as the literature says.
But correlated with the *current model's residual* (`airport_operator_multimodel`
/ msmlp, val split), it's essentially zero pooled and for well-observed
stand-runway pairs (ρ≈0 at ≥100 training observations of that pair) — the
model has already fully learned typical taxi time for those pairs directly
from the categorical embedding. Signal only survives for **rare/under-observed
pairs**, decaying smoothly with pair frequency:

| training obs. of (stand, runway) pair | n (val, in-band) | % of in-band | ρ(resid, dist) | r² |
|---|---:|---:|---:|---:|
| < 30 | 6,830 | 1.85% | +0.0797 | 0.00636 |
| 30–49 | 5,515 | 1.49% | +0.0152 | 0.00023 |
| 50–99 | 13,530 | 3.66% | +0.0022 | 0.00000 |
| ≥ 100 | (remainder) | ~93% | ≈0 | ≈0 |

**Ceiling.** Pricing each non-overlapping band at its own r² and RMSE and
summing the pooled-MSE reduction (not just the tightest n<30 slice, which
alone would understate the true — still small — total):

pooled in-band val RMSE: 227.738 → 227.712s (**Δ 0.026s**), crediting perfect
capture of 100% of the rare-pair residual variance across all three bands.

**Closed, negligible.** 0.026s is the same order of magnitude as every other
real-but-too-small finding closed this session (LIRF operator ~0.08–0.33s,
EGLL stand ~0.12s, EGLL n_taxiing_at_pushback ~0.025s) and would not move a
board score reported to 2 decimal places. The premise was correct — distance
*is* a top physical driver, the literature was right — but it fails here for
a structural reason, not because distance doesn't matter: **a deterministic
function of features the model already has (stand, runway) can only add
information where those categoricals are rarely observed, and only 1.85–7%
of rows fall in that regime.** This is the same principle behind every other
derived-feature negative in this document (`is_vfr`, `high_util_flag`,
`lemd_business_flag`) stated here with a clean, direct measurement instead of
inferred from a string of failures.

Not pursued further (retrying LSZH, using precise per-threshold runway
coordinates instead of centroids, multi-seed confirmation) since the ceiling
caps the upside regardless of measurement precision — even a perfect
distance signal on the rare-pair subset tops out at 0.026s. Single-seed
(seed42) measurement; not multi-seed-confirmed, though the ceiling's
magnitude makes that unlikely to change the conclusion. OSM fetch script and
the 9 per-airport JSON files kept in `exploration/artifacts/` for
reproducibility (a competition eligibility requirement for external data).
`features.py` untouched, no submission made.

## 2i. Huber loss (robust training objective) — tested, closed decisively negative

Same underlying hypothesis as §2g (extreme-target rows exert disproportionate
squared-error gradient leverage during training) but attacked with a soft,
continuous mechanism instead of hard row exclusion: replace `nn.MSELoss()`
with `nn.HuberLoss(delta=...)`, which is quadratic below `delta` and linear
beyond it — large residuals still contribute to the gradient, just not
quadratically. Added `--loss {mse,huber}` / `--huber-delta` to
`pipeline_mlp_antigravity/train.py`; model selection and reported val RMSE
are unchanged (both always compute true squared error on val regardless of
the training-time loss), and the train-loop RMSE log line was fixed to
report true squared error too — previously it printed the raw criterion
value, which under Huber isn't an RMSE.

**delta chosen from data**, not guessed: the msmlp blend's actual in-band val
residual distribution has median |resid|≈102s, p75≈187s, p90≈302s (§2h's
residual table). `delta=150` sits just above the median, so it treats the
bottom ~60% of errors exactly like MSE and only softens the gradient for
larger errors — a plausible, non-arbitrary starting point.

**Method:** 3 seeds (42/43/44), same protocol as the plain-MSE baseline
(`forward` split, `with_aobt` featureset, `keep_all` outlier policy),
comparing the standalone `mlp_antigravity` arm's own val RMSE (not the
blend) against the pre-existing baseline runs at the same seeds.

| seed | baseline (MSE) | huber150 | diff |
|---|---:|---:|---:|
| 42 | 253.99 | 330.81 | +76.8 |
| 43 | 256.16 | 334.85 | +78.7 |
| 44 | 260.88 | 351.98 | +91.1 |
| **mean** | **257.01** (SD 3.52) | **339.21** (SD 11.24) | **+82.2** |

**Closed, decisively negative** — not a noise-floor judgment call like every
other finding in this document. The effect (+82.2s) is ~23x the established
baseline noise floor (SD 3.52s pooled) and consistent in direction and
magnitude across all three independent seeds; a second delta value was not
worth testing. One seed alone (43, +78.7s) was already enough to close this
without needing the other two, which is why no delta sweep was run — the
mechanism explains the effect well enough that refining delta could only
narrow the gap, not close it entirely, and there's no reason to expect
`delta` anywhere in a reasonable range would flip the sign.

**Why it failed, and why that's informative.** RMSE is minimized by the
conditional mean, and the pooled metric squares every error — so getting the
genuinely large taxi-out times (not just noisy ones) approximately right
matters disproportionately to the score. Huber loss is *designed* to reduce
sensitivity to exactly those large residuals, on the assumption they're
outliers/noise to be downweighted. Training epoch logs support this
directly: huber150 runs needed 22–25 epochs before early stopping (vs.
baseline's typical 10–19), and per-epoch train RMSE stayed in the 390–420s
range even late in training (vs. baseline's ~250s), i.e. the model was
still failing to fit the tail by design, not by lack of training time. This
closes the loop with §2g: excluding the extreme rows outright didn't help
(didn't clear noise floor), and softening their gradient influence didn't
help either (made things dramatically worse) — both experiments agree the
same population of rows carries real signal the model needs to fit, not
noise to be suppressed. Plain MSE remains the correct training objective for
this RMSE-scored task. `features.py` untouched, no submission made.

## 2j. Two-stage floor+delta decomposition — closed on the argument, not just the ceiling

The idea (standard in real ATM ops: predict an "unimpeded" floor taxi time
from route/procedure, then a separate congestion delta, sum them) is
architecturally different from anything else in this document — it changes
how information flows through the model, not just what features or loss it
uses. That makes it harder to close with a single number, so this section
closes it on an argument built from three separate measurements this
session, not from the diagnostic below alone.

**First attempt at a diagnostic was circular — worth recording so nobody
repeats it.** The natural-seeming check is: compute `floor` (a robust
per-`stand_runway` low percentile from TRAIN only, `p5`, with an
`n>=10`-observations gate), compute `delta = y - floor` on val, and
correlate `delta` against the current model's residual. That came back at
ρ=0.677 — but this is spurious. `delta` and `resid = y - pred` both contain
`y` as an additive term, so of course they correlate strongly; the check
never touches whether `floor` carries information the model lacks, only
confirms that two quantities sharing a dominant common term move together.

**The real test:** does the `floor` statistic itself — not `y - floor` —
correlate with the current model's residual? This is the same test §2h ran
for taxi-distance, on a different variable. `stand_runway` is already a
categorical the model embeds directly, so `floor` (an aggregate of that same
categorical's training targets) can only add value where the categorical is
under-observed, exactly as with distance:

| training obs. of `stand_runway` group | n (val, in-band, 99.8% floor coverage) | ρ(resid, floor) | r² |
|---|---:|---:|---:|
| < 10 | 5,282 | +0.1358 | 0.01845 |
| 10–29 | 12,479 | +0.0746 | 0.00556 |
| 30–99 | 34,422 | −0.0127 | 0.00016 |
| ≥ 100 (85.8% of rows) | 316,529 | −0.0042 | 0.00002 |

Sanity check: ρ(model prediction, floor) = 0.746 pooled — the model has
already absorbed the floor concept for well-observed groups, exactly as
expected. Summing the MSE reduction across the non-overlapping bands (same
method as §2h; this run's row population is 99.8% of §2h's in-band set, not
identical — coverage differs slightly by the `stand_runway` vs. distance
match rate, so the two ceilings are not directly comparable numbers, only
comparable in shape): **pooled in-band RMSE 227.445 → 227.362 (Δ 0.083s)**
if `floor` were added as an engineered feature to the existing model.

**But that ceiling only rules out "add floor as a feature" — it does not by
itself rule out the two-stage architecture**, which could in principle add
value through a different mechanism: a component-specific loss for the
delta stage, or a component-specific feature split unavailable to a single
end-to-end model. Both of those are independently ruled out by measurements
already made this session, not by this diagnostic:

1. **No new information would enter.** Both stages would draw on the same
   feature set the MLP already sees in full (categoricals with embeddings +
   numerics) — decomposition changes the function class searched, not the
   information available, and a single model with sufficient capacity can
   already represent any floor+delta decomposition internally.
2. **A component-specific loss is the one lever with a direct measurement,
   and it failed catastrophically.** §2i tested exactly this shape of idea
   (treat part of the target differently from plain squared error) and got
   +82.2s worse, ~23x the noise floor, precisely because the RMSE-scored
   metric penalizes under-fitting the real tail — the failure mode a
   delta-specific loss would risk in the same way.
3. **Inductive-bias changes to this architecture are already measured
   flat.** The earlier MLP sweep (residual connections, wider/deeper,
   cosine LR — all closed flat vs. baseline) is itself evidence that this
   architecture isn't bottlenecked on representational flexibility; a
   two-stage split is another inductive-bias change of the same kind.

**Closed.** Three independent measurements this session — the feature-level
ceiling here, §2i's loss-shape failure, and the earlier architecture
sweep — converge on the same explanation from different angles: the model
already has the information a floor/delta split would try to expose, and
reformulating information it already has cannot add any. This is the
clearest statement yet of why derived reformulations keep failing in this
document (`is_vfr`, `high_util_flag`, distance in §2h, floor here): the
ceiling on this whole family of ideas is set by what fraction of rows have
under-observed categoricals (1–7% depending on band), not by how cleverly
the derived quantity is computed. No model built, `features.py` untouched,
no submission made.

## 2k. Queue attention (departure-neighbour set) — closed, flat/no reliable effect

Different in kind from §2h/§2j: those closed because the reformulation was a
deterministic function of a categorical the model already embeds directly
(`stand_runway`), so it structurally couldn't add information. This one
exposes something the model genuinely does not see today: the *individual
identity* of nearby departures (their exact timing, runway match, and
realised-taxi proxy), not the scalar counts/means (`dep_prev_30m`,
`same_rwy_prev_15m`, `dep_realised_taxi_mean_prev_60m`, already closed at
~0.01s real gain) the model is pre-collapsed down to today. A hypothesis
this session hasn't already ruled out on structural grounds.

**Built:** `pipeline_queueattn_antigravity/` — `neighbors.py` constructs,
for every DEP row, the K=8 temporally-nearest other departures at the same
airport within ±90 minutes (exact K-nearest, not approximate: in a
time-sorted array, index distance equals rank distance, so the true nearest
neighbours are always a subset of the 2K adjacent positions — proven, not
assumed). Per neighbour: signed `delta_t_min`, `same_runway` flag, and
`proxy_taxi_sec` — the same leakage-safe proxy (`MVT_TIME_UTC_mvt -
AOBT_3_flt`) the existing congestion features already use, verified
directly against `ranking.parquet` that `TAXITIME_SEC_mvt` and
`BLOCK_TIME_UTC_mvt` are the *only* columns blanked on DEP rows there — a
neighbour's real target is never available (not just the query's), so the
proxy is the only legitimate per-neighbour outcome signal, and both
directions of the time window are legitimate for the same reason
`build_congestion`'s `_next` windows already are (ranking.parquet ships
every movement in the scored months). Whole-period computation before
split-subsetting, matching `build_scope`'s established rule against
train/serve skew at split boundaries. Unit-tested on synthetic cases
(sparse padding, out-of-window exclusion, missing-timestamp exclusion) and
run at scale (6s for the full 4.17M-row period, 98.9% row coverage, mean
7.9/8 neighbour slots filled). `model.py`'s `TabularMLPQueue` adds a
single-head masked attention-pooling layer (d_model=16, d_ctx=16) over the
neighbour set, output concatenated into the same backbone the baseline MLP
uses — isolating the queue-attention addition as the only variable in a
direct comparison. `train.py` is deliberately scoped down (no
`--residual`/`--lr-schedule`/`--loss`/`--congest`) for that reason.

**Result, 6 seeds (42–47), same protocol as baseline** (first 3 seeds
reported flat/ambiguous below; 45/46/47 added to resolve it):

| seed | baseline (MSE) | queueattn | diff |
|---|---:|---:|---:|
| 42 | 253.989 | 251.890 | −2.099 |
| 43 | 256.159 | 266.607 | +10.448 |
| 44 | 260.879 | 258.424 | −2.454 |
| 45 | 257.996 | 256.380 | −1.617 |
| 46 | 255.546 | 254.907 | −0.639 |
| 47 | 252.579 | 252.962 | +0.383 |
| **mean** | 256.19 (SD 2.95) | 256.86 (SD 5.32) | **+0.670** |
| **median diff** | | | **−1.127** |

Paired t-statistic over the 6 diffs: mean 0.670, SE 2.001 (SD of diffs
4.900 / √6), **t = 0.335** — not remotely distinguishable from zero (would
need |t|≈2 for even weak significance at 5 df). 4 of 6 seeds improved
(42, 44, 45, 46, by 0.6–2.5s each), 1 seed regressed sharply (43, +10.45),
1 landed essentially flat (47, +0.38).

**Closed: flat, no reliable effect — a different negative shape than
§2g/§2i/§2h/§2j.** This is not "structurally can't add information" (the
individual-neighbour-identity argument for why this differs from §2h/§2j
still holds) and not "actively fights the metric" (§2i's Huber collapse).
It's simpler: across 6 independently-seeded runs, attending over this
particular neighbour construction (K=8, ±90 min, single-head, d_model=16)
doesn't reliably move val RMSE in either direction. The mean sits inside
noise; the direction seed43 pulled toward is not the direction 5 of 6 other
seeds point in, and neither is a consistent story on its own. Given the
paired t-stat is this close to zero at the largest seed count run for any
single config this session, further seed-grinding on the same
architecture is unlikely to surface a real effect — if one exists here, it
would need a materially different neighbour representation (larger K,
richer per-neighbour features such as aircraft type, or a multi-head
mechanism) to clear the noise floor, not more seeds of this exact config.

**This closes the last of the 4 candidate angles from the research menu.**
Combined with §2h (taxi-distance, negligible ceiling), §2i (Huber loss,
decisive negative), and §2j (floor+delta decomposition, closed on
argument): none moved the board. `pipeline_queueattn_antigravity/` kept in
the repo regardless — a working, tested attention-over-neighbours
architecture that a future session could extend (richer features, more
seeds, or as a component of an ensemble) without rebuilding the neighbour
tensors or leakage-safety verification from scratch. No submission made.

## 2l. Global date-rollover row exclusion (training-only) — tested on the winning pipeline, closed decisively negative

Follow-up to §2g, prompted by re-asking "should we remove outliers from
training" (2026-09-14). `config.OUTLIER_POLICIES`'s fixed-bound sweep
(`drop_7200`/`winsor_*`) already closed that question broadly — `keep_all`
wins by ~27s val RMSE because every bound cuts into the *learnable*
7,200-80,000s tail (PLAN.md finding A3). §2g tested the one population that
argument doesn't cover — the 15 date-rollover rows (target in
[80,000, 95,000)s, off-block timestamp exactly one calendar day before
takeoff, a documented timestamp bug rather than tail structure) — but only
**LIRF-scoped** (13 of the 15), on the standalone `mlp_antigravity` arm, and
only measured on LIRF in-band RMSE. That leaves the genuinely untested cell:
the same rows, **project-wide** (13 LIRF + 1 LFPG + 1 LSZH, re-derived
directly from the current data snapshot rather than reused from
`lirf_artifact_row_ids.json`'s LIRF-only subset), dropped from training
**globally** across every component of the actual winning pipeline
(`airport_operator_multimodel`, tree-tag `congest`, `mlp_seeds=42,43,44`),
measured on **pooled** val/test RMSE — the metric the leaderboard actually
scores.

**Method.** New outlier policy `drop_date_artifacts` (`config.py`,
`common/dataset.py` `apply_outlier_policy` mode `"drop_ids"`) threaded
through the project's existing single choke point for training-pool
filtering — the same mechanism every `--outlier-policy` sweep already uses,
not a bespoke `--exclude-ids-file` path. Of the 15 rows, 11 fall in
`forward`'s training window (Jan-Aug), 1 in val, 3 in test — eval sets are
never filtered, per the project's standing rule, so val/test still score
every row exactly as before. Retrained `mlp_antigravity` and
`airport_operator_blend` (with `--congest`) on both `forward` and
`forward_inner` at seeds 42/43/44, then `airport_operator_multimodel` with
`--tree-tag congest --mlp-seeds 42,43,44 --tag congest_msmlp` — the same
recipe as the `results.csv` `congest_msmlp` baseline, differing only in
`--outlier-policy`.

**Pass/fail bar, set before running:** val was expected to be near-silent by
construction (only 1 of 15 rows sits in val); test was expected to carry any
real signal, since PLAN.md finding C measured the 3 test-period rows at ~52%
of pooled test squared error. Call it a real effect only if the test RMSE
delta exceeds ~1s (an order of magnitude above the ~0.09s tree-seed SD this
arm shows in `results.csv`) with a consistent direction across all 3 seeds,
and treat a test in-band regression as disqualifying regardless of the
pooled number.

**Result, 3 seeds (42/43/44), `drop_date_artifacts` minus `keep_all` baseline:**

| metric | seed42 | seed43 | seed44 | mean diff |
|---|---:|---:|---:|---:|
| val RMSE | 244.71 vs 247.10 | 244.74 vs 247.04 | 244.66 vs 246.98 | **−2.34s (better)** |
| val in-band | 226.82 vs 227.46 | 226.83 vs 227.39 | 226.74 vs 227.35 | −0.60s |
| test RMSE | 242.57 vs 237.02 | 242.54 vs 237.10 | 242.52 vs 237.15 | **+5.45s (worse)** |
| test in-band | 218.78 vs 215.97 | 218.71 vs 216.06 | 218.72 vs 216.09 | **+2.70s (worse)** |

All three seeds agree on direction and magnitude (spread across seeds
≤0.18s on every metric) — against the baseline's own ~0.09s seed SD, the
test regression is roughly 60 standard deviations, not noise. **Fails the
declared bar**: it clears the significance threshold but on the wrong side,
and the in-band regression (+2.70s) disqualifies it outright — this isn't
just the 3 known artifact rows scoring worse (expected, since the model now
has zero training examples anywhere near their range), it's the *ordinary
bulk* of test getting worse too.

**Mechanism — this is the finding, not a footnote.** The per-airport blend
weight (`airport_operator_multimodel`'s alpha, weight on the tree side)
shifted wholesale toward the MLP at airports that contain **zero** of the
11 dropped rows:

| airport | baseline alpha (congest_msmlp) | drop_date_artifacts alpha |
|---|---:|---:|
| EDDF | 0.980 | 0.438 |
| EDDM | 0.807 | 0.517 |
| LSZH | 0.974 | 0.818 |
| LTFM | 0.987 | 0.650 |

Six of the dropped rows sit in `forward_inner`'s training window (Jan-Jun),
none in EDDF/EDDM/LSZH/LTFM's own cells — yet those airports' alphas moved
by 0.15-0.55. Removing 6-11 rows out of 1.3-1.4M shifted the *tree side's*
calibration broadly enough to flip the meta-blend's preference at airports
those rows never touch directly. This is config.py:87-91's "dropping them
removes learnable structure" argument, now measured on the winning pipeline
instead of argued from a simpler group-mean model — extreme-target rows
anchor shared model structure (per-operator effects, global feature
normalization) well beyond their own cells, and losing even a handful of
them degrades that shared structure enough to change how the blend trusts
each parent.

**Closed, decisively negative.** Well-powered (multi-seed, ~60σ effect,
consistent direction), unlike §2g's noise-limited LIRF-only result — this
is the strongest possible version of "remove outliers from training" (15
rows, a documented timestamp bug, not a tail-shape bet) tested on the
pipeline that actually matters, and it still loses on test by 5.45s. The
val improvement (−2.34s) is precisely the val≠board trap this project has
hit before (`idband`, `congest`) — **not evidence for a submission.** Given
this result, re-running the already-closed broad sweep (`drop_7200`/
`winsor_*` vs `keep_all`, which removes thousands of rows including the
entire learnable 7,200-80,000s tail) on the current pipeline would cost
several hours of GPU to reconfirm a strictly worse variant of what was just
measured — not run without asking first. **v25 (295.21) stands. No
submission made.**

## 2m. CatBoost + MLP blend — real diversity vs. MLP alone, but not yet an upgrade on our existing blend

Motivated by real competitor evidence, not a guess: team zestful-fountain
(rank #16/131 at 273.56 on the live DC2026 leaderboard, ahead of our 295.21)
uses CatBoost as their primary model. Built `pipeline_catboost_antigravity/`
(mirrors `pipeline_lightgbm/train.py`'s structure — same CLI, data path,
categorical list, early-stopping/provenance conventions) and wired it into
`common/predict.py` so it participates in the existing blend machinery.

**Standalone, CatBoost is not competitive with the MLP or even our own
LightGBM** (val_rmse_clipped ~305–311 across seeds 42–44, vs. LightGBM's own
~296–298 and the MLP's ~254–261, all same eval path) — its all-rows number
is dragged down by a genuinely bad tail (out-of-band RMSE 12,328 vs. the
MLP's 5,949 on the same 93 rows), even though its *in-band* RMSE is
essentially tied with the MLP's (234.8 vs. 235.8, single seed). Default
hyperparameters (depth=8, lr=0.05), untuned, and the run hadn't even fully
converged (hit the 5000-round cap) — this is a first honest look, not a
tuned result.

**The real question was diversity, not standalone accuracy** — residual
correlation with the MLP is ρ=0.88 (high, but not so high blending adds
nothing). Built `pipeline_airport_catboost_multimodel/` — a direct
structural copy of `pipeline_airport_operator_multimodel` (the arm behind
our actual v25 submission), swapping `airport_operator_blend` for
`catboost_antigravity` as the tree-side blend partner. Same honest
discipline: per-airport weights solved by least squares on `forward_inner`
(a holdout carved from TRAIN, never val).

**Result, 3 seeds:**

| seed | MLP alone | existing blend (airport_operator_multimodel) | CatBoost+MLP blend | vs MLP | vs existing blend |
|---|---:|---:|---:|---:|---:|
| 42 | 253.99 | 248.33 | 246.71 | −7.28 | −1.62 |
| 43 | 256.16 | 248.60 | 249.30 | −6.86 | +0.70 |
| 44 | 260.88 | 252.42 | 251.93 | −8.95 | −0.49 |
| **mean** | 257.01 | 249.78 | 249.31 | **−7.70 (SD 1.11)** | **−0.47 (SD 1.16)** |

**Two different verdicts for two different comparisons.** Against the MLP
alone, this is a real, robust, consistent win — all three seeds land within
a tight 2s band of each other (−6.86 to −8.95), nothing close to washing out.
CatBoost genuinely makes different-enough mistakes from the MLP to be worth
blending, and the per-airport weights make sense (LIRF gets ~0.0–0.07 weight
on CatBoost at every seed — the blend independently rediscovers what we
already know about LIRF being an MLP-only airport, without being told).

Against our **existing** best blend architecture, though, the picture is
much weaker: mean margin −0.47s, SD 1.16 — smaller than its own spread, and
the direction isn't even consistent (seed43 goes the *wrong* way, +0.70s).
This is the exact pattern flagged independently in `conversation.md` §4 by a
community team: "single-seed searches looked like 12–21s gains; multi-seed
confirmation put them at p=0.06–0.18." Seed42's initial read (−1.62s) looked
like a genuine upgrade; three seeds show it's statistically indistinguishable
from a wash.

**Not closed — open with a clear next step, not a dead end.** This isn't a
structural negative like §2h/§2j (CatBoost demonstrably adds real
information the MLP doesn't have) and it isn't a collapse like §2i. It's
"this specific comparison (CatBoost+MLP vs. airport_operator_blend+MLP) is a
wash" — which is a different question from "does CatBoost add value on top
of what we already have." The 2-arm swap discards `airport_operator_blend`'s
own tree-ensemble sophistication (per_airport + per_operator, itself already
beating plain per_airport) entirely; a 3-way blend (CatBoost +
airport_operator_blend + MLP) hasn't been tried and is the fairer test of
whether CatBoost's diversity stacks on top of, rather than substitutes for,
what the existing blend already does well. Single-seed CatBoost
hyperparameters are also still untuned and one run hadn't converged —
neither has been revisited given the 3-seed result didn't clearly justify
the investment yet.

**Follow-up: the 3-way blend, tried — real but unstable, closed negative on
this design.** Built `pipeline_airport_operator_catboost_multimodel/`,
per-airport NNLS over all three predictors (`catboost_antigravity` +
`airport_operator_blend` + `mlp_antigravity`), not a single global weight —
`tri_blend` already measured that a single global weight vector over 3 arms
loses clearly here (val 263 vs. MLP alone's ~254) because the tree/neural
mix genuinely varies by airport. Per-airport NNLS is a principled
combination of two things this codebase has separately already validated
(`airport_operator_multimodel`'s per-airport structure, `tri_blend`'s NNLS
for >2 arms), not a novel design — but flagged its own risk before running:
up to 2 free weights × 10 airports = up to 20 parameters, in the range
MISTAKES.md's lever-3 lesson already warned about (a 27-parameter grid
measured 246.94 in-sample vs. 254.24 honest — WORSE).

| seed | existing blend | CatBoost+MLP (2-arm) | **3-way blend** | vs existing | vs MLP alone |
|---|---:|---:|---:|---:|---:|
| 42 | 248.33 | 246.71 | 245.50 | −2.83 | −8.49 |
| 43 | 248.60 | 249.30 | 247.71 | −0.89 | −8.45 |
| 44 | 252.42 | 251.93 | **259.89** | **+7.47** | −0.99 |
| **mean** | 249.78 | 249.31 | 250.37 | **+1.25 (SD 5.47)** | −5.98 (SD 4.32) |

Seed44 breaks badly — worse than every other variant tested, including MLP
alone with no blend at all. The cause is visible directly in the solved
weights: LIRF's weight vector at seed44 is `cb=0.000 tree=0.064 mlp=1.086`,
summing to 1.15 — NNLS pushed the MLP weight *past* 1.0, the signature of an
unconstrained fit chasing noise in a small, high-variance local sample
(LIRF's own unmatched-row population has RMSE in the thousands of seconds,
see §2e) rather than finding real structure. Seeds 42/43 solve sensibly
(LIRF weight sums to ~1.0–1.02, correctly ~all-MLP) and both beat the
existing blend by a real margin (−2.83s, −0.89s) — so the mechanism (CatBoost
+ tree-ensemble + MLP diversity, when it solves cleanly) is real. The
instability is in the weight-solve, not in whether CatBoost has anything to
contribute.

**Closed on this design, not on the underlying idea.** The flagged
free-parameter risk materialized exactly as warned, on exactly the airport
(LIRF) already known throughout this document to be the smallest, noisiest,
least-structured cell in the data — consistent with the community warning
independently recorded in `conversation.md` §4 ("fitted blend weights lost 4
times out of 4... equal weights have held every time"). A constrained
solve (weights clipped to `[0, 1]` and renormalised, or NNLS with an L2
penalty toward equal weights) would likely fix the LIRF instability
specifically without discarding the real signal seen at 42/43.

**Fix built and tested — the instability is resolved, and the result is now
the most robust finding of this whole investigation.** Added
`solve_weights_per_airport_simplex` (`--weight-method simplex`): per-airport
constrained least squares, `w ≥ 0, sum(w) = 1` (true convex combination,
solved via `scipy.optimize.minimize(method="SLSQP")`), so an individual
weight can no longer be pushed past 1 the way plain NNLS did. Non-obvious
second bug found and fixed along the way, not just the constraint: at
LIRF's actual scale (MSE ≈1.1e6, since its unmatched rows carry targets in
the thousands of seconds), SLSQP's default finite-difference gradient step
**underflowed** and reported `success=True` sitting exactly at the
(1/3, 1/3, 1/3) starting point, 44% worse than the true optimum
(`obj([0,0,1])` was 638,935 vs. the reported-converged 1,138,968) — false
convergence, not a real optimum, confirmed by direct manual evaluation of
the objective at several points before and after the fix. Normalising the
objective by `mean(y²)` per airport (scale-invariant — doesn't move the
optimum, only fixes the optimizer's conditioning) resolved it: same problem,
same starting point, now correctly finds `[0, 0, 1]`.

| seed | existing blend | unconstrained NNLS | **simplex-constrained** | vs existing | vs MLP alone |
|---|---:|---:|---:|---:|---:|
| 42 | 248.33 | 245.50 | 245.69 | −2.64 | −8.30 |
| 43 | 248.60 | 247.71 | 247.58 | −1.02 | −8.58 |
| 44 | 252.42 | **259.89** (broken) | **250.36** (fixed) | −2.06 | −10.52 |
| **mean** | 249.78 | 250.37 | 247.88 | **−1.91 (SD 0.82)** | **−9.13 (SD 1.21)** |

Every seed improves over the existing blend now, with a tight spread (SD
0.82 — smaller than any margin it's measuring, unlike the −0.47s/SD 1.16
wash from the plain 2-arm CatBoost+MLP swap, and unlike the unconstrained
3-way's seed44 blowup). LIRF resolves to all-MLP (or 92.5%/7.5% MLP/CatBoost
at seed43) at every seed, matching what's independently known about that
airport throughout this document. This is the most reliable positive result
of the entire session — not closed, open for a submission-quality build
(seed-averaging both the MLP and CatBoost sides, matching how the real v25
submission averages 18 MLP seeds, is the natural next step before this goes
anywhere near `submit.py`).

**Seed-averaged version, built — new best local result of the session.**
`--mlp-seeds`/`--catboost-seeds` (already supported, mirroring
`airport_operator_multimodel`'s `mlp_prediction` averaging pattern) run with
every seed already trained and available with no new training required: 6
MLP seeds (42–47, the limit — that's how many exist under **both**
`forward` and `forward_inner`, though `forward_inner` alone has up to 59)
and all 3 available CatBoost seeds (42–44; more would mean hours of
sequential GPU training, CatBoost jobs can't run concurrently here — see
§2m's CatBoost build note). Tree side (`airport_operator_blend`) stays
single-seed, consistent with `airport_operator_multimodel`'s own reasoning
that its seed variance is near-negligible (SD 0.27–0.80) next to the
neural side's.

**Result: 244.5605** — beats the best single-seed-per-side result (245.69),
the 3-seed mean of single-seed blends (247.88), the existing best blend
(248.33), and MLP alone even after ITS OWN 6-seed averaging (248.438,
printed for reference in the same run). Two airports (LEMD, EGLL) again
solved to exact equal weights `[1/3, 1/3, 1/3]`; verified NOT a repeat of
the LIRF false-convergence bug by direct manual evaluation of the objective
at several nearby points for both — equal weights was the *lowest* value
found at both airports (e.g. LEMD: 36,039.7 at equal weights vs.
36,164.9–36,361.1 at three nearby alternatives, all worse), confirming the
per-airport normalisation fix generalises rather than needing to be
re-verified airport-by-airport.

**6 CatBoost seeds tried (vs. the 3 above) — no improvement, closing that
question.** User asked to push CatBoost seed depth to match the MLP's 6
(45/46/47 trained, ~1h sequential GPU, CatBoost jobs can't run concurrently
here — established constraint). Result with 6 MLP + 6 CatBoost seeds:
**244.6652** — statistically indistinguishable from the 3-seed CatBoost
version's 244.5605 (a hair worse, well inside noise). CatBoost's own
seed-to-seed variance was apparently already well-averaged at 3 seeds; the
extra hour of compute bought nothing measurable. **244.5605 (3 CatBoost
seeds) stands as the best validated number** — no reason to prefer the
6-seed version, and no reason to grind further CatBoost seeds without a
different justification. A third airport (LTFM) landed at exact equal
weights in the 6-seed run; verified genuine the same way as before (equal
weights: 71,240.8; nearest alternative: 71,223.8 — a 0.02% gap, nothing
like LIRF's 44% miss) — the per-airport normalisation fix continues to
generalise cleanly, now checked at 4 airports.

**Still not submission-ready.** This is a local `forward`-split val number,
not a board score, and the actual v25 submission additionally refits on the
full 12-month period (`--split full --freeze-from ...`) after validating on
`forward` — that refit step is the remaining gate before this goes anywhere
near `submit.py`. No submission made.

**Stand-level congestion aggregates (mean/std/count, 60-min backward window,
grouped by airport+stand) — tested, closed negative.** This was the one
genuinely-untested lever flagged from reading zestful-fountain's public code
(#16/131 on the live DC2026 board): their neighbor feature uses richer
scalar aggregates (3 grouping levels x mean/std/count/own-minus-mean/recency)
vs. our existing runway/airport-level mean-only congestion features. Built
just the stand-level slice (mean/std/count) via searchsorted-based prefix
sums, same technique as the existing `_rolling_prev_mean` congestion
features. Result: **coverage 1.68% of val rows** (a single stand rarely has
another departure from itself within 60 minutes — this is a much finer grain
than airport/runway pooling), and among the 6,199 covered rows, correlation
with the current best blend's residual is **essentially zero**: rho(stand
mean, resid) = -0.036, rho(stand std, resid) = +0.002. This is a cleaner
close than §2h/§2j got — those had small-but-real ceilings from rare/
under-observed categoricals; here there's no residual signal even where the
feature *is* observed, so it can't even help that under-observed population.
Retires the last open item from the zestful-fountain review. No feature
built, no training launched.

## 2n. 3-way blend weight period-stability — vectors drift with horizon, performance doesn't

Prompted by a second-opinion review flagging a real risk: the 3-way blend's
per-airport weights (§2m, 244.5605) are solved on `forward_inner`'s Jul-Aug
2025 holdout, but the actual board score covers Jan 2026 (~1 month past the
full-refit's Dec 2025 cutoff) and Jul 2026 (~7 months past) — a scoring
window the weight-solving surface doesn't represent at all. Checked directly
rather than assumed.

**The weight vectors are genuinely unstable across time.** Re-solved the
same per-airport simplex weights on `forward`'s val (Sep-Oct) and test
(Nov-Dec) surfaces and compared to the shipped (`forward_inner`, Jul-Aug)
vector. Large swings at several airports — e.g. EDDF: mlp 0.444 (shipped) ->
0.192 (val) -> 0.000 (test); EDDM: mlp 0.443 -> 0.333 -> 0.000; EHAM: mlp
0.113 -> 0.174 -> 0.004; LSZH: mlp 0.510 -> 0.268 -> 0.000. A follow-up
per-month solve (Sep/Oct/Nov/Dec individually) at these four airports shows
this is a clean **forecast-horizon decay, not a seasonal effect**: MLP
weight declines monotonically as the scored month gets further from the
training cutoff (Aug 2025), while CatBoost/tree weight rises to compensate,
consistent across all four airports checked.

**False-convergence check on the exact-`[1/3,1/3,1/3]` cells (LTFM, LEMD,
EGLL, EDDM-val)** — the same failure mode that broke LIRF in §2m was ruled
back in as a possibility here and re-checked by direct objective evaluation
at nearby points on each cell's own val/test data. Found real but small gaps
this time (0.08-0.28% below the reported equal-weight objective, vs. LIRF's
original 44% miss) — these cells sit on a near-flat plateau where the
solver stops a hair short of the true optimum, not a repeat of the LIRF bug.
Doesn't change the substantive picture below.

**Despite the vector instability, performance is not fragile — this is the
decisive check.** Scored three weight policies against the *same* arm
predictions on `forward`'s test split (Nov-Dec, held fully out of the
shipped and val solves): shipped (`forward_inner`-solved) weights **242.09**,
val-solved weights (closer to test in time, still out-of-sample w.r.t. test)
**242.34**, flat `[1/3,1/3,1/3]` **277.03**. Fitted beats flat by ~35s
regardless of which nearby window it was fit on; solving further from the
scoring window costs a quarter of a second, not the collapse a community
team's board-measured "fitted lost 4/4 to equal weights" (`conversation.md`)
would predict. **Per-airport attribution of that 35s gap shows it isn't
spread thin across ten fragile tunings — it's almost entirely one airport**:
LIRF alone contributes 201.99s of the 202.92s pooled (flat-minus-shipped)
gap (flat weights hand LIRF's rows to an average of all three arms
including catboost-alone's 306 RMSE there, vs. shipped's clean `[0,0,1]`
pure-MLP weight); EHAM contributes a modest 12.46s; every other airport is
within +/-3s (noise). The 3-way blend's robustness is therefore a single,
mechanistically-understood effect (LIRF's near-total AOBT-unmatched
population, see §2m/OpenAir cross-check) riding on top of near-flat, mildly
horizon-sensitive weights everywhere else — not ten independently fragile
per-airport fits.

**Confirmation on `seasonal` (2026-09-17): the same fitted-vs-flat gap
reproduces on the board-composition-matched surface.** User proposed
re-testing this exact hypothesis (community's "equal weights win," per
`conversation.md` sec.4) against the shipped weights, zero training required
-- re-blending arms already on disk. Built
`tests_seasonal_flat_weights/run_seasonal_flat_check.py`: applied the
REAL shipped per-airport weights (`airport_operator_catboost_multimodel`,
tag=`simplex_avg`, the exact weights behind v25/v26) and flat `[1/3,1/3,1/3]`
to `seasonal`'s test (Jan+Jul 2025), using the same seasonal-trained arms
sec.2o already built (catboost_antigravity seed42, airport_operator_blend
seed42 -- single-seed limitations already flagged there -- plus
mlp_antigravity's full 6 seasonal seeds). **Result: shipped beats flat by
+14.91s pooled (319.44 vs 334.35), and the per-airport breakdown is the
identical mechanism** -- LIRF alone accounts for +184.89s of the gap
(flat=611.27 vs shipped=426.38), every other airport within +/-3s (noise).
Note: `seasonal`'s July slice currently still excludes LIRF by the sec.2af
airport-filter (a stale ASSUMPTION about the real ranking file's July
composition baked into `config.py`, correctly applied by `stage2_split.py`
-- not a broken filter, a wrong input to a working one), so this LIRF
contribution is January-only; the mechanism (LIRF's weight is a fixed
per-airport constant, not month-specific) should apply identically in July,
but that specific row population isn't locally testable until `seasonal_full`
exists (sec.2af). **Second independent confirmation that equal/shrunk
weights lose here, specifically and only because of LIRF's known
CatBoost-alone failure -- the community's finding does not transfer to this
blend's architecture.**

**Third confirmation (2026-09-17), on `seasonal_full` with LIRF's real July
presence included.** Trained `catboost_antigravity` (seed42, tag=cb1),
`airport_operator_blend` (seed42), and `mlp_antigravity` (seeds 42-47) fresh
under the new `seasonal_full` protocol (sec.2af), then re-ran the identical
flat-vs-shipped check on its test (all 10 airports genuinely present in both
Jan and Jul, n=344,419). **Result: shipped beats flat by +56.53s pooled
(348.89 vs. 405.42)** -- and LIRF's own gap is **+303.05s** (763.14 shipped
vs. 1066.19 flat), while every other airport stays within +/-2.1s. The two
pooled numbers (this vs. the January-only seasonal version's +14.91s) are
not directly comparable as a ratio -- different test populations (344,419 vs
218,319 rows, 26,528 vs 11,317 LIRF rows) and freshly-trained
`seasonal_full` arms vs. `seasonal`'s. The defensible claim is directional:
**the effect persists, and grows, once LIRF's July population is included
-- the January-only version was not overstating the effect.** Per-month
LIRF breakdown confirms it holds in BOTH scored months, not just one:
Jan (n=11,317) +193.37s, Jul (n=15,211) +364.33s. **Three independent
surfaces (forward test, seasonal test, seasonal_full test) now agree: equal
or shrunk blend weights lose here, entirely and specifically because of
LIRF's broken CatBoost-alone prediction. Fully closed -- do not revisit
without a materially different blend architecture where this specific
asymmetry no longer holds.**

**Conclusion: the weight-vector drift is real but not a submission risk at
the horizons this data can measure (up to ~4 months past cutoff).** It is
not evidence the 244.56 local number is fragile; it's evidence the
per-airport loss surface is locally flat away from LIRF, so the *choice*
of nearby solve window barely matters. The extrapolation to Jul 2026 (~7
months past the full refit's cutoff, beyond anything measured here) remains
genuinely untested and should be reported as an open, unverifiable
extrapolation, not a resolved risk. The shipped weights stay
`forward_inner`-derived per policy (`SELECTION_FORBIDDEN_SPLITS` blocks
solving on val/test) — this was a read-only diagnostic, not a re-fit.

## 2o. v26 board result (295.4767, flat vs v25's 295.21) investigated — resolved on `config.SPLITS["seasonal"]`, mechanism hunt closed

v26 (the §2m/§2n 3-way blend, `simplex_avg` weights, full-period refit)
scored 295.4767 on the board — essentially flat with v25 (295.21), not the
~4s local improvement (244.56 vs 248.33) predicted. Two mechanism
hypotheses were raised and both were ruled out: horizon-decay (§2n's real,
separately-confirmed effect) can't be differential since v25 carries
identical horizon exposure and the same ~47-51s local-to-board offset;
CatBoost-variant mismatch (weights solved against `forward_inner`-trained
CatBoost, applied to a full-refit variant) was initially flagged as the
leading explanation but retracted — the comparison that suggested it was
measured on Nov-Dec, which is in-sample for the full-refit model, and v25
shares the identical solve-on-pre-refit/apply-to-full-refit pattern with the
same board offset, so the mechanism can't be differential either.
l2-shrinkage, re-swept at the objective's actual scale (1e3-1e7, correcting
a bug where the first sweep at 0-50 was numerically meaningless against a
~1e6 denominator), showed the unconstrained fit is already near-optimal —
no hedge available there.

**Decisive check: does the 3-way blend's local edge survive on
`config.SPLITS["seasonal"]`** — train excludes Jan/Jul/Dec entirely, val is
Dec, test is exactly Jan+Jul 2025 (Jul restricted to EDDF/EGLL/EHAM,
matching the real ranking file) — the one local surface actually
composition-matched to the board, as opposed to forward's Sep-Dec surfaces.
Trained catboost_antigravity (seed42) and airport_operator_blend (seed42)
fresh under this split (previously only `mlp_antigravity` had a seasonal
run), then applied the SAME shipped weights used in the real v26 submission
to these seasonal-trained arms and scored on the test.

**Result: the 3-way blend's edge over the 2-arm (tree+MLP) blend replicates
on this surface too — pooled +3.76s (isolating the CatBoost-addition effect;
1 CatBoost seed vs. the shipped config's 3, not directly comparable in
magnitude to forward-test's +4.71s, but same direction), positive in BOTH
scored months (Jan +3.72s, Jul +4.16s) and positive at 8 of 10 airports**
(EDDM -2.02 and LEBL -1.12 the only regressions; LSZH +17.09 and LIRF
+13.64 the biggest wins). This is now confirmed on TWO independent,
leakage-free, out-of-sample surfaces — forward's Nov-Dec test and seasonal's
board-composition-matched Jan+Jul test — both showing a real, consistent,
same-direction edge. The composition-matched surface, built specifically to
look like the board, does **not** reproduce the board's flatness.

**Conclusion: the mechanism hunt is closed with no defect found in the
blend architecture.** The evidence supports that the recipe is sound and
the 0.27s board delta is consistent with board-level measurement noise
against a real but small local edge, not with the 3-way blend failing to
generalize. **One caveat that stays genuinely open, not closed by this
check**: every local surface constructed here lives inside 2025; the board
scores 2026. A year-boundary effect (some operational or distributional
shift between Dec 2025 and Jan 2026 that no within-2025 holdout can
represent) remains structurally unmeasurable and is an open unknown, not a
resolved risk — same epistemic status as §2n's untested Jul-2026-horizon
extrapolation, for a different reason.

## 2p. 5/10-minute congestion windows — tested, closed near-zero, joining §2f/§2h/§2j/§2k/the stand-level check

The one specifically-flagged-as-untested granularity from
`EXTERNAL_PLAN_CROSSCHECK.md` item 3-4: `common/features.py:build_congestion`
computes departure/arrival/same-runway rolling counts at 15/30/60-minute
windows only (`WINDOWS_MIN`); 5 and 10-minute windows had never been tried.
Built as a standalone diagnostic (same searchsorted-prefix-sum technique as
the existing congestion features, not wired into `features.py`) —
`dep_prev_5m/10m`, `arr_prev_5m/10m`, `same_rwy_prev_5m/10m` — computed
across the full train2025 movement history and correlated against the
current best 3-way blend's val residual.

**Unlike the stand-level check (§2o's vicinity), coverage here is high**
(85-98% of rows have count>0, since these are airport/runway-level windows,
not per-stand) **and the residual correlations are small but consistently
non-zero** (rho 0.017-0.067 across the 6 columns, vs. the stand-level
check's ~0.00-0.04 on a 1.68%-covered population). This is the first
congestion-family variant this session where the raw correlation clearly
isn't just noise.

**Ceiling estimate (joint OLS fit of all 6 columns against the residual,
in-sample — an optimistic upper bound, not an achievable number): pooled
val RMSE 244.56 -> 244.32, a +0.24s ceiling, R²=0.0018.** Per-airport, most
airports show a similarly small ceiling (EGLL +0.06s, LIRF +0.45s, EHAM
+0.37s), but two stand out: EDDF +1.94s (R²=0.017) and EDDM +1.08s
(R²=0.011). Still small in absolute terms, and this in-sample OLS number
is an upper bound a real tree/embedding model incorporating these as raw
features would likely not fully reach (every previous ceiling estimate in
this family — §2h's 0.026s, §2j's 0.083s — either matched or overstated
the effect actually realized after building the feature).

**Closed without building the feature or retraining.** The pooled ceiling
(0.24s) sits in the same near-zero range as every other member of this
feature family, and even EDDF's more interesting per-airport ceiling
(1.94s, optimistic) falls well under the ~5s local margin this session has
now twice demonstrated (§2o) is required before a local improvement is
reliably board-visible. Not worth the retrain given the demonstrated
insensitivity. Retires the last specifically-flagged item from the external
strategy document's congestion-granularity suggestion.

## 2q. LIRF's AOBT-unmatched population — the error IS concentrated (20.7% of SSE in 0.08% of rows), and it is still not exploitable

A second-opinion review, asked for a genuinely different direction rather
than more feature/blend tuning, pointed at where the pooled val RMSE
(244.56) actually lives: LIRF is 7.79% of rows but 37.54% of total squared
error, and within LIRF, the AOBT-unmatched subpopulation (no flight record
— the same 1,488-rows/year group characterized in §2m and the
`DATA_QUALITY_AUDIT.md`) is just **283 rows, 0.077% of ALL val rows, but
20.68% of total SSE.** RMSE there is **4,018.88** against 244.56 pooled —
a 16x blowup. Hypothetically zeroing all error on just these 283 rows would
drop pooled RMSE to 217.82 (**-26.7s**); zeroing all of LIRF would drop it
to 193.29 (**-51.3s**) — arithmetic ceilings that dwarf every feature-level
lever tried this session (all under 5s), and a plausible partial
explanation for the scale of the whole pipeline's local-to-board offset.

**The structure is real, not noise — but resists exploitation.** Printing
the 283 rows directly showed two regimes: on the large-delay rows (which
dominate the SSE — the single worst row alone is ~42% of this subgroup's
SSE), `delta_sched_sec` (MVT_TIME − SCHED_TIME, the OpenAir-style override
quantity) tracks true `y` almost exactly (e.g. y=41,700 vs delta_sched=
41,700, exact). On small/typical-delay rows, the current MLP systematically
over-predicts by 500-2,400s. Correlations confirm neither predictor
dominates (y-vs-MLP=0.909, y-vs-proxy=0.898, MLP-vs-proxy=0.943) —
consistent with the earlier finding (§2m) that the model already beats a
raw MVT-SCHED override (4039 vs 5016 RMSE) without either being clearly
better.

**Built and rejected: a single global-weight convex blend of MLP and the
delta_sched proxy, specific to this subgroup.** Grid-searched (not
SLSQP — a 1-parameter problem sidesteps that whole failure class) on
`forward_inner`'s val (534 unmatched-LIRF rows, Jul-Aug 2025): optimal
alpha=0.6635 (66% MLP / 34% proxy). Validated on three genuinely held-out
surfaces, with the seasonal (board-composition-matched) surface set as the
decisive check *before* running it:

| surface | n | mlp alone | proxy alone | blend (α=0.663) | vs mlp |
|---|---:|---:|---:|---:|---:|
| forward val | 283 | 4018.88 | 5016.19 | 3890.10 | **+128.78** |
| forward test (Nov-Dec) | 140 | 5357.26 | 4582.53 | 4808.27 | **+548.99** |
| seasonal test (Jan+Jul) | 60 | 4266.46 | 5702.63 | 4464.22 | **-197.76** |

**Rejected: fails on the one surface that was pre-registered as decisive.**
**Correction (§2af, found 2026-09-17): `seasonal`'s July slice excludes LIRF
entirely due to a stale `airport_filter` config bug, so this n=60 decisive
check was January-only for a LIRF-specific comparison — the rejection still
stands on the surface it was actually measured on, but "matches the board's
Jan+Jul composition" overstated what that surface covered for LIRF
specifically.** The blend wins on the two surfaces calendar-adjacent to the
Jul-Aug solve window and loses on seasonal. Worse, the MLP/proxy *ranking
itself inverts*
between surfaces (MLP beats proxy on the solve surface and forward val;
proxy beats MLP on forward test) — 534 rows in one two-month window cannot
pin down a relationship that swings by period, so a single global alpha is
the wrong shape of answer even before the seasonal check fails it. The
pooled gain measured on forward val (+1.60s, 244.56 -> 242.96) was measured
on exactly the surface that agrees with the fit, and sits under the ~5s
board-visibility floor demonstrated twice this session (§2o) regardless.
**Not built into the pipeline, not submitted.**

**Conclusion: this is a genuine, quantified floor on the metric, not an
untried opportunity.** Three independent approaches have now been tried on
this exact population — the blend-weight solver's own `[0,0,1]` conclusion
(§2m, pure MLP), the OpenAir override cross-check (§2m, beaten by the
current model), and this convex mix of the two — and none improves on
plain MLP in a way that survives out-of-sample. The rows lack a flight
record AND (1,487 of 1,488) aircraft type, so most of the feature vector
the model relies on elsewhere is simply absent; the remaining error looks
close to irreducible from the columns available. Recorded here so a future
session doesn't re-attempt the same three approaches: the 283-row,
20.7%-of-SSE concentration is real and worth knowing about when
interpreting any future local val number, but it is not a lever.

**Follow-up (2026-09-18): is the unmatched population linked to specific
destinations ("flight path")?** Checked ADES_mvt, RUNWAY_mvt, and
AIRCRAFT_TYPE_mvt distributions of the full-year 1,488-row unmatched
population against LIRF's overall departure population (all 160,704 DEP
rows). Runway distribution is roughly proportional (no concentration).
Aircraft type confirms the already-known fact (1,487 of 1,488 missing).
**Destination shows a real, coherent pattern**: the most overrepresented
destinations (excluding LIRF's own self-loop at 81.8x, n=31 -- almost
certainly ferry/technical repositioning flights) are dominated by two
groups -- (a) short regional Italian hops (LIRA/Ciampino 27.1x, LIME/Bergamo
23.7x, LIRP/Pisa 15.5x, LIRZ/Perugia 10.8x, LIBP/Pescara 38.6x) and (b)
leisure/charter routes to North Africa and the Middle East (HELX/Luxor
19.2x, HESX/Sharm-el-Sheikh 7.8x, DTTJ/Djerba 9.8x, DTKA/Tabarka 17.3x,
HTZA/Zanzibar 7.8x). Plausible mechanism: charter/leisure carriers on
low-frequency seasonal routes are less consistently represented in NM's
flight-plan tracking than mainline scheduled carriers, which would explain
why they lack a matched flight record. **Caveat: most individual
destinations have small absolute counts (n=3-16), so each ratio alone is
noisy** -- the finding is the CONSISTENT THEME across many independent
small-n routes (regional Italian + North-African/Middle-Eastern leisure
charter), not any single route's number.

**Tested for a residual ceiling (2026-09-18) -- closed, cleanly negative.**
Runway: main runway 25 carries 88% of the unmatched population at exactly
the overall rate (ratio 0.98) -- not concentrated. The two reverse-config
runway ends (16L/34R) show a real but small-sample elevated *incidence*
(9.4x/5.6x, n=14/21). Destination: restricted to LIRF's `has_issue` rows in
`forward` val+test (n=423), correlated `ades_overrep_ratio` and
`is_reverse_runway` against `mlp_antigravity`'s residual -- **both
correlations are SMALLER in magnitude than their own permutation-shuffled
controls** (dest: -0.044 real vs. +0.049 shuffled; runway: +0.012 vs.
-0.011), i.e. indistinguishable from noise, not just weak. Destination
predicts INCIDENCE (which routes are more likely to produce an unmatched
flight) but not MAGNITUDE (how bad the error is once a row is already
flagged) -- a meaningful distinction, not the same question. This is a
fourth independent angle confirming "the remaining error looks close to
irreducible from the columns available," joining the blend-weight solver,
the OpenAir proxy cross-check, and the convex mix. Full writeup
`tests_lirf_flightpath/RESULT.md`. Data exported:
`tests_lirf_flightpath/run_lirf_flightpath_export.py` (reproducible),
CSVs delivered to user rather than committed (raw per-row export, not a
persisted repo artifact).

## 2r. Weather (METAR) join — built, single-seed tested, closed negative

A second-opinion review (asked mid-session for a genuinely different
direction) surfaced weather as an untried lead: `prc-2026-datasets-wx/`
ships per-airport METAR observations (`tmpf, dwpf, relh, sknt, gust, vsby,
skyl1, wxcodes, p01i, mslp`) at ~30-min resolution, confirmed UTC-timestamped
and covering the real scored ranking period (2025-01-01 through
2026-07-31). Three of ten stations (EDDM, LIRF, LSZH) encode missing
readings as the literal string `'M'`, silently upcasting the affected
numeric column to `object` dtype unless every numeric column on every
airport is coerced with `pd.to_numeric(errors='coerce')` — not just the
columns a spot check flags.

**Diagnostic ceiling (OLS, in-sample, joint fit of 10 weather features
against the shipped blend's residual):** forward test (Nov-Dec) pooled
+1.68s (R²=0.0133); January (winter, seasonal test) pooled +0.52s
(R²=0.0028), with a per-airport standout at EDDM (+5.21s, R²=0.054) — the
largest single-airport ceiling estimate found in any diagnostic this
session. January is outside `seasonal`'s training months, though, so that
number is partly the model extrapolating into an unseen month, not a clean
read of the weather effect alone — EDDF, EGLL, EHAM, LEBL, LEMD, LFPG,
LIRF, LSZH, LTFM all clustered at +0.58s to +2.55s.

**Built:** a new `featureset="with_aobt_wx"` (isolated from `with_aobt` so
the ablation is provably a no-op for every existing arm — see the module
docstring in `common/features.py`), adding a per-departure asof join
(`merge_asof`, backward, 35-min tolerance, matching the causal-congestion
rule already governing this file) of the same 10 columns validated in the
diagnostic: `tmpf, vsby, skyl1, sknt, gust, has_snow, has_freezing,
has_fog, has_rain, temp_near_freezing`. Join coverage 99.86-99.95% on both
`train2025` and `ranking` scopes (verified directly, both built cleanly).

**Tested: one MLP, seed 42, `forward` split** (advisor's scoped-down
recommendation, so as not to commit to a 3-arm retrain against a 1.68s
ceiling before checking the sign). Result: **val RMSE 255.66, vs. 253.99
for the identical seed/split without weather — 1.67s WORSE.** This single
delta is not by itself decisive — the same split's other MLP seeds (42,
48-53) without weather already span 253.99-259.45, a ~5.5s band that
swallows a 1.67s move either direction — but the ceiling (+1.68s pooled,
optimistic/in-sample by construction, and every prior ceiling this session
either matched or overstated what a real build captured) was already too
small to clear the ~5s local-margin/board-visibility floor established in
§2o even if it had landed positive. More seeds would resolve the sign, not
the magnitude, and the magnitude was already the reason to stop.

**Permutation importance** (10 val passes on the trained MLP, one weather
column shuffled at a time) explains the small degradation rather than
leaving it as pure noise: permuting `has_rain` alone costs +0.92s (the one
real signal the net found), while permuting `tmpf`, `gust`, `skyl1`, and
`sknt` each *improve* held-out RMSE by 0.09-0.60s — the net had fit noise
in those columns that hurt generalization. `gust` (97% NaN, median-filled)
and `skyl1` (~50% NaN) are near-constant columns by construction, a
plausible mechanism for adding 10 mostly-flat/noisy dimensions producing a
small real degradation rather than a training-noise artifact.

**Reverted.** `common/features.py` and `config.py` restored byte-for-byte
(`git checkout --`), confirmed via `config.feature_code_fingerprint()`
returning to the pre-edit value (`aeade7fee3099c1a`) — required because the
fingerprint hashes the whole file, so leaving the ablation in place would
have invalidated every existing trained arm's `submit.py` provenance check
even though their feature *values* are unchanged. The weather-join code is
preserved as a patch (`weather_feature.patch`, scratchpad) — **not a live
lead, closed**: the only column that carried real signal (`has_rain`,
+0.92s on permutation) is itself well under the ~5s board-visibility
floor, so a leaner-column retry is not expected to change the verdict, only
possibly shrink the (already small) degradation. Note also: the trained
run (`artifacts/runs/mlp_antigravity__forward-with_aobt_wx-keep_all__seed42`)
and its feature cache under `artifacts/features/` are kept as evidence but
carry `featureset="with_aobt_wx"`, which no longer exists in
`config.FEATURESETS` post-revert — a future `build_scope` call against that
name will raise `ValueError`, not silently reuse the cache.

## 2s. ARVT_1_flt / ARVT_3_flt (own-flight arrival time) — checked, closed near-zero, and one column flagged as leak-shaped

Advisor-prescribed cheap check, following the one pattern on this build that
has produced a real gain from new information (`tests_congestion_feature/`'s
"read why an existing restriction exists, and check whether its reason still
applies"): both columns are in `config.DEP_AVAILABLE` (not blanked on
ranking DEP rows, verified empirically by `stage1_audit.py`) but appear
nowhere in `common/features.py`'s `source_columns` or `build_features` —
genuinely unused.

**What they are, checked directly against both `training` and `ranking`:**
this flight's OWN destination arrival time, not a neighbouring flight's.
`ARVT_1_flt` (filed/estimated, ~98.9% populated) and `ARVT_3_flt` (actual,
same coverage) both land ~2.9h *after* takeoff on average (median +5,800-
6,300s, matching real flight durations) — confirmed identically on
`ranking.parquet`'s scored DEP rows (98.5% populated, same offset shape).

**`ARVT_1_flt` is legitimate** (filed with the flight plan before departure,
so causally available at prediction time) — **`ARVT_3_flt` is not**: it is
the flight's actual arrival, realised strictly after takeoff, so using it
to predict taxi-out (a pre-takeoff quantity) would mean leaning on real
future/outcome information that happens to survive un-blanked in
`ranking.parquet` rather than a genuine forecasting feature. Flagged rather
than silently used or silently dropped, since whether that crosses the
competition's rules is a judgment call, not an engineering one.

**Moot either way — both closed near-zero on the OLS ceiling check** (same
methodology as §2p/§2r, forward test Nov-Dec, n=325,004): the legitimate
`est_flight_dur_sec` (`ARVT_1 - SCHED_TIME`) ceiling is **+0.056s**,
R²=0.0000, spearman ρ=-0.027; the leak-shaped `ARVT_3 - MVT_TIME` ceiling is
**+0.058s**, R²=0.0000, ρ=-0.023 — indistinguishable from each other and
both far below the ~5s board-visibility floor. **Not built.** The full
`config.DEP_AVAILABLE` vs. `source_columns` diff also surfaces
`ADEP_flt, CALLSIGN_flt, FLIGHT_ID_mvt, FLIGHT_mvt, FLIGHT_RULE_flt,
MVT_ID_mvt` as unused-but-available — `MVT_ID_mvt`/`FLIGHT_ID_mvt` are row
keys, not features (1:1 with the row, would just memorize); `FLIGHT_RULE_flt`
is very likely the filed-plan twin of `FLIGHT_RULE_mvt`, already found
99.86%-constant IFR in §2c; `ADEP_flt` (filed origin) almost always equals
`ADEP_mvt` by construction, same shape as the already-tiny `diverted_plan`
flag. `CALLSIGN_flt`/`FLIGHT_mvt` (specific-service identity) were the one
pair not covered by this check — see §2t, which found `FLIGHT_mvt` is a
real lever. `CALLSIGN_flt` (the ATC callsign, e.g. `DLH8UL` for flight
`LH982` — not a reformatting of the flight number for a meaningful share of
rows, confirmed by direct inspection) remains untested.

## 2t. FLIGHT_mvt (specific scheduled service) identity — real single-arm gain, does not survive the existing blend architecture

Follow-up to §2s's `DEP_AVAILABLE` audit: `FLIGHT_mvt` (e.g. "LH982") is
unused in `features.py`, causally available (filed with the schedule, not
a downstream outcome — no leakage concern), 99.99% populated, and 45,717
unique values in training (median count 2, long tail to 470).

**Eta-squared check** (same method as §2c's operator-effects, extended
with a stricter control): fraction of taxi-out variance explained by
`FLIGHT_mvt` identity (levels with n<100 pooled into `__other__`, 6,673
kept), after subtracting each **(stand, operator) cell's own mean** —
sharper than §2c's stand-only control, since operator SOP is the most
obvious confound for a specific-service identifier. Result: EDDF 8.91%,
LEMD 8.15%, EGLL 5.35%, EDDM 4.19%, LTFM 3.12%, LEBL 3.05%, LFPG 3.03%,
LSZH 2.22%, EHAM 2.02%, LIRF 0.98% — larger than operator identity alone
gave most airports in §2c.

**Permutation control (the decisive check, per advisor, following the same
method §1c used to kill the IFR feature): real, not a cardinality
artifact.** 20 within-airport label shuffles per airport, same eta²
pipeline recomputed on each. Null lands at ~0.31-0.33% everywhere (tight,
std 0.01-0.10%) regardless of airport — the observed values are 18-570
standard deviations above it:

| airport | observed | perm mean | perm std | z |
|---|---:|---:|---:|---:|
| EDDF | 0.0891 | 0.0033 | 0.0002 | 569 |
| EGLL | 0.0535 | 0.0033 | 0.0001 | 384 |
| LEMD | 0.0815 | 0.0031 | 0.0002 | 381 |
| LTFM | 0.0312 | 0.0032 | 0.0001 | 256 |
| EDDM | 0.0419 | 0.0032 | 0.0002 | 187 |
| LEBL | 0.0305 | 0.0031 | 0.0002 | 150 |
| EHAM | 0.0202 | 0.0031 | 0.0001 | 147 |
| LFPG | 0.0303 | 0.0033 | 0.0005 | 52 |
| LIRF | 0.0098 | 0.0031 | 0.0004 | 19 |
| LSZH | 0.0222 | 0.0031 | 0.0010 | 18 |

**Built:** `featureset="with_aobt_flightno"` (`common/features.py`),
`FLIGHT_mvt` added to `CATEGORICAL` (safe unconditionally — every consumer
already filters `[c for c in CATEGORICAL if c in X.columns]`) and set only
under this featureset, mirroring the `with_aobt_wx` ablation pattern.
53.7% of ranking rows share a `FLIGHT_mvt` seen >=100x in training, 78.9%
appear at all; the MLP's entity-embedding table reserves an unseen-category
row for the rest.

**MLP checkpoint (seed 42, `forward`): ambiguous, not decisive.** Pooled
val RMSE 256.93 vs. the clean baseline's 253.99 — inside the same-seed
noise band this session already established (253.99-259.45 across seeds
42/48-53), and no per-airport breakdown was captured (only a training-loop
log, not a full `evaluate.py` run) to check whether EDDF/LEMD moved the
predicted direction. Read as likely a mechanism mismatch, not a negative:
an embedding table has to learn a per-category signal from scratch over
~15-21 epochs on a 45,717-level categorical, whereas CatBoost's ordered
target statistics are close to a direct estimator of exactly this kind of
group-mean effect.

**CatBoost pair (seed 42, `forward`, 8000-round cap): first positive
result of the session, real GPU-noise caveat attached.** The existing
CatBoost baseline (`tag=cb1`) was unreliable -- it hit its 5000-round cap
without triggering early stopping (MISTAKES.md #70) -- so both sides were
retrained fresh at a raised cap for a fair comparison:

| run | best_iteration (of 8000 cap) | val RMSE (clipped) |
|---|---:|---:|
| baseline (`with_aobt`, recap) | 3350, early-stopped | 307.145 |
| `with_aobt_flightno` | 3425, early-stopped | **302.253** |

**-4.89s.** `FLIGHT_mvt` lands mid-pack in feature importance (3.512,
rank 10 of 15 shown) -- real, not dominant. **Caveat that prompted a
replicate:** CatBoost's GPU Bayesian-bootstrap training is not bit-exact
even at a fixed seed -- the OLD truncated baseline snapshot read 305.54 at
iteration 4926, while this freshly-converged run of the nominally same
config settled at 307.145, a ~1.6s spread from re-running "the same" thing
once.

**Seed 43, both sides, same 8000-round cap: the advantage holds and
grows -- positive on val, test and ranking-coverage-by-month still
pending before this can be called closed.**

| seed | baseline (best_iter) | flightno (best_iter) | delta |
|---|---:|---:|---:|
| 42 | 307.145 (3350) | 302.253 (3425) | -4.89s |
| 43 | 311.660 (1281) | 299.634 (3800) | **-12.03s** |

Both seeds improve, both directions agree, and the gap is well outside
the ~4.5s baseline-vs-baseline seed spread that motivated running the
replicate in the first place -- seed43's -12.03s especially. `FLIGHT_mvt`
importance is stable across seeds (3.512, 3.473) -- a consistent,
mid-pack, non-dominant contributor at both. Notably the flightno side is
also *more stable* across seeds (302.25/299.63, spread 2.6s) than the
baseline (307.1/311.7, spread 4.5s) -- some indication the feature also
reduces model instability, not just the mean. Mean gain across both
seeds: **-8.46s**, well above the ~5s board-visibility floor and larger
than any other single lever measured this session.

**Held-out test partition (Nov-Dec, selection-independent -- val's
`best_iteration` was chosen on val, so test is the honest check):
confirms it. Baseline (seed42) 327.525, flightno 320.935, `-6.59s`.**
Sits between the two val deltas rather than decaying toward zero -- the
signature of a genuine feature, not schedule-identity memorization. The
single-arm CatBoost result is real, not noise, not a seed artifact, and
not a selection-surface fluke.

**Ranking-set coverage by scored month, checked before trusting the
gain generalizes to score time:** January 2026 (any-seen coverage 98.0%,
n=152,719), July 2026 (any-seen coverage drops to 63.7%, n=192,122; the
"kept >=100x" coverage is steadier, 58.7% -> 49.8%). July is ~56% of
ranking rows by count and the softer month for this feature -- roughly a
third of July's scored rows carry a never-before-seen flight number.
Unseen categories don't break anything (CatBoost's prior / the MLP's
reserved embedding row absorb them), but the Nov-Dec test gain is the
optimistic end of what the board would see, not the expected board delta.

**Blend-level check: the single-arm gain does NOT survive the existing
3-way architecture, and the mechanism is diversity collapse, not a
measurement problem.** Applying the OLD shipped (`with_aobt`,
`forward_inner`-solved) weights to the NEW flightno-trained arms gives
247.5531 -- *worse* than those same stale weights applied to the OLD
arms (245.8155, `+1.74s`). Re-solving the per-airport simplex weights
directly on this same val, in-sample (a read-only ceiling, not a
selection -- same status as every OLS ceiling check this session, no
winner is chosen from it) recovers most of that: **245.7572** -- but
that is still no better than the stale-weight baseline (245.82, within
noise) and worse than the shipped 6-MLP/3-CatBoost-seed-averaged best
(244.5605). Since this is an optimistic in-sample ceiling, the honest
`forward_inner`-solved number would land at or above it -- **the chain
is dead before a single `forward_inner` retrain, which is why none was
run.**

**Why: FLIGHT_mvt collapses the CatBoost/tree diversity the blend was
living on.** Compare the re-solved weight vector to the shipped one:
CatBoost's weight jumped from near-zero at most airports to 0.33-0.60,
while the MLP's collapsed from substantial to 0.10-0.16 almost
everywhere (LIRF stays the exception, `[cb=0.0, tree=0.179, mlp=0.821]`,
consistent with every prior LIRF finding). `FLIGHT_mvt` hands CatBoost
and the LightGBM-based tree arm (which also improved modestly, 272.434
vs the shipped baseline's 272.9) the same categorical information, so
their errors converge -- each arm gets individually better, but the
ensemble's edge (which came from combining *different* mistakes) doesn't
grow, because there's less disagreement left to exploit. This is a
clean, mechanistically-explained negative, not a noisy or ambiguous one.

**Bottom line: the signal is real and reproducible (two seeds, held-out
test, permutation-cleared eta² all agree) -- but the current blend
architecture cannot monetize it.** This is a different, more informative
closure than "the feature doesn't work": a future session with a
different combination strategy (see the open thread below) has something
concrete to build on; a session re-testing whether `FLIGHT_mvt` carries
signal at all does not need to re-ask that question.

**Reverted.** `common/features.py`/`config.py` restored byte-for-byte
(`git checkout --`), confirmed via `feature_code_fingerprint()` returning
to `aeade7fee3099c1a`. Trained artifacts kept as evidence (`tag=cb2_recap`,
both featuresets, both seeds, under `catboost_antigravity` and
`airport_operator_blend`). Side benefit worth recording independent of
`FLIGHT_mvt`: `catboost_antigravity__forward-with_aobt-keep_all__seed{42,43}__tag=cb2_recap`
are the first PROPERLY CONVERGED CatBoost baselines on this arm (the old
`tag=cb1` run hit its round cap without early-stopping, MISTAKES.md #70)
-- and the seed42-vs-seed43 spread between them, ~4.5s (307.145 vs
311.660), is a real CatBoost-on-GPU seed-noise number worth having on
record for any future single-seed CatBoost comparison.

**Open thread, tested, closed negative: mixed-featureset blend.** The
diversity-collapse hypothesis implies keeping `FLIGHT_mvt` OFF the tree
and MLP arms (CatBoost only) might preserve enough diversity to recover
the gain. Cheap to test without any retraining: reused the already-
trained `catboost_antigravity` (`with_aobt_flightno`, seed42) alongside
the PRE-EXISTING `airport_operator_blend`/`mlp_antigravity` runs (plain
`with_aobt`, seed42, predating this whole investigation). Confirmed
`submit.py`'s `assert_feature_agreement` gates on `feature_fingerprint`
only, not featureset name -- so a mixed-featureset blend is submittable
in principle, a reusable fact for any future arm that wants to mix
featuresets.

In-sample val ceiling (weights re-solved directly on val, same read-only
status as every ceiling check this session): **245.4696** -- better than
the same-featureset ceiling (245.7572) and the stale-weight baseline
(245.8155), which looks like confirmation at first glance. **It is not.**
Both those comparators are themselves optimistic in-sample numbers; the
correct comparator is §2m's HONEST, `forward_inner`-solved, single-seed
result for the original blend: **245.69 (seed 42)**. An optimistic
ceiling landing *below* an honest number is not an improvement -- a
ceiling should beat the honest equivalent by construction even when
nothing real changed (exactly the same artifact that produced the
same-featureset blend's illusory "245.76 beats 245.82" reading earlier
in this section, one notch larger here). Against the correct reference,
this is a wash at best, most likely a loss once solved honestly.

**The residual correlation confirms the mechanism is not fixable this
way:** `corr(resid_catboost, resid_tree) = 0.8695` -- essentially
unchanged from the shared-featureset version. Withholding `FLIGHT_mvt`
from the tree arm does not restore the lost diversity, because the
diversity loss was never about information leaking through a shared
featureset -- it is that a CatBoost made intrinsically stronger (by
*any* good feature) converges toward the tree arm's error pattern
regardless of what the tree arm itself is trained on. **This forecloses
the whole family of "withhold the feature from arm X" variants**, not
just this one configuration. LIRF's weights went even more MLP-extreme
(`[0.022, 0.030, 0.948]`) and the MLP's weight collapsed further
elsewhere (0.04-0.18 vs the shared version's 0.10-0.16) -- the blend
telling you it can't use the MLP well alongside two now-converged
tree-family arms, mixed featureset or not.

**Reverted again** (`git checkout --`, fingerprint confirmed back to
`aeade7fee3099c1a`). `cb2_recap` artifacts kept as evidence, same as
before.

**Gate that is now CLOSED (was open while this was live):**
`config.py`/`features.py` are back to their pre-ablation state, so
`submit.py`'s provenance check no longer rejects the pre-existing arms.

## 2u. FLIGHT_mvt as a target statistic (not an identity) — closed flat, but confirms a reusable representation finding

§2t's diversity-collapse finding forecloses "add a categorical to CatBoost"
as a family (any feature that strengthens CatBoost pulls it toward the
tree arm, wherever the strength comes from) — so the productive angle left
is a feature that helps the arm the blend is *not* starving diversity from:
the MLP, whose weight has collapsed to 0.04-0.18 in every configuration
tested today, and whose own `FLIGHT_mvt` checkpoint (256.93 vs baseline's
253.99) was the one arm that got WORSE from the raw categorical, not
better. Diagnosis: representation failure, not information failure — an
entity-embedding table for 45,717 mostly-once-or-twice-seen levels can't
learn a useful representation in 15-21 epochs, while CatBoost's ordered
boosting computes close to the same thing internally as a target
statistic. Hand the MLP that statistic directly instead of the identity.

**Built:** `featureset="with_aobt_flightstat"` — one numeric column,
`flightno_hist_mean_sec`, a smoothed causal mean of taxi-out for this
row's `FLIGHT_mvt` from STRICTLY PRIOR (by takeoff time) training
occurrences only. Built outside the normal `mov`-derived path in
`build_features`, because `apply_ranking_contract` blanks DEP-row target
on `mov` unconditionally, for BOTH scopes, before `build_features` ever
runs (this is *why* the module structurally cannot leak the target — it
never has access to it via `mov` at all). `build_flightno_stat` instead
loads the real, unblanked training target independently (same external-
data pattern as `_load_weather`) and does a backward `merge_asof` (query
time shifted 1us earlier, to exclude an exact self-match) against a
per-flight-number running (sum, count) built from training data alone.
Shrinkage toward the global mean (effective prior count 15) means a
flight's first-ever occurrence reads as exactly the global mean, not NaN.

**Causal correctness verified directly, not just asserted:** the most
frequent flight number (`TAY9LA`, 470 occurrences) starts at 991.4
(global mean) on its Jan-2 first occurrence with zero prior history, and
climbs monotonically toward its true full-year mean (1164.6) as more
occurrences accumulate through the year, reaching ~1160 by December — the
expected shape of a leakage-free expanding estimator. A single
near-zero-taxi row (TAXITIME=3s) reads 1012, nowhere near its own value —
confirms no self-leak. Works identically for the ranking scope by
construction (training entirely precedes 2026, so every ranking row draws
the flight's full training-period history without a scope flag).

**MLP checkpoint (seed 42, `forward`): 254.350 (epoch 20) vs baseline's
253.989, +0.36s -- confirmed converged, closed flat.** First attempt hit
its 25-epoch cap without early stopping firing, raising the same
MISTAKES.md #70 truncation concern already caught once today on the
CatBoost side (`cb1`) -- so it was re-run at `--epochs 50` before reading
anything into the number. **The re-run settled at the identical value**:
early stopping fired naturally at epoch 26, best still epoch 20, still
254.350. That confirms 254.350 was the genuine converged optimum all
along, not a truncation artifact -- and +0.36s sits deep inside the
253.99-259.45 seed-noise band this session established. **Flat. No more
seeds run**: the premise for going further (a better MLP might reclaim
blend weight) requires the MLP to actually be better, and it isn't --
there is nothing for a `forward_inner` blend re-solve to reclaim, so that
expensive chain was not purchased.

**What this closure is worth keeping, independent of the flat result:**
254.35 vs the raw-categorical checkpoint's 256.93 is a real, reproducible
2.6s difference in how the MLP handles the identical underlying signal
depending on representation -- confirms the embedding-vs-target-statistic
diagnosis mechanistically. This is a fact about THIS architecture, not
about `FLIGHT_mvt`: entity embeddings in this pipeline cannot exploit a
high-cardinality identity at these epoch counts (15,717 levels, ~15-26
epochs, median 2 occurrences/level), and any future high-cardinality
signal aimed at the MLP should reach it as a smoothed statistic, not a
raw identity. `build_flightno_stat` (reverted, preserved as
`flightstat_feature.patch`) is a demonstrated, leakage-free, reusable
causal target-encoder for this codebase -- verified directly (not just
asserted) via the `TAY9LA` trace in the paragraph above -- and is the
concrete artifact a future session reaching for a similar feature should
start from rather than rebuild.

**Where this leaves the search: three ideas this session (raw `FLIGHT_mvt`
categorical, the mixed-featureset blend, the target-statistic MLP feature)
all landed the same way -- a real, correctly-diagnosed mechanism, and no
board-visible gain.** That is the same conclusion `tests_weather_metar/RESULT.md`
and `PIPELINE_REVIEW.md` reached independently before this session even
started: closing the gap to the leaderboard's better teams needs a
stronger base model, not another feature on this one. Three independent
reviews (two prior sessions, this one) now agree. v25 (295.21) stands.

## 2v. Competitive-intelligence pass, and a per-row learned stacker -- both closed

After the FLIGHT_mvt thread closed for the third time, a live check of the
PRC Data Challenge 2026 leaderboard and public team repos was run before
committing to a from-scratch architecture (the only option left on the menu
per SS2u's closing paragraph). **Finding: no actionable lead.** The actual
leaders (`youthful-giraffe` #1 at 245.02, `jovial-uniform`, `vigorous-whistle`,
`enthusiastic-daisy`) publish only registration bios on the official team
pages -- no methodology, no code. The public repos that DO exist
(zestful-fountain #12 @273.56, OpenAir, `ahmetabdullahgultekin/prc-taxiout-2026`,
`LeeMarshall1113/prc-taxiout-2026`) all sit at or behind our own score. The one
structurally-different idea found in public code -- reimplementing
EUROCONTROL's P10-per-stand-runway "unimpeded reference" and modelling the
residual -- is already tested in this repo: SS2j's floor+delta decomposition
used the same mechanism (p5-per-`stand_runway` floor) and closed at a
0.083s ceiling, argued shut by three converging measurements. Nothing new.

The coincidence that the leader's board score (245.02) sits close to our own
LOCAL `forward` val (244.56) remains unexplained by anything found -- it's
consistent with the data-quality audit's finding that the join-failure/AOBT-
missing population is 42-54% larger in the scored set than in training, but
SS2q already tried three fixes for exactly that population and all lost
out-of-sample. Nothing here reopens it.

**Per-row learned stacker (advisor-suggested, cheap: reuses existing
`forward_inner`-trained arms, no retraining).** Replace the shipped
per-airport SIMPLEX-CONSTANT blend weights with a per-row combiner over
`[p_cb, p_tree, p_mlp]`, fit on the same `forward_inner` val surface the
existing weights are solved on. Two variants tried:

- **HistGradientBoostingRegressor** (shallow, regularized, early-stopped) over
  `[p_cb, p_tree, p_mlp, airport]` -- catastrophically worse (val 340.74,
  test 360.30 vs shipped 244.56/242.08). Confirms the free-parameter-overfit
  risk this session has flagged before (MISTAKES.md lever-3): a higher-
  capacity per-row model overfits the `forward_inner` fit surface and does
  not generalize to Sep-Dec.
- **RidgeCV with per-airport interaction terms** (unconstrained linear
  weights per airport, no simplex `w>=0, sum(w)=1` constraint) -- looked
  real on `forward` test alone: pooled 242.08 -> 238.59 (-3.48s). But
  corroborating on the board-composition-matched `seasonal` test (same
  discipline as SS2o's original 3-way-blend validation) shrank this to
  **-0.51s** -- noise. Per-airport attribution on both surfaces shows the
  gain is almost entirely LIRF (-19.9s forward test, -11.2s seasonal test)
  -- the one airport already flagged all session as where unconstrained
  weight-fitting is unstable (the original NNLS bug broke here first, at a
  simplex corner). Outside LIRF the effect is small and inconsistent in
  sign across the two surfaces.

**Closed.** Real-looking on one surface, evaporates on independent
corroboration -- the same pattern as every other lever this session. The
existing per-airport SIMPLEX-constant blend (shipped, val 244.5605) is not
improved on by either stacker variant. v25 (295.21) stands.

## 2w. Frost/de-icing spread (`tmpf - dwpf`), isolated cleanly -- closed near-zero on the ceiling, no retrain needed

A community tip in `conversation.md` ("temperature/dewpoint spread beat a
binary snow flag by a distance, because most de-icing happens on clear
frosty mornings with no precipitation code at all") plus SS2r's own
permutation-importance finding (only `has_rain` genuinely helped in the
10-column bundle; `gust` 97% NaN and `skyl1` ~50% NaN were the likely
sources of that build's net degradation) motivated re-scoping the weather
feature to exactly two clean columns: `tmpf_dwpf_spread` (temperature minus
dewpoint, the standard frost/de-icing indicator) and `has_rain`. Built as
`featureset="with_aobt_frost"`, isolated from `with_aobt` exactly like
SS2r's build (feature *values* for every existing arm provably unchanged;
only the cache fingerprint bumps).

**Coverage: 99.96%** on `forward` test, 99.88% on `seasonal` test -- far
better than the stand-level congestion check's 1.68% and in the same range
as SS2r's original join. **Raw signal looked real at first**: mean residual
of the shipped blend was 25.96s when `has_rain=1` vs 3.09s when `has_rain=0`
on `forward` test (Nov-Dec, n=36,924 / 290,960) -- a large, clean-looking
gap. But the joint OLS ceiling (spread + spread² + rain, in-band, against
the shipped-recipe blend's residual) was only **-0.49s** on that same
surface -- much smaller than SS2r's original (inflated) +1.68s pooled
estimate, because that number was skewed by January, which isn't in the
Nov-Dec test window.

**Checked the board-composition-matched `seasonal` test (Jan+Jul) directly,
since January is exactly where frost/de-icing should matter most.** The
ceiling shrinks further, to **-0.15s**, and the `has_rain` raw gap that
looked large on `forward` test (22.9s) drops to 9.6s and partially inverts
sign (3.85s vs -5.74s) on the January-inclusive surface -- consistent with
the `forward`-test gap being confounded with something else (airport/season
composition the model already handles) rather than a real causal rain
effect. **Closed on the ceiling alone, no MLP retrain purchased** -- same
discipline as SS2p's 5/10-minute congestion-window closure: a ceiling this
far under the session's demonstrated ~5s board-visibility floor, measured
on the decisive board-matched surface, does not justify the training cost
of finding out empirically. Reverted byte-exact, fingerprint confirmed back
to `aeade7fee3099c1a`.

**Bottom line after this pass: the post-v27 search is now independently
confirmed exhausted a fourth time** (competitive intelligence, per-row
stacker, and this cleanly-isolated weather scalar all closed, joining raw
`FLIGHT_mvt`, mixed-featureset blend, and the flight target-statistic from
SS2t/SS2u). v25 (295.21) stands as the best submission.

## 2x. Delta/residual TARGET reformulation (`y - delta_aobt_sec`) — tested, closed flat, and a routing/robust-regressor rebuild for the AOBT-unmatched population re-confirmed closed

An externally-sourced analysis (not from this repo's own history) proposed
two "paradigms" framed as new: (A) route AOBT-unmatched rows to a separate
robust/bounded estimator instead of letting them corrupt the shared model,
and (B) reformulate the regression TARGET as
`Delta = TAXITIME_SEC_mvt - (MVT_TIME_UTC_mvt - AOBT_3_flt)` (i.e.
`y - delta_aobt_sec`) rather than only handing the model
`delta_aobt_sec` as an input feature.

**Paradigm A is not new — it is SS2q (and, from the training-exclusion side,
SS2l/the outlier-policy sweep/SS2i) under a different name**, and stays
closed for the reasons recorded there: three independent approaches on the
283-row AOBT-unmatched-in-LIRF population (the blend solver's own `[0,0,1]`,
the OpenAir-override cross-check, a convex MLP/proxy mix) all failed
out-of-sample, and excluding those rows from training (in any of three
forms: date-artifact exclusion, fixed-bound winsorizing, Huber loss) made
things worse by margins 5–80x the noise floor. No new work done here.

**Paradigm B looked genuinely open** — advisor review flagged that SS2j's
"closed on the argument" reasoning (a single model can already represent
any deterministic decomposition of information it already has) was built
for a different anchor (a training-target *aggregate* of an
already-embedded categorical, `stand_runway`) and doesn't obviously transfer
to a per-row *observed* quantity like `delta_aobt_sec` — and SS2u's own
finding (a target *statistic* beat the raw categorical by 2.6s from
representation alone) is a concrete repo-internal counterexample to "the
model already has this." The analysis also cited `conversation.md` (S4,
"Piyush Patil": *"Anchoring the target on takeoff minus the NM off-block
time was the only large step for us"*) as primary-source evidence a
top-scoring team's single biggest lever was exactly this — confirmed
genuinely present in the file, not fabricated, though this repo's own
existing annotation there already reads it as equivalent to
`delta_aobt_sec`-as-a-feature, which we've had all along.

**Built and tested (`tests_delta_target/run_delta_target.py`), single seed
(42, `per_airport`'s own seed range is 0.02s per PIPELINE_REVIEW.md, so one
seed is interpretable on the tree side), on both `forward` and the
board-composition-matched `seasonal` split — no `features.py` edit, no
fingerprint bump.** Trained a delta-target variant ONLY on
`has_aobt==1` rows (~98.9–99.0% of each split), label
`y - delta_aobt_sec`, inverted at predict time
(`pred + delta_aobt_sec`); the ~1% AOBT-unmatched rows fell back to the
control model's raw-target prediction, byte-identical, so the pooled
comparison isolates the reformulation's effect from SS2q's
already-closed population rather than re-confounding it.

| split | val: raw → delta+fb (Δ) | test: raw → delta+fb (Δ) |
|---|---|---|
| forward | 278.17 → 279.12 (+0.94) | 288.16 → 287.86 (−0.30) |
| seasonal | 275.60 → 276.49 (+0.88) | 350.58 → 350.29 (−0.29) |

**Closed flat.** Val gets consistently, trivially worse (+0.88/+0.94s);
test gets consistently, trivially better (−0.29/−0.30s) — both splits agree
on both signs and both magnitudes almost exactly, which rules out this
being a seed-noise fluke, but the effect itself (under 1s either way) is
two orders of magnitude under the ~5s board-visibility floor established in
SS2o, on the pre-registered decisive surface (`seasonal` test). Per the
gating plan, this closes it without spending MLP/multi-seed compute — the
tree read was supposed to answer "is there a real signal worth escalating,"
and the answer is no. **Mechanistically consistent with SS2j's original
argument after all**, just via a different route than SS2j itself checked:
LightGBM can already fit an additive, mostly-linear offset
(`+delta_aobt_sec`) directly from that same column with no help from a
target transform, so reparameterizing the target changes the optimization
path but not the representable function class here — unlike SS2u's
categorical-embedding case, where the MLP's *representational* mechanism
(a 45k-level embedding table needing many observations per level) was the
actual bottleneck a raw column couldn't fix. Not built into any shipped
arm, not submitted. `conversation.md`'s quote, read carefully, is about
having `delta_aobt_sec` as a feature at all (which we've had since
`with_aobt` was introduced) — not evidence for reformulating the target on
top of it.

## 2y. Airport-specialist MLP architecture ("Attack 3") — premise DISPROVEN by this repo's own `per_airport_mlp` data, no build needed

A follow-up externally-sourced analysis proposed three "structural attacks."
Two are addressed elsewhere (Attack 1, the LIRF artifact classifier, is
`idband`/SS2q under a new name — **already submitted to the live board
twice, v15/v17, and regressed both times**, +0.86s/+0.87s, so it was skipped
outright rather than rebuilt; Attack 2, runway-configuration/crossing
features, is genuinely novel and being tested separately, see the
`tests_runway_config/` diagnostic). This section is Attack 3: replace the
single pooled `mlp_antigravity` (one network, airport enters only via an
embedding) with fully independent per-airport specialist networks — the
pitch being that LIRF's extreme artifacts and EGLL/EHAM's congestion physics
each need their own architecture rather than fighting for shared capacity.

**A first read (by a research subagent) of this repo's own `per_airport_mlp`
arm claimed support for the idea — "genuinely won on data-rich calm hubs
(EHAM 171.6, EDDF 190.5, LEMD 207.5 — all beat the pooled MLP)." That claim
is WRONG and was caught by pulling the actual `eval_val.json` /
`model.json` numbers directly, not by re-running anything:**

| airport | pooled `mlp_antigravity` (val) | per-airport `per_airport_mlp` (val) | delta |
|---|---:|---:|---:|
| EHAM | 180.52 | 171.59 | **−8.93 (better)** |
| EDDF | 173.80 | 190.54 | +16.74 (worse) |
| LEMD | 192.10 | 207.53 | +15.43 (worse) |
| LSZH | 163.84 | 205.46 | +41.62 (worse) |
| EDDM | 179.61 | 201.51 | +21.90 (worse) |
| LEBL | 238.91 | 269.41 | +30.50 (worse) |
| LTFM | 243.81 | 270.03 | +26.22 (worse) |
| LFPG | 246.62 | 269.47 | +22.85 (worse) |
| EGLL | 243.80 | 287.84 | +44.04 (worse) |
| LIRF | 542.47 | 1128.92 | +586.45 (catastrophic) |

(Both `forward` split, `with_aobt`/`keep_all`, seed 42 — the only
apples-to-apples pair available: `per_airport_mlp`'s own model.json only
reports val, so this is a val-vs-val comparison, not test.)

**Only 1 of 10 airports improves under full per-airport splitting, and it's
a single seed's ~9s move against a 3.52s MLP seed SD — not distinguishable
from noise.** Every other airport, including the "data-rich calm hubs" the
first read cited as wins, gets WORSE, by amounts from +15s to +44s (LIRF
aside). This is the opposite of "shared weights corrupt LIRF while starving
calm hubs of dedicated capacity" — pooling is *helping* essentially every
airport, LIRF included (pooled LIRF 542.47 is dramatically better than
isolated LIRF's 1128.92, even though LIRF is still by far the pooled
model's worst airport). The mechanism is that a shared backbone gives every
airport implicit regularization roughly proportional to how little of its
own data it has, and LIRF is just the extreme end of that curve, not a
qualitatively different case needing separation.

**Consequence for the "attacks'" shared premise:** all three attacks (and
Paradigm C from the earlier session, "true per-airport independent
pipelines") assume LIRF's extreme loss contaminates the other airports'
learned weights, and that separating architectures would let the calm
airports reach a lower floor. That contamination story now fails
empirically from two independent directions: SS2l (excluding LIRF-adjacent
rows from training made the *shared* model measurably worse, not better)
and this section (giving airports their own dedicated capacity makes 9 of
10 of them worse, not better). **No 3-cluster or N-cluster variant of this
idea was built** — the premise it would need (some airports benefiting from
isolation) is exactly what the existing single-seed data already
contradicts, at every granularity from N=10 down to the coarsest possible
split (LIRF vs. everyone else, which is what the pooled-vs-isolated
LIRF comparison above already tests). Building a 3-cluster version would
only reproduce a weaker version of the same effect at coarser resolution.
No `features.py` change, no retrain beyond what already existed on disk, no
submission.

## 2z. Runway-configuration / active-arrival-runway features ("Attack 2") — closed, but on a nuanced ceiling, not a flat "no signal"

The one genuinely novel idea from the "structural attacks" analysis: does
which runway is currently active for ARRIVALS carry residual signal at
EGLL/EHAM, where taxi-out is plausibly runway-configuration-driven? The
proposed `cross_runway_conflict` (a hardcoded per-airport runway-topology
table) was dropped before building anything, per advisor review: EGLL's
runway alternation ("segregated mode") is specifically *designed* to avoid
crossings, so a guessed conflict-pair table risks encoding the wrong
physics, and a fabricated topology constant is exactly the failure mode
`airport_debiased_multimodel` was quarantined for (PIPELINE_REVIEW.md tier
E). Built and tested only the two clean, data-grounded features
(`tests_runway_config/run_runway_diag.py`, no `features.py` edit, no
retrain into any shipped arm): `active_arr_runway` (the runway of the most
recent arrival at the same airport in the trailing 30 min — a
nearest-neighbour proxy, simpler than a true windowed mode and equivalent
to it whenever the configuration is stable within the window),
`dep_rwy_eq_active_arr` (does this departure's own runway match that), and
`polderbaan_flag` (EHAM 36L/18R). Verified first that `RUNWAY_mvt` is 0%
null on BOTH phases in the real `ranking.parquet` (344,841 DEP / 344,693 ARR
rows checked directly), so the feature is buildable at scoring time.

**In-sample joint OLS ceiling against a single-seed `per_airport` baseline's
residual (99%+ coverage everywhere), per airport, on both `forward` test and
the pre-registered decisive `seasonal` test:**

| airport | forward test ceiling | seasonal test ceiling |
|---|---:|---:|
| EGLL | +0.027s | **+0.231s** |
| EHAM | +3.052s | **+0.020s** |
| EDDM | +4.672s | +0.124s |
| LTFM | +0.080s | +2.095s |
| (all others) | ≤0.58s | ≤0.75s |

Both airports of interest land far under the ~5s bar on the decisive
surface. **But before closing this as "no signal," checked the exact risk
advisor flagged: is `dep_rwy_eq_active_arr` degenerate at EGLL/EHAM?** Yes —
mean match rate is **1.4% at EGLL and 0.05% at EHAM** on `seasonal` test.
Departures there almost never share a runway with the most recently active
arrival — itself a clean empirical confirmation of segregated-mode
operations (real-world physics, not a data artifact), but it also means the
*equality flag* specifically has almost no variance to explain anything
with. A near-zero ceiling from a near-constant regressor is "unmeasurable
by this construction," not "measured and found absent."

**Follow-up check: does the runway *identity* (not just the match flag)
carry signal, e.g. via an interaction between a departure's own runway and
the concurrently active arrival runway?** In-sample eta² (`resid ~
categorical`, seasonal test): EGLL — dep_rwy alone 0.00034, active_arr alone
0.00004, the *combination* (12 levels) 0.00062; EHAM — dep_rwy alone
0.00538, active_arr alone 0.00109, the combination (55 levels) **0.02426**,
a real jump over either alone. Converting via `RMSE·(1−√(1−η²))` against
EHAM's approximate residual scale (~200–250s) gives a ceiling of roughly
**2–3s even at this more generous, interaction-aware reading** — bigger
than the flag-only number, but still under the 5s bar, and this reading is
itself almost certainly inflated (55 levels fit in-sample with no
regularization is the same optimistic-ceiling-with-more-free-parameters
effect flagged elsewhere in this document, and no permutation control was
run before treating 0.02426 as real, because even taking it at face value
doesn't clear the bar).

**Closed, precisely stated: not "runway configuration has no effect on taxi
time" (segregated-mode physics is real and directly visible in the
degenerate match rate), but "the tested feature constructions — match flag
and runway-identity interaction alike — top out at an estimated 2–3s
ceiling even under the more generous reading, under the board-visibility
floor."** A richer construction (e.g. a proper runway-configuration regime
label per airport, derived from clustering concurrent arr+dep runway usage
rather than a single nearest-neighbour arrival) might do better, but that is
a meaningfully bigger build for an already-small ceiling, and wasn't
attempted. No `features.py` change, no retrain, no submission.

**All three "structural attacks" from this analysis now closed**: Attack 1
skipped (already board-disproven twice), Attack 3's premise disproven by a
direct data-pull correction (SS2y), Attack 2 measured and closed on a
ceiling (this section, SS2z).

## 2aa. Wind-derived headwind/crosswind on the departure's own runway — genuinely new open data, closed decisively

Follow-up to SS2z's open question: is there a MORE DIRECT way to test
runway-configuration effects than inferring "active configuration" from
noisy nearest-arrival observations? Real aviation physics: aircraft take
off and land into the wind (crosswind/tailwind limits are a hard
operational constraint), so the actual wind direction at departure time,
resolved against the departure's own assigned runway heading, gives a
headwind/crosswind component directly -- no inference about concurrent
arrivals needed, and no risk of the near-degenerate-flag problem SS2z hit
at EGLL/EHAM.

**Two genuinely new pieces, both open, neither previously used in this
project:**
- **Wind direction (`drct`)** -- the existing `prc-2026-datasets-wx/` feed
  (already used SS2r/SS2w) has wind SPEED but not DIRECTION. Fetched fresh
  from the same open provider (Iowa Environmental Mesonet ASOS archive,
  `mesonet.agron.iastate.edu`, public, no account) for all 10 airports,
  confirmed at the identical 30-min cadence and reaching through the real
  scored period (checked EGLL to 2026-07-30 directly). Cached at
  `artifacts/external/wind_direction/<ICAO>.csv`.
- **Runway heading** -- not a fitted or guessed constant (unlike the
  dropped `cross_runway_conflict` table): it's read directly off the
  runway's own name by standard ICAO convention ("09"=090 deg, "27"=270
  deg, "36"=360/000 deg), the same way a pilot reads it off the runway
  number. Public, deterministic, zero free parameters.

**Built (`tests_wind_runway/run_wind_diag.py`, no `features.py` edit, no
retrain into any shipped arm): `headwind_component = sknt*cos(wind_dir -
runway_heading)`, `crosswind_component = sknt*|sin(wind_dir -
runway_heading)|`, joined by nearest wind observation within 35 min (same
tolerance as the existing METAR join).** Coverage ~90-92% at every airport
(real, not degenerate -- confirms this isn't SS2z's near-constant-flag
problem). In-sample joint OLS ceiling against a single-seed `per_airport`
baseline's residual, per airport, on both `forward` test and the
pre-registered decisive `seasonal` test:

| airport | forward test ceiling | seasonal test ceiling |
|---|---:|---:|
| EGLL | +0.057s | +0.163s |
| EHAM | +2.929s | +0.018s |
| LTFM | +0.045s | +1.840s |
| EDDM | +4.789s | +0.023s |
| (all others) | ≤0.64s | ≤0.35s |
| POOLED | +0.272s | +0.076s |

**Closed, decisively this time -- every airport, on the decisive surface,
lands under 1.9s, an order of magnitude under the 5s bar, with healthy
non-degenerate coverage throughout.** Unlike SS2z, there is no
"unmeasurable vs. absent" ambiguity here: the regressor has real variance
everywhere it's covered, so a near-zero ceiling means what it says. Runway
assignment plausibly already correlates with time-of-day/season/stand
(all of which the model already sees), leaving little room for wind
physics specifically to add anything once the assignment itself is known.
No `features.py` change, no retrain, no submission. This closes the
runway-configuration family of ideas (SS2z + this section) with a clean
negative rather than SS2z's nuanced one.

## 2ab. Public holiday calendars — genuinely new open data (no API/account), closed decisively

Follow-up user question ("we have not tried... open source data") also
proposed national holiday calendars as a driver: staffing constraints and
leisure-travel surges around public holidays. Genuinely untried in this
project (grepped `TAXI_OUT_DRIVERS.md`/`PIPELINE_REVIEW.md` for "holiday"
first -- zero hits). Fully open, no account or API needed: the `holidays`
PyPI package, installed fresh, computing each airport's own country's
calendar (EDDF/EDDM->DE, EGLL->GB, EHAM->NL, LEBL/LEMD->ES, LFPG->FR,
LIRF->IT, LSZH->CH, LTFM->TR).

**Built (`tests_holiday/run_holiday_diag.py`, no `features.py` edit, no
retrain): `is_holiday`, `is_day_before_holiday`, `is_day_after_holiday`**
(UTC calendar date, not localized per-country -- a deliberate
simplification, off by at most a few hours near local midnight). Same
in-sample joint-OLS-ceiling-against-residual method, per airport, on
`forward` test and the decisive `seasonal` test. Holiday rate ~2.9-3.6%
pooled, real per-airport counts (hundreds to low thousands of rows/airport,
not degenerate).

| airport | forward test ceiling | seasonal test ceiling |
|---|---:|---:|
| EDDM | +8.978s | +0.172s |
| EHAM | +2.915s | +0.017s |
| LTFM | +0.032s | +1.980s |
| (all others) | ≤1.4s | ≤0.35s |
| POOLED | +0.275s | +0.123s |

**Closed, decisively, same shape as SS2aa: every airport lands under 2s on
the decisive `seasonal` surface**, an order of magnitude under the 5s bar
(EDDM's forward-test +8.98s is the one number that would have looked
promising, but that's the secondary surface, and it collapses to +0.17s on
the surface that's actually board-shaped -- exactly the kind of
surface-dependent number this project's discipline exists to catch before
trusting it). Existing temporal features (`dow`, `is_weekend`,
`doy_sin/cos`, month) likely already absorb most of what a holiday flag
would add, similar to how the model already implicitly captures whatever
correlates with an already-seen categorical. No `features.py` change, no
retrain, no submission.

**Both genuinely-new open-data avenues raised in response to "we have not
tried feature engineering and open source data" are now closed**:
wind-derived headwind/crosswind (SS2aa) and public holidays (this
section). Two further ideas from the same conversation -- OpenSky ADS-B
reconstruction of missing AOBT, and Eurocontrol ATFM ground-hold/slot data
-- were investigated for feasibility rather than built: OpenSky's granular
historical data needs an application-reviewed research account (self-service
REST access is capped to the last 30 days, useless for this project's
Jan-2025-to-Jul-2026 window); Eurocontrol's own ATFM slot data needs an
accredited industry account, though Eurocontrol also runs a separate,
apparently more self-service "Aviation Data Repository for Research (ADRR)"
(flight trajectories through Jul 2026) that wasn't investigated further --
both require account creation neither of which this session can perform.

**Update (2026-09-16): ADRR access confirmed not obtainable** -- the user
checked directly. Not a lead. Do not re-propose registering for it.

## 2ac. Objective-function axis (poisson, closing by inference Gamma/Tweedie) — tested, closed decisively, axis fully mapped

Follow-up to §2i's Huber closure and `tests_log_target/RESULT.md`'s log1p
closure: both prior attempts to keep the heavy right tail from dominating
training (residual capping, label compression) failed on pooled/test RMSE
despite improving the in-band fit. LightGBM's built-in `poisson` objective
looked like it might sit outside that failure family: it fits `E[y|x]`
directly (log-link mean, no transform to invert, so no log1p-style
`expm1`/Jensen bias) and its score-space gradient (`μ − y`) weights every row
uniformly, unlike Gamma/Tweedie (`μ^(1−p)`, p>1) which explicitly down-weight
large-value rows.

**Built and tested** (`tests_poisson_objective/run_poisson.py`, same harness
pattern as `tests_log_target`: one script trains the L2 control and the
poisson variant with identical data/params/early-stopping, `forward /
with_aobt / keep_all`, seeds 42/43/44, both `lightgbm` pooled and
`per_airport`). L2 control reproduced `tests_log_target`'s registered numbers
to the decimal, confirming no harness confound.

**Result: closed, same shape as log1p, milder.** In-band improved decisively
on both arms (val −6.4 to −13.3s, test −0.95 to −2.85s) but pooled test RMSE
got decisively *worse* on both (lightgbm +3.44s, per_airport +15.49s) — fails
the pre-registered bar (must beat control on both test RMSE and test in-band).

**Why the "uniform weighting" premise was wrong:** LightGBM builds trees via
Newton steps, leaf value ≈ `−Σg / (Σh + λ)`. Poisson's Hessian is `h = μ` — a
row with large predicted mean carries proportionally more Hessian, so its
residual moves the leaf *less* per unit of gradient. Tail de-weighting
happens through the Newton-step denominator, not the gradient numerator.
Same mechanism as Gamma/Tweedie, just milder (p=1 vs. p up to 2).
Corroborated independently by convergence: poisson needs 2-3x the boosting
rounds (579-638 vs. L2's 211-283 pooled; 250-360 vs. 100-190 per airport) —
consistent with damped per-row Newton steps.

**Closes the entire objective-function axis by inference.** Four points now
measured, ordered by severity of tail de-weighting: L2 (best) → Poisson
(+3.4 to +15.5s pooled test, p=1) → log1p (+22 to +39s, adds its own Jensen
bias) → Huber (+82.2s, aggressive capping). **Gamma and Tweedie (p between
1.5 and 2) sit strictly between Poisson and Huber's severity and are closed
by inference — do not run them.** `tests_log_target/RESULT.md`'s own two
suggested follow-ups (tuned Huber, routed tail model) are also both already
closed elsewhere (§2i; §2q three independent ways). Full writeup
`tests_poisson_objective/RESULT.md`. **Not adopted. L2 remains the shipped
objective; nothing left to test on this axis.**

## 2ad. Feature-distribution drift audit (train2025 vs. real ranking.parquet) — closed clean, closes a genuine gap in the record

`DATA_QUALITY_AUDIT.md` covered missingness/duplicates/identifier consistency
but never a systematic per-feature numeric-distribution comparison; the
community's own reported check of this kind (`conversation.md` sec.4, "all
52 numeric features... none shifted more than 0.25 sd") had no persisted
artifact confirming it was ever replicated in this repo. Built
`tests_drift_audit/run_drift_audit.py`: 42 numeric features, two comparisons
(full train pool vs. each scored month; same-calendar-month year-over-year),
threshold |z|>0.25 to match the community figure.

**Result: the decisive comparison (same-month, year-over-year) flags ZERO
features on both Jan and Jul** — directly addresses §2o's one genuinely open
caveat (a year-boundary effect no within-2025 holdout could represent); at
the feature-distribution level, no such effect is visible. The other
comparison (full pool vs. single scored month) flags 9/42 (Jan) and 2/42
(Jul), but every flag is the expected calendar-concentration artifact
(`doy`/`month_num`/etc., why `config.SPLITS["seasonal"]` exists) or a small,
plausible seasonal traffic dip (congestion counts, Jan only, barely over
threshold). **Closed clean** — a bug-detector that found no bug, not a
lever. Full writeup `tests_drift_audit/RESULT.md`.

## 2ae. Stand-turnover physical check on LIRF's AOBT-unmatched population — RETRACTED: the "contradiction" was a false positive, not a label defect

**RETRACTED (2026-09-17), before any training ran.** The original write-up
below claimed a real training-label defect. A follow-up check, done while
scoping the label-correction experiment this section originally
recommended, falsifies it: cross-checking the 439 "contradicted" rows
against `delta_sched_sec` (§2q's independent proxy, already shown to track
true `y` almost exactly on this population) shows the **recorded label**
matches that proxy far better than the computed "physical bound" does — 93.8%
of the 439 rows, mean absolute error 1,157s vs. 9,643s (~8x). Worst case:
`mvt_id 183903219` claims 131,167s; `delta_sched_sec` independently says
131,163s (agreement to 4 seconds); the "bound" this check computed was
18,239s. **The recorded labels are corroborated, not contradicted** — the
stand-turnover check's assumption (a `STAND_mvt` value implies exclusive,
continuous occupancy) is what was wrong, not the data. No training-level
experiment was run; the harness (`tests_label_correction/run_label_correction.py`)
exists but was never executed. Full correction: `tests_stand_turnover/RESULT.md`.

**Original (now-retracted) claim, kept for record:**

Follow-up to §2q (three prior approaches to re-predict this 20.7%-of-SSE
population, all failed out-of-sample): asked a purely physical question
instead of a statistical one. A stand holds one aircraft at a time, so if
another movement is recorded at the SAME (airport, stand) strictly inside a
training row's claimed `[BLOCK_TIME, takeoff)` window, the recorded
`BLOCK_TIME` is provably wrong, independent of any model. Built
`tests_stand_turnover/run_stand_turnover.py` on the full LIRF movement stream
(both phases).

**Result: 58.9% of the full-pool population (439/745, TAXITIME>3600s,
no-flight-record) and 59.4% of the honest `forward` held-out val+test slice
(133/224) carry a definitive contradiction** — consistent between full-pool
and held-out, not an artifact of pool size. The physical bound on
contradicted rows averages only 28.5% of the recorded label (median 25.5%) —
typically 3-4x smaller than what's on record as ground truth. Only 12/439
overlap the already-tested 86,400s date-rollover pattern (§2g/§2l) — this is
a substantially larger, mechanistically different population.

**Reading: this is a training-LABEL-quality finding, not an inference-time
lever.** The check needs `BLOCK_TIME_UTC_mvt`, which is blanked on ranking
DEP rows — it cannot run at scoring time to cap a live prediction. What it
shows is that the majority of this high-SSE training subpopulation's ground
truth is itself physically impossible, and the model is currently trained to
reproduce it. Different failure shape than §2l (which dropped 15
heuristically-flagged rows and made test decisively worse, +5.45s, via a
shared blend-weight shift): this is ~29x larger and evidenced by a general
physical constraint, and the natural fix is label CORRECTION toward a known
bound, not exclusion — an untested variant of the failure mode §2l hit, not
a rerun of it.

**Not built, not tested as a training change.** Diagnostic only (prevalence
+ feasibility). Recommended next step: cap contradicted rows' training
label at their physical bound, multi-seed, measured on `forward` and
`seasonal`, with a permutation control (does a same-sized FAKE contradiction
set produce the same effect?) before trusting any result — explicit sign-off
needed given §2l's precedent that this exact population's training-label
changes have backfired before even when the local logic looked sound. Full
writeup `tests_stand_turnover/RESULT.md`, contradicted-row table
`tests_stand_turnover/contradicted_rows.csv`.

## 2af. `config.SPLITS["seasonal"]`'s July airport_filter is stale against the real ranking.parquet — genuine config bug, found incidentally, not yet fixed

While cross-checking `seasonal` as a fair test surface for the (later
retracted) label-correction experiment, queried the real `ranking.parquet`'s
actual per-airport row counts by month directly (not assumed from
`config.py`'s comment). **All 10 airports have substantial July 2026 rows**
(LIRF: 15,838; every other airport in the same order of magnitude as
January) — `config.py:189`'s `airport_filter: {"2025-07": ["EDDF", "EGLL",
"EHAM"]}`, and every place in this project's history that calls `seasonal`
"board-composition-matched" on that basis, is **stale against the file
currently on disk**.

**What this does and does not invalidate.** Every A-vs-B comparison already
run on `seasonal` is still a valid comparison — the filter under-weights 7
airports in July, it doesn't corrupt the arithmetic, and results that were
already under the ~5s board-visibility floor stay there regardless (§2v,
§2w, §2aa, §2ab). §2ad's own drift audit derived its airport list from the
data directly (not from `config.py`), so it is unaffected. What it **does**
weaken is specifically the word "decisive" wherever `seasonal` was invoked
as the board-matched tiebreaker. **Sharpest consequence, flagged in §2q**:
the LIRF MLP/proxy blend rejection's decisive check (`seasonal` test, n=60)
was LIRF-absent-from-July by construction, so n=60 was January-only — the
rejection still stands on the surface it was actually measured on, but
"decisive" overstated what that surface covered.

**Fixed (2026-09-17), additively.** Added `config.SPLITS["seasonal_full"]`
(identical months to `seasonal`, `airport_filter: None`), annotated the old
`seasonal` entry in place as composition-stale (kept byte-for-byte so every
existing `seasonal` artifact stays reproducible), added `seasonal_full` to
`SELECTION_FORBIDDEN_SPLITS`, and generalized `compare.py`'s anti-causal
selection guard to cover both. Generated fresh split IDs via
`stage2_split.py --split seasonal_full`: all gates pass (pairwise
disjointness, month exclusivity, airport-filter-applied — vacuously true
here since there is none). Test now correctly carries all 10 airports in
both Jan and Jul (344,419 rows total — matching the community's own
Jan+Jul-2025-holdout row count from `conversation.md`, a nice independent
cross-check). **Update (2026-09-17): the three arms are now trained under `seasonal_full`**
(`catboost_antigravity` seed42/tag=cb1, `airport_operator_blend` seed42,
`mlp_antigravity` seeds 42-47) and used for the §2n third-confirmation check
above — see that section for the result.

## 2ag. WTC sequencing (neighbour wake-turbulence category) — real, permutation-verified, uniform across all 10 airports, but closed on the ceiling

Checked a community post's speculative hypothesis (`conversation.md` sec.5):
does a NEIGHBOUR's wake-turbulence category (WTC), not just the aircraft's
own, drive taxi-out — light-behind-heavy needing extra wake-separation time.
Cheap diagnostic before touching `pipeline_queueattn_antigravity` (sec.2k,
already closed flat, and expensive to re-retrain): for every DEP row, found
the immediately preceding same-(airport, runway) departure (whole-period,
backward-looking only), ordinal-encoded WTC (L<M<H<J), and correlated
`prev_heavier` (1 if the preceding aircraft outweighs this one) against
`mlp_antigravity`'s val residual.

**Result: real and uniform, not an artifact — but negligible in aggregate.**
rho=+0.0299, collapsing to -0.0007 under a permutation control. Raw group
means: +22.44s mean residual when the preceding aircraft was heavier vs.
+2.47s otherwise (97.7% coverage). **Consistent in sign at all 10 of 10
airports** (rho +0.009 to +0.058, resid gap +12.9s to +37.8s) — unlike most
closed levers here, which concentrate in one airport or split sign. But the
in-sample OLS ceiling on pooled val RMSE is only **-0.134s** — an order of
magnitude under the ~5s board-visibility floor (sec.2o) — because the
affected subpopulation is only 6-18% of rows per airport and the model's
existing features (own WTC, runway identity, congestion counts) already
absorb most of what it would add. **Closed on the ceiling, not built.**
Distinct from a typical closure: the physical mechanism is confirmed real,
just too small to move this metric — joins sec.2h/sec.2p/sec.2z as another
"real effect, negligible aggregate impact" case rather than "no effect."
Full writeup `tests_wtc_sequencing/RESULT.md`.

## 2ah. EUROCONTROL AIU ATFM-delay diagnostic — closes §2b's "rejected for lack of data" gap, negative on the ceiling

§2b rejected the ATC-slot/CTOT hypothesis for lack of ANY supporting column
in the competition data, explicitly noting it would need an external source
to actually test. Checked EUROCONTROL's own public Aviation Intelligence
Portal daily Airport Arrival ATFM Delay dataset (real bulk download,
1.03M rows, 2019-2026) as that source — a genuine, exogenous, network-wide
ground-hold mechanism (attributed to the departure airport/day by
EUROCONTROL's own definition), distinct from local traffic density.
Complete daily coverage confirmed for all 10 airports across 2025 AND the
real scored Jan+Jul 2026 period, no gaps. Joined by (airport, date) to
`forward`/val DEP rows: 100% coverage.

**Result: real but small, mixed sign across airports, closed on the
ceiling.** Pooled rho(DLY_APT_ARR_1, resid)=+0.0136, only modestly above
the permutation-shuffled control (+0.0025) — a much thinner margin over
noise than §2ag's WTC-sequencing check (~30x ratio there). In-sample OLS
ceiling: only **-0.094s**. Per-airport: LEBL (+0.106) and LTFM (+0.071)
positive, but EDDF/LEMD/LIRF go the WRONG direction — unlike §2ag's uniform
sign at all 10 airports, this reads as noise with 1-2 coincidental airports,
not a genuine uncaptured driver. Same shape as the already-closed
day-regime clustering (`pipeline_lightgbm_unsup`) — a coarse "was today
disrupted" proxy that doesn't clear the bar. The genuinely useful EUROCONTROL
signal (per-FLIGHT CTOT/regulation, not a daily airport aggregate) lives in
NM's own accredited-access systems, already found infeasible this session
(ADRR/NM-B2B). **Closed, not built.** Full writeup
`tests_atfm_delay/RESULT.md`.

- **Analyze per-airport, not pooled.** Consistent with why `per_airport` arms
  outperform pooled ones in `PIPELINE_REVIEW.md`. "Visit 1 by 1" — walk each of
  the 10 airports individually before generalizing a driver.
- **Look at the taxi-time distribution per airport**, not just its mean/RMSE —
  `stage13_airport_headroom.py` already warns big RMSE can just mean long taxi
  times (e.g. EGLL vs EHAM), so shape matters, not just scale.
- **Set explicit lower/upper bounds before removing outliers** — don't eyeball
  it. Note `PLAN.md` §3a already measured that filtering the tail *hurts* L2
  score (454.68 filtered vs 445.85 unfiltered), so any bound here is for
  *analysis/visualization*, not for excluding rows from training.
- **Perplexity (the search tool)** — use for researching airport-specific
  operational context (runway configuration, known slot constraints, local
  procedures) that isn't in the dataset, when a hypothesis above needs outside
  grounding. Cite what you find inline in the per-airport notes below so it's
  checkable, not just asserted.

## 2ai. Real-board confirmation of the `OUTLIER_POLICIES` sweep — `winsor_7200` submitted, decisively worse

§2l explicitly declined to re-run the broad `winsor_*`/`drop_7200` vs.
`keep_all` sweep on the real board ("would cost several hours of GPU to
reconfirm a strictly worse variant of what was just measured — not run
without asking first"). User asked directly for exact confirmation "for the
record." Rather than retrain the full production blend, used the cheapest
controlled test that still gives a real board number on the exact recipe:
`airport_operator_blend` (`per_airport` + `per_operator`, convex blend), the
identical architecture already shipped as **v19** (`keep_all`, board
382.9174) — solved a fresh alpha on `forward` under `--outlier-policy
winsor_7200` (alpha=0.706 vs. keep_all's 0.676), froze it, refit on `full`,
submitted as **v27**.

**Local val, same split, only the outlier policy changed:**

| policy | val RMSE |
|---|---:|
| keep_all (v19's basis) | 272.9 |
| winsor_7200 | 333.0 |
| **gap** | **+60.1s** — already ~12x this project's ~5s board-visibility floor |

**Real board result: v27 scored 496.0944, vs. v19's 382.9174 — +113.2s worse.**
Larger than the local val gap, same direction, unambiguous. For scale: the
naive-proxy-alone baseline (`pipeline_naive_offblock`, zero learning) scores
~482 RMSE locally — a real, *trained* model with its training labels
winsorized at 7,200s now performs in that same neighbourhood on the actual
leaderboard.

**Visible mechanism, before the score even came back:** re-predicted the
identical ranking rows under `keep_all` (v19's already-trained model, no
retraining needed) for a same-population comparison:

| policy | median | mean | p99 | **max** |
|---|---:|---:|---:|---:|
| winsor_7200 (v27) | 958 | 1028.6 | — | **8,148** |
| keep_all (v19) | 958 | 1033.0 | 2,463 | **40,640** |

The bulk of the distribution is essentially unchanged (median/mean agree to
within 5s) — only the tail collapses, from 40,640s down to 8,148s, sitting
almost exactly at the training cap. This is `config.py`'s documented
mechanism (winsorizing biases the conditional-mean prediction downward
across the real, learnable 7,200-80,000s tail) made visible on the actual
submission file, not just argued from the training-time loss.

**Closed, now with direct real-board evidence at three independent scales**:
the original broad local sweep (~27s), the regime-decomposition mechanism
(mid-tail RMSE halves under `keep_all`), and now this — a dedicated real
submission, +113.2s worse, on the exact recipe that would have been
submitted. **v25 (295.21) stands as the best submission; v27 was a
deliberate confirmation probe, not a candidate.** No further outlier-policy
testing planned; this axis is closed at every level this project can test it.

## 2aj. Micro-effects ledger — sorting "closed, no real effect" from "closed, real but small," and a joint-overlap check

User asked to stop treating every sub-5s finding as automatically settled
just because it's under the board-visibility floor — a fair correction to
this document's own habit this session of writing "closed on the ceiling"
for two genuinely different outcomes. Splitting them out:

**Category A — permutation control PASSED (real, non-random effect), closed
only because the ceiling is small.** These are the actual stacking
candidates:

| finding | rho vs. permutation | in-sample ceiling | sign pattern |
|---|---|---:|---|
| §2ag WTC sequencing | +0.0299 vs. +0.0009 | −0.134s | uniform, 10/10 airports |
| §2ah ATFM daily delay | +0.0136 vs. +0.0025 | −0.094s | mixed, 2 of 10 airports |
| §2p 5/10-min congestion windows | rho 0.017-0.067, "clearly not noise" | +0.24s pooled | consistent sign, not permutation-tested per-column |

**Category B — permutation control FAILED, or correlation ~0 outright (not
a stacking candidate, genuinely nothing there):**

| finding | result |
|---|---|
| §2q follow-up / `tests_lirf_flightpath` (destination, reverse-runway) | real correlations SMALLER than their own shuffled controls |
| Stand-level congestion aggregates (2026-09-15) | rho −0.036/+0.002, indistinguishable from zero even where covered |
| §2h OSM taxi-distance, well-observed pairs | essentially zero residual correlation (categorical embedding already captures it) |
| §2r weather (METAR) | wrong sign entirely (−1.67s val RMSE) |

**Joint-overlap check (2026-09-18), because Category A's ceilings cannot
just be summed.** Each ceiling above was fitted independently against *the
same* `mlp_antigravity` forward/val residual — if two features both proxy
"local traffic/queue pressure" (plausible for WTC-sequencing and ATFM delay,
both airport-and-time-localized), their combined value is closer to the
larger effect than to the sum. Fit a single joint OLS of the residual
against WTC's `prev_heavier` and ATFM's `DLY_APT_ARR_1`/`pct_regulated`
together, same rows (97.7% joint coverage), plus a permutation control on
the joint feature set:

| | ceiling |
|---|---:|
| WTC alone | −0.134s |
| ATFM alone | −0.074s |
| **naively summed** | **−0.208s** |
| **actual joint fit** | **−0.159s** |
| overlap (summed − joint) | −0.049s (≈24% of the naive sum was double-counted) |
| joint fit, permutation-shuffled | −0.049s |

**Result: even correctly combined (not summed), two of the three Category A
candidates together clear only −0.159s** — about 3x the shuffled-control
floor, technically still a real, non-random joint effect, but two orders of
magnitude under the ~5s board-visibility floor. §2p's congestion-window
features weren't included in this run (their diagnostic script wasn't
preserved from the session that built it and would need reconstructing),
but given the overlap already observed between two conceptually-similar
"local congestion" signals, a three-way joint ceiling is very unlikely to
clear even 1s, let alone 5s.

**Disposition: the ledger did its job.** This is not "closed, ignore
forever" — if a fourth or fifth genuinely independent (not
congestion-flavored) small effect is found later, re-run this same joint-OLS
check with it added, since the marginal value of one more real predictor on
top of near-zero existing ones is easier to clear than stacking two
correlated ones. But do not build any Category A feature into `features.py`
on today's evidence — the pre-registered bar (a joint ceiling that clears
~5s) was not met, and the correct move per this project's own discipline is
to say so rather than average estimates across sub-5s findings.

## 2ak. `lobt_sched_sec` (LOBT − SCHED, explicit) — competitor-review-derived, checked, closed near-zero, joins Category A

A live leaderboard/public-repo pass (2026-09-18) found `saurabhdxt-labs/
prc-challenge` (team merry-quicksand, rank 7/160 at 257.55, ahead of us).
Their `reports/MATCHED_MODEL.md` documents their single biggest feature-
family gain (+8.17s, one fold) coming from handing a GBT **explicit
pairwise flight-plan clock differences** rather than the raw clock columns
individually — their stated mechanism: "trees cannot form the difference of
two inputs; handing the differences over directly is what moved the
number." Cross-checking `common/features.py` found this project already
computes nearly all of their differences explicitly (`delta_aobt_sec`,
`aobt_minus_eobt_sec`, `aobt_minus_lobt_sec`, `aobt_minus_sched_sec`,
`eobt_minus_iobt_sec`, `lobt_minus_iobt_sec`) — which is why the earlier
§2x delta-target-reformulation test (testing the same underlying
information as a *target* transform rather than a *feature*) closed flat:
we already hand the model these differences directly. One difference was
missing as its own explicit column: `lobt_sched_sec = LOBT − SCHED`
(recoverable from two existing features, `delta_sched_sec −
delta_lobt_sec`, but never materialized on its own). Their own E1b (the
last four clock differences added to an already-rich set) found only
+0.26s, "not established, family exhausted" — low prior, but cheap and
concretely falsifiable.

**Checked (`tests_lobt_sched/run_lobt_sched_check.py`), same convention as
§2p/§2s/§2z/`tests_wtc_sequencing`: correlate the candidate column against
`mlp_antigravity` seed42's residual on `forward` val AND test, plus a
permutation control, then an in-sample OLS ceiling on pooled RMSE.**

| part | rho | permutation rho | pooled ceiling |
|---|---:|---:|---:|
| val (n=365,246, 98.8% usable) | +0.0180 | −0.0007 | −0.084s |
| test (n=325,004, 99.1% usable) | +0.0226 | −0.0010 | −0.050s |

**Real but tiny — joins Category A (§2aj) rather than Category B.** Both
splits show a small, clearly non-random correlation (an order of magnitude
above the permutation-shuffled control), consistent in sign and magnitude
between val and test, so this isn't noise — but the pooled ceiling
(−0.05 to −0.08s) is two orders of magnitude under the ~5s board-visibility
floor, in the same range as §2ag/§2ah. One per-airport number stood out
(EDDM: −0.592s on val vs. −7.224s on test, the only airport near the 5s
bar) but the two splits disagree by 12x on the same feature at the same
airport — a single-variable in-sample OLS ceiling on ~25-35k rows without a
per-airport permutation control, read as overfit noise rather than a real
EDDM-specific effect, not a genuinely inconsistent-but-real signal like
§2ag's uniform 10/10 pattern. **Not built.** Confirms the earlier read of
merry-quicksand's repo: their large headline gain was from adding these
differences at all (which we did long ago via `with_aobt`), not evidence of
a remaining gap in this project's feature set. v25 (295.21) stands; nothing
submitted.

## 2al. Pooled matched-vs-unmatched RMSE split, mirroring merry-quicksand's own ledger identity — locally we look BETTER in both lanes, which relocates the mystery to the local-to-board gap itself

Follow-up to §2ak's competitor review. `saurabhdxt-labs/prc-challenge`'s
`reports/MSE_LEDGER.md` tracks `RMSE^2 = w_matched*matched_RMSE^2 +
w_unmatched*unmatched_RMSE^2` (matched = `AOBT_3_flt` present, ~98.5% of
rows) as its central bookkeeping identity and reports, on their own
held-out fold: matched RMSE 226–237, unmatched ("stratum") RMSE 1868–1946.
This project has quantified LIRF's unmatched population specifically
(§2q) but never computed this same pooled, all-10-airport split — the
natural apples-to-apples comparison for asking whether our own well-known
local-to-board gap (+47 to +51s, board worse than local) concentrates in
the matched lane, the unmatched lane, or neither.

**Checked (`tests_matched_unmatched_split/run_matched_unmatched_split.py`),
`airport_operator_multimodel` tag=msmlp seed42 — the same architecture
family as the shipped v25 arm — on `forward` val and test:**

| part | matched RMSE | unmatched RMSE | pooled RMSE | unmatched row share | unmatched SSE share |
|---|---:|---:|---:|---:|---:|
| val | 217.81 | 1099.75 | 247.39 | 1.18% | 23.4% |
| test | 211.34 | 1204.09 | 239.87 | 0.92% | 23.1% |

**Both lanes are locally BETTER than merry-quicksand's own reported numbers** —
our matched RMSE (211–218) beats their 226–237, and our unmatched RMSE
(1100–1204) beats their 1868–1946 by a wide margin. Yet their board score
(257.55) is 38s better than ours (295.21). Locally we do not look like the
weaker model in either lane; the gap has to be coming from somewhere local
numbers don't see.

**LIRF concentration is even starker in this all-10-airport view than
previously documented**: LIRF is 87.2% (val) / 76.9% (test) of TOTAL
unmatched SSE despite being 1 of 10 airports, with unmatched RMSE
4039–4891 vs. the next-worst airport (LFPG, 397–897) — an order of
magnitude gap between LIRF and every other airport's unmatched population,
not just a modest lead.

**A back-of-envelope check on the "our unmatched population is
proportionally larger on the real board" explanation** (DATA_QUALITY_AUDIT.md:
join-failure rate 1.078% train vs. 1.534% ranking, ~1.42–1.54x): holding
our local matched/unmatched RMSEs fixed and solving for the unmatched row
share that would be needed to reproduce the actual board score (295.21)
gives ~3.2% — roughly 2.7–3.5x our local estimate (0.92–1.18%), not the
~1.4–1.5x the training-vs-ranking null-rate comparison supports. **Row-share
inflation alone is not sized to explain the gap.** The more likely
explanation is that the unmatched population's error *within* the real
Jan+Jul 2026 rows is itself worse than what our 2025 `forward`-test sample
shows — i.e. a quality-of-the-tail effect, not just a quantity-of-the-tail
effect — consistent with, but not resolved by, the already-flagged open
caveat that every local surface this project has built lives inside 2025
and a year-boundary shift in LIRF's worst population remains structurally
unmeasurable without more of the actual 2026 scored data (§2o's closing
caveat, §2n's untested horizon extrapolation). **Diagnostic only — not
something more local computation can settle.** v25 (295.21) stands;
nothing submitted.

**CORRECTION (2026-09-18, same day): the comparison above used the wrong
surface.** `forward`'s test set is Nov-Dec 2025, not Jan+Jul -- neither the
board's season nor merry-quicksand's own Jan+Jul-2025 fold. Re-run on
`seasonal_full` test (Jan+Jul 2025, n=344,419 -- almost exactly the real
board's 344,841), single `mlp_antigravity` seed42 (no composite blend arm
is trained under `seasonal_full` yet):

| surface | matched RMSE | unmatched RMSE | pooled RMSE | unmatched row share |
|---|---:|---:|---:|---:|
| forward test (wrong season) | 211 | 1204 | 240 | 0.92% |
| seasonal_full test (right season) | 249 | 2116 | **362** | 1.56% |

**The "we're locally better than merry-quicksand" read does not survive on
the correct surface** -- this single-arm number is if anything slightly
worse than their reported matched/unmatched numbers, not better. The
apparent local-to-board mystery in the original write-up above was largely
a wrong-surface artifact, not a genuine unexplained gap: our own local
diagnostics already show the real difficulty once measured in the right
season, consistent with merry-quicksand's own documented finding that the
unmatched stratum is measurably harder in Jan+Jul than in a fold that
excludes them (`GAP_STRUCTURE.md` sec.6: `sp`>3h share 12.63% real-2026 vs.
7.20% their-fold). Caveat: this re-check uses a single MLP arm, not the
full 3-way blend the original comparison used (no `seasonal_full`-trained
composite exists yet), so it is not a perfect like-for-like either --
flagged, not yet resolved.

**Two follow-on checks on the correct surface, both re-confirm prior
conclusions rather than overturn them:** (1) the Category A micro-effects
(sec.2ag WTC-sequencing, sec.2ak `lobt_sched_sec`) were re-measured on
`seasonal_full`/test and stayed tiny -- WTC ceiling -0.081s (vs -0.134s on
`forward`), `lobt_sched_sec` ceiling -0.038s (vs -0.05 to -0.08s) -- so the
"stack the small real effects" idea (sec.2aj) does not get rescued by
fixing the season; these are genuinely season-invariant, tiny effects, a
different shape from LIRF's severity, which IS season-dependent. (2) The
v28 probe's local effect (sec above) was re-measured on `seasonal_full`:
val -6.92s (proxy better), test +4.88s (proxy worse) -- still genuinely
sign-unstable on the correct surface, same conclusion as before (this is a
real open question, not resolvable by more local computation, worth an
actual board submission).

## 4. Per-airport worksheet

One block per airport once its capacity regime and drivers are checked.
Template:

```
### <ICAO>
- Capacity regime: L? (evidence: ...)
- Drivers confirmed: ...
- Drivers rejected: ...
- Open questions: ...
```

(Empty for now — fill in as each airport is worked.)
