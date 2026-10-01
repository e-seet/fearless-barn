# Runway queue features (Strategy 1) and the routed-distance ceiling (Strategy 3) — pre-registration

Written 2026-10-02, before any feature is computed. Board: v48 = 272.1714 (best and last).

The proposal came from an outside analysis the user pasted. Strategy 2 (looser ADS-B matching) was
already measured and closed: every strict-match failure at EDDF and LEMD has **zero** ground points.

## Strategy 1: runway queue and runway mode

Built from the official movement records only (DEP and ARR rows of training and ranking), so the
inputs are not receiver-dependent. For departure *i* at airport *a*, on runway *R*, taking off at
*T*, using **only movements before *T***:

| feature | definition |
|---|---|
| `rq_dep15`, `rq_dep30` | departures from *a* on runway *R* in [*T* − 15 min, *T*) and [*T* − 30 min, *T*) |
| `rq_gap_prev` | seconds since the previous departure on *R* (capped at 3,600) |
| `rq_arr15` | arrivals on *R* in [*T* − 15 min, *T*) |
| `rm_dep_rw`, `rm_arr_rw` | distinct departure and arrival runways used at *a* in the 60 min before *T* |
| `rm_mixed` | 1 if *R* also had an arrival in the 60 min before *T* |

**Corrector R:**
- inputs: these 7 features plus `base`, airport and hour;
- the same LightGBM parameters as v46, trained on residuals of the held-out bases;
- applied to **every row the ADS-B corrector does not touch**: the uncovered rows at the 7 airports
  and all rows at LFPG, LTFM and LIRF (the 156k-flight population in 2026).

**R_perm (control):** the 7 features shuffled within airport, in training and scoring.

**Surface:** leave one month out over Jan, Jul, Aug, Sep, Oct, Nov and Dec 2025, as in round 7.
- **Reference A+:** base + the v46-design ADS-B corrector on its covered rows.
- **Candidate:** A+ plus R on the other rows.
- Scored on all rows of the held-out month.

**PASS requires all of:**
1. pooled RMSE: candidate < A+;
2. candidate < A+ in at least 5 of the 7 months;
3. candidate < R_perm-candidate, pooled;
4. **drift gate:** each feature has a pooled standardized mean difference of at most 0.2 between 2025
   and 2026, in Jan and in Jul, at the airports where R applies;
5. still better with the 5 rows of largest gain removed (the LIRF tail is in R's population).

Per-airport results are reported (LFPG, LTFM and LIRF in particular). The bar is pooled only.

**PASS →** v49 = v48 plus R on its rows; expectation written before upload; if it scores above
272.1714, v48 is re-uploaded last.

## Strategy 3: ceiling of a routed taxi distance (apt.dat), measured, not built

A routed stand → runway distance can only add information where the `stand_runway` identity is
unknown to the model.

**Ceiling:**
- rows of the Jan + Jul 2025 held-out set (seasonal_full test) whose `stand_runway` pair never occurs
  in that split's training data;
- their share of rows and of squared error;
- the pooled gain if their RMSE fell to the RMSE of seen-pair rows at the same airport, which is an
  upper bound that even a perfect distance feature cannot exceed;
- the same for 2026: the share of ranking rows with an unseen pair (against all of 2025).

**Decision:**
- **Ceiling < 0.5 s pooled:** Strategy 3 is closed as not worth building before the deadline.
- **Otherwise:** it is reported to the user with a build estimate.
