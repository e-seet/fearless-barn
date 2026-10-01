# Round 7 (v48 candidate): ADS-B traffic features and corrector seed averaging — pre-registration

Written 2026-10-01, before any feature is computed.

**Board so far:**
- v46 = 272.2619 (best);
- v47 = 272.7321 lost: tabular columns in the corrector passed leave-one-month-out within 2025, then
  failed on the 2026 board. v46 is to be re-uploaded last.

**Lesson carried in:**
- a within-2025 holdout cannot see a 2025 → 2026 shift;
- so every new input must pass a **drift gate** (2025 vs 2026, same month) as well as the holdout;
- inputs stay movement-type.

## Option 1: traffic features (T)

Built from the tracks already fetched, with no new downloads. For each departure *i* at airport *a*
with takeoff *T_i*, other departures are counted only if they were matched (`features.py` strict
estimates). Every value is something an observer of the airport surface would know at *T_i*.

| feature | definition |
|---|---|
| `tr_n30` | matched departures at *a* that took off in [*T_i* − 30 min, *T_i*) |
| `tr_stop30`, `tr_lineup30`, `tr_taxi30`, `tr_gs30` | their mean `stop_s`, `lineup_s`, `e_move` (seen taxi time) and `med_gs` |
| `tr_ground_T` | matched departures *j* ≠ *i* on the ground and moving at *T_i*: started moving (*T_j* − `e_move_j`) ≤ *T_i* ≤ *T_j* |
| `tr_ground_T10` | the same at *T_i* − 600 s |

**Variant T:**
- the v46 inputs plus these 7 features;
- trained and applied on **all rows at the 7 airports**, not only rows with their own track;
- uncovered rows have NaN own-track features, which lets T also correct flights that v46 leaves
  untouched.

**T_perm (control):** T with only the traffic columns shuffled within airport.

## Option 2: corrector seed averaging (S)

v46's corrector (same inputs and rows) trained with seeds 42–46, predictions averaged. Everything else
is identical.

## Surface (both options)

- Leave one month out over Jan, Jul, Aug, Sep, Oct, Nov and Dec 2025, each with its held-out base,
  as in the interaction check.
- **A** = the v46 corrector design (seed 42), retrained per fold.
- Scored over **all rows** of the held-out month.

## Bars

**T PASSES only if all of:**
1. pooled RMSE: T < A;
2. T < A in at least 5 of the 7 months;
3. T < T_perm pooled;
4. **drift gate:** every traffic feature has a pooled standardized mean difference of at most 0.2
   between 2025 and 2026, in Jan and in Jul separately (2026 values computed the same way from the
   fetched 2026 days).

**S PASSES only if:**
1. pooled RMSE: S < A;
2. S < A in at least 5 of the 7 months.

**If both pass:**
- T is rebuilt with 5-seed averaging (TS);
- TS is used only if it beats T pooled.

## Build

- **v48** = v42 plus the winner, trained on all 7 months.
- The expectation is written before upload. A T-based v48 gets a deliberately cautious expectation,
  because of v47.
- If v48 scores worse than v46 (272.2619), v46 is re-uploaded last.

**If neither passes:** v46 is final (option 3).
