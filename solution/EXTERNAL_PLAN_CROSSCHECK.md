# Cross-check against an external 16-point strategy write-up (2026-09-15)

The user shared a detailed, independently-written strategy document for this
exact competition (taxi-out time prediction, DC2026) and asked which parts
are already done in this pipeline, with the specific finding for each. This
file is that mapping. Section numbers match the source document's own
numbering. Verdict is one of: **Done**, **Done — closed**, **Partial**,
**Not done**, **Contradicted**.

---

### 1. Problem definition (taxi-out = takeoff − off-block, scored by RMSE)

**Done.** Exactly this pipeline's setup since before this session began —
`config.TARGET`, `common/metrics.py:rmse`, the whole `forward`/`seasonal`
split design.

### 2. Flight characteristics (aircraft type, WTC, operator, market segment,
flight type, ADEP/ADES, scheduled time, callsign, IFR/VFR)

**Done**, with one deliberate exclusion. All of `AIRCRAFT_TYPE_mvt`,
`AIRCRAFT_OPERATOR_flt`, `MARKET_SEGMENT_flt`, `FLIGHT_TYPE_flt`,
`ADEP_mvt`/`ADES_mvt`, `SCHED_TIME_UTC_mvt`-derived deltas, and
`FLIGHT_RULE_mvt` (IFR/VFR) are already categorical/numeric features
(`common/features.py:CATEGORICAL`). IFR/VFR was additionally tested as an
*engineered* flag (`is_vfr`) — passed a naive before/after check but **failed
a permutation control** (TAXI_OUT_DRIVERS.md §1c-feature), so it stayed a raw
categorical, not a derived one. `CALLSIGN_flt` is loaded (`config.DEP_AVAILABLE`)
but deliberately not used directly — too high-cardinality per-flight, and
redundant with `AIRCRAFT_OPERATOR_flt` at the airline-identity level.

### 3–4. Airport state / rolling traffic counts at 5/10/15/30-minute windows

**Partial.** `common/features.py:build_congestion` computes `dep_prev_Xm`,
`arr_prev_Xm`, `same_rwy_prev_Xm`, `dep_surge_ratio_15_60` (a movement-rate
ratio, analogous to "movement intensity") — but only at **15/30/60-minute**
windows (`config.WINDOWS_MIN`), never 5 or 10. The finer granularity this
document suggests has genuinely not been tried. Given how the *coarser*
version of this same family closed (next item, and TAXI_OUT_DRIVERS.md
§2f/§2k), a real but small effect is the reasonable prior, not zero — but
not tested, not closed either way.

### 5. Runway importance; airport×runway×hour/weekday/month typical-time
features

**Partial.** `stand_runway`/`adep_runway` are categoricals the model already
embeds directly, so the *coarser* version of your idea is already inside the
model implicitly. The specific **explicit aggregate feature** you describe
(a computed "typical taxi time for this airport+runway+hour cell") has not
been built at that exact cross, but the closely-related version — a robust
percentile per `stand_runway` (no hour cross) — was built and closed:
**TAXI_OUT_DRIVERS.md §2j**, ceiling 227.445 → 227.362s (Δ0.083s), because
the model has already absorbed this information through its embeddings for
well-observed groups (ρ(model prediction, floor) = 0.746 pooled). The
mechanism (a deterministic aggregate of a categorical the model already
sees adds value only where that categorical is under-observed — 1–7% of
rows depending on band) would very likely apply to an hour-crossed version
too, but this has not been directly measured at that specific granularity.

### 6. Stand → taxi route → distance

**Done — closed.** This is `TAXI_OUT_DRIVERS.md §2h`: real stand/runway
coordinates pulled from OpenStreetMap (9 of 10 airports; LSZH's Overpass
query failed repeatedly), haversine distance computed, matched to 86.8% of
training rows. Mechanistically confirmed real (ρ(raw taxi-out, distance) =
0.17–0.50 across airports) but the *model's residual* correlation with
distance is ≈0 for well-observed stand-runway pairs and only real for rare
pairs (<100 training observations) — ceiling **0.026s** on pooled in-band
val RMSE, crediting perfect capture of the entire rare-pair signal. Closed
negligible, not because distance doesn't matter physically, but because
`stand_runway` already exists as a categorical the model embeds directly.

### 7. Aircraft weight class (derived widebody/narrowbody/heavy/medium/light)

**Done — closed.** Two dedicated arms tested exactly this: weighting a
per-airport blend by wake-turbulence category. `airport_wtc_multimodel`
(20 cells) and `airport_wtc_rec_multimodel` (30 cells, airport ×
Heavy+/NoRecord/Other) both moved to `overtuned/` — sub-0.5s margins that
reshuffled ranking across seeds 45–47, indistinguishable from noise
(`pipeline_review.csv` rows 48–49).

### 8. Cyclical time features (sin/cos encoding)

**Done.** `doy_sin`, `doy_cos` (`common/features.py:ABSOLUTE_TIME_COLS`),
`tod_sin` (consistently a top-5 feature by importance in every tree model
checked this session, MLP and CatBoost both), plus `dow` (day-of-week,
`X["dow"] = t.dt.dayofweek`) and `month_num`.

### 9. Historical taxi-time features (median/mean by airport,
airport+hour, airport+runway+hour, ..., leakage-safe)

**Done, extensively**, and leakage-safety has been a first-class concern
from before this session (`assert_ranking_computable`, the
`BLANKED_ON_RANKING_DEP` contract, `common/features.py`'s "Causal congestion"
design rule documented at the top of the file). The realised-taxi proxy
family (`dep_realised_taxi_mean_prev_30m/60m`,
`n_taxiing_at_pushback`) is exactly this idea, tested and closed at ~0.01s
real gain historically; §2j (above) is the percentile/floor version, closed
at a 0.083s ceiling. Every variant of "give the model a pre-computed
historical typical-time statistic" tested this session has hit the same
wall: the model already learns it from the raw categorical.

### 10. Model order: median baseline → time-aware baseline → LightGBM →
CatBoost → feature engineering → ensemble → only then MLP/NN

**Contradicted by measurement**, and this is the single most useful
divergence in this whole cross-check. On this exact data:

| model (standalone, same protocol) | val RMSE |
|---|---:|
| LightGBM, default-ish params | ~296–298 |
| CatBoost, untuned, depth=8 | ~305–311 |
| MLP (entity embeddings) | ~254–261 |

The MLP has been the strongest single model throughout this build, not the
trees — the opposite of the generic prior this document states (and states
reasonably, as a prior; it's just wrong for this specific dataset). The
likely reason: 15 categorical columns including very high-cardinality ones
(`stand_runway` has 6,345 unique values) — entity embeddings appear to
represent that geometry more effectively than tree splits do here. Trees
remain genuinely valuable as **blend diversity** (CatBoost's residual
correlation with the MLP is ρ=0.88, low enough that a blend beats either
alone by ~9s — see item 11), just not as the primary model this document's
order assumes they'll be.

### 11. Specialist "hard case" / congestion model + ensemble
(ŷ = w₁ŷ_A + w₂ŷ_B)

**Done, tried multiple independent ways, closed every time on the "special
handling for the tail" framing** — though the *ensemble* framing (different
whole models blended) is the one thing on this list that eventually
**worked**, just not shaped as "normal model + congestion specialist":

- Hurdle model (classify delayed-or-not, then regress magnitude
  separately) — `delay_twostage_antigravity`, LOST clearly, val 315 vs.
  baseline's ~257 (`pipeline_review.csv` row 33).
- Huber loss (down-weight large-residual rows during training, the direct
  "the tail needs different treatment" idea) — §2i, **decisively negative**,
  +82.2s worse than plain MSE across all 3 seeds (~23× noise floor),
  because RMSE rewards fitting the real heavy tail and Huber is specifically
  designed to underweight it.
- Per-airport residual scans specifically hunting for a correctable
  "hard case" pattern at LIRF/EGLL — §2e/§2f, closed (no lever at LIRF;
  real-but-too-small at EGLL, ~0.12s).
- What **did** work, eventually: a whole different model (CatBoost) blended
  with the MLP via honestly-solved per-airport weights — not because
  CatBoost specializes in hard cases, but because it makes different
  mistakes overall. See §2m: **244.56 val RMSE**, beating the previous best
  blend (248.33) by a real, consistent margin. This is the ensemble
  structure the document describes, arrived at from a different angle.

### 12. Time-based validation, no random splits

**Done**, and has been the foundation of this entire build, predating this
session. `config.SPLITS["forward"]`: train Jan–Aug 2025, val Sep–Oct, test
Nov–Dec — strictly time-ordered, mirroring train-2025 → score-2026 exactly
as this document recommends.

### 13. Specifically simulate January and July

**Done, and already built** — worth flagging since the document proposes
this as something to add: `config.SPLITS["seasonal"]` holds out **exactly**
January and July as test (val = December), with July additionally
restricted to the three airports (`EDDF`, `EGLL`, `EHAM`) the real July 2026
ranking file covers, since the competition's own July ranking set isn't
airport-complete. This is a secondary, composition-matched probe alongside
`forward`, not the primary selection surface (`compare.py` still crowns
arms on `forward`), but it exists and has been used.

### 14. Data-quality report on the movement/flight join

**Partial.** No single systematic table (missing-runway / missing-stand /
missing-aircraft / duplicate-flight / ADEP-ADES-mismatch counts, all in one
place) has been built as a standalone deliverable. But the *specific*
mismatches that turned out to matter have been investigated in depth:
the LIRF "unmatched" population (missing `AOBT_3_flt`, exactly 1,488
rows/year, 1,487 with null aircraft type — independently reproduced against
a competitor's identical numbers from their own public research notes) and
`conversation.md` (cleaned up this session), which is a full record of the
competition organizer's own answers about the BLOCK≈SCHED artifact and
AOBT/BLOCK-time disagreement populations. Targeted and hypothesis-driven
rather than a general audit, but the highest-impact rows have specifically
been covered.

### 15. Overall architecture (data → features → tree/MLP models → hard-case
ensemble → submission)

**Done in shape**, with one branch pruned by measurement: the "normal +
congestion specialist" ensemble branch this document proposes doesn't
survive contact with this data (item 11) — the actual winning shape is
"MLP + tree-ensemble blend + CatBoost, blended per-airport," not
"normal-case model + hard-case model."

### 16. "Don't start with neural networks; trees first"

**Contradicted**, per item 10. Reasonable as a general prior for tabular
data; empirically wrong for this specific feature set.

---

## One-line summary

Of 16 points: **10 done** (1, 2, 6, 7, 8, 9, 12, 13, 15-in-shape, and the
ensemble-structure half of 11), **3 partial** (3–4, 5, 14), **1 not
tested** (the 5/10-minute window granularity specifically), and **2
contradicted by direct measurement** (10, 16 — trees have not outperformed
the MLP here, the opposite of the plan's assumption). The one substantive
open lever from this list is the finer-grained congestion window count
(5/10 min) and the richer per-stand scalar aggregates flagged earlier this
session from competitor code — both untested, both low-but-nonzero-prior
given how consistently this whole feature family has closed near-zero.
