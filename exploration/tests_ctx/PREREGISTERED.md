# Context corrector on every row (airport state) — pre-registration

Written 2026-10-09, before any model is fitted. Time-boxed: decision by 2026-10-11 18:00 CEST (the design
freezes with the docs at 23:59 CET). If not decided by then, the final file stays as built (`tests_final/`).

**Idea and credit.** Several published solutions correct every row from the airport's current state:
- the mean (takeoff − NM actual off-block) of the departures that took off in the previous minutes, as
  a live read of how slow the airport is (quirky-honey `taxiout.py`; outgoing-boat `contexto.py`;
  joris-decombe);
- the lateness against schedule of the surrounding departures, and the flight's own lateness minus
  theirs (quirky-honey `_neighbour_delay`, −1.6 on their board).

Our ADS-B corrector reaches ~55 % of rows. The implementation below is ours.

**Features** (from the movements file only; other departures, never the row's own label):
- `nb_proxy_20`, `nb_proxy_60`: mean of clip(MVT − AOBT_3, 0, 7200) over other departures at the airport
  with takeoff in [t − 20 min, t) and [t − 60 min, t);
- `nb_proxy_rwy30`: the same on the same runway, [t − 30 min, t);
- `nb_late_15`, `nb_late_60`: mean of clip(MVT − SCHED, −1 h, 4 h) over other departures within ± 15 / ± 60 min;
- `own_late_rel`: own clip(MVT − SCHED) minus `nb_late_60`;
- plus `F` (the prediction being corrected), airport, hour, own clip(MVT − AOBT_3), own clip(MVT − SCHED),
  and whether the row had an ADS-B correction.

**Model C.**
- LightGBM on y − F: lr 0.03, 800 trees, 63 leaves, min_data_in_leaf 200, feature_fraction 0.8, seeds 42–44.
- Trained on rows with y ≤ 10,800 s and F ≤ 7,200 s; applied to rows with F ≤ 7,200 s; result clipped to
  the LOBT window. Tail rows keep F.

**Surfaces and references.** Both arms are clipped to the LOBT window.
1. **7 months** (Jan, Jul–Dec 2025): F = the held-out recipe of `tests_copymix` (realistic-4 base + K2
   stage 2), leave one month out.
2. **Held-out Feb–Jun 2025**: F = the shipped design there (`tests_final`: base → `model_v61` → LFPG).
   C is trained on all 7 months of surface 1.

**PASS (all required):**
1. 7 months: pooled better, Jan + Jul better, ≥ 5/7 months, better with the best 20 rows removed;
2. Feb–Jun: pooled better, Feb + Jun better, ≥ 4/5 months, better with the best 20 rows removed;
3. body share: the gain from rows with y ≤ 3,600 s is positive on both surfaces;
4. drift gate: for each feature, |mean(2026) − mean(2025)| / sd(2025) ≤ 0.2 on Jan + Jul, computed on
   `final_ranking.parquet` against 2025 training.

**Build if PASS:** a stage 2b in `final_pipeline.py` after stage 2, before the weather offsets, trained on
all held-out rows; the final file is rebuilt and documented. **FAIL:** closed, and the final file stays.
