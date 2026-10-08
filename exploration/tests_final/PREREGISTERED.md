# Final round: re-checking the design on the new months (Feb, Jun) — pre-registration

Written 2026-10-08, before the script below runs. The organizers added a blind final ranking on
Jan + Jul, Feb + Jun and all four months of 2026 (`final_ranking.parquet`, 670,790 departures). The design
freezes on 2026-10-11; the final file is due 2026-10-14 12:00 CEST.

**Question.** Several stages were scoped to the months of the first ranking set:
- frost +30 s: January only;
- thunderstorm +200 s: July only;
- LIRF ADS-B corrector: January only, because LIRF had no ADS-B in July 2026.

In February and June they are off as shipped. Should any of them apply in every month? And do the
LIRF no-record specialist and the LOBT clip (both under consideration) help in these months?

**Surface.** Held-out Feb–Jun 2025 (`holdout_febjun`: base members trained on Jan + Jul–Nov).
- Base weighted per airport like v61's training base (c3d6).
- Stage 2, LIRF and LFPG correctors are the saved models (`model_v61`, `model_v55`, `model_v56`). None
  of them was trained on Feb–Jun 2025.
- The specialist is trained leave one month out on the other 11 months.

**Reference.** The shipped design as it would run on Feb–Jun: base → stage 2 → LFPG corrector.

**Adopt a change (each judged on its own, against the reference) only if all hold:**
1. Feb + Jun 2025 pooled RMSE better;
2. Feb–Jun pooled better;
3. better in at least 2/3 of the months where it changes rows;
4. still better with the 20 largest-gain rows removed.

**Reported:** whether stage 2 and the LFPG corrector themselves help on Feb–Jun (no decision attached;
they are the design).
