# Prediction floor 60 s → 240 s — pre-registration

Written 2026-09-26, before any number. Source: `EXTERNAL_PLAN_CROSSCHECK_2.md` §"Physical
floor": the 60 s floor (`config.PREDICTION_CLIP_MIN_SEC`) was set, never swept; 240 s
is "genuinely untested, worth a cheap check". No training.

Instrument: the v35 analogue on each surface: 3-member per-airport blend
[FTT6, tree, MLP] with weights solved on forward_inner, plus the TabNet LIRF override
at w = 0.459. Floors compared: 60 (current), 120, 180, 240.

**Adopt 240 (or the pre-stated alternative 180) only if** it lowers **both** in-band
and pooled RMSE on forward val, forward test **and** seasonal_full Jan + Jul, relative
to 60. Otherwise closed. The floor must be applied after the TabNet override, as it
would be in the build.
