# FTT at LIRF with a downward-only guard — pre-registration

Written 2026-09-25, **before any number was computed**. Plan 15, item 2. No
training; all predictions exist.

## Idea

At LIRF the shipped prediction is P = 0.45 × MLP + 0.55 × TabNet, and FTT's blend
weight there solves to 0. FTT is *better* on LIRF in-band (val 418.9 vs 429.3,
`tests_ftt/RESULT.md`) but over-predicts ~50–80 extreme-tail rows badly. The
guard lets FTT only pull the prediction **down**:

    P_new = P − w × max(0, P − FTT)        (at LIRF only; other rows unchanged)

`w` ∈ [0, 1] is the only parameter. There is no threshold, and FTT can never
raise a prediction, so its tail over-prediction cannot reach the output.

## Surfaces (the v32 LIRF analogue on each, built from existing predictions)

| surface | MLP | TabNet | FTT |
|---|---|---|---|
| forward_inner val (Jul–Aug): **solve w here** | 18-seed cache | inner seeds 42–44 (as v29's w was solved) | seeds 42–47 |
| forward val (Sep–Oct) / test (Nov–Dec) | 12-seed cache | seeds 42–44 (`lr2e-3p`) | seeds 42–47 |
| seasonal_full test (Jan + Jul) / val (Dec) | 6-seed cache | seeds 42–44 (`sf_lr2e-3p`) | seeds 42–47 |

`w` = the least-squares solution on forward_inner LIRF rows (pooled squared error,
the board's metric), clamped to [0, 1], then **frozen**.

## Bar (all must hold; LIRF rows only)

1. `w` > 0.05 (otherwise there is nothing to ship);
2. LIRF **in-band** RMSE lower than P on forward val, forward test **and** Jan + Jul;
3. **tail guard:** LIRF **pooled** RMSE not worse than P by more than 2.0 s on
   forward test and on Jan + Jul. Pulling predictions down risks the tail, which
   the pooled metric punishes.

**PASS →** build as a candidate on the v32 base (LIRF only, so it composes exactly
with any non-LIRF change) and show the user before `submit.py`. **FAIL →** closed.

**Prior: low-moderate.** Fitting on the tail's shape resembles divergence gating
(plan 07, closed), but with no fitted threshold.
