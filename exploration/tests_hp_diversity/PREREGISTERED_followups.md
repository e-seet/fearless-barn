# Hyperparameter-diverse MLP — follow-ups (pre-registration)

Written 2026-09-23, after the forward screen (`RESULT.md`) and **before either
follow-up was run**.

## F1. `deep` replication on seasonal_full (fresh surface)

`seasonal_full`: train Feb–Jun + Aug–Nov, val **Dec**, test **Jan + Jul** (the
board's months). Nothing from the HP configs has been trained or looked at there.

- `base` and `deep`, seeds 42–47, same script (`run_configs.py --split seasonal_full`),
  cap 50, patience 6 on Dec.
- **Same bar as the forward screen:** mean(B1, B2) − A ≤ −1.0 s in-band on
  Jan+Jul, **and** both halves < 0, **and** Dec mean < 0.
- **Additional guard, because the forward caveat was "standalone, not diversity":**
  report `deep`'s LIRF in-band vs `base` on Jan+Jul. The board weights LIRF ~2×,
  and the MLP carries LIRF alone in every shipped blend.

**PASS on F1** would license *proposing* a production check to the user: the
blend's MLP side with `deep` members, weights solved on forward_inner, and `full`
refits under a new tag. **FAIL** closes `deep` as a forward-test artefact.

## F2. `wide` rerun at a higher cap (forward)

Same forward screen, `wide` only, seeds 42–47, **cap 100** (2× the prior cap;
the last cap-50 bests were 45–49). The capped runs are kept as
`TRUNCATED50_preds_wide_seed*.npz` and are not used.

- Same bar as the forward screen.
- If ≥ 1 seed hits cap 100 again, it is **still unresolved**. It is then recorded
  as "not competitive within 100 epochs" and not pursued further. The standalone
  gap of ~23 s at cap 50 does not justify more GPU.

## Cost

F1: 12 runs, ~3 min each, ~40 min. F2: 6 runs, ~5 min each, ~30 min.
Queued after `tests_refit_epochs/` seasonal_full confirmation.
