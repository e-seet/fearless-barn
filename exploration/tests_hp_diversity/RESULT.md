# Hyperparameter-diverse MLP ensemble — result (forward screen)

Bar: `PREREGISTERED.md`, fixed before any run. Output: `screen.txt`, `screen.json`.

## Screen: `deep` PASSES, `reg` and `bigbatch` FAIL, `wide` UNRESOLVED

In-band RMSE; A = 6 `base` seeds; B1/B2 = 3 `base` + 3 variant, with the halves swapped.

| variant | test B1−A | test B2−A | **test mean** | val mean | resid corr vs base (reference 0.975) | ens3 standalone test (base 242.58) | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| **deep** (128,128,128,128,64) | −5.94 | −4.12 | **−5.03** | −0.68 | 0.966 | **233.71** | **PASS** |
| reg (dropout 0.3, wd 1e-3) | +0.17 | −2.10 | −0.97 | +1.40 | 0.975 | 240.62 | FAIL |
| bigbatch (8192, lr 2e-3) | −1.57 | +1.04 | −0.26 | +0.54 | 0.972 | 242.47 | FAIL |
| wide (1024,) | — | — | — | — | — | — | **unresolved** |

### The caveat on `deep`, stated before anything is built on it

`deep` passes mostly because it is **better standalone on Nov–Dec** (ens3
233.71 vs 242.58, −8.9 s), **not** because it decorrelates. Its residual
correlation, 0.966, is barely below the base-vs-base reference of 0.975. On
Sep–Oct it is nearly a tie standalone (230.00 vs 230.75).

A large test gain with a near-tie on val is the same shape as the FT-Transformer's
forward-test reading, which turned out to be the optimistic outlier of four
surfaces. It also sits against `TAXI_OUT_DRIVERS.md` §2j, where "wider/deeper"
closed flat standalone on val. **So this PASS licenses a replication on a fresh
surface, not a production build.** See `PREREGISTERED_followups.md`.

### `wide` hit the cap: unresolved, not failed

5 of 6 seeds ended at the 50-epoch cap, still improving at val ~278. That is ~23 s
behind `base`, and gaining ~1–2 s per 10 epochs. The runs are kept as
`TRUNCATED50_preds_wide_seed*.npz`. FTT was 43 s behind at its first cap and
later passed, so this is rerun at a higher cap rather than closed; see the follow-ups.

### Scope note

All configs were compared under the early-stopped recipe. `tests_refit_epochs/RESULT.md`
found that an *ensemble* prefers ~2–3× more epochs than per-seed early stopping
picks, so the ranking between configs could shift under that recipe. That is not
tested here.

## Follow-ups (`PREREGISTERED_followups.md`) — both FAIL. **Hyperparameter-diverse MLP ensembling is closed.**

### F1. `deep` on seasonal_full (Dec val / Jan+Jul test): FAIL

| | B1−A | B2−A | mean | resid corr (ref 0.972–0.978) | ens3 standalone (base) |
|---|---:|---:|---:|---:|---:|
| Dec val | −1.77 | −1.79 | −1.78 | 0.965 | 217.08 (219.81) |
| **Jan+Jul test** | −1.09 | **+0.14** | **−0.47** | 0.969 | 254.50 (254.33) |

The test mean misses the −1.0 bar, and one half is worse than A. On the board's
months `deep` is a standalone tie (254.50 vs 254.33). **LIRF guard: `deep` ens6
LIRF in-band 491.18 vs `base` 481.50, +9.7 s worse**, at the airport the board
weights ~2× and where the MLP carries the whole blend. The forward-test PASS was
a Nov–Dec artefact, the same surface effect that sank the ensemble-epoch idea
(`tests_refit_epochs/RESULT.md`).

### F2. `wide` at cap 100: converged, FAIL

All 6 seeds early-stopped (best epochs 38–71; none capped). It is the most
decorrelated MLP variant measured (resid corr 0.927 vs reference 0.975), but
~18 s worse standalone (ens3 248.6 vs 230.8 val). Swapping it in costs **+6.05 s
val** and +0.39 s test. That is the FTT lesson again: decorrelation without
standalone accuracy does not pay.

### Disposition

Closed alongside `tests_arch_diversity` (GRU). Within the MLP family, the
variants either tie and don't decorrelate (`deep`, `reg`, `bigbatch`) or
decorrelate and are too weak (`wide`). The members that paid here (TabNet, FTT)
are different model families.
