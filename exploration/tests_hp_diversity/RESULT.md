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
