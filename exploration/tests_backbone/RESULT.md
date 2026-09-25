# MLP backbone upgrades (PLE, TabM, TabM+PLE) — result

Bar: `PREREGISTERED.md`, fixed before any run. `base` reproduces the production
MLP's epoch curve to within 0.006 s (seed 42, vs `tests_hp_diversity`). Runs that
hit the 50-epoch cap were rerun at cap 100; the capped versions are kept as
`TRUNCATED50_*` and not used. **All three variants FAIL stage 1 (forward). None is
a candidate.**

In-band RMSE of the 6-seed mean, variant minus `base` (negative = better):

| variant | fwd val | fwd test | seeds better (test) | fwd LIRF test | Jan+Jul test | verdict |
|---|---:|---:|---:|---:|---:|---|
| `ple` | +2.37 | **+17.81** | 0/6 | +27.65 | +3.59 (LIRF +25.3) | FAIL |
| `tabm` (k = 32) | **+8.75** | +2.37 | 3/6 | +3.62 | not judged | FAIL (stage 1) |
| `tabm_ple` | **+2.85** | −1.89 | 6/6 | +11.88 | not judged | FAIL (val sign) |

- **PLE** hurts badly on its own: all seeds converged, and best epochs rose from
  10–22 to 22–33.
- **TabM** varies more from seed to seed than the MLP (val 250–278 against
  251–261). Averaging 32 weight-sharing members inside one network does not beat 6
  independent MLPs.
- **TabM+PLE** is the only variant to improve forward test (−1.89, 6/6 seeds), but
  it is worse on val (+2.85) and **+9 to +12 s worse at LIRF on both**. It would
  fail the stage-2 LIRF guard even if the val sign had held. Its Jan+Jul reruns
  were therefore not run.

**Disposition:** closed. With `tests_hp_diversity`, every MLP-family change tried
this week (width, depth, regularization, batch size, input encoding, internal
ensembling) fails. The members that paid here are different model families.
