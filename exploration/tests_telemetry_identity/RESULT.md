# Is the LIRF telemetry identity general? And does tail work reach the leaderboard?

Run: `python stage12_telemetry_identity_audit.py`
Output: `artifacts/audit/telemetry_identity_audit.json`

Read-only. No model trained, nothing fitted, nothing selected. Three questions
that had never been answered, all cheap.

---

## Q1 — the identity is genuinely LIRF-specific. Do **not** generalise it.

`airport_lirf_identity_multimodel` hard-codes `ADEP_mvt == 'LIRF'`. The mechanism
behind it — *no flight record → `BLOCK_TIME` defaults to `SCHED_TIME` → `y`
collapses to `delta_sched_sec`* — contains nothing specific to Rome, so the
obvious next move is to extend it to all ten airports. **That would have been a
mistake, and this is the check that says so.**

Full year, every airport, `has_flight_record == 0`, `|y − delta_sched| ≤ 60 s`:

| airport | band | full-year n | within 60 s | resid std | verdict |
|---|---|---:|---:|---:|---|
| **LIRF** | **25200–50000** | **34** | **100.0 %** | **2.9 s** | **IDENTITY** |
| LIRF | 7200–25200 | 573 | 61.6 % | 5,657 s | partial |
| EHAM | 7200–25200 | 682 | 0.0 % | 3,375 s | not an identity |
| LFPG | 7200–25200 | 966 | 0.0 % | 3,178 s | not an identity |
| EDDF | 7200–25200 | 495 | 0.0 % | 2,777 s | not an identity |
| EGLL | 7200–25200 | 507 | 0.2 % | 3,125 s | not an identity |
| LSZH · EDDM · LTFM · LEMD · LEBL | 7200–25200 | 223–573 | 0.0 % | 2,405–3,979 s | not an identity |
| EHAM | 25200–50000 | 11 | 0.0 % | 7,494 s | not an identity |

**Exactly one cell in the entire year is an identity, and it is the one already
shipped.** Every other airport is at 0.0 %. This is not "weaker elsewhere" — it
is absent elsewhere. Extending the override would replace a model prediction with
a number that is wrong essentially every time, on ~500+ scored rows.

The hard-coding to LIRF was correct. It is now correct **on evidence** rather
than because LIRF is the airport somebody happened to audit.

## Q2 — exposure in the actual scored set

`ranking.parquet` DEP rows = **215,876**. Rows a rule would touch:

| rule | scored rows touched | share |
|---|---:|---:|
| shipped (LIRF, no record, `ds > 25200`) | **10** | 0.0046 % |
| hypothetical all-airport version | 44 | 0.0204 % |

## Q3 — 10 rows is not negligible, and the arithmetic says why

The instinctive read of Q2 is "10 rows out of 215,876 cannot matter". **That is
wrong for this metric**, which is why it is computed rather than asserted.

Pooled RMSE is a *mean of squares*. Measured on the shipped arm's own eval JSONs:

| part | rows above 7200 s | share of rows | **share of pooled squared error** |
|---|---:|---:|---:|
| val | 93 | 0.025 % | **15.4 %** |
| test | 61 | 0.019 % | **20.8 %** |

A fifth of the test metric is decided by 0.019 % of the rows. A row wrong by
~15,000 s weighs like ~3,600 rows wrong by ~250 s.

Measured directly rather than modelled: the `idband` override touches **five** val
rows and moves pooled val RMSE by −0.55 to −1.20 s, and five test rows by −0.58 to
−2.18 s (plan 12, candidate 2). The scored set is *smaller* than the test set
(215,876 vs 328,009), so each corrected row carries **more** weight there. Ten
touched rows is worth roughly **1–3 s of pooled RMSE** — small, real, free, and it
cannot backfire on a row it touches.

## What this does and does not say about the quarantine

It does **not** rehabilitate anything in `overtuned/`. Those arms never touched
these rows: they re-weighted the blend across *all* rows and moved nothing beyond
seed noise. **Tail rules and blend-weight bucketing are different levers** — this
result is about the first; the over-tuning verdict was about the second.

It does sharpen one thing: `airport_debiased_multimodel` was reaching for a real
prize (the tail genuinely decides ~20 % of the metric). Its failure was *method* —
constants tuned against the reported score, a 3-parameter fit on 7 rows — not
target selection. The right way to attack that prize is a verified zero-parameter
identity, which is what `idband` is.

## Not adopted, and why

**`LIRF / 7200–25200`** — 61.6 % within 60 s over 573 full-year rows, and **41
scored rows**, four times the identity band's exposure. Tempting, and explicitly
**not adopted**: 61.6 % is not an identity, the 38 % that miss have residual std
5,657 s, and under L2 a confidently-wrong large prediction costs far more than the
correct ones gain. Adopting it would be a bet dressed as a mechanism — the exact
failure catalogued in `overtuned/README.md`.

If anyone wants it, it needs the plan-12 treatment first: pre-register the
criterion, measure the paired val/test effect across ≥3 seeds, and check in-band
is untouched. Do not merge it on the strength of the row count.
