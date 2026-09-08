# airport_lirf_identity_multimodel — does it hold?

**What it does.** Zero fitted parameters. Takes the base arm
(`airport_multiseed_multimodel`, val 246.65 / test 240.04 / in-band 218.16) and,
for rows where `ADEP == LIRF AND has_flight_record == 0 AND delta_sched_sec >
25200`, replaces the base prediction with the raw `delta_sched_sec` feature.
Deterministic — the base ensemble is over fixed seeds 42/43/44 and the override
is a fixed rule, so there is no seed variance to sweep.

Ran seed 42; per-row breakdown in `diagnose_out.txt`.

## Headline

| | base | override | delta |
|---|---:|---:|---:|
| val RMSE | 246.65 | **243.26** | −3.39 |
| val in-band | 227.14 | 227.14 | **0.00** |
| test RMSE | 240.04 | **233.84** | −6.20 |
| test in-band | 218.16 | 218.16 | **0.00** |
| test LB-weighted | 228.70 | 224.04 | −4.66 |

**In-band is untouched on both parts** — the override only ever hits `y > 25200`,
so it cannot damage the 99.98 % of rows that decide the low-variance metric. No
`forward_inner`, no fitting, nothing to leak. This is structurally clean, and
nothing like `airport_debiased_multimodel` (10-param winter curve fit on the test
months). No row it touched got worse, on either part.

## It holds in two very different halves

### 1. The 25.2k–50k "checked identity" band — HOLDS, cleanly

5 rows on val, 5 on test. `y_true` equals `delta_sched_sec` to **single-digit
seconds** every time:

| part | override residuals on these rows (s) |
|---|---|
| val | 0, 5, 0, 3, 1 |
| test | 2, 3, 1, 0, 4 |

The base model was off by 1,300–11,000 s on the same rows. This is the real
mechanism (BLOCK_TIME defaults to SCHED_TIME with no radar track), and the arm's
own full-year check backs it (34 rows in this band, `y − delta = 0.0 ± 2.9 s`).

**Worth on its own: −0.67 s val, −1.08 s test.** Small, but exact, parameter-free,
zero-risk. This part should just be adopted.

### 2. The >50k "artifact / rollover" band — WORKS ON THESE ROWS, but it's a bet

The 3 tail rows that dominate pooled test RMSE. `delta_sched_sec` is *not* an
identity here (the arm's docstring says so: full-year std 13,488 s). What actually
happened on the 3 test rows:

| y_true | delta_sched (=new pred) | \|err\| base | \|err\| new | note |
|---:|---:|---:|---:|---|
| 87,598 | 60,599 | 35,134 | **26,999** | still 27,000 s wrong |
| 87,480 | 56,520 | 31,808 | **30,960** | barely moved |
| 87,605 | 86,102 | 15,402 | **1,503** | delta landed near y — luck |

**~5.1 s of the −6.2 s test gain is these 3 rows.** The override makes them *less
wrong*, and `delta_sched` is a better-motivated guess than the base model's blind
extrapolation — but two of the three are still 27–31 k s off, and the big win
(row 3) is `delta_sched` happening to land near `y`, not a mechanism. This is the
same "bid in the LIRF tail cell" bet the MLP and the ensemble make, with
`delta_sched` as the bid instead of a high constant.

**If the override is restricted to the checked-identity band only** (skip the >50k
rows): test 240.04 → **238.96** (−1.08), val 246.65 → **245.98** (−0.67).

## Verdict

**Adopt it — but read the number honestly.**

- Structurally sound: in-band exactly preserved, deterministic, zero params, no
  leak path, never backfires on the rows it touches. The `airport_debiased`
  criticism does not apply.
- The **bankable** improvement is **~−1 s** (the 25.2–50k identity — a real fix).
- The other **~−5 s of test RMSE is a tail bet** on ~3 unpredictable rows where
  `delta_sched_sec` is directionally right. It will help on the leaderboard *if*
  an artifact row appears and `delta_sched` is near it — expected value positive,
  but not a 5-second accuracy gain and not something to select on.
- Report this arm as **in-band 218.16 (unchanged from base) + a motivated tail
  override**, not as "test RMSE 233.84".

Recommended: keep the identity-band override unconditionally; treat the >50k
branch as an explicitly declared tail bet (or drop it and let the base model own
those rows). Either way it belongs *on top of* the shippable winner
(`multimodel_antigravity`), not as a competitor to it.

---

## Follow-up: checked-identity band ONLY (tag=idband)

Re-ran with the override restricted to `25200 < delta_sched_sec <= 50000`
(`run_idband_only.py`, `idband_only_out.txt`). Applied on top of
`airport_multiseed_multimodel`:

| metric | base | **idband only** | full arm |
|---|---:|---:|---:|
| val RMSE | 246.65 | **245.98** (−0.67) | 243.26 (−3.39) |
| val in-band | 227.14 | **227.14** (0.00) | 227.14 |
| val LB-weighted | 230.75 | **230.28** (−0.47) | 228.37 |
| test RMSE | 240.04 | **238.96** (−1.08) | 233.84 |
| test in-band | 218.16 | **218.16** (0.00) | 218.16 |
| test LB-weighted | 228.70 | **227.89** (−0.81) | 224.04 |

5 rows touched per part, every one `y = delta_sched_sec ± 4 s`. In-band exactly
preserved. **This is the bankable part: −1.1 s test / −0.8 s test-LB-weighted,
zero parameters, zero risk.** The remaining −5 s the full arm shows is the >50k
tail bet on ~3 rows. `tag=idband` is tier A in `../pipeline_review.csv`; the full
arm stays tier B.
