# EUROCONTROL AIU ATFM delay diagnostic — result

**Question.** `TAXI_OUT_DRIVERS.md` §2b found no CTOT/slot/regulation field
anywhere in the competition data and explicitly flagged that testing an
ATFM-regulation hypothesis would need an external source — "rejected for
lack of data, not tested and found negative." A recommendation (external
analysis) proposed EUROCONTROL's own public Aviation Intelligence Portal
(ansperformance.eu) daily Airport Arrival ATFM Delay dataset (`APT-DLY`) as
that source: a genuinely different, exogenous causal channel from local
traffic density — network-wide flow-management ground-holds, attributed to
the DEPARTURE airport/day per EUROCONTROL's own definition ("aircraft
expected to arrive during a period of congestion are given ATFM delay AT
THEIR DEPARTURE AIRPORT").

**Data.** Downloaded the real bulk file (`Airport_Arrival_ATFM_Delay.xlsx`,
`DATA` sheet, 1.03M rows, 2019–2026). Filtered to the 10 target airports:
**complete daily coverage** — 365/365 days for 2025 at every airport, and
critically, complete coverage for the actual scored period, Jan+Jul 2026,
with no gaps. Joined `DLY_APT_ARR_1` (minutes of ATFM delay that
airport/day) and a derived `pct_regulated` (share of that day's flights
delayed >15min) to every DEP row in `forward`/val by (airport, calendar
date) — **100% join coverage** (every val row has a same-day ATFM record).
Licensing not yet organizer-confirmed (`conversation.md` §1, open question)
— this is a diagnostic only, nothing built into `features.py`.

## Result — real but small, mixed sign across airports, closed on the ceiling

Pooled: rho(DLY_APT_ARR_1, resid) = **+0.0136**, rho(pct_regulated, resid) =
**+0.0129** — both only modestly above the permutation-shuffled control
(+0.0025 / −0.0017), a much thinner margin over noise than the WTC-
sequencing check's ~30x ratio (`tests_wtc_sequencing/RESULT.md`). In-sample
joint OLS ceiling on pooled val RMSE: baseline 253.989 → corrected 253.895,
**only −0.094s**.

**Per-airport breakdown shows mixed sign, not a uniform mechanism**: LEBL
(rho=+0.106) and LTFM (rho=+0.071) show the strongest positive signal;
EDDF (−0.008), LEMD (−0.030), LIRF (−0.021) go the *wrong* direction;
EHAM/LSZH/LFPG sit near zero. Unlike the WTC-sequencing check (consistent
positive sign at all 10 of 10 airports — a real, uniform physical
mechanism), this is closer to a noisy mix with one or two airports pulling
the pooled number, which reads as weak/inconsistent evidence rather than a
genuine uncaptured driver.

## Reading

The daily-airport aggregate is inherently a coarse proxy for "was today
disrupted at this airport" — mechanistically adjacent to (though causally
distinct from) the already-closed day-regime clustering
(`pipeline_lightgbm_unsup`, tier C, "no effect... test WORSE"), and it lands
in the same place: real-sounding mechanism, negligible/inconsistent
measured effect. The genuinely useful signal EUROCONTROL holds — per-
*flight* CTOT/regulation assignment, which would tell you whether THIS
specific departure was individually held, not just "was the airport busy
that day" — is not in this public daily-aggregate dataset; it lives in NM's
own regulation/slot systems, which need accredited access (the same
ADRR/NM-B2B account-gated systems already found infeasible this session).

## Disposition

**Closed, decisively, at the only granularity actually available without
accredited access.** −0.094s ceiling is an order of magnitude under the ~5s
board-visibility floor and, unlike the WTC-sequencing finding, doesn't even
have the consolation of a physically-confirmed uniform mechanism — this
reads as noise with 1-2 airports of coincidental correlation, not a real
effect the model is missing. Not built. This does properly close §2b's
originally-unresolved question ("rejected for lack of data") — it's now
tested, not just untestable.
