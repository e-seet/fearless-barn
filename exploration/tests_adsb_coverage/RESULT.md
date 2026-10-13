# ADS-B (adsb.lol) off-block recovery -- coverage check result

**Closed for LIRF-unmatched rows** under the rule registered before any download
(coverage >= 30% AND recovery >= 50% of covered AND huge-delay agreement >= 50%).
Coverage at LIRF is HIGH (84-89%), but the ADS-B first-ground-sighting does not reproduce the
recorded off-block time (5%), and on the huge-delay unmatched rows it disagrees with the label
outright. Organizer context: open trajectory data is permitted (relayed 2026-09-21, recorded in
`conversation.md` sec.1 item 2 and sec.6), "practically speaking it won't be possible".

## Setup

`run_adsb_coverage.py` (`fetch <day>` then `analyze`). Source: adsb.lol daily releases
(`adsblol/globe_history_2025`, tag `v<YYYY.MM.DD>-planes-readsb-prod-0`, ~3.3-3.6 GB/day, streamed
over HTTP and discarded, ~3.5 MB/s). `mlatonly-0` is not published for 2025 days, `staging-0` is a
same-size sibling and was not used. Days chosen for the most LIRF-unmatched rows: 2025-07-13,
2025-07-14, 2025-06-26 (18,408 departures across the 10 airports, 68 LIRF-unmatched, 30 of them
with y > 3600 s). Matching: our `FLIGHT_mvt` vs the callsign carried in the trace (they equal the
flight-record callsign on only 72.8% of matched rows, so matching noise counts against coverage).
COVERAGE = the callsign has >= 1 `"ground"` point within 8 km of the airport in [T-4h, T+5min].
RECOVERY = |first ground observation - true BLOCK_TIME| <= 180 s (BLOCK_TIME is known in training).

## Result

| LIRF | rows | covered | recovered (of covered) |
|---|---:|---:|---:|
| unmatched | 68 | 84% | 3 of 57 (5%) |
| matched | 1,412 | 89% | 60 (5%) |
| unmatched, y > 3600 s | 30 | 83% | ADS-B taxi within 30% of y: 3 of 25 (12%) |

Coverage by airport (matched rows): LEBL 84%, LIRF 89%, LEMD 48%, LSZH 24%, EDDM 23%, EGLL 8%,
EDDF 3%, LFPG 2%, EHAM 2%, LTFM 0%. Careful about what 'coverage' means: we measured ground-flagged
points within 8 km of the airport in a 4-hour window, NOT stand-level coverage. The forum poster's
point (surface coverage at the gate is poor at LIRF) is SUPPORTED, not refuted: 89% of LIRF
departures have ground points somewhere on the airport, but the first sighting lands a median of
691 s AFTER the recorded block-out, i.e. the aircraft is seen once it is moving on the taxiways, not
at the stand. The poster's EDDM ~40% is close to what we see (23% matched / 59% unmatched). Recovery is only high at a few airports (EDDM 61%, EHAM 67% of a tiny covered set, LEBL
36%), and matched rows already carry `AOBT_3_flt`, so the marginal information is confined to the ~1.5%
unmatched rows outside LIRF, which carry little of the error.

`lirf_offsets.txt`: at LIRF the first ground sighting comes a median of 691 s (IQR 511-973 s)
AFTER the recorded block time for matched rows, i.e. it does not locate the gate departure. For
LIRF-unmatched rows with y > 3600 s (25 covered): ADS-B taxi (takeoff - first sighting) has a
median of 455 s against a median label of 7,741 s, and 80% are under 15 minutes (examples: label
6,487 s vs 285 s, 10,436 s vs 230 s). This corroborates the earlier finding (`TAXI_OUT_DRIVERS.md`
sec.2q; the stand-turnover retraction) that these labels track MVT_TIME - SCHED_TIME (a recording
artefact), not a physical taxi time: no physical off-block source can reconstruct them, however
good its coverage.

## Caveats

Three days; 68 LIRF-unmatched rows (30 huge-delay); callsign matching is imperfect; only `prod-0`.
The conclusion (closed) is insensitive to those: the huge-delay rows fail on agreement, not on
coverage, and 5% recovery on 1,256 covered matched LIRF rows is not a small-sample effect.

## Disposition

Not built, no retrain, no submission. Per-day row-level CSVs (`coverage_<day>.csv`) are kept
locally and not committed (raw per-row export).
