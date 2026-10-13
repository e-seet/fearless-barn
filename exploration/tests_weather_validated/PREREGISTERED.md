# Weather offsets for flights without ADS-B, validated out of sample — pre-registration

Written 2026-10-02, before any offset is fitted. Board: v50 = 271.8748 (best and last).
- **v49:** +30 s on January frost flights without ADS-B (271.944).
- **v50:** +200 s on July thunderstorm flights without ADS-B (271.8748).

Both offsets were chosen and measured on the same Jan + Jul 2025 data. v51–v54 (per-airport frost,
freezing rain, fog, rain) were never tested out of sample, and uploading several to keep the best
would fit the hidden labels.

## Conditions

These are antigravity's own definitions (`tests_weather_metar/evaluate_candidate_v51_on_2026.py`).
The METAR used is the last observation at or before takeoff, within 90 min, from
`prc-2026-datasets-wx`.

| condition | definition |
|---|---|
| `frost` | tmpf ≤ 34 and tmpf − dwpf ≤ 3 |
| `ts` | `TS` in wxcodes |
| `fz` | `FZ` in wxcodes |
| `fog` | not frost, and (FG or BR in wxcodes, or vsby < 1) |
| `rain` | not ts, and RA or DZ in wxcodes |

A row takes the first matching condition in this order: ts, fz, frost, fog, rain.

**Population:** rows **without** an ADS-B correction (the A corrector's uncovered rows), at all
airports.

## Variants (leave one month out over Jan, Jul, Aug, Sep, Oct, Nov, Dec 2025)

The reference **A+** is the base plus the v48-design ADS-B corrector (5 seeds), as in the Feb–Jun test.

- **V (v50 as built):** +30 s on frost rows in January, +200 s on ts rows in July. Fixed, nothing
  fitted.
- **W (fitted, pooled across airports):**
  - for each condition, offset = mean residual (y − A+) of that condition's uncovered rows in the
    6 training months, shrunk by n / (n + 200);
  - applied in every month, not only Jan or Jul.
- **WA (fitted per airport × condition):** the same, shrunk toward W's condition offset with
  k = 200.

**W PASSES only if all of:**
1. pooled RMSE W < A+;
2. W < A+ in at least 5 of the 7 months;
3. W still beats A+ with the 5 rows of largest gain removed.

**WA is adopted over W only if:**
- WA < W pooled;
- WA < W in at least 5 of 7 months.

V is reported for information. It is not a candidate, because its numbers were picked on this data.

## Use

- **If W (or WA) passes:** its offsets, refitted on all 7 months, **replace v50's two hand offsets**
  in v55 (the Feb–Jun build), on the 2026 rows without ADS-B. This is decided now, before results.
- **If W fails:** v55 carries v50's offsets unchanged, as `tests_febjun/PREREGISTERED.md` says, and
  the write-up states that they are not supported out of sample.
- **No weather-only upload is made** (no v51–v54 probes).
