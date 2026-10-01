# Weather offsets for flights without ADS-B, validated out of sample — result

Pre-registration: `PREREGISTERED.md`. Leave one month out over 7 months; offsets fitted on the
residuals of the other 6 months (after the v48-design ADS-B corrector), shrunk by n / (n + 200).

**Rows without ADS-B, by condition:** rain 28,033; fog 10,727; frost 7,287; ts 1,493; fz 587;
none 558,065.

| month | A+ | V (v50 hand offsets) | W (fitted, per condition) | WA (per airport) |
|---|---:|---:|---:|---:|
| Jan | 349.834 | −0.066 | **+0.187** | +0.490 |
| Jul | 382.167 | −0.310 | −0.274 | −0.212 |
| Aug | 262.100 | 0 | −0.032 | −0.020 |
| Sep | 274.643 | 0 | −0.012 | −0.003 |
| Oct | 214.686 | 0 | −0.242 | −0.258 |
| Nov | 270.945 | 0 | −0.476 | −0.583 |
| Dec | 242.416 | 0 | −0.374 | −0.439 |
| **pooled** | 290.237 | −0.073 | **−0.161** | −0.123 |

**W: PASS.**
- pooled −0.161; better in 6/7 months;
- with the best 5 rows removed: 277.533 → 277.408.

**WA:** better than W in only 3/7 months, so not adopted. The per-airport frost amounts that
v51–v54 were built on are **not supported**.

**Offsets fitted on all 7 months (used in v55):** ts +199 s, fz +203 s, frost +72 s, fog +40 s,
rain +11 s.

**Caveats:**
- **January is the one month where W hurts (+0.19).** The frost offset learned without January
  (+129 s, mostly from Nov–Dec) overshoots January 2025. Jan is half of the scored months, so this
  is the main risk of W on the board.
- **v50's offsets (V) are mildly supported** (−0.07 pooled; Jan −0.07, Jul −0.31), even though they
  were picked in-sample.

**Decision** (fixed in the pre-registration): W's offsets replace v50's two hand offsets in v55.
There is no weather-only upload.
