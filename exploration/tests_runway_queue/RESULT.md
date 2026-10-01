# Outside-analysis strategies 1–3 — result (all closed)

Pre-registration: `PREREGISTERED.md`. Board at the time: v48 = 272.1714 (best and last).

## Strategy 2: looser ADS-B matching at EDDF and LEMD — closed (measured on the data)

Every strict-match failure (sampled days 2026-01-15, 2026-07-15, 2026-07-22 and 2025-07-15) has
**zero ground points**: the aircraft were seen only airborne. Looser matching would add rows with no
ground movement, which is the only thing the corrector uses.

## Strategy 3: routed taxi distance from apt.dat — closed on its ceiling (`routed_ceiling.txt`)

- 0.15 % of Jan + Jul 2025 held-out rows have an unseen `stand_runway` pair.
- A perfect distance feature on those rows would gain at most **−0.14 s** pooled, under the 0.5 s bar.
- 2026: 0.72 % of rows have unseen pairs (EDDF 2.4 %, EDDM 4.6 %), so the 2026 ceiling is about
  0.5 s at most, before any real-world loss.

## Strategy 1: runway queue and runway mode — FAIL

**Features** (`queue_features.py`, from the movement records, only movements before *T*):
- departures on the same runway in the last 15 and 30 min;
- the gap to the previous departure on that runway;
- arrivals on that runway in the last 15 min;
- distinct departure and arrival runways in the last hour;
- mixed mode.

**Drift gate:** passed; every |SMD| ≤ 0.09. These record-based features are stable between years.

**Leave one month out** (A+ = the base plus the v46-design ADS-B corrector; candidate = A+ plus corrector
R on the 606k rows A does not touch):

| month | A+ | candidate | change |
|---|---:|---:|---:|
| Jan | 349.90 | 352.69 | +2.79 |
| Jul | 382.19 | 386.82 | +4.63 |
| Aug | 262.12 | 264.73 | +2.62 |
| Sep | 274.66 | 274.06 | −0.60 |
| Oct | 214.71 | 217.49 | +2.79 |
| Nov | 271.06 | 269.78 | −1.29 |
| Dec | 242.54 | 245.28 | +2.74 |
| **pooled** | **290.29** | **292.39** | **+2.11** |

- Worse in 5/7 months, and still worse with the best 5 rows removed (275.73 → 278.98).
- Per airport: LTFM −0.8, LEMD −0.4, LFPG +0.9, **LIRF +10.0**.
- The shuffled control is worse still (293.73), so the queue features carry a little information.
  But a residual corrector on rows without ADS-B mostly refits the base's own errors and loses: the
  same failure as the stage-2 residual boosting closed earlier (`TAXI_OUT_DRIVERS.md` §2v), and
  consistent with the earlier congestion and runway-configuration closures.

**Verdict: FAIL** on pooled, months and best-5-removed. v48 stays final.
