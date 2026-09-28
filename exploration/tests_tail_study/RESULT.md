# Tail study (2026-09-29) — what the 2025 tail looks like, and why it does not transfer

Read-only. `explore.py` → `explore.txt` (all 2,084,659 labelled 2025 departures);
leaderboard pulled from the public competition endpoint (164 teams, 3,000 entries).

## Where we stand

v41 = 286.80 is **#70 of 164** by best score (#67 by last submission). The leader
(strong-crane) is at 221.60, #10 at 237.0, #25 at 252.1, the median team at 292.4. The
leader has **40 % less squared error** than us. `tests_tail_diagnosis` puts 44 % of our
squared error (on the Jan + Jul 2025 analogue) in ~150 rows above 7,200 s. A team that
handled the tail and was otherwise no better than us would land at ≈ 287 × √0.56 ≈ 215–220.

## The 2025 tail is almost entirely LIRF without a flight record, and structured

| LIRF, no flight record, scheduled delay `ds` | rows | y = ds (±60 s) | y ≥ 80 k ("one-day") |
|---|---:|---:|---:|
| 25.2 k – 50 k | 34 | **100 %** | 0 % |
| 50 k – 60 k | 9 | 67 % | 33 % |
| 60 k – 70 k | 6 | 33 % | 67 % |
| ≥ 70 k | 5 | ~20 % | ~80 % (y ≈ 86,400 + ~800–1,200 s) |
| other airports, any `ds` | — | ~0 % | ~0 % (normal taxi ~1,000 s) |

16 rows ≥ 80 k in all of 2025; 14 are LIRF.

## Why this is not a lever

The 25.2 k–50 k identity is exactly the `idband` rule, and **it regressed on the 2026
board twice** (v15, v17: +0.86). The pattern that is perfect in 2025 does not hold on the
scored 2026 rows. The v18 cap (491) also shows our large predictions are mostly *right*
in aggregate, so the tail is not simply over-predicted. A 2025-learned tail rule is
therefore not safe to ship, and no rule is proposed.

## What the leaderboard suggests about the gap

- Submission count correlates with rank (Spearman −0.64; median highest version: top 10
  v66, ranks 51–100 v18, bottom v6). Because the score is a mean of squared errors, a
  one-row change between two uploads reveals that row's truth, so heavy submitters *can*
  learn the tail from the board. This is leaderboard probing, which we have ruled out, and
  which a hidden final stage (the moderator: "possibly 1 final submission") would erase.
- **But the gap is not only probing:** vigorous-whistle is #2 (224.5) at v41, the same
  count as us, and three top-10 teams sit at 235–237 with 22–29 uploads. There is a
  genuine ~50 s methodological gap at similar effort. The rank-3 team reports using
  adsb.lol ADS-B, METAR weather and X-Plane stand coordinates, and the moderator confirmed
  open data is allowed.

## Implication

Model-side tuning (FTT variety: −0.64, −0.25, −0.23) cannot close a ~50 s gap. The
plausible sources are external data aimed at the tail, or something about the tail's 2026
behaviour we cannot see locally. Our earlier ADS-B check (`tests_adsb_coverage`) tested
off-block recovery, not the long-delay LIRF rows specifically.
