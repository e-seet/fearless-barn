# LIRF destination/runway "flight path" link — residual correlation result

**Question 1 (runway): can the corrupted rows be singled out by runway?**
Mostly no. Runway 25 carries 88% of the unmatched population at the SAME
rate as the overall population (overrepresentation ratio 0.98 — exactly
proportional). The two reverse-configuration ends of LIRF's minor runway
(16L/34R, used when wind favours the opposite direction from the dominant
setup) show a real but small-sample elevated *incidence* (9.4x / 5.6x,
n=14/21 of 1,488) — everything else, including `NA` (missing runway,
49.5x but only 11 rows), is small-count noise territory. The vast majority
of the corrupted population uses the exact same main runway as everyone
else.

**Question 2: does the destination pattern found in `run_lirf_flightpath_export.py`
(regional Italian hops + North-African/Middle-Eastern leisure charter
routes) actually help — i.e. does it predict WHICH already-flagged rows
have the worst error, not just which routes are more likely to be flagged
in the first place?**

**Result: no. Both signals collapse to noise once tested against the
model's residual.** Restricted to LIRF's `has_issue` (no flight record)
rows in `forward` val+test (n=423, consistent with §2q's earlier val-only
count of 283 plus a proportional test share):

| predictor | rho(resid) | permutation-shuffled control |
|---|---:|---:|
| `ades_overrep_ratio` | −0.044 | **+0.049** |
| `is_reverse_runway` | +0.012 | −0.011 |

**The real correlation is SMALLER in magnitude than a single random-shuffle
draw** in both cases — this is not "weak but real," it's indistinguishable
from noise. Mean |residual| for reverse-runway rows (2,552.8s, n=16) vs.
everyone else (2,483.1s, n=407) is a 2.8% difference on a tiny, high-variance
subsample — not a real effect. The in-sample OLS "ceiling" (−161.7s on a
4,485s baseline within this subpopulation) is exactly the kind of overfit
artifact this project's discipline exists to catch: fitting 2 free
parameters on n=423 of a heavy-tailed target, when the underlying
correlations don't even clear their own permutation baseline, produces an
in-sample number that will not survive out-of-sample.

## Reading — an important distinction

The destination pattern from `run_lirf_flightpath_export.py` is still real
as an **incidence** finding: certain routes (regional Italian hops,
North-African/Middle-Eastern charter destinations) are more likely to
produce a flight with no NM record in the first place, plausibly because
low-frequency charter carriers are less consistently tracked than mainline
scheduled ones. **But knowing a flight is headed to one of those
destinations does not tell you how large its resulting taxi-time error will
be once you already know it lacks a flight record.** Incidence and
magnitude are different questions, and this population's error magnitude
looks like it depends on something not captured by destination, runway, or
(per §2q) anything else in the available columns.

## Disposition

**Closed.** Reinforces §2q's conclusion ("the remaining error looks close
to irreducible from the columns available") from a fourth independent
angle (destination and runway, joining the blend-weight solver, the OpenAir
proxy cross-check, and the convex mix). Not built. No feature, no retrain.
