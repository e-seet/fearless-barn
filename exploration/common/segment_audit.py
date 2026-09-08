"""Segment-viability audit: how much data actually backs a candidate segment.

Why this exists
----------------
`pipeline_airport_debiased_multimodel` fits a 3-parameter ridge regression
(intercept + weight on the base prediction + weight on `delta_sched_sec`) on
LIRF rows with `has_flight_record == 0` in two delay bands. Checked directly
against the data (not assumed): the "mid" band (25,200-50,000s delay) has 34
rows in the ENTIRE year and the "extreme" band (>50,000s) has 20 -- of which
only 15 and 7 respectively land in the forward_inner fitting surface, and only
5 and 4 in the real val split. A free regression fit on a handful of points
and judged by its effect on an even smaller handful of evaluation rows is the
textbook shape of overfitting to high-leverage points, not a discovered
pattern -- regardless of how good the resulting story sounds.

What this tool actually shows, so that call doesn't have to be guessed:
- how many rows exist in the full year, the honest fitting surface, and each
  evaluation part, for any candidate segment;
- whether a physical identity claimed for that segment (y approx a candidate
  feature) is exact (near-zero residual variance -- safe to encode as a
  zero-parameter RULE) or merely correlated (residual variance is real, so
  ANY fitted correction is fitting noise at these sample sizes);
- an explicit OVERFIT RISK flag once fitted free parameters exceed the rule
  of thumb of >=30 rows per parameter, scaled to the SMALLEST surface that
  matters (the fitting surface, not the full year -- a segment can look
  plausible on 34 yearly rows and still be fit on 15).

Use this BEFORE building a pipeline around a new segmentation idea, not after
its val/test number already looks good.
"""
from __future__ import annotations

import numpy as np

MIN_ROWS_PER_FREE_PARAM = 30


def segment_report(name: str, *, n_full_year: int, n_fit_surface: int,
                   n_val: int, n_test: int, n_free_params: int = 0,
                   residual=None, residual_label: str = "y - candidate") -> dict:
    """Print and return a viability report for one candidate segment.

    `residual`, if given, is the array of (y_true - candidate_baseline) values
    on the FULL-YEAR rows in this segment -- pass it whenever a "this segment
    is basically deterministic" claim is being checked, so the claim is judged
    by its actual variance rather than by how good the resulting RMSE looked.
    """
    min_safe = n_free_params * MIN_ROWS_PER_FREE_PARAM
    at_risk = n_free_params > 0 and n_fit_surface < min_safe

    print(f"--- segment: {name} ---")
    print(f"  rows: full_year={n_full_year}  fit_surface(forward_inner val)={n_fit_surface}  "
          f"real_val={n_val}  real_test={n_test}")
    if n_free_params:
        print(f"  free parameters to fit: {n_free_params}  "
              f"(rule of thumb needs >={min_safe} rows on the fitting surface)")
        print(f"  {'OVERFIT RISK -- fitting surface is below the safe threshold' if at_risk else 'sample size adequate for the parameter count'}")

    resid_stats = None
    if residual is not None and len(residual) > 0:
        r = np.asarray(residual, dtype="float64")
        resid_stats = {"n": int(r.size), "mean": float(r.mean()), "std": float(r.std()),
                       "min": float(r.min()), "max": float(r.max())}
        exact = resid_stats["std"] < 30.0   # a few seconds/rounding, not real variance
        print(f"  {residual_label}: n={resid_stats['n']} mean={resid_stats['mean']:.1f} "
              f"std={resid_stats['std']:.1f} range=[{resid_stats['min']:.1f}, {resid_stats['max']:.1f}]")
        print(f"  -> {'looks like a near-EXACT identity (safe to encode as a zero-parameter rule)' if exact else 'real residual variance -- a fitted correction here is fitting noise at this sample size, not learning a pattern'}")

    return {
        "name": name, "n_full_year": n_full_year, "n_fit_surface": n_fit_surface,
        "n_val": n_val, "n_test": n_test, "n_free_params": n_free_params,
        "overfit_risk": at_risk, "residual_stats": resid_stats,
    }
