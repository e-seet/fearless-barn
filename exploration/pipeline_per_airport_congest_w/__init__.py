"""SCRATCH robustness variant of pipeline_per_airport_congest.

Identical in every respect EXCEPT extra.WINDOWS_MIN = (15, 120) instead of
(30, 60). Exists only to answer "does the congest gain DEPEND on the window
length that was fixed up front?" -- i.e. to measure fragility, not to search for
a better window. Nothing here is a candidate for adoption: picking a window by
its val score is exactly the threshold-tuned-on-the-evaluation-surface failure
that put six arms in overtuned/.

Note: `dep_realised_taxi_n_prev_60m` is hardcoded at 60 min in extra.py (its name
and its lookback are literals, not derived from WINDOWS_MIN). It is deliberately
left at 60 here, so the ONLY thing that changes between the two arms is the two
mean-window lengths.
"""
