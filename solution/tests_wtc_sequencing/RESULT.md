# WTC sequencing (neighbour wake-turbulence category) — result

**Question.** A community post (`conversation.md` sec.5, explicitly
speculative, not a claim) asked whether the SEQUENCE of aircraft wake-
turbulence category (WTC) drives taxi-out time: ATC must sequence departures
for wake-separation minima, so a light aircraft immediately following a
heavy one might need a longer taxi/hold than the reverse ordering would.
The aircraft's own WTC (`WK_TBL_CAT_flt`) is already a native feature every
arm sees; untested was a NEIGHBOUR's WTC relative to this flight's — the
closest existing work, `pipeline_queueattn_antigravity` (sec.2k), attended
over each departure's 8 nearest neighbours but never fed in neighbour
aircraft type/WTC, and that architecture closed flat on what it did use.

**Setup — cheap diagnostic before touching the (expensive, closed-flat)
architecture,** matching this project's established discipline (sec.2h,
sec.2o, sec.2p, sec.2z all ran a correlation/OLS-ceiling check before
deciding whether to build). `tests_wtc_sequencing/run_wtc_sequencing.py`:
for every DEP row, whole-period (all 12 months, before split-subsetting),
found the immediately PRECEDING departure at the same (airport, runway) —
backward-looking only, since wake-separation is a backward physical
constraint by definition, unlike general congestion features. Ordinal-
encoded WTC (L=0 < M=1 < H=2 < J=3, ICAO wake-strength order) and built
`prev_heavier` (1 if the preceding aircraft's WTC ordinal exceeds this
flight's own — the wake-relevant direction, light behind heavy) and
`wtc_diff` (signed, more general). Correlated against `mlp_antigravity`
(forward split, seed42) val residual — the same baseline the closed
queue-attention architecture extends — plus a permutation control (shuffle
neighbour assignment).

## Result — a real, mechanistically consistent effect, capped well under the board-visibility floor

**Coverage is excellent**: 97.7% of val rows have a usable same-runway
preceding-departure pair with both WTCs known (WTC itself is 98.9% populated
overall).

**The correlation is real, not a construction artifact**: rho(prev_heavier,
resid) = **+0.0299**, collapsing to −0.0007 under the permutation control
(shuffled neighbour assignment). Raw group means show a striking gap: mean
residual is **+22.44s** when the preceding aircraft was heavier vs. **+2.47s**
otherwise (n=45,953 vs. 315,146) — the model systematically UNDER-predicts
taxi time specifically when a heavier aircraft just used the same runway,
exactly the direction wake-separation theory predicts.

**Consistent in sign at all 10 of 10 airports** — unlike most levers closed
in this project (which typically concentrate in one airport or split
sign), every single airport shows rho > 0 and a positive resid gap (12.9s
to 37.8s), from LIRF's smallest (rho=+0.009, resid gap 12s) to EHAM's
largest (rho=+0.058, resid gap 28s). This is a genuinely uniform, physically
well-motivated mechanism, not an airport-specific artifact.

**But the pooled ceiling is negligible.** In-sample OLS correction (fit
resid ~ prev_heavier, apply the fitted correction, recompute pooled val
RMSE): baseline 253.989 -> corrected 253.856, a ceiling of **only -0.134s**.
The affected subpopulation (`prev_heavier=1`) is only 6-18% of rows per
airport, and the model's existing features (own WTC, runway identity,
congestion counts, stand_runway) evidently already absorb most of what this
signal would add — the same "the model already has this information in some
form" pattern that closed most other candidate features this project has
tested (own-WTC as a direct categorical, runway congestion counts, and
stand-level aggregates all partially proxy for "was a heavy aircraft here
recently").

## Disposition

**Closed, decisively, on the ceiling — but confirms the underlying physics,
which most closed levers in this project do NOT.** -0.134s is an order of
magnitude under the ~5s board-visibility floor this project has repeatedly
demonstrated (sec.2o) is needed before a local gain is reliably board-
visible. Not built into `features.py`, no retrain, no extension to
`pipeline_queueattn_antigravity` (which would cost a multi-seed GPU retrain
for a ceiling this small even under optimistic conditions). Worth recording
distinctly from a typical closure: the forum poster's physical intuition was
correct and empirically confirmed (a real, permutation-verified, uniformly-
signed effect across every airport) — it simply isn't large enough in
aggregate to move this specific pooled-RMSE metric, joining sec.2h/sec.2p/
sec.2z as another case where "real physical mechanism, negligible aggregate
effect" turned out to be the answer rather than "no effect at all."
