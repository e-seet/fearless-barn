# Copy-mixture member — result (2026-10-08)

**Pre-registered verdict: PASS.** Pooled 266.056 → 265.046 (−1.010); 6/7 months (Jan +0.015 the loss); best-5 removed
265.642 → 264.985; 64 % of the squared-error gain from body rows (y ≤ 3,600 s).

**Post-hoc decomposition (not pre-registered, reported because it changes the expectation).** The reference F was
not clipped to the LOBT window, while the combination was. Against F *with* the LOBT clip (the v62 analogue):

| month | Jan | Jul | Aug | Sep | Oct | Nov | Dec | all 7 | Jan+Jul |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| copy blend vs clipped F | +0.04 | +0.37 | −1.26 | −0.52 | −1.10 | −1.05 | −1.94 | −0.63 | **+0.22** |

The July gain in the pre-registered table was the clip (0.02 % of rows, −1.63). The copy member itself helps
Aug–Dec and not the two board months.

**LFPG construction check** (`lfpg_check.py`): on rows where the stage-5 LFPG corrector fires, the blend with the
corrector delta added on top is still better (258.52 → 256.35), so the build adds the stage 3–5 deltas back in full.

**2026 drift** (`weights_v63.json`, `build_M.log`): mean p on ranking stays near the 2025 copy rates, but
MVT − SCHED shifts (Jan 2026 medians 300–400 s later at EDDF, LSZH, LTFM; July earlier). The copy branch uses
MVT − SCHED directly, so this is the main transfer risk.

**Build:** `fearless_barn_v63.parquet` = lobt_clip((1 − w)·S2 + w·M + (v61 − S2)); S2 from
`final_pipeline.py --s2-out` (v61 rebuild byte-identical). 280,232 rows changed, RMS 23.1 s vs v61.
Expectation written in `board_scores.csv` before any upload.

Credit: two-stage copy idea and the LOBT window from team outgoing-boat (github.com/EnioAguiar/prc-taxiout-2026).
