# Season-weighting board probe — decided after the local FAIL

Written 2026-09-26, **before any probe run**. The user asked for a board probe of a
candidate that **FAILED its pre-registered local bar** (`RESULT.md`): Jan + Jul
in-band +1.70, LIRF +10.5, while forward was −6.09. The two local surfaces disagree
strongly; the board is asked to settle it. **Expectation: worse than v35
(287.9247)**, mainly through LIRF, which is MLP-only in the blend and weighs
~2× on the board. **Ranking uses the last submission:** if the probe loses, v35 is
rebuilt byte-identically and uploaded last.

## Build (fixed now)

- **MLP slot** = season-weighted MLPs (τ = 3, as the screen), **18 seeds (42–59)** to
  match v35's 18-seed MLP, one model per target month; each row takes the model for
  its own month.
  - `forward_inner` (train Jan–Jun, val Jul–Aug, the weight-solve surface): targets
    7 and 8, early stopping on val (cap 50, patience 6, as the screen).
  - `full` (all 2025, no val): targets 1 and 7, **fixed 17 epochs** = the
    median best epoch of the seasonal_full t01/t07 screen runs (17 / 16.5 → 17).
- **Everything else as v35:** tree (airport_operator_blend seed 42), FTT 6 small
  seeds, 3-member per-airport weights re-solved on forward_inner with the default
  solver, and the TabNet LIRF override at w = 0.459 (5 seeds). FTT's LIRF weight
  must solve to 0.
- Built by `build_probe.py`, deterministic and committed. This is **not** a registered
  `submit.py` arm; if it wins it gets registered before any further use.
- Pre-upload label-free size: RMS(probe − v35), reported against the v31→v32
  (20.4 → −1.79) and v32→v34 (7.5 → tie) yardsticks.
