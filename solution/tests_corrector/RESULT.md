# Month-fold residual corrector — result: FAIL

Bar: `PREREGISTERED.md`. `run.py` → `result.txt` / `result.json`.

The corrector (LightGBM on the blend's residuals, trained on September 2025, early-stopped on
October) stopped after **16 rounds**: there was almost nothing in the residuals it could learn
that held from September to October.

| surface | in-band, base → corrected | pooled, base → corrected |
|---|---|---|
| Nov–Dec 2025 | 215.118 → 214.783 (−0.336) | 238.373 → 238.052 (−0.320) |
| **Jan + Jul 2025** | 251.532 → 251.898 (**+0.366**) | 337.957 → 338.240 (**+0.282**) |

It helps the months next to its training data and hurts the board's months: the seventh
instance of that pattern. **Closed** alongside plan 04's residual booster and §2v's stacker.
