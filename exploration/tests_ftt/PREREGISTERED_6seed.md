# FT-Transformer 3 → 6 seeds — pre-registration

Written 2026-09-23, **before seeds 45–47 were trained**.

## Why

FTT is the one component confirmed on the board this round (v30: −3.87 vs v25).
The blend-level seed curve, taken from existing predictions with the frozen
3-member weights (weights were solved for the 3-seed mean, so k < 3 is slightly
disadvantaged):

| FTT seeds | 1 | 2 | 3 |
|---|---:|---:|---:|
| forward val in-band marginal | −0.14 | −0.68 | −0.86 |
| forward test in-band marginal | −2.58 | −3.37 | −3.63 |

The gain is still growing but flattening. Expected 3→6 is **~0.2–0.3 s local
(~0.5 board)**, below single-submission resolution. The case is low risk on a
confirmed component, not a large expected gain.

## Training (identical recipe to seeds 42–44; `run_ftt.py`, lr 1e-3, patience 10)

| split | seeds | cap / epochs | purpose |
|---|---|---|---|
| `forward_inner` | 45, 46, 47 | cap 260 (1.5 × max observed best, 171) | weight solve |
| `forward` | 45, 46, 47 | cap 175 (1.5 × max observed best, 117) | the check below |
| `full` | 45, 46, 47 | fixed 112 (as seeds 42–44) + `--predict-ranking` | submission |

A seed that hits its cap is **unresolved**: rerun at a higher cap, and never
screen or ship it truncated. **No converged seed is dropped for its score**
(forward_inner seed 44, the "failed-looking" LIRF draw, was kept for the same reason).

## Check (before any build)

Re-solve the 3-member per-airport weights on forward_inner with the **same
12-seed MLP** as `solve_vs_strong_mlp.py`, twice:
- **W3** from the FTT mean of seeds 42–44 (reproduces `solve_vs_strong_mlp.json`);
- **W6** from the FTT mean of seeds 42–47.

Apply each, frozen, to forward val and test with the matching FTT mean.

**PASS requires all of:**
1. the 6-seed blend's in-band RMSE is **lower** than the 3-seed blend's on
   **both** forward val and forward test (any margin, since the expected gain is small);
2. **LIRF unchanged:** FTT's LIRF weight solves to 0 in W6, as it has in every
   solve so far. A non-zero LIRF weight halts the build for investigation;
3. every new seed converged (`hit_cap` false on forward_inner and forward).
4. **Amendment 1 (2026-09-24, before any 6-seed result existed):** the 6-seed
   blend's in-band RMSE is also lower than the 3-seed blend's on **seasonal_full
   test (Jan + Jul)**, with weights solved on forward_inner and frozen, following
   `confirm_seasonal_full.py`. Reason: on 2026-09-23 two leads passed forward and
   failed on seasonal_full (`tests_refit_epochs`, `tests_hp_diversity`).
   Needs seasonal_full seeds 45–47 (cap 200, as seeds 42–44, whose bests were
   116–123), queued after the `full` seeds; ~5 GPU hours.

**PASS →** re-solve the `full` weights exactly as `solve_final_weights.py` did
for v30, with 6 FTT seeds. Register a new `ftt_multimodel` tag, then a
`tabnet_lirf_override` run on it, as v31 was built. Show the user the candidate
numbers before `submit.py`. The board prediction is v31's score plus the
expected small change; it cannot be cleanly confirmed.

**FAIL →** FTT stays at 3 seeds, and the result is recorded in `RESULT.md`.

## Cost

forward_inner up to ~2 h/seed (cap 260, ~28 s/epoch; seeds usually stop earlier),
forward ~1.5 h/seed, full ~1.6 h/seed. **~13–15 GPU hours total**, queued after
the current HP-diversity jobs.
