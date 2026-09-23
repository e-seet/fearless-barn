# Transductive masked-feature pretraining for the FT-Transformer — pre-registration

Written 2026-09-23, **before any code was run**. Plan 14 §4b.

## Hypothesis, stated honestly

Plan 14 motivated this with the +42 % join-failure rate in the scored set
(`DATA_QUALITY_AUDIT.md` §2). **That motivation does not hold and is withdrawn.**
The gap is 1.53 % vs 1.08 % of rows, only 0.45 percentage points. Those rows are
hard because their `_flt` columns are *missing*: `encode_data` fills them with
training medians, and no representation can recover information that isn't in
the input. The audit itself says as much.

What remains is the generic transductive hypothesis. Pretraining the FTT
encoder to reconstruct masked features on the scored period's own feature
distribution may adapt the representation to 2025→2026 covariate shift.
Fine-tuning on labels then starts from a better representation.

Two limits, stated up front:
- **The local screen measures only a within-2025 analogue.** Pretraining sees
  forward train + val + test + ranking features. The val/test rows play the
  role of "unlabeled target period".
- **The board cannot resolve it.** The local→board residual sd is ~9 s, and
  the FTT effect this sits on is ~1.5 s. This is a speculative lever whose
  board effect cannot be falsified with one submission.

## Design (the only differences from `tests_ftt/run_ftt.py`)

- **Pretraining corpus:** features only, from forward train ∪ val ∪ test ∪
  ranking (~2.43 M rows). `cat_maps`, medians and the scaler are fit on
  **train only**, exactly as in `run_ftt.py`, so tokens mean the same thing.
- **Objective:** mask 30 % of feature tokens per row and replace each with a
  learned per-feature mask embedding. Do not zero x, because x=0 still yields
  bias b, which reads as "value at the mean". Reconstruct numerical tokens
  with MSE on the standardized value and categorical tokens with
  cross-entropy. **Cells that were missing before the median fill, and
  categoricals that mapped to "unknown", are excluded from the loss.**
  Otherwise the loss would reward predicting the imputed median.
- **Pretraining stop:** early-stop on the reconstruction loss of a held-out
  2 % of corpus rows (patience 3, cap 40). `hit_cap` is saved, and a capped
  pretrain is *unresolved*, not failed (plan 14 §6).
- **Fine-tuning:** identical to the baseline in lr 1e-3, weight decay 1e-5,
  batch 1024, clipping 5.0, patience 10, and early stopping on forward val.
  Cap **175** (1.5 × the baseline best epochs of 102–117). The regression
  head is freshly initialised exactly as in the baseline. The RNG is re-seeded
  before fine-tuning, so batch order matches the baseline seed for seed.
  **Pretraining is the only difference.**
- **Seeds:** 42, 43, 44, paired against `tests_ftt/preds_seed{42,43,44}_lr1e-3_e132.npz`.

## Pass bar (fixed now)

**Screen:** `screen.py` takes the 3-member per-airport weights from
`tests_ftt/solve_vs_strong_mlp.json` (solved on forward_inner, 12-seed MLP),
**frozen**. It builds the 3-member blend twice: once with the FTT 3-seed mean,
once with the PT-FTT 3-seed mean in the same slot.

PASS requires **all** of:
1. forward **test** in-band RMSE improves by **≥ 1.0 s** (PT blend minus FTT blend ≤ −1.0);
2. forward **val** in-band has the same sign (< 0). Val is PT's early-stopping
   surface, as it is the baseline's, so it corroborates but is not the bar;
3. at least 2 of 3 paired seeds improve standalone on forward test in-band.

Anything else is a FAIL, including a test gain under 1.0 s.

**Early kill after seed 42:** stop if PT seed 42's best forward-val RMSE is worse
than the worst baseline seed (**267.18**). This catches outright failure only;
passing it is not evidence of a pass.

**Control (only if the screen passes):** pretrain on train rows only, same
recipe. This attributes the gain to the *transductive* part, as opposed to
pretraining per se. Shipping does not depend on it.

## Cost

About 2.5 h per seed (≤ 1 h pretrain + ~1.5 h fine-tune), ~7.5 GPU hours for the
screen. If it passes: forward_inner seeds (for an honest weight) and `full`
refits add roughly another 9 h before anything is submittable.
