# Transductive masked-feature pretraining for the FT-Transformer — result

**Status: FAIL. The pre-registered kill fired after seed 42. Closed.**

Bar, design and kill rule: `PREREGISTERED.md`, fixed before any run. The
original motivation (the scored set's higher join-failure rate) was withdrawn
before launch; see there.

## The kill

```
seed 42: pretrain best @ 26 (0.1748); finetune best_ep 90 val 283.11 ran 100/175
PRE-REGISTERED KILL: seed 42 best val 283.11 > 267.18 -- stopping
```

The same seed without pretraining (`tests_ftt/preds_seed42_lr1e-3_e132.npz`)
reached a best val of **263.32**, so the pretrained run is **+19.8 s worse**.
Seeds 43 and 44 were not trained.

**Convergence.** Both stages stopped on patience, not on a cap
(`pt_hit_cap = False`, `hit_cap = False`):
- pretraining stopped at epoch 29, with the best held-out reconstruction loss at epoch 26;
- fine-tuning stopped at epoch 100, with the best val at epoch 90, under the
  same patience-10 rule the baseline used.

## Best val so far, by epoch (same seed, same batch order)

| epoch | baseline | pretrained |
|---:|---:|---:|
| 10 | 352.0 | 351.0 |
| 25 | 332.8 | 330.6 |
| 40 | 319.9 | 320.2 |
| 50 | 309.1 | 312.5 |
| 60 | 298.9 | 305.8 |
| 70 | 290.1 | 300.3 |
| 80 | 281.8 | 294.7 |
| 90 | 275.2 | 283.1 |
| 100 | 268.8 | 283.1 (stopped) |

Pretraining gives a ~2 s head start up to epoch 25. From epoch 40 the pretrained
run falls **steadily** further behind (3 → 7 → 10 → 13 s). A lag that widens
over 50 epochs is structural, not one unlucky val bounce.

**The verdict does not depend on where patience fired.** Credit the pretrained
run with *all* of the baseline's improvement after epoch 90 (275.2 → 263.32 =
−11.9 s). It would end at 283.11 − 11.9 ≈ **271.2**, still above the 267.18 kill line.

## Standalone, seed 42 (descriptive)

| | baseline | pretrained | Δ |
|---|---:|---:|---:|
| val in-band | 236.68 | 248.89 | **+12.21** |
| test in-band | 228.37 | 231.60 | **+3.23** |
| val non-LIRF in-band | 210.29 | 213.15 | +2.86 |
| test non-LIRF in-band | 214.31 | 215.32 | +1.01 |

The pretrained run is worse on both surfaces and both strata. Its residual
correlation with the baseline is 0.85. That is **not** a reason to keep it as an extra
member: `tests_ftt/RESULT.md` established that decorrelation is necessary but
not sufficient, and a member that is worse standalone drags the blend. Adding
it as a fourth member would be a new hypothesis made up after the fact.

## What exactly failed, and what was not tested

**Failed:** this recipe:
- 30 % token masking with a learned per-feature mask embedding;
- missing cells excluded from the loss;
- pretraining on forward train+val+test+ranking, stopped on held-out
  reconstruction loss;
- then the full encoder fine-tuned at the baseline's lr 1e-3.

**Not tested** (low prior, since the motivation was already withdrawn; none
to be run without a new pre-registration and the user's go-ahead):
- a lower fine-tuning lr, or layer-wise lr decay. The widening lag is
  consistent with the pretrained encoder being pulled away from a
  reconstruction optimum that suits regression poorly;
- a shorter pretrain;
- the train-only control, which was only for attribution after a pass.

Cost: 1 seed, 6,867 s (~1.9 h GPU).
