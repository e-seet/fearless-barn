# FT-Transformer screen — bar registered BEFORE any run

Written 2026-09-21, before a single FT-Transformer epoch was trained. Nothing
in this file may be edited after the first number exists; results go in
`RESULT.md`.

## Why this architecture

Four architectures have been closed in this build by varying how the model
consumes the **movement stream**: the GRU state encoder (`tests_seqrnn`,
residual corr **0.980** vs the MLP), bidirectional queue-attention
(`pipeline_queueattn_antigravity`), the matched/unmatched router (0.898/0.876),
and TabR-style retrieval (failed at the screen). `tests_seqrnn/RESULT.md`
states the conclusion outright: richer per-flight neighbour/sequence
representation is foreclosed without a different data source.

One architecture has paid: **TabNet**, residual corr **0.88–0.94**, shipped as
a LIRF-only override (v29 = 293.6685, −1.54 vs v25). The pattern across all
five is that **residual correlation against the MLP ensemble predicted the
outcome** — ≥0.95 died, 0.88–0.94 paid.

FT-Transformer attends over **features**, not neighbours. It is the one
tabular family with no entry in this repo, and its inductive bias is the
axis the closed four have in common.

## What is being tested

Only whether column-attention produces **decorrelated** errors worth an
ensemble slot. Standalone accuracy is secondary — TabNet merely *tied* the MLP
standalone and still paid, while the router beat a pooled LightGBM by 50 s and
was worthless because it lost to the arm it would have to replace.

## The bar

Screen = `decorrelation.py`, the same construction as
`tests_tabnet/decorrelation.py`: paired random draws over the 24-seed MLP pool
in `tests_lirf_seeds/`, comparing

* **A** = 3 MLP + 3 *other* MLP seeds (equal member count baseline)
* **B** = the same first 3 MLP + 3 FT-Transformer seeds

**PASS requires all four:**

1. residual `corr(FTT-ens3, MLP-ens3)` **< 0.95** on both val and test;
2. mean gain (A − B, positive = FTT better) **≥ 2 s** on test pooled;
3. mean gain **> 0** on val pooled;
4. B beats A in **≥ 80 %** of draws on val.

Reported separately but NOT part of the bar: LIRF vs non-LIRF split (the board
is LIRF-driven, `tests_local_vs_board`), and any weight sweep, which is
descriptive only and must not be used to select.

## Convergence gate (added because of this session's audit)

A seed whose run ended **at** `--epochs` is truncated, not converged, and is
**excluded from the screen** — the same gate `tests_tabnet` applies (which
rejected final seeds 47 and 51). `run_ftt.py` saves `hit_cap` in every npz.
The first run's observed best epoch sets the real cap at ≥1.5× it, per
`tools/ml_common/convergence.py`; the initial 60 is a placeholder, not a
measurement.

## Seeds and stopping rule

Three seeds (42, 43, 44) for the screen. This build's rule is multi-seed before
trusting any delta — seed 42 alone looked like a real win for both the GRU and
TabNet, and averaging erased it in both cases.

**If the screen fails, it closes here**, with no hyperparameter sweep. The
`--lr` flag exists to check optimiser stability (a rerun is warranted if a seed
shows TabNet's lr-2e-2 signature: best epoch ≤ 4, or a val spike of several
hundred seconds), **not** to search for a configuration that passes the bar.
Re-running a failed screen at new settings is exactly how the `overtuned/`
ladder happened.

---

## Amendment 1 (2026-09-21): lr 1e-4 -> 1e-3, on convergence grounds

**Made before any screen number existed, and it could not have been otherwise:
`decorrelation.py` refuses to screen a seed that hit its cap, so no PASS/FAIL
was computable at the time of this decision.**

The lr 1e-4 run (`run_lr1e-4_ABORTED.log`, seed 42, aborted at epoch 15 of 60)
was monotonically improving with no sign of a plateau:

| epoch | 1 | 5 | 10 | 15 |
|---|---|---|---|---|
| val RMSE | 499.80 | 483.42 | 451.68 | 419.73 |

~6 s per epoch, still falling, against an MLP reference of ~256. Extrapolating,
it would still have been improving at epoch 60 — i.e. it would have ended AT
the cap and been excluded by this file's own convergence gate. That is an
under-trained run, not a verdict on column attention.

**The decision rule being applied is the shape of the convergence curve
(monotone, no plateau, would hit the cap), not the screen's outcome.** This is
the distinction the whole convergence audit turns on, and it has a direct
precedent in this folder's sibling: TabNet's original "decisive loss" at lr
2e-2 was an optimiser artefact and was retracted after a rerun at 2e-3. That
was lr too high; this is lr too low. Same error, opposite sign.

New setting: **lr 1e-3, weight decay 1e-5, everything else unchanged.** 1e-3 is
the registered `mlp_antigravity` learning rate, which also tightens the
"only the architecture differs" claim rather than loosening it.

**This amendment does not touch the bar.** All four PASS conditions stand
exactly as written above. If lr 1e-3 is unstable — the TabNet-2e-2 signature,
best epoch <= 4 or a val spike of several hundred seconds — the fallback is
3e-4, recorded here in advance. **No third lr.** If a converged run fails the
bar, it closes; re-running a failed screen at new settings is how the
`overtuned/` ladder happened.
