# TabNet — CURRENT STATUS (read this section first)

Last updated 2026-09-21. **BOARD RESULT (submitted 2026-09-21):** v29 (LIRF mix w=0.55) scored
**293.6685**, v28 (w=0.80) scored **294.4929**, vs v25 295.2054 -> new best v29, -1.54 (v28 -0.71).
The LIRF override works on the board but the gain is small. Board pooled MSE drop: 905 (v29), 420 (v28);
the same quantity locally is 673 on the forward test (Nov-Dec) and 7,788 on seasonal_full (mostly
July-2025), so the board behaves like the forward-test surface, ~9x below seasonal_full: the
scored LIRF tail is not as extreme as July 2025's, and my LIRF-model board prediction (+3.7 to +23)
was too optimistic. Lower TabNet weight did better on the board (0.55 > 0.80); a quadratic fit
of board MSE vs w gives an optimum near 0.46 (~0.1 better than 0.55, not worth a slot).
Everything below this line is the pre-submission analysis (local 2025 surfaces).

**Where it stands**
- TabNet at lr 2e-3 converges and TIES the MLP standalone. The original lr 2e-2 analysis
  (bottom of this file) is SUPERSEDED: its "decisive loss" and "3x noisier" claims were
  optimiser-instability artefacts. Do not cite it.
- As an ensemble member it is decorrelated from the MLP (residual corr 0.88-0.94 vs 0.98
  for the GRU state model) and the gain is entirely at LIRF: a LIRF-only mix
  `pred_LIRF = (1-w)*v25 + w*TabNet` (v25's LIRF prediction is 100% MLP).
- Variant matrix (weights solved only on forward_inner, evaluated on forward test and
  seasonal_full test; `blend_variants.py`, ungated inner seeds): LIRF w solved = 0.55.
  B1 PASS (V1 beats shipped by +11.20 pooled on seasonal_full, +1.30 on forward). The full
  per-airport 3-way simplex, a global TabNet weight and a gated LIRF mix all fail to beat
  the LIRF-only recipe by 2s on both surfaces, so LIRF-only stays.
- Uncertainty (`bootstrap_gain.txt`, day-clustered bootstrap of the LIRF RMSE gain): pooled
  over the 4 test months +45.7 (95% CI +8.9 to +85.4); forward test alone +9.0 (CI -27.6 to
  +42.5, P>0 = 69%); seasonal_full alone +68.6 (CI +14.7 to +124.7). By month: Jan +31.6,
  Jul +86.6, Nov +35.9, **Dec -26.1**. TabNet wins on extreme-tail months and loses on a
  moderate-tail month, so the size of any effect depends on how extreme LIRF's scored tail is.

**Epoch cap 60 -> 75 (2026-09-21): the two cap-rejected seeds were already at their optimum.**
Final seeds 47 and 51 had been excluded by the health gate for ending AT the 60-epoch cap
(best epochs 50 and 47). Re-run at cap 75 (= 1.5x the observed natural best of 50, the rule in
`tools/ml_common/convergence.py`) they early-stopped at 66 and 63 with the **same best epochs**
and **byte-identical predictions** -- 0 of 344,841 ranking rows and 0 of 165,677 Dec-val rows
differ. The cap had only stopped patience-15 from confirming a stop that had already happened.
So `hit_cap` in the gate means *unresolved*, not *failed*: raise the cap and re-run to resolve
it rather than discarding the seed. Healthy seeds 9 -> 11; `train_tabnet_pred.py` now defaults
to cap 75 and saves the model binary, not only predictions. Truncated runs preserved as
`build/TRUNCATED60_final_seed{47,51}.*`. Re-ran `seed_curve.py` with the recovered seeds
(`seed_curve_out_cap75.txt`): original 5 seeds LIRF Dec-val 347.7 -> all 11 healthy 345.9,
**+1.8 s against the >= 6 s bar. Still closed, nothing to submit.** Full write-up:
`../CONVERGENCE_AUDIT.md` sec.3.

**Extra seeds (2026-09-21, `seed_curve.py`): CLOSED, bar not met.** Six more final seeds (49-54): 4 healthy, 51 hit the epoch cap, 54 poor (Dec val 294); with seed 47 also cap-excluded, 3 of 12 final runs were excluded by the gate, 9 healthy. TabNet-ensemble LIRF RMSE on Dec val (mean of random subsets): k=1 384.2, 3 358.2, 5 351.8, 9 348.2; the original 5 seeds score 347.7 vs 348.2 for all 9 (-0.5 s against the registered >= 6 s bar). No new candidate; v29 stands.

**Open risks (why no build is final)**
1. Transfer to the scored months: the seasonal_full gain is mostly July 2025; the scored July
   is 2026, ~7 months past the refit cutoff. Not testable locally.
2. `idband` precedent: a local-neutral LIRF override cost +0.86 board twice (v15, v17).
   Much larger locally here, same category.
3. Local-to-board mapping (tests_local_vs_board): board ~ -246 + 2.24 x local, residual sd
   ~9 board-seconds inside the MLP cluster; the LIRF model (k ~ 2.05) is an assumption.
4. Weights are frozen from v25 (not re-solved); w solved on forward_inner Jul-Aug with 3 TabNet
   seeds (seed 42 val 375 is weak; the gated solve with a replacement seed is pending).
5. Final-model seeds (Jan-Nov 2025 train, Dec 2025 early-stopping val): 6 aligned seeds trained
   (Dec val 228.5 / 229.5 / 231.0 / 237.3 / 240.9, seed 47 = 234.9 but ran the full 60-epoch cap,
   best epoch 50, so the gate excludes it). 5 healthy seeds are used.

**Weight `w` is unstable across inner-seed sets (a finding, not a footnote)**
- forward_inner solve, all 3 inner seeds incl. weak seed 42 (val 375): w = 0.55 (`blend_variants_gateoff.txt`).
- Gated solve, healthy seeds 43/44/45 (val 321/319/321 vs MLP-ens18 311.6): w = 0.80, the upper
  guard [0.3, 0.8] (`blend_variants_gated.txt`). Verdicts identical either way: B1 PASS
  (seasonal_full +13.68, forward +1.13 at w=0.8); 3-way simplex / global g / gated-LIRF do not
  beat LIRF-only by 2s on both surfaces, so LIRF-only stays.
- `w_sensitivity.txt`: pooled LIRF RMSE gain over the 4 local months rises with w
  (+29.0 at 0.3, +42.9 at 0.5, +45.7 at 0.55, +54.8 at 0.8, +56.6 at 1.0) but so does the
  December downside (-11.6, -22.9, -26.1, -44.6, -61.8) and the bootstrap interval widens
  (w=0.55: +9.0..+84.1; w=0.8: +4.3..+109.3). A risk/return choice, left to the user.

**Candidate files (LOCAL ONLY, not in submissions/, not uploaded)**
- `build/candidate_v28_w0.8.parquet` (registered protocol) and `build/candidate_v28_w0.55.parquet`
  (ungated solve): v25 with the 26,899 LIRF rows mixed with the mean of 5 healthy TabNet seeds;
  every other row byte-identical to v25; all integrity checks pass (ids/order, dtype, clip range,
  no NaN, non-LIRF rows unchanged, no runaway tail). On the ranking rows the LIRF predictions
  correlate 0.990 with v25's; mean |change| 56.6s (w=0.8) / 38.9s (w=0.55), p99 572s / 393s.

**Errors found and fixed in this thread (kept on record)**
- lr 2e-2 instability (above) -> reran at lr 2e-3.
- `train_tabnet_pred.py final` bug: passed the unfiltered y with an outlier-filtered X, shifting
  labels after the first dropped row; seeds 42/43/44 trained on scrambled labels (val ~480, best
  epoch 2-3). Retracted my mistaken "TabNet fails ~1 run in 4" claim. Fixed with an X/y length
  assert, a corr(delta_aobt_sec, y) alignment check, a `labels_aligned` marker on saved seeds, and
  a health gate in `build_candidate.py` (Dec val <= 260, best epoch >= 10, early-stopped).

---

## Stage-by-stage notes (chronological)

**CORRECTION (2026-09-20): the original closure below was confounded by
unstable optimisation and is superseded by this section.** The original
runs used lr 2e-2; seeds 43/44 hit their best val at epoch 4 and seed 43's
val spiked to 1112 at epoch 10 (an optimiser-instability signature, not a
converged result). Rerun at lr 2e-3 (cap 60 epochs, patience 15), seeds
42/43/44, `TABNET_LR=0.002 TABNET_TAG=lr2e-3`. All three converged: early
stopping fired (best epochs 29/41/29, ran 45/57/45 of 60, none at the cap).

| seed | val | test |
|---|---:|---:|
| 42 | 265.71 | 275.98 |
| 43 | 258.57 | 240.63 |
| 44 | 260.98 | 278.37 |
| mean (sd) | 261.75 (3.63) | 264.99 (21.14) |

vs `mlp_antigravity` 3-seed: val 257.01 (3.52), test 269.78 (8.72).
Paired TabNet minus MLP: val +11.7 / +2.4 / +0.1 (mean **+4.74**), test
+16.3 / -34.1 / +3.5 (mean **-4.78**). **A tie within noise, not a loss**:
the val gap shrank from +14.6 to +4.7 once optimisation was stable, and the
test difference has the wrong sign for "worse" but a test sd of 21 (single-seed
test ranges 241-278), so neither direction is resolvable. Still ~6x noisier on
test than the MLP, and the val curve stays bouncy (last-8-epoch val sd 4.6-7.4).
The original text's "loses to the MLP on val by +14.57s, decisive" and "3x
noisier" are retracted as artefacts of lr 2e-2. Not yet tested: whether TabNet
is decorrelated from the MLP (predictions were not saved) -- the one remaining
reason it could still matter as an ensemble member.

**DECORRELATION CHECK (2026-09-20, `decorrelation.py`, `blend_mix.py`,
`lirf_focus.py`; TabNet lr 2e-3, seeds 42-44, predictions saved):** first
positive result of the architecture investigation, but narrow.
- **Residual correlation with the MLP ensemble:** 0.937 val / 0.880 test
  (MLP vs another MLP ensemble: 0.976 / 0.974; the GRU was 0.98).
- **3 MLP + 3 TabNet vs 6 MLP** (300 random draws from a 24-seed MLP pool):
  pooled +3.10 val / +12.25 test, 100% of draws; passes the pre-registered bar.
  Not driven by one seed: every TabNet subset beats the same number of extra
  MLP seeds (leave-one-out, single seeds; worst seed alone +0.7 val / +2.8 test).
- **Inside the shipped per-airport tree+MLP blend** (per-airport weights frozen
  from msmlp; TabNet on the neural side, 3 MLP + 3 TabNet vs 6 MLP): pooled
  +2.33 val (100%) / +1.70 test (98%) -- **misses the 2s test bar (1.70)**.
  The whole gain is LIRF (+14.0 val / +10.2 test); non-LIRF is flat (-0.02 /
  +0.19) because the trees already supply diversity there and LIRF's blend is
  100% MLP. Rough board prediction via the LIRF model (k=2.05): +5.1 val /
  +3.7 test board-seconds.
- **LIRF detail:** TabNet-ens3 alone 525.4 val / 484.7 test vs MLP-ens24 534.5 /
  489.3; a 50/50 mix gives 521.6 / 480.4 (flat optimum w=0.4-0.6). The gain is
  in the tail: rows with y>3600s (MLP 3434 -> mix 3249 val, 4512 -> 4397 test)
  while TabNet is slightly WORSE in-band (val 366 vs 350). Single-seed LIRF test
  RMSE ranges 413-639, so 3 seeds is thin.
- **Not yet established:** second surface (seasonal_full), more TabNet seeds,
  re-solved (not frozen) blend weights, and a `full` refit; no submission built.

**SECOND-SURFACE CONFIRMATION (2026-09-20, `confirm_seasonal_full.py`,
`lirf_weight_seasonal_full.txt`; TabNet lr 2e-3 on seasonal_full, seeds 42-44,
all early-stopped: best epochs 17/27/26, ran 33/43/42 of 60):** the lead
CONFIRMS on the scored-composition surface (train Feb-Jun+Aug-Nov, val Dec,
test Jan+Jul, all 10 airports). Blend (shipped per-airport weights, frozen)
with 3 registered MLP + 3 TabNet vs 6 registered MLP, all 20 triples:
pooled test +10.12 (min +7.36, 100% of triples), Dec val +4.73; LIRF test
+62.7 (763.1 -> 699.0; Jan 424.0 -> 394.5, Jul 939.1 -> 858.1), Dec val
LIRF +37.4; every other airport within +/-0.5 (non-LIRF -0.05). Residual corr
TabNet-ens3 vs MLP-ens3 = 0.940 test / 0.908 val (MLP vs MLP 0.980 / 0.969).
Passes both pre-registered confirm bars. The rough "predicted board" figure on
this surface (+22.75) is NOT trustworthy (LIRF already 763 local, the LIRF
model is extrapolating); the local pooled +10.1 is the number to use.
LIRF alone, TabNet-ens3 beats MLP-ens6 outright on this surface (val 364.2 vs
410.2, test 673.9 vs 763.1); LIRF-mix optimum is w=0.8-1.0 here vs 0.5-0.6 on
forward (w=0.5 recovers ~87% of the seasonal_full test gain). Again a TAIL
effect: rows y>3600s (test n=460, 1.73%): MLP 4778 vs TabNet 3720; in-band
TabNet is worse (466.8 vs 435.7). Caveats: only 3 TabNet seeds; seed 42's val
climbed to ~348 after its epoch-17 best (best-epoch weights used, but the
instability signature persists); blend weights frozen, not solved; seasonal_full
is selection-forbidden so the LIRF mix weight w cannot be chosen from it (needs
forward_inner TabNet arms); no `full` refit exists; nothing submitted.

---

## SUPERSEDED -- original lr 2e-2 analysis (do NOT cite; kept for the record)

Everything below this line was written before the instability was found. Its conclusions
("closed, negative", "3x noisier", "loses to the MLP on val by +14.57s") are retracted by
the sections above.

# TabNet (deep tabular architecture) — result

**Closed, negative against the correct baseline — and the single promising
seed was noise.** Beats plain LightGBM decisively and consistently; does not
beat the existing entity-embedding MLP once seed variance is accounted for,
and carries roughly 3x the MLP's own seed noise.

## Setup

`run_tabnet.py`. `forward`/`with_aobt`/`keep_all`, seeds 42/43/44. TabNet
(`pytorch-tabnet`, newly installed — not previously used in this repo),
`n_d=n_a=64`, `n_steps=5`, categorical embedding sizes matched to the
existing MLP's own sizing rule (`min(48, max(4, sqrt(card)))`) for a fair
architecture-only comparison, not a capacity-budget difference. Preprocessing
mirrors `pipeline_mlp_antigravity/train.py` exactly (train-only category
vocab with a shared unknown code, train-median-filled + standardized
numerics). GPU (RTX 3080 Ti), ~17-24 epochs per seed before early stopping,
~1000-1400s per seed.

## Result

| seed | val RMSE | test RMSE |
|---|---:|---:|
| 42 | 262.87 | 245.05 |
| 43 | 282.87 | 297.30 |
| 44 | 269.00 | 274.47 |
| **mean (sd)** | **271.58 (10.25)** | **272.27 (26.19)** |

vs. `mlp_antigravity` (same protocol, 3-seed mean/sd from `results.csv`):
val 257.01 (3.52), test 269.78 (8.72).

**Seed 42 alone looked like a real win** (test 245.05 vs MLP's own seed42 of
259.71, a -14.66s delta — large enough to have been worth building on).
**That was noise.** Seeds 43/44 came in decisively worse on both val and
test, and TabNet's own across-seed spread (test sd 26.19) is roughly 3x
larger than the MLP's (8.72) — this is a genuinely higher-variance
architecture on this dataset, not merely an unlucky draw or two. Averaged
properly: TabNet loses to the MLP on val by +14.57s (larger than either
model's own seed-sd, decisive) and is a statistical wash on test (+2.49s,
well inside TabNet's own noise band) but with far less reliability.

**Does beat plain LightGBM decisively and consistently** (val -14.9 to
-34.9s, test -18.5 to -70.7s across all three seeds) — TabNet is a real,
working deep-tabular architecture on this data, it just doesn't clear the
bar the existing MLP already set, and the bar it needs to clear is the MLP,
not the tree baseline (the same lesson as the matched/unmatched router:
picking the right comparator changes the verdict).

## Disposition

**Not built further.** No decorrelation-gate/blend check run — moot, since
the standalone result already fails on the primary comparator (same
reasoning as the router closure: a weaker-or-equal, higher-variance
component isn't worth blend-testing before it at least matches what it
would replace). Closes the "deep tabular architecture" direction for this
specific construction (TabNet with matched capacity to the existing MLP).
Not tested: FT-Transformer or NODE specifically — TabNet was chosen as the
most differentiated inductive bias (sequential sparse feature attention) and
representative of the "materially different deep-tabular architecture"
option; a different deep-tabular variant remains untested in principle, but
given TabNet already underperforms the MLP mean while adding 3x the
variance, and this project's broader finding (per §2v's competitive-
intelligence pass) that no top team publishes methodology suggesting a deep-
tabular edge, further deep-tabular variants are low-prior without a more
specific reason to expect a different outcome.
