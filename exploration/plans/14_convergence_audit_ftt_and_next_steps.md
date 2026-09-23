# Plan 14 — Convergence audit, the FT-Transformer, and what to do next

Written 2026-09-23, covering the 2026-09-21 → 09-23 session. Picks up from
[plan 13](13_live_board_next_steps.md), which ended at **v29 = 293.6685** with
the note that "improving from here requires a real modelling change, not more
seeds."

Evidence files produced this session:
[`../CONVERGENCE_AUDIT.md`](../CONVERGENCE_AUDIT.md),
[`../tests_ftt/RESULT.md`](../tests_ftt/RESULT.md),
`../tests_ftt/PREREGISTERED.md`, `../tests_ftt/PREREGISTERED_seasonal.md`,
`../tests_mlp_convergence/`.

---

## 1. What was asked

Two questions: **suggest another architecture to test**, and **audit the
submission folder for things not done correctly — we should be training to
convergence.** Both were answered, and the convergence question turned out to
be the more productive of the two.

## 2. What was done

### 2a. Convergence audit (`CONVERGENCE_AUDIT.md`)

Scanned all 435 runs for cap-hit flags. Findings, in order of consequence:

| finding | outcome |
|---|---|
| **12 of 18 `forward_inner` MLP seeds never early-stopped** — ended at the 25-epoch cap, so `best_epoch` was "the last epoch paid for", not a selected quantity. These set v25's blend weights. | Real defect. Retrained (§2d). |
| TabNet's 60-epoch cap excluded final seeds 47 and 51 | Re-run at cap 75: **same best epochs, byte-identical predictions**. The cap blocked *confirmation*, not accuracy. |
| `queueattn` closed on 6 seeds, **2 of which hit their cap** | Closure caveated, not overturned; the GRU pillar of that argument is clean. |
| `per_airport__full` / `per_operator__full` `hit_max_rounds: true` at every airport | **False positive** — on a refit the cap *is* the frozen best_iteration, so it is true by construction. |
| CatBoost `cb1` hit its 5000 cap | Already handled: `cb2_recap` converged at cap 8000 and scored *worse*. No re-open. |

Code fixes: `mlp_antigravity --epochs` default 25 → 45; TabNet cap 60 → 75 and
it now saves model binaries; the health gate's `hit_cap` condition documented as
**unresolved, not failed** (raise the cap and re-run rather than discarding).

### 2b. Submission-folder hygiene (audit §6)

| problem | fix |
|---|---|
| **v29 (best score) was `not_reproducible_via_submit_py`** — built by ad-hoc scripts | Registered `tabnet_lirf_override` arm. Verified: `rint(rebuilt base)` reproduces v25 **exactly on all 344,841 rows**; the v29 rebuild differs only on LIRF rows, only by +1, because `submit.py` rounds where `build_candidate.py` truncated. |
| **v26 was silently unrebuildable** — `airport_operator_catboost_multimodel` had a pipeline, a predict branch and a `PARENTS` entry but was missing from `config.ARMS` | Registered. |
| **v18 was missing from the generated index entirely** — `VERSION_RE` didn't match its `_cap*` filenames | Regex widened, `variant` column added, sidecars written retroactively. |
| The quarantine block had been live for 11 days after its rebuild finished | Reverted. |
| Board scores lived in hand-added columns the generator wipes | Moved to `submissions/board_scores.csv`, which the generator only reads. |

### 2c. FT-Transformer — the new architecture (`tests_ftt/`)

Column attention (attention over *features*, not neighbours), chosen because
every closed architecture here varies how the model consumes the movement
stream, and `tests_seqrnn/RESULT.md` states that axis is foreclosed.

Confirmed on **three independent periods** against bars fixed in advance:

| period | surface | in-band | non-LIRF |
|---|---|---:|---:|
| Sep–Oct 2025 | forward val | −0.86 | −1.47 |
| Nov–Dec 2025 | forward test | −3.63 | −4.22 |
| Dec 2025 | seasonal_full val | −1.71 | −2.00 |
| Jan + Jul 2025 | seasonal_full test | −1.45 | −1.83 |

**Honest effect size ~1.5 s in-band**; the −3.63 reading was the optimistic
outlier of four. Residual correlation 0.75–0.88 vs the MLP — the lowest ever
measured here. **FTT's weight at LIRF solved to exactly 0.000 in five
independent solves**, so its contribution is entirely off-LIRF by the solver's
own choice.

Shipped as **v30** (`ftt_multimodel`, registered *before* its first submission).
Validated PASS; tail controlled (max identical to v25, +3 rows over 7,200 s);
LIRF 26,898 of 26,899 rows byte-identical to v25.

### 2d. Rec 3 — retraining the 12 truncated seeds

All 12 retrained at cap 45. **5 were genuinely truncated (−2.0 to −11.4 s each),
7 were unaffected** (their optimum already sat inside the cap). Paired mean
−2.63 s; the 18-seed ensemble improves **311.585 → 309.514 (−2.07 s)**, so it
survives averaging.

Cross-check: alphas re-solved from the *old* ensemble reproduce the shipped v25
weights **exactly to four decimals**, which validates the reconstruction.

**But it is not a submission lever.** The retrain fixed the `forward_inner`
MLPs, which only *solve weights*. The predictions in v25/v29/v30 come from the
`full` MLPs, which were not retrained. The only part reaching a submission is a
+0.0085 mean alpha shift — 0.85 percentage points, not worth a slot alone.

## 3. Errors made and corrected — kept on the record

1. **Three verdicts read off unconverged runs.** TabNet's cap, then FTT at cap
   60 and again at cap 90. Each was wrong in the same direction; had FTT been
   closed at cap 60 as was nearly done, v30 would not exist.
2. **An unpaired comparison presented as evidence of no effect.** The audit's
   original "measurably NOT costing score" compared *different* seeds, where
   between-seed variance (~6 s) swamps the effect. The paired retrain showed
   −2.0 to −11.4 s on the seeds the cap actually bound. **Compare the same seed
   with and without the change.**
3. **Buffered logs misread as stalled jobs**, twice — once leading to a
   recommendation to kill another project's process. Check CPU time or output
   files, not the log.
4. **A kill that left the loop alive.** `taskkill /IM python.exe` killed the
   training child but not the bash loop, which advanced and began overwriting a
   run directory seconds after it had been restored from backup. Fixed with a
   PID file, a signal trap and a STOP flag.

## 4. What to do next — ranked

### 4a. Compose TabNet + FTT (no training required)

**The highest-value move on the table.** v29 carries TabNet's LIRF-only
override; v30 carries FTT at *exactly zero weight at LIRF*. They act on
disjoint row sets — 26,899 LIRF rows vs the other 317,942 — and both sets of
`full`-split predictions already exist on disk. Cost: a build script and a
registered arm, ~2 h, no GPU training.

Caveat: this stacks two sub-2 s local effects, and this build's ledger warns
against summing ceilings. Unlike most stacking, though, these genuinely do not
overlap.

### 4b. Transductive self-supervised pretraining

> **Correction (2026-09-23, before any run):** the join-failure motivation below
> does not hold. The gap is 1.53 % vs 1.08 % of rows (0.45 pp). Those rows are
> hard because their `_flt` columns are *missing*, so pretraining cannot recover
> them; `DATA_QUALITY_AUDIT.md` says so itself. What remains is the generic
> covariate-shift hypothesis, which is speculative and cannot be resolved on the
> board. It is being tested anyway at the user's request, under the bar in
> `../tests_ftt_pretrain/PREREGISTERED.md`.
>
> **CLOSED — FAIL (2026-09-23).** The pre-registered kill fired on seed 42: best
> val 283.11 against 263.32 for the same seed without pretraining. Both stages
> stopped on patience, and it fails even with generous extrapolation. See
> `../tests_ftt_pretrain/RESULT.md`. Do not re-open without a new pre-registration.

The only new-architecture idea that targets a **measured** problem.
`DATA_QUALITY_AUDIT.md` found the scored set has a **+42 % higher join-failure
rate** than training (+54 % at LIRF), and those rows are the model's worst
segment. The scored set's *features* are given — only labels are blanked — so
masked-feature pretraining (SAINT/VIME style) on train **+** ranking rows,
then fine-tuning on labels, attacks the 2025→2026 shift directly. Reuses the
FTT code that just cleared the bar. Nothing here has used the scored features
for representation learning.

### 4c. The `full`-refit epoch count (audit §2)

The 18 `full` MLP members train at 10 or 14 epochs, frozen from whichever
single seed was passed to `--freeze-from`, against per-seed forward optima
spanning 10–21. Arbitrary and inconsistent. **Direction undetermined** — at a
fixed batch size, `full`'s 1.5× data makes 10 epochs ≈ 15 forward-epochs of
updates — so this is a measurement to make, not a fix to apply. Cheapest honest
version: freeze from the median, same count for every member.

### 4d. Lower priority

* **NODE** (differentiable oblivious trees) — different bias, but may land close
  to the tree side. Screen on decorrelation before spending epochs.
* **Hyperparameter-diverse ensembling** instead of seed-diverse: the seed curve
  floored at 12–18, but every member is architecturally identical.

### 4e. Do NOT re-open

* The **neighbour/sequence axis** — GRU (r = 0.980) and queue-attention.
* The **objective axis** — log1p, Huber, Poisson, quantile ladder, all closed
  decisively in the same direction.
* **Hyperparameter tuning** of existing arms — measured at 6.5 s for a full
  sweep against 27 s for one correctly-reasoned constant.
* **More MLP seeds** — 12→18 bought −0.54 board and the curve is flattening
  faster than 1/√n.

## 5. Open items requiring the user

1. **Upload v30.** Built and validated at
   `submissions/fearless_barn_v30.parquet`. Expectation **~−3 board seconds**
   (293.67 → ~290.7) against a local→board mapping whose residual sd is ~9, so
   one submission cannot cleanly verify it. Ranking takes the best score across
   all submissions, so a weak result cannot hurt standing.
2. **Record the score** in `submissions/board_scores.csv` when it lands.
3. **Decide on 4a** (the composite).

## 6. Method note to carry forward

Two rules this session paid to learn:

* **Set caps by the 1.5×-observed-best rule before the first run**, and treat a
  cap-hit as *unresolved*, not *failed*.
* **Compare the same seed with and without the change.** Group means across
  different seeds hide effects that paired comparisons find immediately.

---

## 7. Addendum (2026-09-23) — 4a built as a registered run; board prediction pre-registered

**Run:** `tabnet_lirf_override__full-with_aobt-keep_all__seed42__tag=lirf_w0.55_ftt3`
— the existing override arm with `--base-arm ftt_multimodel --base-tag ftt3_mlp18
--w 0.55 --tabnet-seeds 42,43,45,46,48`. No new model code.

* **Seeds pinned explicitly.** `tests_tabnet/build/` now holds `final_seed42..54`;
  the arm's default glob would have gated in seeds v29 never used, and w=0.55 was
  solved for the 5-seed mean only.
* **w transferred, not re-solved.** Valid because v30's LIRF rows equal v25's on
  26,898/26,899 (FTT's LIRF weight solved to 0.000) — the base under the override
  is the one w was solved against.
* **`submit.py` fix:** `PARENTS` hard-coded the override's base as
  `airport_operator_multimodel`, which would have recorded an unused parent for
  this run. The static entry is removed; the chain now comes from the run's own
  `base_arm`, plus the static parents of every arm reached that way. v29's chain
  resolves unchanged (checked).

**Scratch acceptance check (passed):** non-LIRF rows == v30 on 317,942/317,942;
LIRF rows within 1 s of v29 on 26,899/26,899 (13,374 exact — the rint-vs-truncate
difference already documented in the arm); `validate()` clean; max 81,837,
89 rows > 7,200 s; provenance chain fingerprint-uniform (`aeade7fee3099c1a`).

**Pre-registered board prediction.** The two edits touch disjoint rows, so MSE is
additive:

    RMSE_composite = sqrt(RMSE_v30² − (295.2054² − 293.6685²)) = sqrt(RMSE_v30² − 905.04)

| v30 lands at | composite should land at |
|---:|---:|
| 290.7 (expected) | 289.14 |
| 292.0 | 290.45 |
| 293.0 | 291.45 |
| 295.2 (FTT null) | 293.66 |

The rint-vs-truncate shift on LIRF moves this by well under 0.01 s. **If both are
uploaded and the composite deviates from the formula by more than ~0.05 s, the
composition is wrong somewhere; investigate before trusting either number.**
Because the prediction is almost exact, uploading the composite alone would also
reveal FTT's board effect by subtraction. v30 is therefore optional as a
measurement, but it is useful as the cross-check.

### 7a. Board result (2026-09-23)

| | predicted | board |
|---|---:|---:|
| v30 (FTT) | ~290.7 (−3 vs v25) | **291.3371** (−3.87 vs v25) |
| v31 (v30 + TabNet LIRF) | sqrt(291.3371² − 905.04) = **289.7797** | **289.7783** |

The composition check lands within **0.0014 s**: the two edits add exactly as
MSE additivity says, so both are confirmed as built. **v31 = 289.7783 is the
new best.** FTT's board effect (−3.87) matches its local estimate mapped
through `board ≈ −246 + 2.24 × local` (~−3), the first sub-5 s lever in this
build whose board result lands where the local evidence said.
