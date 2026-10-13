# Convergence + submission-folder audit (2026-09-21)

Two questions, answered against the artifacts on disk rather than from the
plan documents: **(1) is anything in the shipped chain trained to a
convergence criterion that actually fired?** and **(2) what in
`submissions/` is not done correctly?**

Evidence script: `tests_mlp_convergence/alpha_cap_vs_converged.py`
(output: `alpha_cap_vs_converged.txt`). No training was run for this audit —
every number below comes from runs already on disk.

---

## 1. The headline defect: 12 of 18 `forward_inner` MLP seeds never early-stopped

`forward_inner` is the surface that solves v25's ten per-airport blend
weights, and v25 is the base of v28 and v29 (the current best board score).

| budget | seeds | best epoch | early stop fired? |
|---|---|---|---|
| `--epochs 25` | 42, 43, 44, 48, 49, 50, 51, 53, 54, 55, 57, 58 | 19–25 | **no — run ended at the cap** |
| `--epochs 25` | 52, 56, 59 | 17–18 | yes |
| `--epochs 45` | 45, 46, 47 | 23, 27, 25 | yes |

For those 12 seeds `best_epoch` is not a selected quantity — it is "the last
epoch we paid for." `hit_max_epochs: true`, `early_stop_fired: false`,
`converged: false` are written into each `model.json`. Seed 46, given a
45-epoch leash, put its optimum at **27** — past the cap the other twelve
were held at, so the cap is binding, not incidental. `tests_snapshot_avg`
independently put the val bottom at epochs 13–25.

This is a real protocol defect: patience-6 early stopping is declared in
`pipeline_mlp_antigravity/train.py` and never got to run for two-thirds of
the seeds that set the shipped weights.

### But it is measurably NOT costing score — and that is the useful finding

The mechanism worth worrying about is: *a truncated MLP is a worse MLP, so
the least-squares solve under-weights it, and those under-weighted alphas
get applied at submission time.* This repo has seen that exact shape once
before — fixing the stale-fingerprint `forward_inner` MLPs (v22) shifted
weight toward the MLP and bought −1.4 on the board.

Re-solving the same per-airport alpha from a capped trio vs. the converged
trio (`alpha_cap_vs_converged.txt`):

| airport | capped 42/43/44 | converged 45/46/47 | shift |
|---|---:|---:|---:|
| EDDF | 0.4739 | 0.4442 | −0.030 |
| EDDM | 0.4939 | 0.6145 | **+0.121** |
| EGLL | 0.6089 | 0.6154 | +0.006 |
| EHAM | 0.8556 | 0.8583 | +0.003 |
| LEBL | 0.8002 | 0.7472 | −0.053 |
| LEMD | 0.6098 | 0.5778 | −0.032 |
| LFPG | 0.4874 | 0.5072 | +0.020 |
| LIRF | 0.0000 | 0.0000 | 0.000 |
| LSZH | 0.3208 | 0.3197 | −0.001 |
| LTFM | 0.5427 | 0.5386 | −0.004 |
| | | **mean shift** | **+0.003** |

**No systematic shift.** The mean is +0.003, the signs are mixed, and the
one large move (EDDM +0.121) goes the *wrong* way for the mechanism — the
converged seeds got *less* MLP weight there, not more. Per-seed
`forward_inner` val RMSE: capped mean **327.79** (n=12) vs converged
**325.49** (n=6), a 2.3 s gap against a seed-to-seed spread of 318–338
(sd ≈ 6). Inside noise by this project's own margin-vs-seed-swing rule.

Read the trio-vs-trio comparison, not that aggregate: the six-seed
"converged" group mixes three seeds given a genuinely longer budget
(45/46/47 at `--epochs 45`) with three that merely happened to early-stop
inside 25 (52/56/59, best epoch 17–18), so the aggregate confounds "longer
leash" with "draw that plateaued early." The 42/43/44 vs 45/46/47 alpha
table above is the clean contrast and it is what carries the conclusion.

### CORRECTION (2026-09-23, from the rec-3 retrain): the truncation DID cost per-seed accuracy

The retrain produced the paired measurement this section never had, and it
changes part of the conclusion above. **All 12 seeds, complete** (finished
2026-09-23 10:44), each retrained at cap 45:

| seed | old best | new best | old val | new val | delta |
|---|---:|---:|---:|---:|---:|
| 54 | 23 | **32** | 335.73 | 324.37 | **−11.36** |
| 44 | 22 | **31** | 337.93 | 328.44 | **−9.49** |
| 48 | 23 | **33** | 331.46 | 324.78 | **−6.68** |
| 43 | 25 | **31** | 318.64 | 316.59 | −2.05 |
| 55 | 20 | **26** | 322.00 | 320.00 | −2.00 |
| 42 | 23 | 23 | 326.09 | 326.09 | +0.00 |
| 49 | 19 | 19 | 333.12 | 333.12 | +0.00 |
| 50 | 19 | 19 | 330.26 | 330.26 | +0.00 |
| 51 | 25 | 25 | 318.24 | 318.24 | +0.00 |
| 53 | 19 | 19 | 328.96 | 328.96 | +0.00 |
| 57 | 22 | 22 | 328.74 | 328.74 | +0.00 |
| 58 | 22 | 22 | 322.30 | 322.30 | +0.00 |

**Paired mean −2.63 s. Five improved, seven unchanged, none worse.**

The split is clean and mechanical: the seven unchanged seeds had their optimum
far enough inside 25 that patience-6 fired anyway, so the cap never bound --
the same pattern as TabNet seeds 47/51 (sec.3). The five that moved were cut
off mid-improvement, and their best epoch jumped to 26-33.

**Where the earlier reading went wrong: it compared the wrong things.** The
"capped mean 327.79 vs converged 325.49, inside a seed spread of ~6" above is
an **unpaired** comparison across *different* seeds, where between-seed
variance swamps the effect. The retrain gives the **paired** comparison — the
same seed, with and without the cap — and there the cap costs up to 9.5 s.

Two distinct claims, and only one of them survives:

* **"The truncation does not bias the blend weights" — still stands.** The
  alpha table is unaffected; that measurement compared weight solves, not model
  quality, and nothing about it depended on the unpaired RMSE aggregate.
* **"It is measurably NOT costing score" — withdrawn as stated.** Five of the
  twelve were genuinely truncated mid-improvement and are 2–11.4 s better with
  a longer leash. The other seven landed on the *same* best epoch, i.e. the cap
  was not binding for them — the TabNet 47/51 pattern. Note that a group mean
  over all twelve (−2.63 s) *understates* the effect on the seeds it actually
  touched, where it is −2.0 to −11.4 s.

### It DOES survive averaging -- but it lands on the weight-solve surface, not the submission

`alpha_paired_old_vs_new.py` / `.txt`, run once all 12 had converged:

| quantity | old (truncated) | new (converged) | delta |
|---|---:|---:|---:|
| per-seed val, paired mean | -- | -- | **-2.63** |
| **18-seed MLP ensemble val** | **311.585** | **309.514** | **-2.071** |
| blend val (*in-sample for the alphas*) | 306.831 | 304.369 | -2.462 |

**The gain survives averaging almost in full** (-2.63 per member -> -2.07 on the
18-seed ensemble). It does not wash out, which is the outcome the earlier
unpaired reading would have predicted wrongly.

**Cross-check that validates the whole comparison:** the alphas re-solved from
the OLD ensemble reproduce the shipped v25 weights **exactly** -- 0.3972,
0.3888, 0.5552, 0.8595, 0.6903, 0.5145, 0.4555, 0.0000, 0.2501, 0.4792, to four
decimals. So this reconstruction of the weight solve is the one that actually
shipped, and the "new" column differs only by the retrain.

| airport | shipped v25 = old ens | new ens | shift |
|---|---:|---:|---:|
| EDDM | 0.3888 | 0.4114 | +0.0226 |
| EGLL | 0.5552 | 0.5694 | +0.0141 |
| EHAM | 0.8595 | 0.8732 | +0.0137 |
| EDDF | 0.3972 | 0.4096 | +0.0125 |
| LFPG | 0.4555 | 0.4645 | +0.0090 |
| LSZH | 0.2501 | 0.2560 | +0.0060 |
| LEBL | 0.6903 | 0.6954 | +0.0051 |
| LEMD | 0.5145 | 0.5156 | +0.0011 |
| LTFM | 0.4792 | 0.4798 | +0.0006 |
| LIRF | 0.0000 | 0.0000 | 0.0000 |
| | | **mean shift** | **+0.0085** |

Note the sign is now **consistent across all nine non-LIRF airports** (the
earlier unpaired comparison had mixed signs). A *better* MLP receives slightly
*less* weight -- alpha is the weight on the tree side -- which is small
(0.85 percentage points) but systematic.

### What this does NOT mean, stated plainly

**This does not make the shipped submissions better, and it is not a
submission lever on its own.** The retrain fixed the **`forward_inner`** MLPs,
which exist only to *solve the blend weights*. The predictions in v25/v29/v30
come from the **`full`** MLPs -- separate `refit_only` runs at frozen epoch
counts of 10/14, which were NOT retrained and are sec.2's issue, not sec.1's.

So of the two things this measurement produced:

* the **-2.07 s ensemble gain** is on `forward_inner` val, a surface no
  submission is scored on. It does not transfer.
* the **+0.0085 alpha shift** is the only part that would reach a submission,
  by applying slightly different per-airport weights to the same unchanged
  `full` predictions. A 0.85-percentage-point weight change is very unlikely to
  be worth a slot on its own.

The honest disposition: **rec 3 delivered what it was billed as -- correctness,
not score** -- while also proving that the defect was real and the original
"not costing score" reading was wrong. Getting a score out of it would mean
retraining the 18 `full` MLPs at a properly-chosen epoch count, which is sec.2,
whose direction is undetermined (more data at a fixed batch size means 10
epochs on `full` is ~15 forward-epochs of updates).

**This also means B'' was closed on the same flawed comparison** — see the
following subsection, which should be read with this correction in mind.

### This closes plan 13's B'' without spending the GPU run

B'' (`--patience 12 --epochs 40`, architecture unchanged) was pre-registered
and never run. It does not need to be: runs with a longer leash **already
exist on both surfaces** and they do not win.

`forward` split (the surface with an honest val, 12 seeds on disk):

| budget | seeds | best epoch | val RMSE |
|---|---|---|---|
| `--epochs 60` | 43, 44 | 19, 14 | 256.16, 260.88 |
| `--epochs 25` | 42, 45–51 | 10–16 | 252.58–259.45 |
| `--epochs 25` (capped) | 52, 53 | 21, 20 | 257.13, 256.99 |

The two seeds given 60 epochs land **mid-pack and worse than the median**.
Best epochs span 10–21 with no relationship to the budget. **B'': answered,
no measurable gain, ~6h GPU saved.** Fix the cap for hygiene (so the
declared criterion is the one that fires), not for score.

## 2. The `full` refit epoch count is frozen from one seed, not measured per seed

Every `full` MLP member is a `refit_only` run whose epoch count comes from
`frozen["best_epoch"]` of a single `forward` run:

- seeds 42–51 → **10 epochs** (frozen from forward seed 42, best epoch 10)
- seeds 52–59 → **14 epochs** (frozen from forward seed 44, best epoch 14)

while the per-seed forward optima span **10–21, median 12.5**. So the 18
averaged members are trained at two arbitrary epoch counts inherited from
two other seeds, and seed 42's 10 is the *shortest* optimum observed.

Stated precisely, because the obvious reading is wrong: this is **not**
demonstrably under-training. At a fixed batch size of 2048, `full`
(2,084,659 rows) gives 1.50× the gradient steps per epoch of `forward`
(1,387,105), so 10 epochs on `full` is ~15 forward-epochs of updates. The
direction is genuinely undetermined. The defect is that the count is
**arbitrary and inconsistent across members**, chosen by which seed happened
to be passed to `--freeze-from`. And `full` has no holdout, so any change
here is adopted on principle, exactly like the original full-refit decision.

Cheapest honest fix if touched at all: freeze from the **median** forward
best epoch rather than seed 42's, and use the same count for every member.

## 3. TabNet's 60-epoch cap discarded 2 usable seeds — RESOLVED, and the answer is instructive

`tests_tabnet/RESULT.md`: final seeds 47 and 51 hit the 60-epoch cap (best
epochs 50 and 47) and the health gate excluded both; seed 54 was excluded as
poor. 9 of 12 survived. A third of the seed budget was discarded for
non-convergence, on the one architecture that has actually moved the board
(v29 = 293.67).

**Re-run at cap 75** (= 1.5× the observed natural best of 50, the rule in
`tools/ml_common/convergence.py`), 2026-09-21, ~2.5 h GPU:

| seed | cap | best epoch | ran | early stop fired | Dec val |
|---|---|---|---|---|---|
| 47 | 60 | 50 | 60 | **no** | 234.86 |
| 47 | **75** | **50** | 66 | yes | **234.86** |
| 51 | 60 | 47 | 60 | **no** | 236.21 |
| 51 | **75** | **47** | 63 | yes | **236.21** |

**The predictions are byte-identical** — 0 of 344,841 ranking rows and 0 of
165,677 Dec-val rows differ, for both seeds. They had already found their
optimum at cap 60; the cap only stopped patience-15 from *confirming* it.

So the gate's `not hit_cap` condition means **unresolved, not failed**. It was
right to be cautious — at cap 60 with best at 50, epochs 61–65 were genuinely
unobserved and the run *might* have kept improving — but the correct response
to a rejection there is to raise the cap and re-run, which resolves it, rather
than discard the seed. `gate.py` now says so, and the default cap is 75.

**No submission comes out of it**, and the registered bar said so in advance.
Re-running `seed_curve.py` with the recovered seeds (11 healthy, up from 9):

| healthy seeds k | 1 | 3 | 5 | 7 | 9 | 11 |
|---|---|---|---|---|---|---|
| LIRF Dec-val RMSE | 382.2 | 356.5 | 350.3 | 347.9 | 346.8 | **345.9** |

Original 5 seeds 347.7 → all 11 healthy 345.9 = **+1.8 s against a
pre-registered ≥6 s bar. Closed.** The value of this exercise was the
convergence measurement, not a score: it establishes that TabNet's cap was
costing seed *availability*, not seed *quality*.

## 4. A closed negative that partly rests on truncated runs: queue-attention

`pipeline_queueattn_antigravity` was closed as "flat at 6 seeds, mean diff
+0.670 s worse, not distinguishable from zero." **2 of those 6 seeds hit the
25-epoch cap:**

| seed | best epoch | capped | val RMSE |
|---|---:|---|---:|
| 42 | 14 | no | 251.89 |
| 43 | 9 | no | 266.61 |
| 44 | 16 | no | 258.42 |
| **45** | **22** | **yes** | 256.38 |
| 46 | 16 | no | 254.91 |
| **47** | **19** | **yes** | 252.96 |

This is the same question §5's CatBoost entry answers with a clean "no" —
it was simply never asked here. The two truncated seeds are the 2nd and 3rd
*best* of the six, so truncation is not what produced the flat verdict; but
the direction of the untested effect is toward queue-attention, not away.
Caveat the closure rather than treating it as settled: if anyone wants to
reopen neighbour-attention, re-run seeds 45 and 47 at a higher cap first
(two runs, ~1 h). §7's argument does not depend on it — the GRU pillar of
that closure is clean (early-stopped at 18–24 epochs, residual corr 0.980).

**The TabNet `seasonal_full` baselines are clean** — all six
`mlp_antigravity__seasonal_full` seeds early-stopped (best epoch 12–18, none
capped), so `confirm_seasonal_full.py`'s "+11.20 pooled" is not sitting on
truncated comparators. Two `seasonal` (not `seasonal_full`) MLP seeds, 42
and 46, *are* capped; that split is not the TabNet comparison surface.

## 5. Three things that look like convergence failures and are not

- **`per_airport__full` / `per_operator__full` report `hit_max_rounds: true`
  at every airport and every operator.** This is **true by construction, not
  a failure**: on a `refit_only` run `PerAirportLGBM` sets
  `num_boost_round = frozen["best_iteration_per_airport"][a]`, so best == cap
  necessarily. Do not read these as ten real truncations. (The substantive
  point stands separately: `full` uses the *identical* round counts as
  `forward` — EDDF 185, EDDM 149, EGLL 133, … — on 1.50× the rows.
  Given the board has shown the tree side is worth almost nothing marginal
  (v19 no-MLP = 382.92 vs 306.32), scaling those rounds is low-value.)
- **CatBoost was not dismissed while truncated.** `cb1` did hit its 5000
  cap (best 4926, `converged: false`), but it was re-run as `cb2_recap` at
  cap 8000, where early stopping fired at 3551 (best 3350) — and the
  converged run scored **worse** (307.15 vs 305.54). The hygiene already
  worked here; no re-open.
- The LR schedule is not implicated: `--lr-schedule` defaults to `"none"`
  (flat AdamW), so the refit's shorter epoch count does not compress a
  cosine `T_max`.

---

## 6. `submissions/` — what is not done correctly

Ranked by consequence. **Status as of 2026-09-21: 6a, 6b and 6c are fixed, 6d
partly.** What each fix was is recorded inline below.

### 6a. The current best score cannot be rebuilt through the registered path

`fearless_barn_v29.parquet.meta.json` (and v28) carry
`"not_reproducible_via_submit_py": true`. v29 = 293.6685, the best score on
the board, was assembled by ad-hoc scripts
(`tests_tabnet/{train_tabnet_pred,blend_variants,build_candidate}.py`) as a
post-hoc LIRF override on v25's parquet, with no `--arm` that `submit.py`
can dispatch.

This has bitten before: the **Sep 4 data refresh forced a rebuild of all 17
submissions**, and v17 was rejected outright by the scorer for staleness. If
`ranking.parquet` grows again, v25 rebuilds through `submit.py` and v29 does
not — the best result would have to be hand-reassembled from scratch, with
the TabNet binaries and the gate decisions reconstructed from prose.

**FIXED 2026-09-21.** `pipeline_tabnet_lirf_override/` is now a registered
arm with a `predict_ranking()` branch and a `PARENTS` entry, the way
`airport_lirf_identity_multimodel` was registered when it hit the same wall.
The healthy-seed gate that lived only inside `build_candidate.py` is lifted
into `pipeline_tabnet_lirf_override/gate.py`, and the selection is frozen at
train time and *replayed* at predict time, so a seed trained later cannot
silently join an ensemble that was already scored. Both the arm and
`submit.py` fail loudly if the saved TabNet predictions no longer cover every
scored row — the staleness failure that got v17 rejected, caught locally.

Two things surfaced while verifying it:

* **The full chain reproduces v25 byte-for-byte, and v29 to within a
  rounding convention.** Measured, not asserted:

  | check | result |
  |---|---|
  | `rint(rebuilt base)` vs stored **v25** | **exact on all 344,841 rows** (0 differing) |
  | rebuilt **v29** vs stored v29, non-LIRF rows | **0 differing** |
  | rebuilt v29 vs stored v29, LIRF rows | 13,525 of 26,899 differ, **every one by exactly +1**, mean +0.503 |
  | max abs delta anywhere | **1** |

  The whole residual is one convention: `submit.py` writes
  `np.rint(pred).astype("int32")`, `build_candidate.py` cast with
  `.astype(int32)`, which *truncates*. (The raw float base sits a uniform
  0–0.5 from v25's stored ints — mean 0.2503, signed mean +0.0004 — which is
  exactly the `rint` signature and initially looked like a real discrepancy.)
  The arm keeps `rint`: consistency with every other submission in the build
  beats byte-matching one historical file. Consequence, stated rather than
  buried — the board's **293.6685 belongs to the truncated artifact**, and a
  rebuilt v29 is a hair different from the file that scored it.
* **v26 (295.4767) was also not rebuildable, and nobody had noticed.**
  `airport_operator_catboost_multimodel` has a full pipeline, a
  `predict_ranking()` branch *and* a `PARENTS` entry, but was missing from
  `config.ARMS` — so `submit.py --arm` rejected it and the index had been
  reporting it as NOT REBUILDABLE. Added.

### 6b. The quarantine block in `config.py` was never reverted

`config.ARMS` still contains, live and uncommented:

```
"airport_unsup_multimodel", "airport_multiseed_multimodel",
"airport_debiased_multimodel",
```

under a comment that says *"TEMPORARILY RE-ENABLED (data-refresh rebuild)
… Revert this line to restore the quarantine."* That rebuild finished on
2026-09-10, so for eleven days `submit.py --arm` would still accept all three
tier-E arms — including `airport_debiased_multimodel`, the one whose own
docstring justifies its constants by the test score they produce.

**FIXED 2026-09-21.** Reverted. The rebuilt v4–v12 files stay valid (only
`submit.py --arm` reads the list) and the code stays importable via
`OVERTUNED_DIR`, so re-enabling for a future refresh is still a one-line
change — deliberately one that has to be made on purpose.

### 6c. `submissions.csv` has hand-edited rows that will be silently wiped

The index is complete (29 versions, all present, none missing) but:

- Ten of the 39 rows are not submissions: four blank rows and six pasted
  shell notes (`list files`,
  `mc ls dc26/…`, `mc cp "submissions\fearless_barn_v14.parquet" …`) sitting
  in the **`version` column**. Any consumer that parses `version` as an int
  has to skip them.
- Three hand-added columns — `personal notes`, `actual `, `Date of
  submission` (note the trailing space in `actual `) — hold the only local
  record of several board scores. `build_submission_index.py` writes only
  its fixed columns, so **the next regeneration deletes them**. The README
  already warns about this and it has not been acted on.

Board scores are load-bearing evidence. They belong in a file the generator
does not own.

**FIXED 2026-09-21.** `submissions/board_scores.csv` now holds the full board
history (version, board name, score, date, status, note), and
`build_submission_index.py` only ever *reads* it, joining five `board_*`
columns into the index and warning about any submitted version with no score.
Dates that were blank in the original hand-kept column were left blank rather
than inferred from neighbouring rows.

### 6d. Minor

- `fearless_barn_v18_cap7200/10800/14400.parquet` had no `.meta.json`
  sidecars (hand-built post-processing of v14, never went through
  `submit.py`). **This was worse than it looked: `VERSION_RE` matched only
  `_v<N>.parquet.meta.json`, so v18 — the probe that established the tail
  predictions are load-bearing — was missing from the generated index
  entirely.** FIXED: sidecars written retroactively (and marked as such), the
  regex widened to capture an optional variant suffix, and a `variant` column
  added. The board score attaches only to the file actually uploaded
  (`cap7200`), never to its unsent siblings. Re-derived `n_rows_clamped` = 62
  for cap7200, matching the plan-13 record exactly.
- `submissions/README.md` still described the state as of v13 — "9 of 13
  submissions came from arms now quarantined" — 16 submissions out of date.
  **FIXED:** rewritten around what is and is not rebuildable, the board-score
  file, and the rounding caveat.
- v27's meta records `feature_fingerprint: null` for the `per_airport` and
  `per_operator` provenance entries while `airport_operator_blend` records
  `aeade7fee3099c1a`. The feature-agreement gate is the build's main
  anti-staleness control; a null there is an unverified link.

---

## 7. Architecture: what to test next

The repo has closed GRU-sequence, queue-attention, the matched/unmatched
router, TabR retrieval, the quantile ladder, snapshot/SWA averaging, the
wide MLP, and the entire log1p/Huber/Poisson objective axis. One thing has
paid: **TabNet, and only as a decorrelated override at the single airport
where the blend is 100% MLP** (v29 = 293.67, −1.54 vs v25).

That suggested a screening number this repo already measures — **residual
correlation against the 18-seed MLP ensemble** — and rec 2 below was run on
that basis. **It came back and corrected the criterion, so the table now reads:**

| candidate | resid corr vs MLP | standalone vs MLP | outcome |
|---|---:|---:|---|
| GRU state model | 0.980 | tie | dead |
| *(MLP seed vs MLP seed)* | *0.93* | *—* | *reference: no diversity* |
| matched/unmatched router | 0.898 / 0.876 | worse | dead |
| **TabNet** | **0.88–0.94** | **tie** | **paid, at LIRF (v29)** |
| **FT-Transformer** | **0.75–0.81** | **+11 to +20 s worse, still improving** | **OPEN — swap gain now +0.80 test / −1.28 val** |

**Correlation is necessary but not sufficient** — a candidate has to clear
*both* axes, decorrelated **and** close enough standalone that its diversity
outweighs its error. TabNet paid because it tied the MLP *and* decorrelated.

**But do not read the FT-Transformer row as a closed negative.** An earlier
draft of this section did, off a run that had hit its epoch cap at 60: it
recorded a +43 s standalone gap and a 9–12 s swap loss. Re-run at cap 90 the
same seed reached best epoch 88 and was *still improving* — the gap more than
halved and the test-side swap gain crossed zero. That verdict is retracted; see
`tests_ftt/RESULT.md`. The candidate is live and needs a genuinely converged
run (cap 132, then multi-seed) before anything is concluded.

That mistake is worth keeping on the record precisely because it happened
inside the audit that documents it: the reading was taken off a truncated run,
the same error as TabNet's retracted lr-2e-2 verdict and the same error as
§3's cap-rejected seeds.

Screen on both (`tests_ftt/decorrelation.py` reports them together), at the
target airport. In that frame, ranked:

1. **Raise TabNet's epoch cap and recover the excluded seeds.**
   This is the convergence fix and the architecture answer in one: it
   strengthens the only decorrelated member that has ever moved the board,
   instead of starting a new architecture at zero. Seed 47 (best epoch 50 at
   a 60 cap) is a discarded *good* run — its Dec val 234.9 sits inside the
   healthy band. Set the new cap by the repo's own rule, quoted in the `cb1`
   convergence note — *"set the cap from an OBSERVED natural best-epoch with
   at least 1.5x headroom"* — which on an observed best of 50 gives **~75**,
   not a number picked for comfort.
2. **Feature-attention, not neighbour-attention** — **RUN, and OPEN: see
   `tests_ftt/RESULT.md`.** The inductive-bias argument held better than
   expected: residual corr 0.75–0.81, the lowest ever recorded in this build.
   Still under-trained at cap 90 (best epoch 88) with the test-side swap gain
   already positive. This is the live candidate. Original text: — FT-Transformer / SAINT-style attention *over
   columns*. Every closed architecture above varies how the model consumes
   the *movement stream*; `tests_seqrnn/RESULT.md` states outright that
   neighbour/sequence representation is closed (GRU r = 0.98, cleanly
   converged — queue-attention agrees but carries §4's caveat).
   Column-attention has a different inductive bias and
   is the one family with no entry in this repo. Gate it on the
   decorrelation screen before spending epochs.
3. **A decorrelated member at EHAM — only if it clears a harder bar.** EHAM
   is the mirror of LIRF in the weight table (α = 0.856–0.870, the most
   tree-dominated cell) and has never been probed. But do **not** frame this
   as "screen a neural member against the tree parent": the MLP already *is*
   a neural member decorrelated from the trees, it was on the solver's menu
   at EHAM, and the solver assigned it ≈ 0.14. That question has an answer.
   A real EHAM candidate must be decorrelated from **both** the tree parent
   and the 18-seed MLP — a much harder bar, and `TAXI_OUT_DRIVERS.md` §2m's 3-way CatBoost blend
   ("a wash") is prior evidence against the obvious tree-family candidate.
   Ranked last for that reason.

What not to spend on: more MLP seeds (the 12→18 step bought −0.54 and the
curve is flattening faster than 1/√n), the B'' patience change (§1),
hyperparameter tuning (measured at 6.5 s val for a full sweep vs 27 s for
one correctly-reasoned constant), and the tree side generally (v19).

---

## 9. Addendum (2026-09-24): were the closed OUTLIER verdicts read off unconverged runs?

Asked before any new tail work, because plan 14 §3 records three verdicts that
were wrong for exactly this reason. Every run behind every closed outlier or
objective verdict was checked, reading the `model.json` convergence fields.
**No verdict changes.**

| verdict | model | convergence | stands? |
|---|---|---|---|
| outlier policy (`keep_all` over winsor/drop), `artifacts/clip/policy_cap_decision.json` | LightGBM | all early-stopped, best 214–499 of cap 5000 | **yes**, and board-confirmed (v27 = 496) |
| prediction cap (v18 = 491) | post-hoc on converged preds | n/a | **yes** (board) |
| log1p target (`tests_log_target`) | LightGBM + per_airport | early stop fired, best 260–289, no cap hit | **yes** |
| Poisson (`tests_poisson_objective`) | LightGBM + per_airport | early stop fired, best 579–638, no cap hit | **yes** |
| v27 winsor_7200 blend | per-airport/per-operator LightGBM | best 89–204 / ≤ 284 | **yes** |
| hurdle, `delay_twostage_antigravity` | LightGBM clf + reg | best 238 / 164 | **yes** |
| hurdle / clamping / divergence gating (plan 07) | post-hoc on converged preds | n/a | **yes** |
| date-artifact exclusion (`TAXI_OUT_DRIVERS.md` §2g) | MLP forward, seeds 42–44 | early stop fired at 15/13/16 (cap 25) | **yes** |
| **Huber150 (§2i)** | MLP forward | **seeds 42 and 44 hit the 25-epoch cap** (44 still improving at 25); seed 43 early-stopped at 16 | **yes**: the converged seed 43 alone is **+78.7 s**, ~22× the seed SD. The mechanism (Huber fits a quantity between median and mean, and the train RMSE stayed at 390–420 s) is a bias that more epochs do not remove |

**Correction to §2i of `TAXI_OUT_DRIVERS.md`:** it says the Huber runs "needed
22–25 epochs before early stopping". Two of the three **never early-stopped**;
they ended at the 25-epoch cap. The verdict rests on the converged seed and the
mechanism, not on those two.

**Weight-solve surface only (no verdict depends on it):** 2 of 3
`forward_inner` `drop_date_artifacts` MLP seeds hit cap 25. That is the §1 defect class,
measured there as costing ~0 (mean alpha shift +0.003).

**Not a convergence issue but worth stating:** every outlier verdict predates
TabNet and FTT. None of them tested the tail behaviour of the current v31
components. The FTT write-up already found FTT over-predicts the extreme LIRF
tail (12,430 vs 7,643 on 50 test rows), which is why its LIRF weight solves to 0.
