# FT-Transformer (column attention) — result

**Status: CONFIRMED on three independent periods.** The screen passed, it
survives against a 12-seed MLP partner, and a pre-registered third-surface
check on `seasonal_full` (Jan + Jul 2025) confirmed it. The honest effect size
is **~1.5 s in-band, not the 3.6 s the forward-test reading suggested** -- that
reading was the optimistic outlier of four. FTT's weight at LIRF is exactly
0.000 in all four independent solves.

Nothing is built and no submission slot has been spent; that decision is the
user's, with all four surfaces below.

**Checked against the SHIPPED blend and it survives:** mixing FTT into
`airport_operator_multimodel` at non-LIRF improves it by 2-4.6 s on val AND
test. That is the comparator that killed the router. **Still not established:**
an honestly-solved weight (needs FTT seeds on `forward_inner`), and board
transfer -- congest, idband and candidate 9 all won locally and lost on the
board. Nothing built or submitted.

Two earlier readings off truncated runs (cap 60, cap 90) were wrong in the same
direction and are retracted below -- the convergence history is kept in full
because it is the finding.

Bar and rationale: `PREREGISTERED.md`, fixed before any run. Amendment 1 (lr
1e-4 → 1e-3) was made on convergence-curve grounds before any screen number
existed.

## Setup

`model.py` + `run_ftt.py`. FT-Transformer (Gorishniy et al. 2021): every
feature becomes a token (per-feature weight+bias for numericals, embeddings for
categoricals), a [CLS] token is prepended, 3 pre-norm blocks of MHSA + FFN,
readout from [CLS]. d_token 64, 8 heads, 1.27M params, 58 tokens
(15 categorical + 42 numerical + CLS).

`forward` / `with_aobt` / `keep_all`, identical data pipeline, optimizer,
gradient clipping, early stopping and output format to
`tests_lirf_seeds/train_seeds.py` (the fast `mlp_antigravity` loop), so any
delta is attributable to the architecture.

## Convergence

| lr | outcome |
|---|---|
| 1e-4 (paper default) | **aborted at epoch 15/60.** Monotone, ~6 s/epoch, 499.80 → 419.73, no plateau. Would have ended at the cap. `run_lr1e-4_ABORTED.log`. |
| **1e-3** (the MLP's own rate) | stable, no instability signature. 449.92 → **298.89 at epoch 58**, then **hit the 60-epoch cap still improving** (~1 s/epoch, decelerating). |

**Seed 42 is therefore truncated, not converged**, and `decorrelation.py`
refuses to screen it — the same gate that rejected TabNet final seeds 47 and
51. Seeds 43/44 were not trained: with seed 42 failing the gain bar by 9–12 s
there was no case for 80 more minutes of GPU to average a decisive margin.

## Descriptive numbers (NOT the screen — seed 42 is capped)

Computed from the saved best-epoch predictions, reported because they are
decisive in one direction and remarkable in the other.

| | FTT seed 42 | MLP single seed | MLP ens3 |
|---|---:|---:|---:|
| val pooled | **298.89** | 255.11 | 249.77 |
| val LIRF | **790.84** | 551.47 | 539.60 |
| test pooled | **320.25** | 277.96 | 260.72 |
| test LIRF | **912.02** | 512.13 | 480.49 |

**Standalone gap: +43.8 s val, +42.3 s test** against a *single* MLP seed, and
worse still at LIRF — so it is not a LIRF specialist either, which is where
TabNet earned its place.

### The decorrelation is the best this repo has ever measured

| member | residual corr vs MLP | outcome |
|---|---:|---|
| GRU state model | 0.980 | dead |
| MLP seed vs another MLP seed | 0.93 | (reference: what "no diversity" looks like) |
| matched/unmatched router | 0.898 / 0.876 | dead |
| TabNet | 0.88–0.94 | **paid** (v29) |
| **FT-Transformer** | **0.761 val / 0.729 test** | see below |

Column attention genuinely does make different mistakes — decisively more
different than TabNet, which is the only architecture that ever paid here.

### But the swap still loses, and by a lot

Descriptive equal-weight swap (3 MLP seeds → 3× this FTT), the same
construction as the registered screen:

| | A (3+3 MLP) | B (3 MLP + FTT) | gain A−B | bar |
|---|---:|---:|---:|---:|
| val | 248.15 | 257.46 | **−9.32 s** | ≥ +2 |
| test | 258.29 | 270.11 | **−11.82 s** | ≥ +2 |

## What this refines

`../CONVERGENCE_AUDIT.md` §7 proposed residual correlation as *the* screen,
reading the pattern GRU 0.98 → dead, TabNet 0.88–0.94 → paid. **This result
shows correlation is necessary but not sufficient.** A member must also be
close enough standalone for its diversity to be worth its error: at 0.73 the
FT-Transformer is the most decorrelated member ever measured here and still
drags the ensemble mean by 9–12 s, because it is ~43 s worse to begin with.
TabNet paid because it *tied* the MLP standalone **and** decorrelated. The
screen needs both axes, not one.

## 3-seed screen (`screen_seeds424344_e132.txt`) -- PASS

All three seeds converged; none hit the 132 cap:

| seed | best epoch | ran | val |
|---|---:|---:|---:|
| 42 | 102 | 112 | 263.32 |
| 43 | 112 | 122 | 267.18 |
| 44 | 117 | 127 | 263.38 |

| side | corr | pooled gain | B better in | bar | verdict |
|---|---:|---:|---:|---|---|
| val | 0.885 | **+4.036** | **100 %** | < 0.95, > 0, >= 80 % | **PASS** |
| test | 0.841 | **+3.447** | 93 % | < 0.95, >= +2 | **PASS** |

**The effect grew on averaging** (+2.31 -> +4.04 val, +3.11 -> +3.45 test).
That is the opposite of what happened to the two previous promising seed-42
draws: the GRU and TabNet both looked like real wins at seed 42 and were erased
by averaging. This one strengthens, which is the behaviour of a real effect
rather than a lucky draw.

FTT-ens3 is still *worse standalone* than MLP-ens3 (val 253.73 vs 249.77, test
269.20 vs 260.72). It earns its place purely on decorrelation -- 0.84-0.88
against 0.97 for MLP-vs-MLP.

### The split is the structural finding

| stratum | val gain | B better | test gain | B better |
|---|---:|---:|---:|---:|
| non-LIRF mix | +4.380 | **100 %** | **+20.196** | **100 %** |
| LIRF | +6.806 | 89 % | **-68.671** | **0 %** |

TabNet pays **only** at LIRF (v29 = 293.6685, a LIRF-only override). This pays
**only** off LIRF, and hurts badly at LIRF on test. If it holds against the
right comparator the two are complementary, not competing: a non-LIRF member
alongside v29's LIRF-only override.

### Against the SHIPPED BLEND (`vs_shipped_blend.txt`) -- survives, with one framing corrected

No training; scores the already-trained `airport_operator_multimodel` (tag
msmlp, the forward-split shipped-class blend, val 247.39) on the same rows.

**1. It still decorrelates, but less than the headline suggested.**

| | vs MLP-ens3 | vs SHIPPED blend | vs shipped, non-LIRF only |
|---|---:|---:|---:|
| val | 0.8745 | 0.8829 | **0.9281** |
| test | 0.8114 | 0.8802 | **0.9106** |

At non-LIRF -- the stratum where it wins -- correlation against the shipped
blend is 0.91-0.93, much closer to the 0.95 bar than the pooled 0.88. The tree
side evidently already captures part of what column attention finds.

**2. It nonetheless improves the shipped blend at non-LIRF, on BOTH surfaces:**

| w (FTT weight at non-LIRF) | val non-LIRF | test non-LIRF |
|---:|---:|---:|
| 0.0 (shipped) | 204.65 | 209.15 |
| 0.1 | 203.39 (-1.25) | 207.46 (-1.68) |
| 0.2 | 202.44 (-2.21) | 206.15 (-2.99) |
| 0.3 | 201.78 (-2.87) | 205.22 (-3.93) |
| 0.5 | 201.38 (-3.27) | 204.50 (-4.65) |

Same sign and similar magnitude on val (Sep-Oct) and on test (Nov-Dec, a later
held-out period). This is where the router failed and this does not: the router
lost to the shipped blend on both strata.

**This sweep is DESCRIPTIVE and w must not be selected from it** -- it is read
on the same surfaces it reports. An honest w has to be solved on
`forward_inner`, which needs FTT seeds trained there (not yet done).

**3. CORRECTION -- the "mirror image of TabNet" framing was too tidy.**

The screen's -68.7 s LIRF test gap is a **tail artefact, not a structural
weakness**:

| | n | shipped | FTT-ens3 | MLP-ens3 |
|---|---:|---:|---:|---:|
| val LIRF in-band (y <= 7200) | 28,729 | 429.33 | **418.89** | 423.93 |
| val LIRF tail (y > 7200) | 80 | 6178.19 | 7365.89 | 6349.34 |
| test LIRF in-band | 23,346 | 335.52 | 339.37 | 336.16 |
| test LIRF tail | 50 | 7642.54 | **12430.00** | 7434.19 |

FTT is *better* than the shipped blend on LIRF in-band at val and level at
test. The entire LIRF deficit is 50-80 tail rows, where it over-predicts badly:
the tail's share of LIRF squared error rises from 52.6 % (shipped) to 74.2 %
(FTT) on test. That is this repo's well-known pooled-RMSE lottery
(PIPELINE_REVIEW.md: ~3 date-artifact rows carry ~52 % of pooled test squared
error), not evidence of a LIRF-vs-non-LIRF specialisation.

So the correct reading is **not** "TabNet owns LIRF, FTT owns the rest". It is:
FTT adds real bulk accuracy across the board and must be kept away from the
extreme tail. A non-LIRF-only restriction is one way to do that and happens to
work, but the mechanism is tail handling, not geography.

## THIRD-SURFACE CONFIRMATION (`confirm_seasonal_full.txt`) -- CONFIRMED

Bar fixed before any seasonal_full number existed
(`PREREGISTERED_seasonal.md`): in-band gain >= 1.0 s AND non-LIRF gain >= 1.0 s
on seasonal_full **test (Jan + Jul 2025)**. Protocol copied from
`tests_tabnet/confirm_seasonal_full.py`: seasonal_full is in
`config.SELECTION_FORBIDDEN_SPLITS`, so **both** blends' per-airport weights
were solved on forward_inner and applied **frozen**. Six MLP seeds (all that
exist as registered seasonal_full runs), matched between solve and apply.

All three FTT seeds converged: best epochs 123 / 116 / 116, none hit the 200
cap. Dec val 265.92 / 265.28 / 263.79.

| seasonal_full | pooled | in-band | LIRF | non-LIRF |
|---|---:|---:|---:|---:|
| **TEST (Jan+Jul)** 2-member | 349.36 | 252.54 | 763.14 | 289.21 |
| **TEST** 3-member (+FTT) | 347.97 | 251.10 | 763.14 | 287.38 |
| **TEST marginal** | **-1.40** | **-1.45** | 0.00 | **-1.83** |
| Dec val (*not a bar*) 2-member | 224.45 | 213.29 | 410.23 | 202.77 |
| Dec val 3-member | 222.78 | 211.58 | 410.23 | 200.77 |
| Dec val marginal | -1.68 | -1.71 | 0.00 | -2.00 |

**CONFIRMED.** Dec val is reported but was excluded from the bar in advance --
it is FTT's own early-stopping surface.

### All four readings, and the honest effect size

| period | surface | in-band | non-LIRF |
|---|---|---:|---:|
| Sep-Oct 2025 | forward val | -0.86 | -1.47 |
| Nov-Dec 2025 | forward test | **-3.63** | **-4.22** |
| Dec 2025 | seasonal_full val | -1.71 | -2.00 |
| Jan + Jul 2025 | seasonal_full test | -1.45 | -1.83 |

**Every reading helps, and the sign is consistent across four periods spanning
the whole year.** But the magnitudes are not: forward test (-3.63) is roughly
2.5x the other three, which cluster at **-0.9 to -1.7 s in-band**. The honest
estimate is **~1.5 s in-band / ~1.8 s non-LIRF**, and the earlier -3.6 s should
not be quoted as the expected value.

**FTT weight at LIRF = 0.000 in all four solves** (3-seed, 12-seed and 6-seed
MLP partners, plus six-start multistart). Its contribution is entirely off
LIRF, by the solver's own choice, and LIRF RMSE is identical between the
2-member and 3-member blends on every surface.

### What a submission would be worth

`tests_local_vs_board`: `board ~ -246 + 2.24 x local`, residual sd ~9 board
seconds. A ~1.4 s local pooled gain maps to roughly **-3 board seconds**, which
would put the current best (v29 = 293.6685) near ~290.7. For scale, v29's own
gain over v25 was -1.54. The residual sd of the mapping (~9) is larger than the
predicted effect, so **a single submission cannot cleanly verify this** -- the
same limitation that applies to every sub-5 s lever this build has tested.

Cost to get there: FTT refit on `full` (12 months) x3 seeds, a `full` weight
solve, and a registered arm -- roughly 5-6 h GPU plus the arm work.

## DEFLATION CHECK: does FTT still add against a STRONG MLP side?

`solve_vs_strong_mlp.py` / `.txt` / `.json`. The -2.9 s above was measured
against `msmlp`, whose MLP side is **3 seeds**; v25 ships **18**. Seed-averaging
is this build's largest measured lever (1/3/6/12/18 seeds = board
324/306/299/295.75/295.21), so that was an upper bound. Clean ablation: **both**
blends re-solved from scratch on forward_inner with the same **12-seed** MLP
(12 = every seed present on both surfaces), frozen, applied to forward val/test.
Only difference between them is FTT's presence.

| surface | | pooled | in-band | LIRF | non-LIRF |
|---|---|---:|---:|---:|---:|
| **val** | 2-member (tree+MLP12) | 245.58 | 224.95 | 537.53 | 202.45 |
| | 3-member (+FTT) | 244.46 | 224.09 | 537.53 | 200.98 |
| | **FTT marginal** | **-1.12** | **-0.86** | 0.00 | **-1.47** |
| **test** | 2-member (tree+MLP12) | 245.34 | 219.42 | 519.71 | 209.93 |
| | 3-member (+FTT) | 242.00 | 215.78 | 519.71 | 205.71 |
| | **FTT marginal** | **-3.34** | **-3.63** | 0.00 | **-4.22** |

**It survives, but the magnitude is surface-dependent.** Against the 3-seed
baseline the in-band gain was -1.94 val / -2.87 test; against 12 seeds it is
**-0.86 val / -3.63 test**. Val deflated by ~55 %, test did not deflate at all.
The two surfaces disagree about size by 4x, which is a real caveat and not
something to average away: val is Sep-Oct, test is Nov-Dec.

**Where it acts is consistent.** non-LIRF is the stable read: -1.47 val,
-4.22 test, and the non-LIRF baseline is nearly identical between the 3-seed
and 12-seed comparators (209.15 vs 209.93 on test). **LIRF gets FTT = 0.000
from every solve and every multistart**, so FTT contributes exactly nothing
there by construction.

### Solver check: the equal-weight airports are NOT stalls

Four airports solved to exactly [1/3, 1/3, 1/3] with the 12-seed MLP (one did
with 3 seeds), which is the optimiser's own start point and the signature of
the SLSQP failure this repo documents. Checked by multistart from six starts:

| airport | obj at 1/3 | best found | equal weights leaves |
|---|---:|---:|---:|
| EDDF | 0.039086 | 0.039085 | **0.002 %** |
| LTFM | 0.055431 | 0.055406 | **0.045 %** |
| EGLL | 0.033201 | 0.033133 | **0.204 %** |
| LEMD | 0.030776 | 0.030698 | **0.253 %** |
| *EHAM (control)* | 0.040068 | 0.038601 | *3.661 %* |

SLSQP stops at w0 exactly when the surface is flat enough that moving does not
pay, and it moves when there is something to gain (EHAM, 3.7 %). Benign; the
reported gains are not handicapped by it.

### Side-finding: the 12-seed baseline is WORSE on test than the 3-seed one

Worth recording because it cuts against this build's central seed-averaging
result. The 2-member baseline with 12 MLP seeds scores **test 245.34** where
the shipped 3-seed `msmlp` arm scores **239.87** -- 5.5 s worse with 4x the
seeds. It is entirely LIRF: **519.71 vs 486.99** at LIRF, while non-LIRF is
flat (209.93 vs 209.15).

LIRF on forward test is 23,396 rows of which ~50 carry most of the squared
error, so this is the pooled-RMSE lottery in the one stratum where it bites
hardest, not evidence against seed-averaging (the board result for more seeds
is real and was measured on the actual scored set). But it means **no single
local LIRF number should be trusted to rank two blends**, which is the same
lesson `tests_lirf_seeds` reached from the other direction.

## HONEST WEIGHT SOLVE -- vs the 3-seed shipped blend

`solve_three_member.py` / `.txt` / `.json`. Weights solved on **forward_inner
val (Jul-Aug)**, then **frozen** and applied to forward val (Sep-Oct) and test
(Nov-Dec). Three members, reusing `pipeline_airport_operator_catboost_multimodel`'s
own per-airport simplex solver (w >= 0, sum = 1) with FTT in the slot CatBoost
occupies there. Baseline is the SHIPPED two-member blend, not an MLP pool.

Seed 43 was re-run at cap 260 after hitting 180 -- best epoch 171 again, and
**byte-identical predictions** (0 of 381,895 rows differ). That is the third
instance of this session's cap finding, after TabNet seeds 47 and 51: the cap
had only blocked confirmation, not changed the model.

### Solved weights [FTT, tree, MLP]

| airport | FTT | tree | MLP |
|---|---:|---:|---:|
| EDDF | 0.333 | 0.333 | 0.333 |
| EDDM | 0.138 | 0.477 | 0.385 |
| EGLL | 0.382 | 0.443 | 0.176 |
| EHAM | 0.501 | 0.450 | 0.049 |
| LEBL | 0.238 | 0.698 | 0.065 |
| LEMD | 0.319 | 0.503 | 0.178 |
| LFPG | 0.296 | 0.382 | 0.322 |
| **LIRF** | **0.000** | **0.000** | **1.000** |
| LSZH | 0.586 | 0.278 | 0.136 |
| LTFM | 0.368 | 0.395 | 0.236 |

**The solver drove FTT to exactly zero at LIRF on its own** -- no non-LIRF
restriction was imposed. This was predicted in advance from the tail
decomposition and is the strongest confirmation that FTT's LIRF deficit is a
real, learnable property of the member rather than an artefact of how the
screen was constructed. LIRF resolves to 100 % MLP, exactly as the shipped
two-member blend does, so LIRF RMSE is *identical* between the two.

### Frozen-weight results

| surface | | pooled | in-band | LIRF | non-LIRF |
|---|---|---:|---:|---:|---:|
| **val** | shipped | 247.39 | 227.74 | 538.34 | 204.65 |
| | 3-member | **244.69** | **225.80** | 538.34 | **201.09** |
| | delta | **-2.70** | **-1.94** | 0.00 | **-3.55** |
| **test** | shipped | 239.87 | 219.03 | 486.99 | 209.15 |
| | 3-member | **236.99** | **216.16** | 486.99 | **205.58** |
| | delta | **-2.88** | **-2.87** | 0.00 | **-3.56** |

**The in-band deltas are the ones that matter** (-1.94 val, -2.87 test). This
build's pooled RMSE is a lottery -- ~3 date-artifact rows carry ~52 % of pooled
test squared error -- so a pooled gain alone would prove nothing. In-band
improves on both surfaces by a similar amount, so this is a bulk-accuracy gain,
not tail luck. It is also the first bulk gain this build has found since the
congestion feature, and that one did not survive the board.

### Solver sanity check (EDDF's equal weights)

EDDF solving to exactly [1/3, 1/3, 1/3] -- the optimiser's own starting point --
looks like the SLSQP failure this repo documents (`success=True` at a bad
point). Checked by multistart from all three corners: best objective 0.039203
vs 0.039208 at equal weights, a **0.013 % difference**. EDDF's objective is
genuinely flat across the simplex, so equal weights is near-optimal there, not
a stall. EHAM shifts by 0.07 %, LIRF returns [0, 0, 1] from every start.
Re-solving with multistart would not move the reported numbers.

### What is still NOT established

1. **The comparator uses a 3-seed MLP** (`msmlp`), while the shipped submission
   v25 uses **18**. A stronger MLP side may absorb part of what FTT
   contributes, so -2.9 s is an upper bound on what this would be worth on top
   of v25. The forward_inner MLP seeds 42-59 and forward seeds 42-53 exist on
   disk, so this is measurable without new training and is the next check.
2. **Board transfer.** congest (-0.60 s corrected val), idband (3/3 seeds
   improved) and candidate 9 (-1.27 s val) all won locally by margins of this
   size and lost on the board. Local-to-board here is roughly
   `board ~ -246 + 2.24 x local` with residual sd ~9 board-seconds, so a -2.9 s
   local gain maps to perhaps -6 board-seconds *if* it transfers at all.
3. Nothing is built and nothing is submitted.

### forward_inner seeds (the weight-solve surface): variance is ALL at LIRF

Trained Jan-Jun, val Jul-Aug -- the surface `solve_alpha_per_airport` fits on.

| seed | best ep | ran | pooled | LIRF | non-LIRF |
|---|---:|---:|---:|---:|---:|
| 42 | 152 | 162 | 388.51 | 1092.00 | 244.93 |
| 43 | 171 | **180 (hit cap)** | 386.53 | 1079.90 | 246.21 |
| 44 | 52 | 62 | **476.11** | **1477.22** | **236.46** |

Seed 44 looks like a failed draw at 476 pooled -- 90 s worse than the other
two -- and it is not. It is the **best** of the three at non-LIRF (236.46) and
catastrophic only at LIRF. The non-LIRF spread across seeds is ~10 s; the
pooled spread is 89.58 s. **All of the variance is LIRF.**

Contrast the forward split, whose val (Sep-Oct) has a milder tail: FTT val
263.32 / 267.18 / 263.38, spread **3.87 s**.

forward_inner's val is Jul-Aug, and this build already established that LIRF's
long-taxi tail is heaviest in summer (~390 July rows over 3600 s,
`tests_weather_metar` Part 2). So the weight-solve surface is the one place
FTT's tail weakness is maximally exposed -- the same mechanism as the 50-row
LIRF test deficit above, not a separate problem.

Two consequences:

* Seed 44 must **not** be dropped for being "unhealthy". It converged, early
  stopping fired, and its weakness is the known tail behaviour rather than a
  failed optimisation. Dropping a converged seed after seeing its number is
  post-hoc selection, which is what put six arms in `overtuned/`.
* The per-airport solve is the right instrument precisely because of this: it
  gets to give FTT a LIRF weight of its own, and on this evidence it should
  drive that weight toward zero while keeping FTT elsewhere.

Seed 43 is re-running at cap 260 (1.5x its observed best of 171); the truncated
run is kept as `TRUNCATED180_preds_seed43_forward_inner_lr1e-3.npz`.

### The comparator that decides it -- partially run

Everything above is measured against a 24-seed **MLP-only** pool
(`tests_lirf_seeds`). The shipped arm is `airport_operator_multimodel`: a
per-airport blend of the tree family and an 18-seed MLP. `tests_matched_unmatched_router`
is the cautionary precedent -- it beat a pooled LightGBM by 38-50 s and by
576-776 s on the unmatched stratum, then **lost to the shipped blend on both
strata**, because the fresh specialist re-derived a weaker version of structure
the shipped pipeline already had.

Required before any build: re-solve the per-airport blend on `forward_inner`
with FTT as a third member alongside the tree parent and the MLP, and compare
against the shipped two-member solve on forward val/test -- at non-LIRF
specifically. Local-vs-board remains a separate risk on top of that (congest,
idband and candidate 9 all won locally and lost on the board).

## Cap 132: converged, seed 42 alone

Seed 42 at cap 132 **early-stopped at epoch 112 with best epoch 102, val
263.32** -- the first genuinely converged FT-Transformer run. `hit_cap` false,
so `decorrelation.py` will screen it.

### The convergence history is the headline

| | cap 60 | cap 90 | **cap 132** | MLP single |
|---|---:|---:|---:|---:|
| best epoch / ran | 58 / 60 (capped) | 88 / 90 (capped) | **102 / 112 (early-stopped)** | — |
| val pooled | 298.89 | 275.19 | **263.32** | 255.11 |
| val gap | +43.77 | +20.08 | **+8.20** | — |
| test pooled | 320.25 | 289.42 | **278.06** | 277.96 |
| test gap | +42.29 | +11.46 | **+0.09** | — |
| residual corr (val/test) | 0.76 / 0.73 | 0.81 / 0.75 | 0.85 / 0.81 | (0.97 MLP-vs-MLP) |
| swap gain val | −9.32 | −1.28 | **+1.93** | |
| swap gain test | −11.82 | +0.80 | **+4.25** | |

**On test it now ties a single MLP seed to 0.09 s.** Every intermediate reading
was a measurement of an under-trained model, and each one would have closed the
architecture as a clear negative.

### The screen (`screen_seed42_e132.txt`, 300 paired draws)

| side | corr | pooled gain | B better in | verdict |
|---|---:|---:|---:|---|
| val | 0.855 | **+2.314** | 98 % | **PASS** (bar: < 0.95, > 0, >= 80 %) |
| test | 0.809 | **+3.113** | 91 % | **PASS** (bar: < 0.95, >= +2) |

### And it is the mirror image of TabNet

| stratum | val gain | test gain | B better (test) |
|---|---:|---:|---:|
| non-LIRF mix | +2.298 | **+11.660** | **100 %** |
| LIRF | +4.865 | **−34.365** | **0 %** |

TabNet paid **only** at LIRF and is shipped as a LIRF-only override (v29). This
model does the opposite: it is strongly positive on the non-LIRF bulk and
strongly negative at LIRF on test. If that survives seeds 43/44, the two are
complementary rather than competing, and the natural construction is a
non-LIRF-only override sitting alongside v29's LIRF-only one.

**Not yet claimed.** This is one seed. `PREREGISTERED.md` requires three
precisely because seed 42 alone looked like a real win for *both* the GRU and
TabNet and averaging erased it both times. Seeds 43/44 are training.

## (retracted) Cap 90: the closure-on-standalone-grounds reading was WRONG

Seed 42 re-run at cap 90 reached **best epoch 88, val 275.19 — and hit the cap
again, still improving.** The judgement recorded above, that the standalone gap
was decisive and convergence would move 322.9 "toward maybe 305, not toward
256", is retracted: 30 more epochs bought **23.7 s** and the curve has not
turned over.

| | cap 60 (best ep 58) | cap 90 (best ep 88) | MLP single | bar |
|---|---:|---:|---:|---:|
| val pooled | 298.89 | **275.19** | 255.11 | |
| val gap vs MLP single | +43.77 | **+20.08** | — | |
| val LIRF | 790.84 | **655.31** | 551.47 | |
| test pooled | 320.25 | **289.42** | 277.96 | |
| test gap vs MLP single | +42.29 | **+11.46** | — | |
| test LIRF | 912.02 | **752.86** | 512.13 | |
| residual corr vs MLP-ens3 | 0.76 / 0.73 | 0.81 / 0.75 | (0.93 MLP-vs-MLP) | < 0.95 |
| **swap gain, val** | −9.32 | **−1.28** | | > 0 |
| **swap gain, test** | −11.82 | **+0.80** | | >= +2 |

**The test-side swap gain has crossed zero and val is near break-even, on a
model that is still under-trained.** Three of the four preconditions are now
either met or close, and the one number that was supposed to be decisive
against it — the standalone gap — has more than halved and is still falling.

This is the failure mode this whole session documented, arriving in the work
that documented it: a verdict read off a truncated run. The cap-60 numbers
above are not a weak result for column attention, they are a measurement of an
under-trained model, exactly as the lr-1e-4 curve was, and exactly as TabNet's
lr-2e-2 "decisive loss" was before it was retracted.

**Status: OPEN, and the candidate is live.** Resolving it needs seed 42 at cap
132 (1.5x the observed best of 88, the repo's cap rule) to find where the curve
actually turns over, then seeds 43/44 for the multi-seed requirement before any
delta is trusted -- roughly 4.5 h GPU. No disposition until then, and none of
the numbers above should be cited as a verdict on this architecture.

## (superseded) Open question as originally framed

Seed 42 was still improving at the cap, so by this build's own rule
(`pipeline_tabnet_lirf_override/gate.py`: `hit_cap` means *unresolved*, not
*failed*) the standalone number is a floor, not a verdict. Closing on it
without resolving that would apply a standard this session explicitly rejected
for TabNet.

Being resolved now: **seed 42 alone at cap 90** (1.5× the observed best of 58,
the repo's cap rule). One seed, ~1 h, because only the convergence question is
open — not the screen, which fails by 9–12 s against a +2 s bar and would need
the standalone gap to close by roughly 30 s to come into play.

Result of that run and the final disposition go below.
