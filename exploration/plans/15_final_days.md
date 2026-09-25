# Plan 15 — the final days (from 2026-09-25)

**Standing:** v32 = 287.9898 is the best and the designated final. Each item below can
only add to it: a failure leaves v32 in place. This assumes ranking uses the best
score. **If the organisers rank a designated or private submission, stop after
item 1 and designate v32.**

## What the week established

- **Pays on the board:** new model *families* that are both decorrelated and
  competitive (TabNet at LIRF, FTT elsewhere), and FTT seed averaging (3 → 6:
  −1.79 board, ~4× the local estimate).
- **Does not pay:** anything inside the MLP family (wider, deeper, regularized,
  big-batch, PLE, TabM), tail rules, epoch changes, pretraining, and TabNet seed
  averaging (v33, 11 seeds: +0.91).
- **Instrument:** a forward (Nov–Dec) pass must replicate on seasonal_full
  (Jan + Jul). Three leads failed exactly there. Local surfaces *under*-read seed
  averaging on the board.

## Queue, in order

### 1. ~~Finish the backbone screen~~ DONE 2026-09-25: all three FAIL (`tests_backbone/RESULT.md`)
TabM+PLE stage 1, forward, with seeds 43 and 46 rerun at cap 100. Stage 2 runs
only if stage 1 passes. `tests_backbone/`. A FAIL closes the MLP-family question.

### 2. FTT at LIRF with a tail guard: no GPU, ~1 h
FTT is *better* than the shipped blend on LIRF in-band (val 418.9 vs 429.3,
`tests_ftt/RESULT.md`). Its LIRF weight solves to 0 only because it over-predicts
~50–80 extreme tail rows (12,430 vs 7,643 on test). Test at LIRF only:
`FTT_guarded = min(FTT, base)`, which lets FTT pull predictions *down* but never up,
as a fourth weight in the LIRF solve.
- Weight solved on forward_inner (9 FTT seeds exist there), frozen; judged on
  forward val/test and **seasonal_full Jan + Jul** (9 seeds exist).
- Bar (to be written into `tests_ftt_lirf_guard/PREREGISTERED.md` before any
  number): LIRF in-band better on all three surfaces **and** LIRF pooled not worse
  by more than 2 s on Jan + Jul.
- **Honest prior: low-moderate.** A fitted gate on the tail is close to the
  divergence-gating family closed in plan 07. `min()` has no fitted threshold,
  which is the difference.

### 3. FTT 7-seed board probe: no GPU, 1 slot
Full-data seed 48 is already trained (the 9-seed queue was stopped after it).
v34 = v32 with FTT 42–48 and weights re-solved exactly as v32's. Label-free
sizing: RMS move 7.5 s against v31→v32's 20.4 s, which bought −1.79. Expected
board −0.3 to −0.7. A *board probe after v32's result*, not a pass; the local
9-seed check failed on Jan + Jul (+0.06).
- *Optional:* seeds 49/50 on `full` (~3.3 GPU h) for a 9-seed version, only if
  v34 improves on v32.

### 4. Compose any winners
Items 2 (LIRF only) and 3 (non-LIRF only) touch disjoint rows, so they compose
exactly, as v31 did. Predict the composite score before uploading it.

### 5. Close-out (last day, no GPU)
Back up the untracked rebuild set (`tests_repro/dependency_manifest.csv`, 0.43 GB
plus v32's added FTT files); rerun `tests_repro/rebuild_check.py` for the final
version; finish `WRITEUP.md`; designate the final submission.

## Not planned, and why

- More MLP-family variants: six closed this week.
- More TabNet seeds: board-rejected (v33).
- New neighbour/sequence or retrieval models: residual correlation 0.98.
- Tail identity rules: board-rejected twice (v15, v17).
- Anything needing > 1 day of GPU plus a seasonal replication: no longer fits the time left.
