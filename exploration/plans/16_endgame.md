# Plan 16 — endgame (from 2026-09-26; deadline 2026-10-11)

**Standing:** v35 = 287.9247 is the best **and last** submission. **Ranking uses the last
submission.** Every experiment must end with the chosen file uploaded last.

## The week's evidence

Five model-side leads in a row improved the forward months (Sep–Dec) and failed
seasonal_full Jan + Jul: tree-seed averaging, FTT 9 seeds, the larger FTT (C1 and C2),
the `deep` MLP, and the ensemble-epoch effect. Where the board could check it (FTT 3→6
and 6→7 seeds), Jan + Jul matched the board both times. Also closed: MLP backbones (PLE,
TabM), tail rules, pretraining, extra TabNet seeds, the FTT LIRF guard, and the
prediction floor.

**Instrument caveat (not a reason to re-open anything).** seasonal_full's July sits
between training months (interpolation), and its January precedes them. The board
is a full-year refit predicting 1 and 7 months ahead. A +0.03 "tie" means *no
detectable gain*, not harm.

## Sweep of items the repo itself left open (2026-09-26)

| item | status |
|---|---|
| CatBoost 3-way ("open, next step" in `TAXI_OUT_DRIVERS.md`) | built as v26; board 295.48, worse than v25. Closed |
| label correction for stand-turnover rows | retracted in its own RESULT before running. Closed |
| prediction floor 240 s | tested today: FAIL (`tests_pred_floor`) |
| TabNet decorrelation | answered: TabNet paid (v29) |
| **recency-weighted retrain** (plan 13 §6) | **untested at the board's horizon**, and untestable locally. Plan 13 notes it helps January while hurting July. ~2 days of retraining, board-only verdict. **User decision** |
| **C2 board probe** (6 small + 3 d192 FTT) | **running**: d192 `full` seeds 43/44 (~3 GPU h); expect a tie. `tests_ftt_large/PROBE_C2.md` |

No further board fits: v35 used the only set of existing uploads that lie on one
line in prediction space. More would need uploads built only to fit test parameters,
which is ruled out unless the organizers confirm it's allowed.

## Endgame schedule

- **Now → ~10-08:** at most the C2 probe and, if the user chooses, recency weighting.
  Each ends with the best file uploaded last.
- **Freeze 2026-10-09:** after it, only a byte-identical rebuild of the chosen final
  file may be uploaded. Confirm the result JSON's `"file"` line.
- **Close-out:** rerun `tests_repro/rebuild_check.py` on the final version (v35 also
  depends on FTT seeds 45–47 and the TabNet w = 0.459 run), back up the untracked
  rebuild set, finish `WRITEUP.md`.

## Needed from the user

1. The deadline **time and time zone** on 10-11.
2. **The final deliverable:** code release, paper or report, its format and due date.
   The organizers stress model design and documentation, so this may be the most
   valuable remaining work.
3. **Backup location** for the untracked rebuild set.

## Process fix

Queues now run through `run_queue.py` (PID file, STOP flag, per-step log), and each
launch is confirmed a few minutes later from the log and the GPU state.
