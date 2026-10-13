# Copy-mixture member (two-stage idea) — pre-registration

Written 2026-10-07 ~23:30, before any model is fitted. User decision ("we can try and see how it work"),
time-boxed. Board: **v61 = 269.9888** (best and last).

**Idea and credit.** The two-stage design is from team outgoing-boat's public repository
(github.com/EnioAguiar/prc-taxiout-2026, README §1.1):
- a classifier of "the recorded off-block is a copy of the schedule" (|BLOCK − SCHED| ≤ 60 s);
- an L2 regressor trained on the non-copy flights only (target clipped at 7,200 s);
- an expectation mixture ŷ = p·(MVT − SCHED) + (1 − p)·ŷ_normal, never an argmax.

The LOBT window (|BLOCK − LOBT| ≤ 3,606 s, 100 % of 2025 departures) is taken from the same README.
None of their code is used. The implementation below is ours.

## Member M (all LightGBM on the 57 `with_aobt` features; trained without the month it predicts)

- **Classifier:** binary on `copy`; 500 rounds, lr 0.05, 127 leaves.
- **Regressor:** L2 on non-copy rows, y clipped at 7,200 s; 800 rounds, lr 0.05, 255 leaves,
  min_data_in_leaf 100.
- **Mixture:** M = p · clip(MVT − SCHED, 0, 86,400) + (1 − p) · ŷ_normal.
  - p is set to 0 when |SCHED − LOBT| > 3,606 s (a copy would break the LOBT window).
  - M is then clipped to the LOBT window [MVT − LOBT − 3,606, MVT − LOBT + 3,606] ∩ [0, ∞).

## Reference F (our current recipe, held out)

The round-13 setup: the realistic-4 base plus the K2 stage-2 corrector trained on the round-12 base
(as in v61), leave one month out.

## Combination

- final′ = (1 − w_a) · F + w_a · M, with one weight per airport a, w_a ∈ [0, 1];
- w_a is fitted by least squares on the **other six** months' held-out (F, M); it is never fitted on
  the month being scored;
- final′ is then clipped to the LOBT window.

**Surface:** leave one month out over Jan, Jul, Aug, Sep, Oct, Nov, Dec 2025; scored on all rows.

## PASS (final′ against F), all required

1. pooled RMSE better;
2. better in at least 5 of 7 months;
3. still better with the 5 largest-gain rows removed.

**Reported, not gating:**
- the split of the gain between body (y ≤ 3,600 s) and tail;
- outgoing-boat found that tail-only gains did not transfer to the board.

## Build

- **PASS →** v63 = v62 (or v61) combined with M through the airport weights fitted on all 7 months
  (M from models trained on all of 2025), then clipped to the LOBT window.
- The expectation is written before upload. If v63 scores above the then-best, that file is
  re-uploaded last.
- **FAIL →** closed.

**Cut-off:** decision by 2026-10-08 18:00.
