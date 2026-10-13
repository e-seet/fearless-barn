# Carrier-echo hedge for LIRF no-record departures — result: FAIL (decisive)

Bar: `PREREGISTERED.md`, fixed before any comparison with model predictions. `evaluate.py` →
`evaluate.txt` / `evaluate.json`.

The exploratory finding holds: within LIRF no-record departures the carrier predicts whether the
recorded off-block echoes the scheduled time, out of sample (leave-one-month-out Brier score
0.255 → 0.178, in both January and July). **But the hedge built on it is far worse than the
model** on every surface:

| surface | pooled RMSE, model → H1 | LIRF no-record RMSE, model → H1 |
|---|---|---|
| **Jan + Jul 2025** | 341.0 → **401.3 (+60.3)** | 4,747 → 7,832 |
| Sep–Oct | 245.7 → 286.1 (+40.3) | 3,839 → 6,539 |
| Nov–Dec | 257.2 → 315.7 (+58.5) | 5,099 → 10,224 |
| Jul–Aug | 306.2 → 377.9 (+71.6) | 4,430 → 7,394 |

H2 (the average of H1 and the model) is also worse everywhere (Jan + Jul +19.4). The shipped
model is already a much better hedge on these rows than any rule built on carrier and scheduled
delay alone. `p × ds` is costly on non-echo rows with large `ds`, and the model evidently uses
more of the row than the carrier. **Closed.** The carrier signal is real, but the model is not
missing it in a form a rule can add.
