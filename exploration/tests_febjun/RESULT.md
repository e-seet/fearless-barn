# More training months for the ADS-B corrector (Feb–Jun 2025) — result: FAIL

Pre-registration: `PREREGISTERED.md`.

**Training:**
- split `holdout_febjun`: train Jan + Jul–Nov, val Dec, test Feb–Jun;
- models: tree, MLP seeds 42–47 (all early-stopped at epochs 15–19 of 45), small FTT seeds 42–44
  (best epochs 106–122 of 180, none at the cap);
- Feb–Jun base RMSE 294.2; ADS-B coverage in Feb–Jun 49–55 %.

**Fairness control:** the regenerated seasonal_full MLP base matches the cache to 0.004 s.

**Leave one month out** (A = v48 design trained on the other 6 scored months; A5 = the same plus
Feb–Jun, which nearly doubles the training rows):

| month | A | A5 | change |
|---|---:|---:|---:|
| Jan | 349.834 | 349.681 | −0.154 |
| Jul | 382.167 | 382.236 | +0.069 |
| Aug | 262.100 | 262.187 | +0.087 |
| Sep | 274.643 | 274.821 | +0.177 |
| Oct | 214.686 | 214.897 | +0.211 |
| Nov | 270.945 | 270.905 | −0.040 |
| Dec | 242.416 | 242.631 | +0.215 |
| **pooled** | 290.237 | 290.308 | **+0.071** |

**FAIL:** pooled worse, better in only 2/7 months, and still worse with the best 5 rows removed.

**Likely reason:** the extra rows come from a different kind of base.
- The Feb–Jun predictions come from models trained on only six months, mostly *after* the months
  they predict.
- Their residuals behave differently from the bases the corrector is applied on, so the corrector
  learns a mixed signal.
- More data of a different kind does not help. Earlier gains came from adding months whose bases were
  built the same way (forward-split bases).

**Consequence:**
- no v55 under this pre-registration;
- the weather pre-registration tied its fitted offsets (W) to v55 and ruled out a weather-only
  upload, so **v50 stays final**.
- On the two scored months, v50's hand offsets (V: Jan −0.07, Jul −0.31) do at least as well locally
  as W (Jan +0.19, Jul −0.27), so nothing is lost.
