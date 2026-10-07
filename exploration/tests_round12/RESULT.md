# Round 12 — result: PASS (small)

Pre-registration: `PREREGISTERED.md`. Leave one month out over 7 months; both correctors use the K2
setting and are applied to the realistic base.

| month | realistic base | A (trained on equal base) | B (trained on realistic base) | B − A |
|---|---:|---:|---:|---:|
| Jan | 348.452 | 339.042 | 338.808 | −0.234 |
| Jul | 347.443 | 338.794 | 338.796 | +0.001 |
| Aug | 265.315 | 254.299 | 254.265 | −0.034 |
| Sep | 260.125 | 246.189 | 246.197 | +0.008 |
| Oct | 231.562 | 214.691 | 214.585 | −0.106 |
| Nov | 255.464 | 236.752 | 236.677 | −0.075 |
| Dec | 231.264 | 206.741 | 206.576 | −0.165 |
| **pooled** | 280.456 | 266.838 | **266.759** | **−0.079** |

**PASS:** pooled better, 5/7 months, and still better with the best 5 rows removed (266.809 →
266.739).

The first run was interrupted after January by a session restart. It was rerun from scratch, and
January reproduced exactly.

**v61:**
- `final_pipeline.py` on the v59 base with stage 2 = `tests_adsb_full/model_v61` (K2 trained on
  realistic-base residuals over all 7 months, 627,042 rows, 5 seeds, reload identical);
- every other stage is as in v60.

**Expectation written before upload: 270.32–270.48 (central about 270.40)** against v60 = 270.4667.
- A near-tie is plausible.
- The realistic training base uses v42's 3-member weights, while the v59 base also contains
  CatBoost and the wide FTT.
- If it scores above 270.4667, v60 is re-uploaded last.
