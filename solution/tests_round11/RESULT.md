# Round 11 — result

Pre-registration: `PREREGISTERED.md`.

**Part A (7-airport corrector, leave one month out over 7 months, against K = `model_v57`):**

| variant | pooled | months better | best 5 rows removed | verdict |
|---|---:|---:|---|---|
| **K2: 3,000 trees, 127 leaves, lr 0.015** | 289.638 → **289.363 (−0.275)** | **7/7** | 289.592 → 289.322 | **PASS** |
| K3: 3,000 trees, 63 leaves, lr 0.01 | 289.638 → 289.638 (±0.000) | 2/7 | worse | FAIL |

**Part B (the K setting on the small correctors):**

| corrector | pooled | months better | verdict |
|---|---:|---:|---|
| LIRF | 603.585 → 605.517 (+1.93) | 0/6 | **FAIL** |
| LFPG | 301.850 → 301.894 (+0.04) | 0/6 | **FAIL** |

Bigger models overfit the small LIRF (45k rows) and LFPG (7k rows) training sets, so both keep their
v55 / v56 settings.

**v60:**
- built by `final_pipeline.py` on the v59 base with stage 2 = `tests_adsb_full/model_v60` (K2 trained
  on all 7 months, 5 seeds, reload byte-identical);
- every other stage is as in v59;
- `final_pipeline.py --stage2-model tests_adsb_full/model_v57` on the v59 base still rebuilds v59
  byte-identical.

**Expectation written before upload: 270.40–270.75 (central about 270.55)** against v59 = 270.81. If
it scores above 270.81, v59 is re-uploaded last.

**Board: v60 = 270.4667** (−0.34 against v59; inside the written expectation of 270.40–270.75). It is the best score and the last upload.
