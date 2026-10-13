# Round 10 — result

Pre-registration: `PREREGISTERED.md`. Leave one month out over 7 months; A = the v48 corrector design
(pooled 290.237).

| variant | pooled vs A | months better | best 5 rows removed | verdict |
|---|---:|---:|---|---|
| **K: 1,500 trees, 63 leaves, lr 0.02** | **−0.599** | **7/7** | 290.174 → 289.582 | **PASS** |
| F: average with CatBoost | −0.348 | 7/7 | 290.181 → 289.843 | PASS |
| M: + member disagreement | +0.028 | 2/7 | worse | FAIL |

**Combination:** K + F = −0.591, which does **not** beat K alone (−0.599), so K alone is used, as
pre-registered. KFM (−0.653) includes the failed M and was not eligible.

**v57:**
- built by `final_pipeline.py` with stage 2 = `tests_adsb_full/model_v57` (K trained on all 7
  months, 5 seeds, reload byte-identical);
- the weather, LIRF and LFPG stages are unchanged; because "no ADS-B correction" is defined as
  "stage 2 left the base unchanged", 4 more frost rows and 1 more thunderstorm row receive offsets
  than in v56;
- `final_pipeline.py --stage2-model tests_adsb_full/model_v48` still rebuilds v56 byte-identical.

**Expectation written before upload: 270.70–271.30 (central about 271.0)** against v56 = 271.5625.
If it scores above 271.5625, v56 is re-uploaded last.

**Board: v57 = 270.8809** (−0.68 against v56; inside the written expectation of 270.70–271.30). It is the best score and the last upload.
