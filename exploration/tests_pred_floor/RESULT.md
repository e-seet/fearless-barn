# Prediction floor 60 → 240 s — result: FAIL, keep 60

`check.py` → `result.txt` / `result.json`, on the v35 analogue. Raising the floor
**hurts on every surface and both metrics**, monotonically in the floor: Jan + Jul
in-band 251.532 at 60 s, 251.546 at 180 s, 251.563 at 240 s; forward test 215.118 →
215.125 → 215.135. Only 561–589 predictions per surface fall below 240 s, and about
0.5 % of true taxi-outs are that short (1,689–2,027 rows per surface), so the low
predictions are mostly right. Closed; `config.PREDICTION_CLIP_MIN_SEC` stays 60.
