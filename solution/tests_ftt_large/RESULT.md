# Larger FT-Transformer (d_token 192) — result: FAIL (both candidates)

Bar: `PREREGISTERED.md` (+ amendment 1: the lr-1e-3 runs hit the pre-registered
divergence rule; everything was redone at lr 5e-4, tag `d192lr5`). `check.py` →
`check.txt` / `check.json`. All screen runs early-stopped (none capped).

Standalone the d192 model is **worse** on every surface. The 3-seed FTT-alone in-band
figures: forward test 219.6 vs 217.6 (6 small), Jan + Jul **266.8 vs 257.2**.
Per-seed forward val: 303.0 / 268.6 / 281.4 against 263.3 / 267.2 / 263.4 for the small
model. Bigger tokens did not make a better FTT on this data.

In-band RMSE of the 3-member blend (weights re-solved on forward_inner per candidate):

| | forward val | forward test | **Jan + Jul** | verdict |
|---|---:|---:|---:|---|
| comparator (6 small FTT) | 223.875 | 215.484 | 250.971 | — |
| C1: 3 d192 | −0.238 | +0.272 | **+1.090** | FAIL |
| C2: 6 small + 3 d192 | −0.283 | −0.560 | **+0.030** | FAIL |

C2 is the fifth lead this week to help the forward months (Sep–Dec) and not
Jan + Jul. Closed. The `full` stage was stopped after seed 42 (99 epochs) because no
candidate passed; its outputs are unused.
