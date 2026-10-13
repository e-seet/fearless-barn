# v34 — FT-Transformer 6 → 7 seeds (board probe)

Written 2026-09-25, **before upload**. Plan 15, item 3. A **board probe decided after
v32's result**, not a pre-registered pass: the local 6 → 9 check failed on Jan + Jul
(+0.062 in-band), but local surfaces under-read FTT seed averaging by ~4× (v32).

- v32 with FTT seeds 42–48 (seed 48's `full` run exists; the 9-seed queue was
  stopped after it). Per-airport weights re-solved exactly as v32's
  (`final_weights_ftt7.json`); FTT's LIRF weight is 0.
- Acceptance (scratch): LIRF identical to v32 on 26,899/26,899 rows; clean
  validation; max 81,837; 87 rows > 7,200 s (v32: 88).
- **Label-free size:** RMS(v34 − v32) = 7.454 s, against RMS(v32 − v31) = 20.431 s,
  which bought −1.79. **Expected board −0.3 to −0.7; a tie is plausible.**
- Better than 287.9898 → FTT seed averaging still pays on the board. Worse →
  v32 stays final and no more FTT seeds are trained.

## Result (2026-09-25)

**v34 = 288.03**, +0.04 vs v32: a tie. FTT seed averaging has saturated on the board at 6 seeds. No seeds 49/50 on `full`, and no v35.
