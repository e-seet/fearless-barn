# v33 — TabNet LIRF override with every gate-healthy seed (board probe)

Written 2026-09-25, **before upload**. This is a **board probe decided after v32's
result**, not a pre-registered pass. The local 5 → 9 seed check (`RESULT.md`,
2026-09-21) closed at −0.5 s against a ≥ 6 s bar.

**Why probe anyway.** On two seed-count questions the board paid far more than the
local surfaces predicted: MLP 1 → 3 seeds was −0.9 local and −18 board, and FTT
3 → 6 was −0.13 to −0.30 local and −1.79 board (v32). The local instrument cannot
resolve seed-averaging effects of this size.

**What changes.** v32 with the LIRF override's TabNet mean taken over **all 11
seeds that pass the recorded health gate** (42, 43, 45, 46, 47, 48, 49, 50, 51, 52,
53; seed 54 is rejected at Dec val 294.1 > 260). The shipped 5 were
42, 43, 45, 46, 48. w = 0.55 is unchanged and frozen. Non-LIRF rows are
byte-identical to v32.

**Sizing (label-free, scratch):** RMS(v33 − v32) = 17.0 s on the scored set, all at
LIRF, against RMS(v32 − v31) = 20.4 s, which bought −1.79. That bounds the size
of the possible change, not its sign.

**Reading the result:** better than v32 means the board rewards TabNet seed
averaging as it did FTT's. Worse means the 5-seed mean was a favourable draw on
2026, and v32 stays the final model. **This is the last probe;** no further seed
experiments follow it.
