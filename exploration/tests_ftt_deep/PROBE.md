# Deep-narrow FT-Transformer as a third FTT-slot component — written before training

2026-09-27. Base: **v39 = 287.0370** (FTT slot = 6 small d64×3 + 6 large d192×3, large
share 1/3), best and last. If 6 → 9 large seeds (queued first) produces a better base,
this probe is built on that base instead.

**Why.** The board has repeatedly paid for *variety* among attention models in the FTT slot
even when the added model was worse alone (large FTT: −0.64, then −0.25), and local
surfaces under-read it (C2 tied locally). Every MLP-family change failed. This adds a
**deeper, narrower** transformer: 6 blocks, d_token 96, 8 heads. More attention
rounds change how features combine, not just capacity.

**Training** (`run_ftt.py --d-token 96 --n-blocks 6 --amp --lr 5e-4 --tag d96b6`; 5e-4 as
the large model needed, bf16 for speed), seeds 42–44:
- forward_inner (cap 260): the weight solve;
- forward (cap 200): the median best epoch fixes the `full` count, as for the other FTTs;
- full: ceil(median forward best epoch), `--predict-ranking`.
Any capped run is rerun at a higher cap; a run whose val diverges early (best val
> 400 on forward) is recorded, not silently dropped.

**Build.** FTT slot = small : large : deep at fixed shares **1/3 : 1/3 : 1/3** (per-model
weights set accordingly), per-airport weights re-solved on forward_inner (default solver),
TabNet LIRF w = 0.459, FTT's LIRF weight must be 0. Scratch check against the base; the user
approves before `submit.py`.

**Expectation:** tie to about −0.4; a board probe, since the local check cannot judge FTT-slot
mixes. The fallback is a byte copy of the best file uploaded last.

## Amendment 1 (2026-09-27, before any deep result): fold in the 9 large seeds

User decision: the v40 candidate (6 small + 9 large, RMS move vs v39 only 3.86, expected
−0.05 to −0.15) is **not uploaded alone**; its 9 large seeds are folded into this probe.
**v41 FTT slot = 6 small (d64×3) + 9 large (d192×3) + 3 deep (d96×6), each family 1/3**:
per-model weights small 1, large 2/3, deep 2 (each family totals 6). The comparison base
stays **v39 = 287.0370** (best and last). One upload tests both changes; if v41 loses, v39
is re-uploaded as a byte copy.

## Amendment 2 (2026-09-28 02:40, user decision, before any forward or full deep run)

To save ~10 GPU hours (the deep model runs ~84 s/epoch, 3.3 h per forward_inner seed), the
forward stage is **skipped**. The `full` epoch count = **ceil(median forward_inner best
epoch of the 3 deep seeds)** instead of the forward median. Caveat on record: for the small
FTT the forward_inner median (~152) ran well above the forward median (112), so this count
may be longer than the forward rule would have given. For the large FTT the two were closer
(forward_inner 56–100 vs forward 99). The build and upload rules are unchanged.

## Result (2026-09-29)

**v41 = 286.8032, -0.234 vs v39 (287.0370).** Inside the pre-upload range (−0.6 to +0.3). Adding a third, deeper FT-Transformer family to the FTT slot pays again; the weight solver leaned further into the slot (EHAM 0.79, LSZH 0.82, LTFM 0.60). This includes the folded-in 6 → 9 large seeds (v40, not uploaded alone).
