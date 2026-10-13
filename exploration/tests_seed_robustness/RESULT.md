# Seed-robustness of the `airport_*_multimodel` "champion" ladder

**Question.** `airport_unsup_multimodel` was crowned "unanimous 3-seed champion"
over its sibling arms on seeds 42/43/44, by val margins under 1 s. Is that a real
ranking, or does it reshuffle once more seeds are added?

**Method.** Re-ran the family (`airport_operator/wtc/runway/wtc_rec/unsup_multimodel`
+ `multimodel_antigravity`) at seeds **45, 46** (47 finishing) on the canonical
protocol, retraining the stochastic parents (per-airport LightGBM, the MLP) and
the blend at each seed. `analyze.py` reads every seed's `eval_{val,test}.json` off
disk. Numbers table: `result_seeds_42-46.txt`.

## Result — the ranking is seed noise

5-seed means and the win margin vs the seed spread:

| metric | #1 | mean | #2 | mean | #1→#2 margin | #1 seed SD | **margin ÷ noise** | order held across seeds? |
|---|---|---:|---|---:|---:|---:|---:|---|
| val | unsup | 249.38 | wtc_rec | 249.82 | 0.44 | 1.90 | **0.23** | top-4 held; `mm_antigravity` passed `operator` at s45; `runway` last at s45/46 |
| test | unsup | 244.85 | wtc_rec | 245.14 | 0.28 | 3.99 | **0.07** | **broke at s46**: `wtc_rec 247.25 < unsup 247.41` |
| val in-band | **runway** | 227.45 | unsup | 227.49 | 0.04 | 1.33 | **0.03** | #1 is a different arm than on val/test |
| test in-band | unsup | 219.10 | wtc_rec | 219.37 | 0.26 | 0.56 | **0.47** | **broke at s46**: `wtc_rec 218.63 < unsup 218.90 < mm_antigravity 218.99` |

Full 6-arm family spread of means: **1.8 s (test-ib) to 4.8 s (test)** — i.e. no
larger than re-seeding any single member.

### What changed when seeds 45–46 were added

- On **seeds 42/43/44** the order was a clean `unsup < wtc_rec < wtc < operator <
  mm_antigravity` on val and test — the basis for the "unanimous champion" claim.
- On **test** and **test in-band**, seed 46 puts `airport_wtc_rec_multimodel`
  ahead of `airport_unsup_multimodel`. The "unanimous" champion is now #2 on the
  two metrics that matter most.
- On **val in-band**, the 5-seed #1 is `airport_runway_multimodel` — an arm the
  build's own notes say uses an axis (runway) that "barely" changes LIRF's weight.
- Below the top two, `multimodel_antigravity` (usually last) jumps to 3rd–4th at
  seeds 45 and 46, beating `operator` on val and test-in-band.
- `multimodel_antigravity` seed 44 (val 262.8, +14 vs its other seeds) is a
  single bad MLP draw and inflates its 5-seed mean by ~3 s — a reminder the whole
  family's numbers ride on the MLP parent's seed.

## Conclusion

The six `airport_*_multimodel` arms are **one arm** — the per-airport tree×MLP
blend (= `multimodel_antigravity`) with the weight bucketed six slightly different
ways. They land in a ~3 s cloud and their internal order is decided by the seed,
not the bucketing. Every "X beats Y" claim in the ladder has a margin 2–30× smaller
than the seed spread. `airport_unsup_multimodel`'s extra machinery (k-means K=6,
4-way regimes, λ=2000 shrinkage) buys ~0.3 s of that noise.

**Ship `multimodel_antigravity` (or `cross_blend w_inner` — a tie) and treat the
rest of the family as within-noise duplicates.** Verdicts folded into
`../pipeline_review.csv` (tier E).
