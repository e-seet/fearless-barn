# Audit of closed ideas: were any closed early or on weak evidence?

Written 2026-10-05, at the user's request, while round 10 and the wide-FTT queue run. Each closed
item in `EXPLORATION_SUMMARY.md` §3 was checked against its own record.

## Closed on solid evidence: no reopen

| idea | why the closure holds |
|---|---|
| Tree seed averaging (1 → 6) | Checked on Jan + Jul, the surface that predicted the board both times for FTT seeds; +0.16 there (`tests_tree_seeds`) |
| MLP backbones (PLE, TabM, TabM+PLE), MLP widths and depths | Converged reruns (capped runs redone); all fail forward, and TabM+PLE is +9 to +12 at LIRF |
| CatBoost truncation | Not dismissed while truncated: rerun at cap 8000, converged, scored *worse* (`CONVERGENCE_AUDIT.md` §5) |
| TabNet 5 → 11 seeds; small FTT 7 seeds | Board-tested (+0.91; +0.04) |
| Snapshot averaging | Converged second attempt; fails |
| ATFM delay statistics | Real data with full 2026 coverage; correlation barely above its permutation control |
| Holidays, OSM distance, runway configuration, routed distance | Ceilings measured; all under about 0.2 s |
| LIRF identity override | Board-tested twice (+0.86) |
| Runway queue features | Full leave-one-month-out test: +2.11 |
| Tabular columns in the ADS-B corrector | Board-tested (v47 +0.47) |

## Closed in an older setup, never re-tested in the current one

| idea | gap | prior |
|---|---|---|
| **CatBoost as a blend family** | Board-tested once (v26, 295.48 against v25 295.21), when the blend was trees + MLP only. It has **not** been tried as a family in today's blend (trees + MLP + three FTT shapes + TabNet). The board has repeatedly paid for model variety that local checks under-read (MLP seeds, FTT seeds, large FTT). Against that: CatBoost's errors track the trees' (residual correlation 0.87, §2m), so its diversity is smaller than an FTT shape's. | low to moderate |
| XGBoost as a blend family | Only compared standalone with LightGBM; never tried as a blend member | low (same tree-like errors) |

## Never tried

These are architecture options:
- NODE, SAINT, T2G-Former, ExcelFormer;
- a GNN over the taxiway graph;
- an explicit departure-queue simulation.

None fits before the 9 Oct freeze with proper validation.

## Recommendation

**The only genuine "closed but not really checked" item is CatBoost as a family in the current
blend.** It can be trained with the converged recipe (`cb2_recap`, cap 8000):
- runs: `forward_inner` (weights) and `full` (submission), 3 seeds;
- timing: after the wide-FTT queue on the GPU;
- build: folded into the same v58 base probe (four-plus-one families, per-airport weights
  re-solved).

The solver gives it zero weight where it adds nothing, so the risk is mainly the GPU time.
