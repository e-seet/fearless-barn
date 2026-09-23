#!/usr/bin/env python3
"""Pre-check before spending 4h on forward_inner FTT seeds: does the FT-Transformer
decorrelate from the SHIPPED BLEND, or only from an MLP-only ensemble?

The 3-seed screen passed against a 24-seed MLP pool (tests_lirf_seeds). That is
not the arm FTT would have to join. `tests_matched_unmatched_router` is the
precedent: it beat a pooled LightGBM by 38-50s and then LOST to
airport_operator_multimodel on both strata, because the fresh specialist
re-derived weaker versions of structure the shipped pipeline already had.

Also decomposes the LIRF asymmetry (-68.7s test in the screen) into in-band
(y <= 7200) and tail, because this repo's pooled RMSE is lottery-prone:
PIPELINE_REVIEW.md opens with ~3 date-artifact rows carrying ~52% of pooled
test squared error. If the asymmetry is a handful of rows, the "mirror image of
TabNet" reading is too tidy.

No training. Scores the already-trained shipped arm on the same forward rows.
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import config
from common import dataset as D
from common.predict import predict_arm

ARM, SEED, TAG = "airport_operator_multimodel", 42, "msmlp"
FS, POL, SPLIT = "with_aobt", "keep_all", "forward"
SEEDS = [42, 43, 44]
rm = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))
LS = ROOT / "tests_lirf_seeds"

Xtr, _, _ = D.load_part(SPLIT, "train", FS, filter_target=True, outlier_policy=POL)
for part, pk in (("val", "pv"), ("test", "pt")):
    X, y, meta = D.load_part(SPLIT, part, FS, filter_target=False)
    X = D.align_columns(Xtr, X)
    y = np.asarray(y, float)
    ship = predict_arm(ARM, SPLIT, part, FS, SEED, POL, X=X, meta=meta, Xtr=Xtr, tag=TAG)
    ftt = np.mean([np.load(HERE/f"preds_seed{s}_lr1e-3_e132.npz")[pk] for s in SEEDS], axis=0)
    mlp = np.mean([np.load(LS/f"preds_seed{s}.npz")[pk] for s in range(48, 72)][:3], axis=0)
    ap = meta["ADEP_mvt"].astype(str).to_numpy()
    assert len(ship) == len(ftt) == len(y), (len(ship), len(ftt), len(y))
    L, NL = ap == "LIRF", ap != "LIRF"
    print(f"\n===== {part.upper()}  (n={len(y)}, LIRF {int(L.sum())}) =====")
    print(f"  shipped blend  pooled {rm(y,ship):7.2f} | LIRF {rm(y[L],ship[L]):8.2f} | non-LIRF {rm(y[NL],ship[NL]):7.2f}")
    print(f"  FTT-ens3       pooled {rm(y,ftt):7.2f} | LIRF {rm(y[L],ftt[L]):8.2f} | non-LIRF {rm(y[NL],ftt[NL]):7.2f}")
    print(f"  MLP-ens3       pooled {rm(y,mlp):7.2f} | LIRF {rm(y[L],mlp[L]):8.2f} | non-LIRF {rm(y[NL],mlp[NL]):7.2f}")
    c_ship = float(np.corrcoef(y-ftt, y-ship)[0, 1])
    c_mlp  = float(np.corrcoef(y-ftt, y-mlp)[0, 1])
    print(f"  resid corr FTT vs SHIPPED BLEND = {c_ship:.4f}   (vs MLP-ens3 = {c_mlp:.4f})   bar < 0.95")
    for nm, msk in (("non-LIRF", NL), ("LIRF", L)):
        cs = float(np.corrcoef(y[msk]-ftt[msk], y[msk]-ship[msk])[0, 1])
        print(f"    {nm:9s} corr vs shipped {cs:.4f}")
    # Does mixing FTT into the shipped blend help, at non-LIRF? descriptive sweep.
    print("  descriptive mix at NON-LIRF, w*FTT + (1-w)*shipped:")
    base = rm(y[NL], ship[NL])
    for w in (0.0, 0.1, 0.2, 0.3, 0.5):
        mixed = w*ftt[NL] + (1-w)*ship[NL]
        print(f"     w={w:.1f}: {rm(y[NL],mixed):7.2f}  ({rm(y[NL],mixed)-base:+.2f} vs shipped)")
    # LIRF asymmetry: in-band vs tail
    ib = y <= 7200
    for nm, msk in (("LIRF in-band", L & ib), ("LIRF tail y>7200", L & ~ib)):
        n = int(msk.sum())
        if n:
            print(f"  {nm:18s} n={n:6d}  shipped {rm(y[msk],ship[msk]):9.2f}  FTT {rm(y[msk],ftt[msk]):9.2f}  MLP {rm(y[msk],mlp[msk]):9.2f}")
    sse = lambda p, msk: float(np.sum((y[msk]-p[msk])**2))
    print(f"  LIRF tail share of LIRF SSE: shipped {sse(ship,L&~ib)/sse(ship,L):.1%}  FTT {sse(ftt,L&~ib)/sse(ftt,L):.1%}")
