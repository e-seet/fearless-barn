#!/usr/bin/env python3
"""Carrier-echo hedge H1 vs the shipped LIRF recipe on held-out months (PREREGISTERED.md)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT))
from common import io as cio
from common import dataset as D
from tests_tail_diagnosis.diagnose import CACHE
K, TW = 10.0, 0.459
rm = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

r = cio.load_training()
d = r[r.PHASE_mvt.eq("DEP")].copy()
t = lambda c: pd.to_datetime(d[c], utc=True, errors="coerce")
d["y"] = pd.to_numeric(d.TAXITIME_SEC_mvt, errors="coerce")
d["ds"] = (t("MVT_TIME_UTC_mvt") - t("SCHED_TIME_UTC_mvt")).dt.total_seconds()
d["echo"] = (t("BLOCK_TIME_UTC_mvt") - t("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 60
d["carrier"] = d.FLIGHT_mvt.astype(str).str.extract(r"^([A-Z]{2,3}|[A-Z]\d|\d[A-Z])", expand=False).fillna("?")
d["mon"] = d.month.astype(str)
L = d[d.ADEP_mvt.eq("LIRF") & d.FLIGHT_ID_mvt.isna() & np.isfinite(d.y) & (d.y > 0)].set_index("MVT_ID_mvt")


def h1_params(exclude_months):
    tr = L[~L.mon.isin(exclude_months)]
    base = tr.echo.mean()
    st = tr.groupby("carrier").echo.agg(["sum", "size"])
    rate = (st["sum"] + K * base) / (st["size"] + K)
    b = float(tr.y[~tr.echo].median())
    return rate, base, b


def surface(name, ids, y, ap, mlp, tab, months):
    base = mlp.copy()
    Lm = ap == "LIRF"
    base[Lm] = (1 - TW) * mlp[Lm] + TW * tab[Lm]      # shipped LIRF recipe; other rows untouched below
    new = base.copy()
    idx = pd.Index(ids)
    mask = idx.isin(L.index) & Lm
    sub = L.reindex(idx[mask])
    # H1 with p, b learned only from months outside this surface's test months
    rate, pbase, b = h1_params(months)
    p = sub.carrier.map(rate).fillna(pbase).to_numpy()
    ds = np.clip(sub.ds.to_numpy(), 0, 86400)
    h1 = p * ds + (1 - p) * b
    new[mask] = h1
    ok = np.isfinite(y)
    res = {"n_rows": int(ok.sum()), "n_lirf_norec": int(mask.sum()),
           "pooled": (rm(y[ok], base[ok]), rm(y[ok], new[ok])),
           "lirf_norec": (rm(y[mask], base[mask]), rm(y[mask], new[mask]))}
    h2 = base.copy(); h2[mask] = 0.5 * (h1 + base[mask])
    res["pooled_H2"] = rm(y[ok], h2[ok])
    # single-row robustness: drop the row with the largest SE improvement
    gain = (y[mask] - base[mask]) ** 2 - (y[mask] - h1) ** 2
    j = np.argmax(gain)
    keep = np.ones(ok.sum(), bool)
    pos = np.flatnonzero(mask)[j]
    keep_all = ok.copy(); keep_all[pos] = False
    res["pooled_drop_best_row"] = (rm(y[keep_all], base[keep_all]), rm(y[keep_all], new[keep_all]))
    res["per_month"] = {}
    for mo in sorted(set(pd.Series(sub.mon).dropna())):
        mm = (sub.mon == mo).to_numpy()
        yy = y[mask][mm]
        res["per_month"][mo] = (rm(yy, base[mask][mm]), rm(yy, h1[mm]), int(mm.sum()))
    print(f"\n== {name} ({'/'.join(months)}) : {res['n_lirf_norec']} LIRF no-record rows of {res['n_rows']} ==")
    print(f"  pooled all rows : model {res['pooled'][0]:.3f} -> H1 {res['pooled'][1]:.3f} ({res['pooled'][1]-res['pooled'][0]:+.3f}) | H2 {res['pooled_H2']:.3f}")
    print(f"  LIRF no-record  : model {res['lirf_norec'][0]:.1f} -> H1 {res['lirf_norec'][1]:.1f}")
    print(f"  pooled, best row removed: {res['pooled_drop_best_row'][0]:.3f} -> {res['pooled_drop_best_row'][1]:.3f} "
          f"({res['pooled_drop_best_row'][1]-res['pooled_drop_best_row'][0]:+.3f})")
    for mo, (a, b_, n) in res["per_month"].items():
        print(f"    {mo}: LIRF no-record model {a:.1f} -> H1 {b_:.1f} (n={n})")
    return res


out = {}
# seasonal_full test, Jan + Jul
X, y, meta = D.load_part("seasonal_full", "test", "with_aobt", filter_target=False)
tab = np.mean([np.load(ROOT / "tests_tabnet" / f"preds_seed{s}_sf_lr2e-3p.npz")["pt"] for s in (42, 43, 44)], axis=0)
out["janjul"] = surface("seasonal_full test", X.index.to_numpy(), np.asarray(y, float), meta["ADEP_mvt"].astype(str).to_numpy(),
                        np.load(CACHE / "mlp6_seasonal_full_test.npy"), tab, ["2025-01", "2025-07"])
# forward val + test, Sep-Dec
for part, key, mon in (("val", "pv", ["2025-09", "2025-10"]), ("test", "pt", ["2025-11", "2025-12"])):
    X, y, meta = D.load_part("forward", part, "with_aobt", filter_target=False)
    tab = np.mean([np.load(ROOT / "tests_tabnet" / f"preds_seed{s}_lr2e-3p.npz")[key] for s in (42, 43, 44)], axis=0)
    out[f"forward_{part}"] = surface(f"forward {part}", X.index.to_numpy(), np.asarray(y, float), meta["ADEP_mvt"].astype(str).to_numpy(),
                                     np.load(CACHE / f"mlp12_forward_{part}.npy"), tab, mon)
# forward_inner val, Jul-Aug
X, y, meta = D.load_part("forward_inner", "val", "with_aobt", filter_target=False)
tab = np.mean([np.load(ROOT / "tests_tabnet" / "build" / f"inner_seed{s}.npz")["pv"] for s in (42, 43, 44)], axis=0)
out["julaug"] = surface("forward_inner val", X.index.to_numpy(), np.asarray(y, float), meta["ADEP_mvt"].astype(str).to_numpy(),
                        np.load(CACHE / "mlp18_forward_inner_val.npy"), tab, ["2025-07", "2025-08"])

J = out["janjul"]
c1 = J["pooled"][1] < J["pooled"][0] and all(v[1] < v[0] for v in J["per_month"].values())
c2 = all(out[k]["pooled"][1] < out[k]["pooled"][0] for k in ("forward_val", "forward_test", "julaug"))
c3 = J["pooled_drop_best_row"][1] < J["pooled_drop_best_row"][0]
out["verdict"] = {"janjul_pooled_and_each_month": c1, "corroborating_same_sign": c2, "not_one_row": c3, "PASS": bool(c1 and c2 and c3)}
print(f"\nPRE-REGISTERED: Jan+Jul pooled + each month {c1} | Sep-Dec & Jul-Aug same sign {c2} | not one row {c3} => {'PASS' if (c1 and c2 and c3) else 'FAIL'}")
(HERE / "evaluate.json").write_text(json.dumps(out, indent=2, default=float))
