#!/usr/bin/env python3
"""Addendum 4 build: the frozen configuration (all_nd, EDDF/EHAM/LEBL only) trained on every fetched
Jan + Jul 2025 day (held-out seasonal_full base), applied on top of v42 for Jan + Jul 2026.

Writes tests_adsb_full/v43_candidate.parquet (same format as submissions/fearless_barn_v42.parquet)
and build_report.json/.txt. Does NOT upload and does not touch submissions/: the upload is the
user's decision (rules point on post-pushback ADS-B), made after seeing this report.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import features
from common import dataset as D
from common import io as cio
from tests_tail_diagnosis.diagnose import CACHE

SHIP = sys.argv[1].split(",") if len(sys.argv) > 1 else ["EDDF", "EHAM", "LEBL"]
TAG = sys.argv[2] if len(sys.argv) > 2 else "v43"          # v43 = the original three airports
FEAT_ADSB = ["e_first", "e_disp", "e_move", "liftoff_dt", "stop_s", "lineup_s", "med_gs"]   # all_nd
FEAT = FEAT_ADSB + ["base", "apc", "hour"]
PARAMS = dict(objective="l2", learning_rate=0.03, n_estimators=300, num_leaves=15, min_data_in_leaf=50,
              feature_fraction=0.8, random_state=42, verbose=-1)
BOARD_V42 = 286.5256
rm = lambda r: float(np.sqrt(np.mean(np.square(r))))
lines = []
def say(s):
    print(s, flush=True); lines.append(s)


def load_estimates(prefixes):
    est, days = [], []
    for f in sorted((HERE / "points").glob("*.parquet")):
        if f.stem[:7] in prefixes:
            est.append(features.load_day(f.stem)); days.append(f.stem)
    e = pd.concat(est).sort_values("liftoff_dt")
    return e[~e.index.duplicated()], days


# ---------------- 2025 training rows (held-out base, as in the test) ----------------
X, y, meta = D.load_part("seasonal_full", "test", "with_aobt", filter_target=False)
ap = np.asarray(meta["ADEP_mvt"]).astype(str)
tree = np.load(CACHE / "tree_seasonal_full_test.npy"); mlp = np.load(CACHE / "mlp6_seasonal_full_test.npy")
ftt = np.mean([np.load(ROOT / "tests_ftt" / f"preds_seed{s}_seasonal_full_lr1e-3.npz")["pt"] for s in range(42, 51)], axis=0)
base = (tree + mlp + ftt) / 3
raw = cio.load_training(columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]).set_index("MVT_ID_mvt")
t = pd.to_datetime(raw.MVT_TIME_UTC_mvt.reindex(X.index), utc=True)
tr = pd.DataFrame({"y": np.asarray(y, float), "base": base, "ap": ap,
                   "day": np.asarray(t.dt.strftime("%Y-%m-%d")), "hour": np.asarray(t.dt.hour)}, index=X.index)
e25, days25 = load_estimates({"2025-01", "2025-07"})
tr = tr[tr.day.isin(days25) & np.isfinite(tr.y)].join(e25[FEAT_ADSB])
tr["covered"] = tr.e_first.notna() & tr.ap.isin(SHIP)
tr["apc"] = pd.Categorical(tr.ap.where(tr.ap.isin(SHIP)), categories=SHIP)
tr["mon"] = tr.day.str[5:7]
say(f"2025 training: {len(days25)} days, {len(tr)} rows, {int(tr.covered.sum())} covered at {SHIP}")

# cross-month on all fetched days: the per-corrected-row squared-error drop, for the board projection
drops = {}
for a, b in (("01", "07"), ("07", "01")):
    m = lgb.LGBMRegressor(**PARAMS).fit(tr[(tr.mon == a) & tr.covered][FEAT], (tr.y - tr.base)[(tr.mon == a) & tr.covered])
    te = tr[(tr.mon == b)]; c = te.covered.to_numpy()
    p = te.base.to_numpy().copy(); p[c] += m.predict(te.loc[c, FEAT])
    se0, se1 = (te.y - te.base) ** 2, (te.y - p) ** 2
    drops[f"{a}->{b}"] = {"pooled": (rm(te.y - te.base), rm(te.y - p)), "se_drop_per_corrected_row": float((se0 - se1)[c].mean()),
                          "n_corrected": int(c.sum())}
    say(f"cross-month {a}->{b} (all days): pooled {drops[f'{a}->{b}']['pooled'][0]:.2f} -> {drops[f'{a}->{b}']['pooled'][1]:.2f}, "
        f"SE drop per corrected row {drops[f'{a}->{b}']['se_drop_per_corrected_row']:.0f} s^2 on {c.sum()} rows")

# final corrector: every covered 2025 row at the shipped airports
final = lgb.LGBMRegressor(**PARAMS).fit(tr[tr.covered][FEAT], (tr.y - tr.base)[tr.covered])
MODEL_DIR = HERE / ("model" if TAG == "v43" else f"model_{TAG}"); MODEL_DIR.mkdir(exist_ok=True)
final.booster_.save_model(str(MODEL_DIR / "adsb_corrector.txt"))
(MODEL_DIR / "config.json").write_text(json.dumps({"airports": SHIP, "features": FEAT, "adsb_features": FEAT_ADSB,
    "params": PARAMS, "train_days": days25, "takeoff_tol_s": features.TAKEOFF_TOL, "disp_m": features.DISP_M,
    "base_file": "submissions/fearless_barn_v42.parquet"}, indent=2))
cc = tr[tr.covered]
say(f"bias check: mean(y - base) on covered 2025 rows at {SHIP} = {float((cc.y - cc.base).mean()):+.1f} s; "
    f"by month {({k: round(float(v), 1) for k, v in (cc.y - cc.base).groupby(cc.mon).mean().items()})}")

# ---------------- 2026: apply on top of v42 ----------------
v42 = pd.read_parquet(ROOT / "submissions" / "fearless_barn_v42.parquet")
rk = cio.load_ranking(columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]).drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
e26, days26 = load_estimates({"2026-01", "2026-07"})
missing = [d.strftime("%Y-%m-%d") for m_ in ("2026-01", "2026-07") for d in pd.date_range(m_ + "-01", periods=31)
           if d.strftime("%Y-%m-%d") not in days26]
say(f"2026: {len(days26)} of 62 days fetched; missing {missing}")
te = pd.DataFrame({"base": v42.TAXITIME_SEC_mvt.astype(float).to_numpy()}, index=v42.MVT_ID_mvt.astype("int64"))
rk.index = rk.index.astype("int64")
te["ap"] = rk.ADEP_mvt.reindex(te.index).astype(str).to_numpy()
tt = pd.to_datetime(rk.MVT_TIME_UTC_mvt.reindex(te.index), utc=True)
te["hour"] = tt.dt.hour.to_numpy()
e26.index = e26.index.astype("int64")
te = te.join(e26[FEAT_ADSB])
te["covered"] = te.e_first.notna() & te.ap.isin(SHIP)
te["apc"] = pd.Categorical(te.ap.where(te.ap.isin(SHIP)), categories=SHIP)
c = te.covered.to_numpy()
corr = np.zeros(len(te)); corr[c] = final.predict(te.loc[c, FEAT])
new = np.clip(np.rint(te.base.to_numpy() + corr), 0, 86400).astype("int32")
out = v42.copy(); out["TAXITIME_SEC_mvt"] = new
assert len(out) == 344841 and out.MVT_ID_mvt.equals(v42.MVT_ID_mvt)
out.to_parquet(HERE / f"{TAG}_candidate.parquet", index=False)

cov = te.groupby("ap").covered.mean().round(3).to_dict()
say(f"2026 corrected rows: {int(c.sum())} of {len(te)} ({c.mean():.1%}); by airport {({k: v for k, v in cov.items() if k in SHIP})}")
say(f"correction: mean {corr[c].mean():+.1f} s, mean |.| {np.abs(corr[c]).mean():.1f} s, 1-99 % [{np.percentile(corr[c], 1):.0f}, {np.percentile(corr[c], 99):.0f}]")
say(f"rows changed vs v42: {(out.TAXITIME_SEC_mvt != v42.TAXITIME_SEC_mvt).sum()}")

# board projection: naive = local SE drop per corrected row x corrected share; expectation = a third to a half of it
drop = np.mean([d["se_drop_per_corrected_row"] for d in drops.values()])
naive = float(np.sqrt(max(BOARD_V42 ** 2 - drop * c.sum() / len(te), 0))) - BOARD_V42
say(f"PROJECTION: naive {naive:+.2f} s vs v42 ({BOARD_V42}); written expectation {naive/3:+.2f} to {naive/2:+.2f} s "
    f"(weaker local base, 2026 coverage mix; GREKI: new inputs 'came in well under the estimate'). "
    f"Board result above v42 -> re-upload v42 last as a byte copy.")
(HERE / f"build_report{'' if TAG == 'v43' else '_' + TAG}.json").write_text(json.dumps({"days25": days25, "days26": days26, "missing26": missing, "cross_month": drops,
    "n_corrected_2026": int(c.sum()), "coverage_2026": cov, "naive_projection": naive,
    "expectation": [naive / 3, naive / 2]}, indent=2, default=float))
(HERE / f"build_report{'' if TAG == 'v43' else '_' + TAG}.txt").write_text("\n".join(lines) + "\n")
