#!/usr/bin/env python3
"""Harvest every run's eval JSON into one flat table -> results.csv.

This is the generator behind `result.md`. `result.md` is the *reading* of the
numbers (ownership, overlaps, what is stale); `results.csv` is the raw pivot
table. Re-run this whenever a new arm/seed/tag finishes:

    python3 harvest_results.py

Columns
-------
owner              claude | antigravity   (assigned by folder-name suffix; see result.md)
arm, protocol, seed, tag
val_rmse, val_inband, test_rmse, test_inband   (pooled RMSE + tail-excluded companion)
test_trivial_mean                              predict-the-training-mean floor on test
selection_surface                              where free params were fitted, if the run records it
val_fitted                                     True if val labels were used to fit params (in-sample val)
status                                         has_results | val_only | built_not_run
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "artifacts" / "runs"
OUT = HERE / "results.csv"

# Ownership is by folder-name convention: the `_antigravity` suffix marks an arm
# built by the Antigravity agent, everything else by Claude. README section 4
# corroborates (it labels the same arms "third-party"). Fix it here if wrong and
# every row follows.
ANTIGRAVITY_SUFFIX = "_antigravity"

RUN_RE = re.compile(r"^(?P<arm>.+?)__(?P<proto>.+?)__seed(?P<seed>\d+)(?:__tag=(?P<tag>.+))?$")


def owner_of(arm: str) -> str:
    return "antigravity" if arm.endswith(ANTIGRAVITY_SUFFIX) else "claude"


def load_eval(run_dir: Path, part: str) -> dict | None:
    f = run_dir / f"eval_{part}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def selection_bits(run_dir: Path) -> tuple[str, str]:
    """(selection_surface, val_fitted) from model.json, best-effort across arms."""
    mj = run_dir / "model.json"
    if not mj.exists():
        return "", ""
    try:
        j = json.loads(mj.read_text(encoding="utf-8"))
    except Exception:
        return "", ""
    surface = j.get("weight_split") or j.get("selection_set") or j.get("selection_surface") or ""
    vf = j.get("weights_are_val_fitted")
    if vf is None and surface:
        vf = surface in ("val", "validation")
    return str(surface), ("" if vf is None else str(bool(vf)))


def round2(x):
    return round(x, 2) if isinstance(x, (int, float)) else ""


def main() -> int:
    rows = []
    known_arms = set()
    for d in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        m = RUN_RE.match(d.name)
        if not m:
            print("skip (unparsed):", d.name)
            continue
        arm = m["arm"]
        known_arms.add(arm)
        ev = load_eval(d, "val")
        et = load_eval(d, "test")
        surface, val_fitted = selection_bits(d)
        if ev is None and et is None:
            # `full` split has no val/test part by design -- it is a refit for the
            # submission, not a scorable run. Don't flag it as unrun.
            status = "refit_only" if m["proto"].startswith("full-") else "built_not_run"
        elif et is None:
            status = "val_only"
        else:
            status = "has_results"
        tb = ""
        if et:
            tb = round2(et.get("trivial_baselines", {}).get("train_mean", {}).get("rmse"))
        rows.append({
            "owner": owner_of(arm),
            "arm": arm,
            "protocol": m["proto"],
            "seed": int(m["seed"]),
            "tag": m["tag"] or "",
            "val_rmse": round2(ev.get("rmse")) if ev else "",
            "val_inband": round2(ev.get("rmse_in_band")) if ev else "",
            "test_rmse": round2(et.get("rmse")) if et else "",
            "test_inband": round2(et.get("rmse_in_band")) if et else "",
            "test_trivial_mean": tb,
            "selection_surface": surface,
            "val_fitted": val_fitted,
            "status": status,
        })

    # arms with a pipeline folder but no run dir at all -> built_not_run
    for pf in sorted(p for p in HERE.glob("pipeline_*") if p.is_dir()):
        arm = pf.name[len("pipeline_"):]
        if arm not in known_arms:
            rows.append({
                "owner": owner_of(arm), "arm": arm, "protocol": "", "seed": "",
                "tag": "", "val_rmse": "", "val_inband": "", "test_rmse": "",
                "test_inband": "", "test_trivial_mean": "", "selection_surface": "",
                "val_fitted": "", "status": "built_not_run",
            })

    cols = ["owner", "arm", "protocol", "seed", "tag", "val_rmse", "val_inband",
            "test_rmse", "test_inband", "test_trivial_mean", "selection_surface",
            "val_fitted", "status"]
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")

    # canonical protocol view, ranked on val
    canon = [r for r in rows if r["protocol"] == "forward-with_aobt-keep_all" and r["seed"] == 42]
    canon.sort(key=lambda r: r["val_rmse"] if isinstance(r["val_rmse"], float) else 9e9)
    print("\n=== forward / with_aobt / keep_all, seed 42 -- ranked on val ===")
    print(f"{'arm':<28}{'tag':<16}{'owner':<12}{'val':>9}{'test':>9}{'val_ib':>9}{'test_ib':>9}  surface")
    for r in canon:
        print(f"{r['arm']:<28}{r['tag']:<16}{r['owner']:<12}"
              f"{str(r['val_rmse']):>9}{str(r['test_rmse']):>9}"
              f"{str(r['val_inband']):>9}{str(r['test_inband']):>9}  {r['selection_surface']}")

    write_overview(rows)
    return 0


def write_overview(rows: list) -> None:
    """Join the hand-maintained technique catalog with the live canonical-protocol
    numbers -> pipelines_overview.csv. One row per pipeline folder. Attributes
    (technique, segmentation, ...) are declared once in pipeline_catalog.csv;
    numbers come from this harvest so they never go stale. Add a row to the
    catalog when a new pipeline_* folder lands."""
    cat_path = HERE / "pipeline_catalog.csv"
    if not cat_path.exists():
        print("(no pipeline_catalog.csv -- skipping pipelines_overview.csv)")
        return
    with cat_path.open(encoding="utf-8") as fh:
        catalog = list(csv.DictReader(fh))
    cat_arms = {c["pipeline"] for c in catalog}

    # per-arm rollup over the canonical protocol (untagged run = the representative)
    def canon_rows(arm):
        return [r for r in rows
                if r["arm"] == arm
                and r["protocol"] == "forward-with_aobt-keep_all"]

    def seeds_for(arm):
        s = sorted({r["seed"] for r in canon_rows(arm)
                    if r["tag"] == "" and isinstance(r["seed"], int)})
        return "/".join(str(x) for x in s) if s else "-"

    def pick(arm):
        # prefer the untagged seed-42 run; else any seed-42 run; else blank
        c = [r for r in canon_rows(arm) if r["seed"] == 42]
        for r in sorted(c, key=lambda r: (r["tag"] != "", r["tag"])):
            return r
        return None

    out = HERE / "pipelines_overview.csv"
    # review_tier / location come straight from the catalog. `location` is the
    # folder the pipeline actually lives in: arms judged over-tuned are moved to
    # overtuned/ and dropped from config.ARMS, so a reader of this table can see
    # which rows are quarantined without cross-referencing PIPELINE_REVIEW.md.
    #
    # APPENDED, not inserted: result.md documents awk one-liners against this
    # file by field POSITION ($3, $4, $8), so adding columns in the middle would
    # silently break every one of them.
    cols = ["pipeline", "owner", "model_family", "segmentation", "technique",
            "features_vs_base", "weight_fit_surface", "parents", "tests",
            "val_rmse", "test_rmse", "val_inband", "test_inband",
            "seeds_run", "status", "review_tier", "location"]
    n = 0
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for c in catalog:
            r = pick(c["pipeline"])
            any_row = next(iter(canon_rows(c["pipeline"])), None)
            status = (r or any_row or {}).get("status", "built_not_run")
            w.writerow({
                "pipeline": c["pipeline"], "owner": c["owner"],
                "review_tier": c.get("review_tier", ""),
                "location": c.get("location", "pipeline_" + c["pipeline"]),
                "model_family": c["model_family"], "segmentation": c["segmentation"],
                "technique": c["technique"], "features_vs_base": c["features_vs_base"],
                "weight_fit_surface": c["weight_fit_surface"], "parents": c["parents"],
                "tests": c["tests"],
                "val_rmse": (r or {}).get("val_rmse", ""),
                "test_rmse": (r or {}).get("test_rmse", ""),
                "val_inband": (r or {}).get("val_inband", ""),
                "test_inband": (r or {}).get("test_inband", ""),
                "seeds_run": seeds_for(c["pipeline"]),
                "status": status,
            })
            n += 1
    print(f"wrote {out} ({n} pipelines)")

    # catalog / folder drift check. overtuned/ is scanned too: a quarantined arm
    # still HAS a folder, it is just not a candidate any more, so it must not
    # show up here as catalog drift.
    folders = {p.name[len("pipeline_"):]
               for p in list(HERE.glob("pipeline_*")) + list((HERE / "overtuned").glob("pipeline_*"))
               if p.is_dir()}
    missing = folders - cat_arms
    extra = cat_arms - folders
    if missing:
        print(f"  [WARN] pipeline_* folders with no catalog row: {sorted(missing)}")
    if extra:
        print(f"  [WARN] catalog rows with no pipeline_* folder: {sorted(extra)}")


if __name__ == "__main__":
    raise SystemExit(main())
