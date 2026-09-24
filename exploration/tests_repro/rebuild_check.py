#!/usr/bin/env python3
"""Can the best submissions be rebuilt, and from exactly which files?

1. Rebuilds v29, v30 and v31 through submit.py's own code path (provenance
   gate, predict_ranking, np.rint to int32 in template order) and compares
   each row against the uploaded parquet.
2. Records EVERY file opened during those rebuilds (sys.addaudithook "open"),
   so the dependency list is measured, not guessed. Each file is classified
   as git-tracked or not, with size and sha256. The untracked ones are what a
   backup must hold.

Nothing is written to submissions/. Output: rebuild_check.txt,
dependency_manifest.csv.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

OPENED: set[str] = set()


def _hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        mode = args[1] if len(args) > 1 else "r"
        if mode is None or "w" not in str(mode):
            try:
                OPENED.add(os.path.abspath(os.fsdecode(args[0])))
            except Exception:
                pass


sys.addaudithook(_hook)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import config  # noqa: E402
import submit  # noqa: E402

TARGETS = [  # version, arm, tag, expected
    (29, "tabnet_lirf_override", "lirf_w0.55", "LIRF rows within 1 s (upload truncated, submit.py rounds)"),
    (30, "ftt_multimodel", "ftt3_mlp18", "byte-identical"),
    (31, "tabnet_lirf_override", "lirf_w0.55_ftt3", "byte-identical"),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    tmpl = pd.read_parquet(config.SUBMITTING_PARQUET)
    lines = []
    for v, arm, tag, expected in TARGETS:
        prov = submit.provenance_chain(arm, "full", "with_aobt", 42, "keep_all", tag=tag)
        submit.assert_feature_agreement(prov)
        pred, meta = submit.predict_ranking(arm, "full", "with_aobt", 42, "keep_all", tag=tag)
        reb = pd.DataFrame({config.ID: tmpl[config.ID].to_numpy(),
                            config.TARGET: np.rint(pred).astype("int32")})
        up = pd.read_parquet(ROOT / "submissions" / f"fearless_barn_v{v}.parquet")
        mm = reb.merge(up, on=config.ID, suffixes=("_reb", "_up"))
        d = (mm[f"{config.TARGET}_reb"] - mm[f"{config.TARGET}_up"]).to_numpy()
        ap = meta["ADEP_mvt"].astype(str).reindex(mm[config.ID].to_numpy()).to_numpy() \
            if config.ID not in meta.columns else None
        ident = int((d == 0).sum())
        line = (f"v{v} ({arm}, tag {tag}): {len(mm)}/{len(up)} rows joined | identical {ident} | "
                f"max |diff| {int(np.abs(d).max())} | rows differing {int((d != 0).sum())} | expected: {expected}")
        if ap is not None and (d != 0).any():
            line += f" | differing rows all LIRF: {bool((ap[d != 0] == 'LIRF').all())}"
        print(line, flush=True)
        lines.append(line)

    tracked = set(subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True,
                                 encoding="utf-8").stdout.splitlines())
    rows = []
    for p in sorted(OPENED):
        P = Path(p)
        if not P.is_file():
            continue
        try:
            rel = P.resolve().relative_to(REPO.resolve()).as_posix()
        except ValueError:
            continue  # site-packages etc.
        if rel.endswith((".py", ".pyc")) or "__pycache__" in rel:
            continue
        rows.append({"path": rel, "git_tracked": rel in tracked, "bytes": P.stat().st_size,
                     "sha256": sha256(P)})
    with open(HERE / "dependency_manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "git_tracked", "bytes", "sha256"])
        w.writeheader()
        w.writerows(rows)
    un = [r for r in rows if not r["git_tracked"]]
    summ = (f"\ndependencies opened: {len(rows)} files | git-tracked {len(rows) - len(un)} | "
            f"NOT tracked {len(un)} ({sum(r['bytes'] for r in un) / 1e9:.2f} GB)")
    print(summ)
    by_dir: dict[str, list] = {}
    for r in un:
        top = "/".join(r["path"].split("/")[:3])
        by_dir.setdefault(top, []).append(r["bytes"])
    for k, b in sorted(by_dir.items(), key=lambda x: -sum(x[1])):
        print(f"  {k:60s} {len(b):4d} files {sum(b) / 1e6:9.1f} MB")
    (HERE / "rebuild_check.txt").write_text("\n".join(lines) + summ + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
