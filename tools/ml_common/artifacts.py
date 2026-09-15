"""Run-keyed artifact paths and an overwrite guard.

Encodes MISTAKES.md entries #35, #38, #50, #67, #69, and the #38 recurrence audit.

One rule, two halves:

  * A path that identifies a run must contain EVERY dimension that varies between runs
    -- arm, protocol, seed, resolution, checkpoint. #35: subfolder88 cached embeddings
    as ``{arm_id}_test.npz``. A reduced 5-arm seed-1 sweep left 3 stale seed-0 ``.npz``
    files on disk; Stage 4 globbed the directory and scored them against seed 1's
    labels. All three came back IDENTICAL to seed 0 to four decimal places -- the split
    JSON is written class-by-class, so the label SEQUENCE is byte-identical between
    seeds, and the fabricated rows were indistinguishable from real ones in the very
    artifact used to claim "the ranking survives a repeat-seed sweep". #38 is the same
    defect in a sibling path the #35 fix never touched (per-arm checkpoint dirs), which
    silently overwrote the seed-0 convergence evidence cited in two retrospectives.
  * Writing a result artifact over an existing one requires an explicit ``force``.
    #50: a hardcoded output path made a seed-1 verification run overwrite the
    registered seed-0 result -- a verification step destroying the result it verified.
    #67: a repeat-seed sweep with no ``out_dir`` began overwriting the registered
    checkpoint from its first epoch. #69: the sweep passed ``--seed``/``--name`` to a
    script with no argparse at all, so the flags went into ``sys.argv`` unread, the
    "seed 1" run trained seed 42 again, and it clobbered the registered run.

Before any sweep, run ``python <script> --help`` and confirm the seed / output-dir
flags are actually PARSED (#69). An unrecognised flag that argparse rejects with exit 2
is safer than one silently dropped.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

__all__ = ["run_key", "run_dir", "guard_overwrite", "write_json_guarded"]

_UNSAFE = re.compile(r"[^A-Za-z0-9._+-]+")


def _slug(value: Any) -> str:
    s = _UNSAFE.sub("_", str(value)).strip("_")
    if not s:
        raise ValueError("path component {!r} slugified to nothing".format(value))
    return s


def run_key(
    arm: str,
    protocol: Optional[str] = None,
    seed: Optional[int] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    """Build the canonical ``arm__protocol__seedN[__k=v...]`` run key.

    Every dimension that changes the CONTENTS belongs in the key (MISTAKES.md #35).
    Windows-safe: no colons, no spaces -- ``protocol="open:set"`` becomes ``open_set``.
    """
    parts = [_slug(arm)]
    if protocol is not None:
        parts.append(_slug(protocol))
    if seed is not None:
        parts.append("seed{}".format(int(seed)))
    for k in sorted(dict(extra or {})):
        parts.append("{}={}".format(_slug(k), _slug(dict(extra)[k])))
    return "__".join(parts)


def run_dir(
    base: os.PathLike | str,
    arm: str,
    protocol: Optional[str] = None,
    seed: Optional[int] = None,
    extra: Optional[Mapping[str, Any]] = None,
    *,
    create: bool = True,
) -> Path:
    """``base / arm__protocol__seedN`` -- the only shape a per-run artifact dir should take.

    Do NOT write to ``base / arm``. That is the exact path shape behind MISTAKES.md #35
    and #38: a later seed silently overwrites an earlier one's checkpoints and history,
    and the loss is invisible because nothing errors. When a "reproducibility" artifact
    reproduces a previous run's number to full precision, treat that as a SUSPICION
    rather than a reassurance and check the two runs actually consumed different data.
    """
    path = Path(base) / run_key(arm, protocol, seed, extra)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def guard_overwrite(path: os.PathLike | str, force: bool = False) -> Path:
    """Refuse to overwrite an existing result artifact unless ``force=True``.

    Encodes MISTAKES.md #50 / #67 / #69. A silent overwrite produces no error, which is
    precisely why it destroyed a registered checkpoint three separate times in this
    workspace. Call this immediately before writing any result file, and give each
    checkpoint / seed / configuration its OWN output path (see :func:`run_dir`) rather
    than reaching for ``force``.

    ``force=True`` is for a deliberate, stated re-run of the same configuration -- never
    as the way past this guard during a sweep.
    """
    p = Path(path)
    if p.exists() and not force:
        raise FileExistsError(
            "refusing to overwrite existing artifact {}. Give this run its own path "
            "(artifacts.run_dir keys on arm/protocol/seed) or pass force=True "
            "deliberately. MISTAKES.md #50/#67/#69: a verification run overwriting the "
            "result it was verifying leaves no error behind to catch it by.".format(p)
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_json_guarded(
    path: os.PathLike | str,
    payload: Any,
    *,
    force: bool = False,
    indent: int = 2,
) -> Path:
    """``json.dump`` behind :func:`guard_overwrite`, always UTF-8 with a trailing newline.

    Content written to a FILE may use Unicode freely; only ``print()`` must stay ASCII
    (MISTAKES.md #57 -- a cp1252 console crashed on an arrow in a final print statement
    after the JSON had already been written correctly). Keep committed artifacts free of
    absolute machine paths (#12): store paths relative to the solution root.
    """
    p = guard_overwrite(path, force=force)
    p.write_text(json.dumps(payload, indent=indent) + "\n", encoding="utf-8")
    return p
