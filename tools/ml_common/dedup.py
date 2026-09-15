"""Duplicate / near-duplicate audit primitives (a PARTIAL equivalence-class recipe).

Encodes MISTAKES.md entries #7, #28, #30, and implements steps 1, 4-part, 6 and 7 of
methodology/common/splits-and-grouping.md -> "The equivalence-class recipe":

    step 1  exact-duplicate grouping by content hash          -> md5_groups
    step 4  all-pairs candidate scan over a fingerprint       -> hash_pairs_below
    step 6  calibrate the instrument on a POSITIVE control    -> reencode_positive_control
    step 7  union-find the exact + near edges into classes    -> connected_components

The fingerprint itself (which hash, which confirmation statistic, which threshold) is
deliberately NOT fixed here: MISTAKES.md #28 is the case where the documented default
8x8 dHash plus ``mean|diff| <= 2/255`` had 11.5 percent recall at JPEG quality 90 and
0 percent at quality 60 on 28x28 rock textures, and where an UNRELATED flat pair scored
more similar (2.392) than a genuine re-encoded duplicate (2.213), so no threshold on
that statistic could separate them. The instrument was replaced, not retuned. Pick the
fingerprint for the corpus, then prove it works with the positive control below.

The rule that makes all of this honest: A NULL RESULT FROM AN UNCALIBRATED INSTRUMENT
IS NOT A FINDING. "0 near-duplicates beyond the exact-MD5 pairs" is only meaningful once
the instrument has been shown to recover planted duplicates it should recover.
"""
from __future__ import annotations

import collections
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "md5_file",
    "md5_groups",
    "hash_pairs_below",
    "connected_components",
    "sign_hash_from_images",
    "reencode_positive_control",
    "assert_positive_control_ran",
]

_CHUNK_BYTES = 1024 * 1024


def md5_file(path: Any) -> str:
    """Streaming MD5 of a file's bytes (constant memory, safe on multi-GB images/videos)."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(_CHUNK_BYTES)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def md5_groups(paths: Iterable[Any]) -> Dict[str, List[int]]:
    """Step 1: group item indices by exact content hash.

    Exact matching is necessary and NOT sufficient. MISTAKES.md #7 (subfolder79,
    CORD-v2) had 21 equivalence classes of near-duplicate images split across train and
    test that MD5 missed entirely, because the duplicates were RE-ENCODED -- same visual
    content, different file bytes. Always follow this with a near-duplicate pass whose
    recall you have measured (:func:`reencode_positive_control`).

    Returns ``{md5_hex: [index, ...]}``; groups of size 1 are included so the caller can
    report how many items were checked, not only how many collided.
    """
    out: Dict[str, List[int]] = collections.defaultdict(list)
    for i, p in enumerate(paths):
        out[md5_file(p)].append(i)
    return dict(out)


def sign_hash_from_images(arr: np.ndarray) -> np.ndarray:
    """Full-resolution horizontal-gradient difference hash as a +-1 float matrix.

    ``arr`` is ``(N, H, W)`` or ``(N, H, W, C)`` of any numeric dtype. Returns
    ``(N, H*(W-1))`` of +1/-1, so an all-pairs Hamming scan is one BLAS matmul:
    ``fraction_differing = (n_bits - X @ X.T) / (2 * n_bits)``.

    Full resolution rather than the conventional 8x8 dHash because a resize to 9x8
    throws away most of the linear resolution of a small image; on subfolder88's 28x28
    corpus that produced BOTH false positives (133 of 140 candidate pairs at Hamming<=3
    were genuinely different images sharing a coarse gradient pattern) and false
    negatives (the re-encode control recovered only 57-79 percent of known duplicates).
    On larger images a downscale is a reasonable trade -- but measure it, do not assume
    the technique transfers (MISTAKES.md #28).
    """
    a = np.asarray(arr)
    if a.ndim == 4:
        g = a.astype(np.float32).mean(axis=3)
    elif a.ndim == 3:
        g = a.astype(np.float32)
    else:
        raise ValueError("expected (N,H,W) or (N,H,W,C), got shape {}".format(a.shape))
    if g.shape[2] < 2:
        raise ValueError("images must be at least 2 pixels wide to form a gradient hash")
    bits = g[:, :, 1:] > g[:, :, :-1]
    return np.where(bits.reshape(len(g), -1), 1.0, -1.0).astype(np.float32)


def hash_pairs_below(
    H: np.ndarray,
    threshold: float,
    *,
    chunk: int = 2048,
) -> Tuple[List[Tuple[int, int, float]], float, Optional[Tuple[int, int]]]:
    """EXACT chunked all-pairs scan; every pair emitted ONCE, always with ``i < j``.

    Encodes MISTAKES.md #30 (subfolder88). The original scan masked the self-diagonal
    and every column before the current chunk, but NOT the within-chunk lower triangle,
    so every pair whose two members landed in the same chunk was emitted twice. The
    audit reported 224 candidate pairs where 167 existed, and both the "candidates
    beyond exact-MD5" count and the precautionary-grouping count derived from it were
    inflated. Nothing announced the error: the counts were plausible, and the duplicate
    pairs were visible only because one printed listing showed the same pair twice with
    the file names transposed. The sibling ``exact_max_correlation`` scan had the
    identical structure and was fixed in the same pass -- check siblings, not only the
    place the bug was caught.

    ``H`` is a ``(N, B)`` +-1 matrix (see :func:`sign_hash_from_images`). Returns
    ``(pairs, global_min_fraction, argmin_pair)``; the global minimum is what makes a
    "zero found" statement exact rather than a sampling artifact.

    The paired test asserts that 3 identical images yield exactly ``C(3,2) = 3`` pairs.
    """
    Hf = np.asarray(H, dtype=np.float32)
    if Hf.ndim != 2:
        raise ValueError("H must be 2-D (N, B)")
    n, nb = Hf.shape
    found: List[Tuple[int, int, float]] = []
    gmin, gpair = 1.0, None
    for s in range(0, n, int(chunk)):
        e = min(s + int(chunk), n)
        frac = (nb - Hf[s:e] @ Hf.T) / (2.0 * nb)
        rows = np.arange(e - s)
        frac[rows, s + rows] = 1.0          # self-pairs
        frac[:, :s] = 1.0                   # columns already covered by earlier chunks
        # within-chunk LOWER TRIANGLE -- without this every same-chunk pair is emitted
        # twice, as (i,j) and (j,i). This is MISTAKES.md #30.
        ii, jj = np.tril_indices(e - s, k=0)
        frac[ii, s + jj] = 1.0
        if frac.size:
            m = float(frac.min())
            if m < gmin:
                a, b = np.unravel_index(int(frac.argmin()), frac.shape)
                gmin, gpair = m, (s + int(a), int(b))
        for a, b in np.argwhere(frac < float(threshold)):
            i, j = s + int(a), int(b)
            assert i < j, "pair emitted with i >= j -- the triangle mask is wrong"
            found.append((i, j, round(float(frac[a, b]), 6)))
    return found, gmin, gpair


def connected_components(n: int, edges: Iterable[Sequence[int]]) -> List[List[int]]:
    """Step 7: union-find the exact + near-duplicate edges into equivalence classes.

    Duplicates are TRANSITIVE by intent -- if a~b and b~c then a, b and c must all land
    on the same side of the split, even when a and c were never directly matched. Split
    whole classes, never individual items. Returns every component including singletons,
    each sorted, ordered by first member.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for edge in edges:
        i, j = int(edge[0]), int(edge[1])
        if not (0 <= i < n and 0 <= j < n):
            raise IndexError("edge ({}, {}) out of range for n={}".format(i, j, n))
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a

    comp: Dict[int, List[int]] = collections.defaultdict(list)
    for i in range(n):
        comp[find(i)].append(i)
    return [sorted(v) for v in sorted(comp.values(), key=lambda v: min(v))]


def reencode_positive_control(
    images: np.ndarray,
    *,
    fingerprint=sign_hash_from_images,
    qualities: Sequence[int] = (95, 90, 80, 70, 60),
    n: int = 400,
    seed: int = 0,
) -> Dict[str, Any]:
    """POSITIVE CONTROL for a near-duplicate instrument. Run it BEFORE trusting a null.

    Encodes MISTAKES.md #28 (subfolder88). A "0 near-duplicates beyond the 7 exact-MD5
    pairs -- a genuine negative result" verdict was written into the session before any
    positive control had been run. The control then measured the instrument's recall at
    11.5 percent at JPEG quality 90 and 0 percent at quality 60: it would have missed
    essentially every re-encoded duplicate. The "clean" verdict was an artifact of a
    blind instrument.

    This plants the transform the corpus could plausibly hide -- a JPEG re-encode -- and
    reports the fingerprint distance distribution per quality. Read
    ``true_positive_ceiling_p95`` as the smallest threshold that still recovers 95
    percent of planted duplicates; if that ceiling sits above the distances of genuinely
    unrelated pairs, the instrument cannot separate them and must be REPLACED, not
    retuned. When a confirmation statistic disagrees with a real example, swap the
    instrument.

    Requires Pillow. Raises a clear error if it is missing rather than silently skipping
    -- an audit whose positive control did not run has no verdict to report.
    """
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PIL (Pillow) is required for the re-encode positive control. Install it "
            "(python3 -m pip install Pillow) -- do NOT skip the control: a null "
            "near-duplicate result from an uncalibrated instrument is not a finding "
            "(MISTAKES.md #28)."
        ) from exc

    import io

    arr = np.asarray(images)
    if arr.ndim not in (3, 4):
        raise ValueError("images must be (N,H,W) or (N,H,W,C)")
    rng = np.random.default_rng(int(seed))
    pick = rng.permutation(len(arr))[: min(int(n), len(arr))]

    out: Dict[str, Any] = {
        "control": "reencode_positive_control",
        "n_probed": int(len(pick)),
        "instrument": getattr(fingerprint, "__name__", str(fingerprint)),
        "by_quality": {},
    }
    for q in qualities:
        fracs: List[float] = []
        for i in pick:
            src = arr[int(i)]
            img = Image.fromarray(np.asarray(src, dtype=np.uint8))
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=int(q))
            back = Image.open(io.BytesIO(buf.getvalue()))
            back = back.convert(img.mode)
            re_arr = np.asarray(back)
            pair = np.stack([np.asarray(src, dtype=np.uint8), re_arr])
            Hp = fingerprint(pair)
            nb = Hp.shape[1]
            fracs.append(float((nb - float(Hp[0] @ Hp[1])) / (2.0 * nb)))
        a = np.asarray(fracs, dtype=np.float64)
        out["by_quality"][str(int(q))] = {
            "distance_median": float(np.median(a)),
            "distance_p95": float(np.percentile(a, 95)),
            "distance_max": float(a.max()),
        }
    out["true_positive_ceiling_p95"] = max(
        v["distance_p95"] for v in out["by_quality"].values()
    )
    out["note"] = (
        "the instrument's RECALL on a planted re-encode is measured BEFORE any "
        "zero-found result is believed; if this ceiling overlaps the distances of "
        "unrelated pairs, replace the instrument rather than moving its threshold "
        "(MISTAKES.md #28)"
    )
    return out


def assert_positive_control_ran(report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Gate: a near-duplicate audit may not report a null result without its control.

    Call this in the split/audit stage's quality gate, next to the "0 near-duplicates
    found" line it protects.
    """
    if not report or "true_positive_ceiling_p95" not in report:
        raise AssertionError(
            "near-duplicate audit has no positive-control report. A null result from an "
            "instrument that was never shown to recover a planted duplicate is not a "
            "finding (MISTAKES.md #28) -- run reencode_positive_control first."
        )
    return {
        "gate": "neardup_positive_control_ran",
        "n_probed": report.get("n_probed"),
        "true_positive_ceiling_p95": report["true_positive_ceiling_p95"],
    }
