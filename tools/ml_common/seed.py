"""Determinism primitives: process seeding, stable derived seeds, per-item RNGs.

Encodes MISTAKES.md entries #6, #10, #11, #19, #25, #51, #55, #57.

The whole point of this module is that "seeded" code which is not actually
deterministic is worse than unseeded code, because it manufactures confidence.
Three failure modes have been observed in this workspace and are all fixed here:

  * builtin ``hash()`` used to derive a reproducible seed (#6 subfolder82,
    #19 subfolder84, #25 subfolder86) -- CPython salts str/bytes/tuple hashes per
    process, so the "seeded" decision changes on every fresh interpreter;
  * a single mutable RNG stored on a Dataset and advanced inside ``__getitem__``
    (#10 subfolder65) -- every DataLoader worker forks its own copy and advances it
    in OS-scheduled order, so the augmentation stream depends on ``num_workers``;
  * a control that flips process-global cuDNN flags and never restores them
    (#51 subfolder95), so the real training run that follows it in the same
    process silently runs under a different numerical regime.

All console output in this package is plain ASCII (#57): a Unicode arrow in a
print() crashes on a Windows cp1252 console after the real work is already done.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import random
from typing import Any, Iterator, Optional

import numpy as np

__all__ = [
    "seed_everything",
    "stable_seed",
    "item_rng",
    "item_seed",
    "preserve_determinism_flags",
    "torch_available",
]


def torch_available() -> bool:
    """True if torch can be imported. Never import torch at module scope here."""
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def seed_everything(seed: int, deterministic: bool = False) -> int:
    """Seed python / numpy / torch (if installed) and return the seed for logging.

    Encodes MISTAKES.md #11 and #55: a sweep or validation script that claims to
    reproduce a registered run must replay that run's ENTIRE RNG-consuming call
    sequence, not merely call this function with the same integer. Calling
    ``overfit_one_batch`` before the real model is constructed (as the registered
    script does) but omitting it from the sweep script consumed different RNG state
    and moved accuracy by 18 points at an identical seed. If a sweep cannot replay
    the sequence exactly, say so explicitly rather than presenting its seeds as
    bit-reproductions of a registered checkpoint.

    ``deterministic=True`` additionally pins cuDNN. It is off by default because
    full determinism costs roughly 10-25 percent throughput on convolution-heavy
    training; the workspace's stronger guarantee is a repeat-seed sweep. Note that
    ``PYTHONHASHSEED`` is set here for completeness only -- the interpreter's hash
    salt is fixed BEFORE any Python code runs, so this cannot make builtin
    ``hash()`` reproducible. Use :func:`stable_seed` instead (#19, #25).
    """
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    try:
        import torch
    except Exception:
        return int(seed)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return int(seed)


def stable_seed(*parts: Any, bits: int = 32) -> int:
    """A reproducible integer seed derived from arbitrary parts, via blake2b.

    NEVER USE THE BUILTIN ``hash()`` FOR THIS. MISTAKES.md #6 / #19 / #25 are three
    separate recurrences of the same bug in this workspace:

      * #6  subfolder82 -- ``hash(cls) % 10_000`` chose per-class split offsets, so
            two fresh-process runs of the same script produced different splits.
      * #19 subfolder84 -- ``hash((split, doc_id))`` seeded a negative-pair sampler,
            so a resumed training run drew different negatives than an uninterrupted
            one at the same SEED, and a repeat-seed sweep silently varied two things.
      * #25 subfolder86 -- ``abs(hash(session_id))`` formed a group key, so carcasses
            were re-keyed between runs and group-disjointness broke with no error.

    CPython salts the hash of str / bytes / tuple per process (``PYTHONHASHSEED``),
    and the salt is fixed at interpreter start -- before ``seed_everything`` can run.
    An in-process check therefore CANNOT detect this bug; #25's regression test spawns
    a real subprocess for exactly that reason.

    Parts are stringified and joined with ``|``, so ``stable_seed("a", 1)`` and
    ``stable_seed("a|1")`` collide by construction -- pass structured parts, not
    pre-joined strings, if that matters.
    """
    if not parts:
        raise ValueError("stable_seed() needs at least one part")
    if bits < 1 or bits > 63:
        raise ValueError("bits must be in 1..63")
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).hexdigest()
    return int(digest, 16) % (2**bits)


def item_seed(seed: int, idx: int, *extra: Any) -> int:
    """Stable per-item seed from (seed, idx, *extra). See :func:`item_rng`."""
    return stable_seed("ml_common.item", seed, idx, *extra)


def item_rng(seed: int, idx: int, *extra: Any) -> np.random.Generator:
    """A FRESH RNG derived from (seed, idx) -- for use inside ``Dataset.__getitem__``.

    Encodes MISTAKES.md #10 (subfolder65). The broken pattern is::

        class D(Dataset):
            def __init__(self, seed):
                self._rng = np.random.RandomState(seed)   # WRONG
            def __getitem__(self, idx):
                jitter = self._rng.randint(...)           # mutates shared state

    With ``num_workers > 0`` each worker process inherits an identically-seeded copy
    and advances it independently in OS-scheduled order, so the jitter sequence
    depends on the worker count and does not reproduce run to run. The fix is a
    stateless RNG derived per call::

        def __getitem__(self, idx):
            rng = item_rng(self.seed, idx)
            jitter = int(rng.integers(0, 5))

    which is bit-identical under ``num_workers=0`` and ``num_workers=2``. Pass
    ``*extra`` (e.g. the epoch) when the draw is meant to change across epochs.
    """
    return np.random.default_rng(item_seed(seed, idx, *extra))


@contextlib.contextmanager
def preserve_determinism_flags() -> Iterator[Optional[dict]]:
    """Save and restore torch's process-global cuDNN determinism flags.

    Encodes MISTAKES.md #51 (subfolder95). A negative control that sets
    ``torch.backends.cudnn.deterministic = True`` / ``benchmark = False`` for its own
    reproducibility (#49) leaks those settings into whatever runs next in the same
    process. ``run_all.py`` calls the control immediately before real training, so the
    orchestrator's training run would have executed under a different cuDNN regime
    than the standalone invocation that produced the registered checkpoint -- a
    discrepancy nothing errors on and cuDNN does not announce.

    No-op (yielding None) when torch is not installed.
    """
    try:
        import torch
    except Exception:
        yield None
        return
    prev = {
        "deterministic": torch.backends.cudnn.deterministic,
        "benchmark": torch.backends.cudnn.benchmark,
    }
    try:
        yield dict(prev)
    finally:
        torch.backends.cudnn.deterministic = prev["deterministic"]
        torch.backends.cudnn.benchmark = prev["benchmark"]
