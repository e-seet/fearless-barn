# Path helper for scripts written when solution/ and exploration/ were one folder.
#
# The scripts address everything as ROOT / "<name>". After the split, <name> may live in either
# folder. dual(ROOT) returns an object that behaves like the path ROOT, except that
# ROOT / "<name>" resolves to whichever of solution/<name> or exploration/<name> exists (the
# script's own folder wins; a name that exists in neither stays in the script's own folder).
# It also puts both folders on sys.path, so `import config`, `from common import ...` and
# `from tests_x.y import z` keep working from either side.
#
# Usage (a script executes this file, then wraps its root):
#     exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
import sys as _sys
from pathlib import Path as _Path

_REPO = None


class _DualRoot:
    def __init__(self, primary, other):
        self._p, self._o = _Path(primary), _Path(other)

    def __truediv__(self, key):
        a = self._p / key
        if a.exists():
            return a
        b = self._o / key
        return b if b.exists() else a

    def __str__(self):
        return str(self._p)

    def __fspath__(self):
        return str(self._p)

    def __repr__(self):
        return f"dual({self._p!s})"

    def __getattr__(self, name):          # parent, name, exists(), glob(), ...: the script's own folder
        return getattr(self._p, name)


def dual(root):
    """Wrap a path that points at solution/ or exploration/ (see the header)."""
    root = _Path(root)
    repo = root.parent
    names = {"solution", "exploration"}
    if root.name not in names:
        return root
    other = repo / (names - {root.name}).pop()
    for p in (other, root):               # both importable; scripts insert their own folder first anyway
        if str(p) not in _sys.path:
            _sys.path.append(str(p))
    return _DualRoot(root, other)
