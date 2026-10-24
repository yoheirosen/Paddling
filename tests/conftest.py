"""Make ``paddling_expt`` importable when running these tests directly.

Adds ``paddling/`` (the parent of ``tests/``) and the repository root to
``sys.path`` so the package resolves whether pytest runs from the repository root
or from ``paddling/``, installed or not.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for p in (_ROOT, _ROOT.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
