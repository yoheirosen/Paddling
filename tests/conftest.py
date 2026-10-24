"""Make ``paddling_expt`` importable when running these tests directly.

Adds the repository root (the parent of ``tests/``) to ``sys.path`` so the
package resolves whether it is installed or not.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
