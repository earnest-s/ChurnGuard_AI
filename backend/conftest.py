"""Pytest configuration for the backend.

Two jobs:

1. Put ``backend/`` on ``sys.path`` (so ``import main`` / ``import train`` work
   when pytest is invoked from the backend directory or from the repo root).
2. Point the service at throw-away directories *before* ``main`` is imported, so
   the tests never load a real artifact nor write to the real prediction log.
"""

import os
import sys
import tempfile
from pathlib import Path

_TMP_DIR = Path(tempfile.mkdtemp(prefix="churnguard-tests-"))

os.environ["CHURNGUARD_MODEL_DIR"] = str(_TMP_DIR / "models")
os.environ["CHURNGUARD_DATA_DIR"] = str(_TMP_DIR / "data")

_BACKEND_DIR = str(Path(__file__).resolve().parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
