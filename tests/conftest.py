"""
Auto-inject a numpy-backed torch mock when real torch is unavailable
(e.g. CPU-only CI without CUDA libraries).
"""

from __future__ import annotations

import os
import sys

try:
    import torch  # noqa: F401
except Exception:
    # Insert mock so the rest of the test suite can import
    sys.path.insert(0, os.path.dirname(__file__))
    import _mock_torch as _mt  # type: ignore[import]

    sys.modules["torch"] = _mt  # type: ignore[assignment]
