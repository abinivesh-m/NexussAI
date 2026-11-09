"""Serverless entrypoint.

Vercel imports this module and looks for an ASGI application named ``app``.
The project root is added to the path first because Vercel places the handler
inside ``api/`` while the application package lives at the repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import app  # noqa: E402  (path must be set first)

__all__ = ["app"]
