"""Root entrypoint for Streamlit Community Cloud deployment.

Streamlit Community Cloud defaults to looking for 'streamlit_app.py' in the
repository root. This entrypoint forwards execution to 'webapp/streamlit_app.py'.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
WEBAPP_DIR = ROOT_DIR / "webapp"
ENTRYPOINT = WEBAPP_DIR / "streamlit_app.py"

if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))

runpy.run_path(str(ENTRYPOINT), run_name="__main__")
