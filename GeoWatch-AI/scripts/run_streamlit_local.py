from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = ROOT / ".runtime"
EARTH_PACKAGES = ROOT / ".earth_runtime"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))
if EARTH_PACKAGES.exists():
    sys.path.insert(0, str(EARTH_PACKAGES))

from streamlit.web import cli as stcli


if __name__ == "__main__":
    sys.argv = [
        "streamlit",
        "run",
        str(ROOT / "app.py"),
        "--server.port",
        "8501",
        "--server.headless",
        "true",
        "--global.developmentMode",
        "false",
        "--browser.gatherUsageStats",
        "false",
    ]
    raise SystemExit(stcli.main())
