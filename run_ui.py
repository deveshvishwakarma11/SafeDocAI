"""
SafeDocAI UI launcher.

Usage:
    python run_ui.py

Equivalent to:
    streamlit run src/ui.py

The app runs on localhost only. No external assets, no cloud calls.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        print("Streamlit is not installed. Install it with:")
        print("    pip install streamlit>=1.30")
        print("Then run:  python run_ui.py")
        return 1

    # Avoid the Streamlit prompt for an email on first run.
    sys.argv = [
        "streamlit",
        "run",
        str(PROJECT_ROOT / "src" / "ui.py"),
        "--server.headless",
        "true",
        "--server.port",
        "8501",
        "--browser.gatherUsageStats",
        "false",
        "--server.address",
        "localhost",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    raise SystemExit(main())
