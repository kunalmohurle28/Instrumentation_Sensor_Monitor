"""Main Entry Point for Industrial Sensor Monitoring & Predictive Fault Analysis System.

Executes the top-level Streamlit application launcher and configures page defaults.

To run:
    streamlit run main.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is available on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Streamlit Page Configuration must be the first Streamlit command invoked
st.set_page_config(
    page_title="Industrial Sensor Monitoring System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.dashboard import render_dashboard


def main() -> None:
    """Execute the dashboard application."""
    render_dashboard()


if __name__ == "__main__":
    main()
