"""Streamlit Cloud entrypoint.

Streamlit Cloud installs the package via uv-sync, but the shim also
adds `src/` to ``sys.path`` so the entrypoint works regardless of
install state.

DIAGNOSTIC MODE: we render a heartbeat banner *before* any other
imports so we can tell at a glance whether Streamlit is even running
our script. If the page is blank, you're seeing browser/Cloud issues.
If you see the banner but the rest is missing, something downstream
crashed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# ---- Heartbeat — this is the FIRST visible output --------------------------
st.set_page_config(
    page_title="CanAID — Contact Center",
    layout="wide",
    initial_sidebar_state="expanded",
)
_DIAG_BOX = st.empty()
_DIAG_BOX.success(
    "✅ streamlit_app.py loaded — proceeding to import canaid.ui.streamlit_app …"
)

# Now bootstrap canaid.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from canaid.ui.streamlit_app import *  # noqa: F401, F403, E402
except Exception as exc:  # noqa: BLE001
    import traceback as _tb

    _DIAG_BOX.error(
        f"**Import error in canaid.ui.streamlit_app**\n\n"
        f"`{type(exc).__name__}: {exc}`"
    )
    with st.expander("Full traceback", expanded=True):
        st.code(_tb.format_exc(), language="python")
    st.stop()
else:
    # Successful import — clear the heartbeat (the real app has rendered
    # its own UI by now, so leaving the banner just adds clutter).
    _DIAG_BOX.empty()
