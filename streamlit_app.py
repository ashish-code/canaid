"""Streamlit Cloud entrypoint.

Streamlit Cloud installs `requirements.txt` with pip but doesn't
guarantee the project's own package is installed (an editable install
of a hatch-built package on Streamlit Cloud's container is brittle).
Add `src/` to ``sys.path`` *before* importing canaid so the entrypoint
works whether or not `pip install -e .` succeeded.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Now load the real app — this runs its module-level code (st.set_page_config,
# st.title, sidebar render, chat_input, …).
from canaid.ui.streamlit_app import *  # noqa: F401, F403, E402
