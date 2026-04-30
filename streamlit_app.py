"""Streamlit Cloud entrypoint shim.

Streamlit Cloud expects an app file at the repo root by default. This
module re-exports the real app so we don't have to duplicate UI code.
"""

from canaid.ui.streamlit_app import *  # noqa: F401,F403
