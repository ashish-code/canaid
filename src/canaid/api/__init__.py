"""FastAPI HTTP surface.

`canaid.api.server` requires the ``api`` extra (fastapi + uvicorn +
sse-starlette). Don't import it at package init — that breaks the
embedded deploy on Streamlit Cloud where those deps are absent.
Callers that need the FastAPI app should ``from canaid.api.server
import app, create_app`` directly.
"""
