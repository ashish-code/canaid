# syntax=docker/dockerfile:1.7
# CanAID API container.
#
# Two-stage build:
#   1. builder — uv installs all base + rag deps into /opt/venv
#   2. runtime — slim Python with /opt/venv copied in
#
# uv is fast enough that we don't bother with wheel-cache mounts; the
# install step is O(seconds) on warm caches.

FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    cp /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml /app/
COPY src /app/src
COPY README.md /app/

# Install into a system location we can copy in stage 2.
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --upgrade pip && \
    uv pip install -e ".[rag]"

# ---------- runtime ----------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        ca-certificates curl && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --uid 10001 app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

WORKDIR /app
USER app
EXPOSE 8000

# Healthcheck hits /health — must work without AWS creds.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "canaid.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
