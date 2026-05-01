"""Application configuration.

All CanAID-specific settings live in `Settings` and are loaded from CANAID_*
environment variables (or `.env`). AWS credentials/region are *not* managed
here — boto3's default credential chain handles them via AWS_PROFILE,
AWS_ACCESS_KEY_ID, IAM roles in ECS/Lambda, etc. This separation matters in
production: the same image runs in dev with a developer's profile and in
prod with a task-role-attached IAM identity, with no code change.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Top-level config. Read once, cached for the process lifetime."""

    model_config = SettingsConfigDict(
        env_prefix="CANAID_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- Per-agent Bedrock model IDs ------------------------------------------
    # Different LLM per agent on purpose:
    #   - supervisor / qualifier / rag : Sonnet — needs reasoning, multi-turn
    #   - intent                       : Haiku — runs every turn, must be fast/cheap
    #   - tool                         : Llama 3.3 — cross-vendor portability proof
    #   - summarizer                   : Nova Lite — AWS-native, lowest cost
    supervisor_model: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    intent_model: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    qualifier_model: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    rag_model: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    tool_model: str = "meta.llama3-3-70b-instruct-v1:0"
    summarizer_model: str = "us.amazon.nova-lite-v1:0"
    embed_model: str = "amazon.titan-embed-text-v2:0"

    # -- Guardrails (Phase 5) ------------------------------------------------
    # Created once via `scripts/setup_guardrail.py`. Empty → no-op.
    guardrail_id: str | None = None
    guardrail_version: str = "DRAFT"

    # -- Caching + memory (Phase 6) ------------------------------------------
    cache_ttl_seconds: int = 3600
    cache_enabled: bool = True
    use_postgres_checkpointer: bool = False

    # -- Single-process / Streamlit Cloud mode -------------------------------
    # When `embedded=True`, the Streamlit app calls the LangGraph workflow
    # in-process instead of HTTP-streaming through the FastAPI server. Pairs
    # naturally with `use_faiss=True` (FAISS in-memory store, no Postgres).
    embedded: bool = False
    use_faiss: bool = False

    # -- API + UI -------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_url: str = "http://localhost:8000"  # used by Streamlit

    # -- Stores ---------------------------------------------------------------
    pg_dsn: str = "postgresql://canaid:canaid@localhost:5432/canaid"
    redis_url: str = "redis://localhost:6379/0"

    # -- Observability --------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://us.cloud.langfuse.com"

    @property
    def aws_region(self) -> str:
        """Resolve the region the same way boto3's default chain would."""
        return (
            os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or "us-east-1"
        )

    @property
    def aws_profile(self) -> str | None:
        """Active AWS profile name, if any.

        Default is `vscode-user` to match the AskAI-Mahabharat reference
        repo's local profile (already exists in ~/.aws/credentials with
        Bedrock access). Override per-host via the AWS_PROFILE env var.

        Returns None on hosts where env-var creds are present (Streamlit
        Cloud, ECS task role, GitHub OIDC) — see `make_aws_session`.
        """
        if os.getenv("AWS_ACCESS_KEY_ID"):
            return None
        return os.getenv("AWS_PROFILE", "vscode-user")

    def postgres_dsn_host_only(self) -> str:
        """Strip credentials from the Postgres DSN for safe logging."""
        try:
            from urllib.parse import urlparse

            u = urlparse(self.pg_dsn)
            return f"{u.hostname}:{u.port or 5432}/{(u.path or '').lstrip('/')}"
        except Exception:
            return "<unparseable>"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def make_aws_session():
    """Single boto3 Session per process, picking the right credential path.

    The pattern mirrors AskAI-Mahabharat's `get_bedrock_client` to keep
    behavior consistent across the two demos:

      * If AWS_ACCESS_KEY_ID is set (Streamlit Cloud secrets, ECS task
        role with web-identity, GitHub OIDC), build a Session WITHOUT a
        profile name. Passing profile_name on a host with no
        ~/.aws/config raises ProfileNotFound — this branch avoids that.
      * Otherwise honor AWS_PROFILE (default `vscode-user`, matching
        the local profile in ~/.aws/credentials).
    """
    import boto3  # local import keeps cold-start light

    s = get_settings()
    if os.getenv("AWS_ACCESS_KEY_ID"):
        return boto3.Session(region_name=s.aws_region)
    return boto3.Session(profile_name=s.aws_profile, region_name=s.aws_region)


def boto_client_factory_safe(service: str):
    """Build a boto3 client; return None if credentials aren't available.

    Used by guardrail / PII paths that should *no-op gracefully* when run in
    a dev environment without AWS configured (e.g., unit tests).
    """
    try:
        from botocore.exceptions import NoCredentialsError, ProfileNotFound

        s = get_settings()
        try:
            return make_aws_session().client(service, region_name=s.aws_region)
        except (NoCredentialsError, ProfileNotFound):
            return None
    except Exception:
        return None
