"""Bedrock runtime client wrapper.

Why a wrapper, not raw boto3?
  - We want every LLM call traced (model, latency, tokens, stop reason) in a
    single place. Wrapping makes that uniform across agents.
  - Retries / circuit breakers / cost tagging plug in here in later phases.
  - The Converse / ConverseStream APIs unify Anthropic / Meta / Amazon Nova /
    Mistral / Cohere — letting us keep one code path while swapping models per
    agent. (Compare to vendor-specific SDK calls that diverge per provider.)

Boto3 clients are thread-safe — one process-wide instance is fine. We expose a
factory (`get_bedrock_client`) so test suites can swap it via dependency
injection rather than monkey-patching globals.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import structlog
from botocore.config import Config as BotoConfig
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from canaid.config import get_settings, make_aws_session

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    """Normalized result from a non-streaming Converse call."""

    text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str
    latency_ms: int
    model_id: str
    raw: dict[str, Any]
    tool_uses: list[dict[str, Any]]  # parsed `toolUse` blocks if any

    def first_tool_use(self) -> dict[str, Any] | None:
        return self.tool_uses[0] if self.tool_uses else None


class BedrockClient:
    """Thin wrapper over `bedrock-runtime`'s Converse / ConverseStream APIs."""

    def __init__(self, region: str | None = None) -> None:
        cfg = get_settings()
        self._region = region or cfg.aws_region
        # `adaptive` retry mode handles Bedrock throttling more gracefully
        # than the default `legacy` mode. The session picks the right
        # credential path (profile vs env-var) — see `make_aws_session`.
        self._client = make_aws_session().client(
            "bedrock-runtime",
            region_name=self._region,
            config=BotoConfig(
                retries={"max_attempts": 3, "mode": "adaptive"},
                read_timeout=120,
                connect_timeout=10,
            ),
        )

    @retry(
        retry=retry_if_exception_type(Exception),  # narrowed in Phase 8
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=8),
        reraise=True,
    )
    def converse(
        self,
        *,
        model_id: str,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        tool_config: dict[str, Any] | None = None,
        inference_config: dict[str, Any] | None = None,
    ) -> LLMResponse:
        from canaid.guardrails.policy import guardrail_config_for_converse

        kwargs: dict[str, Any] = {"modelId": model_id, "messages": messages}
        if system is not None:
            kwargs["system"] = (
                [{"text": system}] if isinstance(system, str) else system
            )
        if tool_config is not None:
            kwargs["toolConfig"] = tool_config
        if inference_config is not None:
            kwargs["inferenceConfig"] = inference_config
        guardrail = guardrail_config_for_converse()
        if guardrail is not None:
            kwargs["guardrailConfig"] = guardrail

        t0 = time.perf_counter()
        resp = self._client.converse(**kwargs)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        out = resp["output"]["message"]
        blocks = out.get("content", [])
        text = "".join(b.get("text", "") for b in blocks)
        tool_uses = [b["toolUse"] for b in blocks if "toolUse" in b]
        usage = resp.get("usage", {})
        log.info(
            "bedrock.converse",
            model_id=model_id,
            input_tokens=usage.get("inputTokens"),
            output_tokens=usage.get("outputTokens"),
            stop_reason=resp.get("stopReason"),
            latency_ms=latency_ms,
            tool_calls=len(tool_uses),
        )
        return LLMResponse(
            text=text,
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
            stop_reason=resp.get("stopReason", ""),
            latency_ms=latency_ms,
            model_id=model_id,
            raw=resp,
            tool_uses=tool_uses,
        )

    def converse_stream(
        self,
        *,
        model_id: str,
        messages: list[dict[str, Any]],
        system: str | None = None,
        inference_config: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Yield text deltas as they arrive. The metadata event at end-of-stream
        is logged but not yielded — callers only see content."""
        kwargs: dict[str, Any] = {"modelId": model_id, "messages": messages}
        if system is not None:
            kwargs["system"] = [{"text": system}]
        if inference_config is not None:
            kwargs["inferenceConfig"] = inference_config

        t0 = time.perf_counter()
        stream = self._client.converse_stream(**kwargs)
        for event in stream["stream"]:
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    yield delta["text"]
            elif "metadata" in event:
                meta = event["metadata"]
                usage = meta.get("usage", {})
                log.info(
                    "bedrock.converse_stream",
                    model_id=model_id,
                    input_tokens=usage.get("inputTokens"),
                    output_tokens=usage.get("outputTokens"),
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                )


_singleton: BedrockClient | None = None


def get_bedrock_client() -> BedrockClient:
    global _singleton
    if _singleton is None:
        _singleton = BedrockClient()
    return _singleton
