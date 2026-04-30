"""LangChain ``ChatBedrockConverse`` factory.

LangGraph nodes that need *streaming* tokens use this. The token stream
flows through LangGraph's ``astream_events`` pipeline — letting the FastAPI
endpoint forward every chunk as an SSE event without any per-node plumbing.

For non-streaming calls (intent classification, summarization), keep using
the plain ``BedrockClient`` from ``canaid.llm.bedrock``. We deliberately use
both:
  * ``BedrockClient`` for one-shot calls + native tool-use (forced-JSON).
  * ``ChatBedrockConverse`` for streaming compatibility with LangGraph.

This dual-path is on purpose. LangChain's structured-output helpers can be
flaky across vendors, but its streaming integration is excellent.
``BedrockClient`` gives us a battle-tested escape hatch when LangChain's
abstractions don't pull their weight.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_aws import ChatBedrockConverse

from canaid.config import get_settings
from canaid.llm.registry import AgentName, get_model_spec


@lru_cache(maxsize=8)
def get_chat_model(
    agent: AgentName,
    *,
    temperature: float = 0.3,
    max_tokens: int = 800,
) -> ChatBedrockConverse:
    from canaid.guardrails.policy import guardrails_for_chat_bedrock

    spec = get_model_spec(agent)
    region = get_settings().aws_region
    kwargs: dict = dict(
        model=spec.model_id,
        region_name=region,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    guardrails = guardrails_for_chat_bedrock()
    if guardrails is not None:
        kwargs["guardrails"] = guardrails
    return ChatBedrockConverse(**kwargs)
