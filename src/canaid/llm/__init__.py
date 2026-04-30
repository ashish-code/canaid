"""LLM gateway — Bedrock client + per-agent model registry."""

from canaid.llm.bedrock import BedrockClient, LLMResponse, get_bedrock_client
from canaid.llm.registry import AgentName, ModelSpec, get_model_spec

__all__ = [
    "AgentName",
    "BedrockClient",
    "LLMResponse",
    "ModelSpec",
    "get_bedrock_client",
    "get_model_spec",
]
