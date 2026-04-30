"""Embedding client.

Wraps Bedrock's Titan Embed v2 for both batch (ingestion) and per-query
embedding. We pin Titan v2 because:
  * It's the highest-quality general-purpose embedding on Bedrock today.
  * 1024-dim is a good cost/quality trade-off (Titan v2 also supports 256
    and 512 if storage matters; 1024 is hard-coded in the pgvector schema).
  * It returns L2-normalized vectors, so cosine similarity in pgvector
    behaves predictably.

Same wrapper is used at query time in the retriever and at build time in
`scripts/build_index.py`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from functools import lru_cache

import structlog

from canaid.config import get_settings
from canaid.llm.bedrock import get_bedrock_client

log = structlog.get_logger(__name__)

EMBED_DIM = 1024


class TitanEmbedder:
    """Client for `amazon.titan-embed-text-v2:0` via Bedrock."""

    def __init__(
        self,
        model_id: str | None = None,
        dimensions: int = EMBED_DIM,
    ) -> None:
        self.model_id = model_id or get_settings().embed_model
        self.dimensions = dimensions
        # Reuse the same boto3 client as the chat path — same retries, same
        # adaptive backoff config, fewer connections.
        self._bedrock = get_bedrock_client()._client

    def embed_one(self, text: str) -> list[float]:
        body = {
            "inputText": text,
            "dimensions": self.dimensions,
            "normalize": True,
        }
        resp = self._bedrock.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(resp["body"].read())
        return list(payload["embedding"])

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        # Titan v2 has no batch endpoint on Bedrock — we serialize. For tiny
        # corpora (this demo's ~25 chunks) that's fine. A thread-pool
        # fan-out lands in Phase 8 if/when the index grows.
        out: list[list[float]] = []
        for i, text in enumerate(texts):
            out.append(self.embed_one(text))
            if (i + 1) % 25 == 0:
                log.info("embed.progress", processed=i + 1)
        return out


@lru_cache(maxsize=1)
def get_embedder() -> TitanEmbedder:
    return TitanEmbedder()
