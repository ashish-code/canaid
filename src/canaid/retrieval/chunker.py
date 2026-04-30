"""Chunking strategy.

Two doc types, two strategies — picked because each matches the document's
information density:

  * **SKU** (one JSON record per product) →  ONE chunk per SKU.
    Each SKU is short (~200-400 tokens once flattened) and self-contained.
    Splitting it would scatter specs across chunks and tank retrieval.

  * **Policy** (multi-section markdown docs) →  RECURSIVE character split,
    600 chars per chunk with 100-char overlap. We use `RecursiveCharacterTextSplitter`
    so chunk boundaries respect paragraph/heading breaks before falling back
    to sentences.

Chunk text format for SKUs is normalized into a human-readable card so the
LLM has high-quality context — JSON dumps are *terrible* RAG input compared
to flowing text.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(slots=True)
class Chunk:
    doc_id: str
    doc_type: str           # "sku" | "policy"
    chunk_index: int
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _format_sku_card(sku: dict[str, Any]) -> str:
    """Flatten a SKU record to a readable card. Avoid JSON noise in chunks."""
    parts: list[str] = []
    parts.append(f"# {sku['name']}")
    parts.append(f"SKU: {sku['sku']}")
    parts.append(f"Category: {sku.get('category', 'Unknown')}"
                 f" / {sku.get('subcategory', '')}".rstrip(" /"))
    parts.append("")
    parts.append(sku.get("description", ""))
    parts.append("")
    specs = sku.get("specs") or {}
    if specs:
        parts.append("## Specs")
        for k, v in specs.items():
            parts.append(f"- {k.replace('_', ' ')}: {v}")
        parts.append("")
    parts.append(f"Packaging: {sku.get('packaging', '-')}")
    parts.append(f"Case pack: {sku.get('case_pack', '-')}")
    parts.append(f"Min order qty: {sku.get('min_order_qty', '-')}")
    parts.append(f"Lead time: {sku.get('lead_time_days', '-')} business days")
    certs = sku.get("certifications") or []
    if certs:
        parts.append(f"Certifications: {', '.join(certs)}")
    parts.append(f"Country of origin: {sku.get('country_of_origin', '-')}")
    return "\n".join(parts)


def chunk_catalog(jsonl_path: Path) -> Iterable[Chunk]:
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sku = json.loads(line)
            content = _format_sku_card(sku)
            yield Chunk(
                doc_id=sku["sku"],
                doc_type="sku",
                chunk_index=0,
                title=sku["name"],
                content=content,
                metadata={
                    "sku": sku["sku"],
                    "category": sku.get("category"),
                    "subcategory": sku.get("subcategory"),
                    "lead_time_days": sku.get("lead_time_days"),
                    "country_of_origin": sku.get("country_of_origin"),
                },
            )


_POLICY_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
)


def chunk_policies(dir_path: Path) -> Iterable[Chunk]:
    for md_path in sorted(dir_path.glob("*.md")):
        text = md_path.read_text()
        # First non-empty heading line, used as document title.
        title = next(
            (ln.strip("# ").strip() for ln in text.splitlines() if ln.startswith("# ")),
            md_path.stem,
        )
        for i, piece in enumerate(_POLICY_SPLITTER.split_text(text)):
            yield Chunk(
                doc_id=md_path.stem,
                doc_type="policy",
                chunk_index=i,
                title=title,
                content=piece,
                metadata={"source": md_path.name},
            )
