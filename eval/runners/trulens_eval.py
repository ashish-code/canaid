"""TruLens RAG triad on the rag_golden dataset.

What we measure (the canonical "RAG triad"):
  * **Groundedness**     — is each claim in the answer supported by context?
  * **Answer relevance** — does the answer address the question?
  * **Context relevance**— do the retrieved chunks relate to the question?

TruLens orchestrates the feedback functions; the *judge* is Bedrock
(Claude Sonnet 4.5). Results land in TruLens's local SQLite DB which the
Phase 10 demo polish surfaces in a Streamlit panel.

Run:
    uv sync --extra eval
    uv run python -m eval.runners.trulens_eval
"""

from __future__ import annotations

import asyncio
import sys

from canaid.agents.rag import rag_retrieve_node
from canaid.config import get_settings
from canaid.llm.lc import get_chat_model
from canaid.llm.registry import get_model_spec
from canaid.observability.logging import configure_logging, get_logger
from eval.runners._io import banner, load_jsonl, write_results


def main() -> int:
    configure_logging()
    log = get_logger(__name__)

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from trulens.core import TruSession
        from trulens.providers.bedrock import Bedrock as TruBedrock
    except ImportError as e:
        print(f"TruLens not installed: {e}", file=sys.stderr)
        print("Run: uv sync --extra eval", file=sys.stderr)
        return 1

    judge_model_id = get_model_spec("rag").model_id
    region = get_settings().aws_region
    log.info("trulens.start", judge_model=judge_model_id, region=region)

    session = TruSession()
    session.reset_database()
    provider = TruBedrock(model_id=judge_model_id, region_name=region)

    golden = load_jsonl("rag_golden.jsonl")
    rows: list[dict] = []
    chat = get_chat_model("rag", temperature=0.0, max_tokens=400)

    for ex in golden:
        state = {"messages": [HumanMessage(content=ex["query"])]}
        retrieve_out = rag_retrieve_node(state)
        citations = retrieve_out.get("citations") or []
        contexts = [c["content"] for c in citations]
        passages = "\n\n".join(
            f"[{c['id']}] {c['title']}\n{c['content']}" for c in citations
        )
        composed = HumanMessage(
            content=f"Question: {ex['query']}\n\nPassages:\n{passages}"
        )
        sys_msg = SystemMessage(
            content="Answer using ONLY the numbered passages. Cite [n]. Be concise."
        )
        resp = asyncio.get_event_loop().run_until_complete(
            chat.ainvoke([sys_msg, composed])
        )
        answer = resp.content if isinstance(resp.content, str) else str(resp.content)

        # Score each turn directly through the feedback function. We do this
        # rather than wrapping the agent in TruApp because the agent is async
        # and lives inside LangGraph; for a small offline eval it's cleaner
        # to call providers directly.
        scores = {
            "Groundedness": float(
                provider.groundedness_measure_with_cot_reasons(
                    "\n".join(contexts), answer
                )[0]
            ),
            "AnswerRelevance": float(
                provider.relevance(ex["query"], answer)
            ),
            "ContextRelevance": float(
                provider.context_relevance(ex["query"], "\n".join(contexts))
            ),
        }
        rows.append(
            {
                "id": ex["id"],
                "query": ex["query"],
                "answer": answer,
                **scores,
            }
        )
        log.info("trulens.row", id=ex["id"], **scores)

    avg = {
        k: round(sum(r[k] for r in rows) / len(rows), 3)
        for k in ("Groundedness", "AnswerRelevance", "ContextRelevance")
    }
    payload = {"n": len(rows), "averages": avg, "rows": rows}
    out_path = write_results("trulens_eval", payload)

    banner("TruLens RAG triad")
    for k, v in avg.items():
        print(f"  {k:18} {v}")
    print(f"\n  results → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
