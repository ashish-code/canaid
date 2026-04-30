"""RAGAS suite on the rag_golden dataset.

What we measure:
  * **faithfulness**       — does the answer stick to the retrieved context?
  * **answer_relevancy**   — does it actually answer the question?
  * **context_precision**  — is the retrieved context on-topic?
  * **context_recall**     — does the retrieved context cover the ground truth?

Bedrock is the LLM judge AND the embedding model — keeps the eval inside
one provider and avoids API key sprawl. ``ChatBedrockConverse`` and
``BedrockEmbeddings`` from ``langchain-aws`` plug into RAGAS via its
LangChain adapters.

Run:
    uv sync --extra rag --extra eval
    uv run python -m eval.runners.ragas_eval

Requires Bedrock model access for Sonnet 4.5 and Titan Embed v2, and a
populated pgvector index (run ``scripts/build_index.py`` first).
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


def _build_dataset():
    """Build the RAGAS-format dataset by running real retrieve + generate."""
    from langchain_core.messages import HumanMessage, SystemMessage

    golden = load_jsonl("rag_golden.jsonl")
    rows: list[dict] = []

    for ex in golden:
        # Retrieve real chunks
        state = {"messages": [HumanMessage(content=ex["query"])]}
        retrieve_out = rag_retrieve_node(state)
        citations = retrieve_out.get("citations") or []
        contexts = [c["content"] for c in citations]

        # Generate a real answer (one-shot, no streaming) for grading
        passages_text = "\n\n".join(
            f"[{c['id']}] {c['title']}\n{c['content']}" for c in citations
        )
        chat = get_chat_model("rag", temperature=0.0, max_tokens=400)
        composed = HumanMessage(
            content=f"Question: {ex['query']}\n\nPassages:\n{passages_text}"
        )
        sys_msg = SystemMessage(
            content=(
                "Answer using ONLY the numbered passages. Cite passage numbers in [n]. "
                "Be concise."
            )
        )
        resp = asyncio.get_event_loop().run_until_complete(
            chat.ainvoke([sys_msg, composed])
        )
        answer = resp.content if isinstance(resp.content, str) else str(resp.content)

        rows.append(
            {
                "user_input": ex["query"],
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": ex.get("ground_truth", ""),
                "id": ex["id"],
            }
        )
    return rows


def main() -> int:
    configure_logging()
    log = get_logger(__name__)

    try:
        from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as e:
        print(f"RAGAS not installed: {e}", file=sys.stderr)
        print("Run: uv sync --extra eval", file=sys.stderr)
        return 1

    region = get_settings().aws_region
    judge_model_id = get_model_spec("rag").model_id  # reuse RAG model as judge
    embed_model_id = get_settings().embed_model

    log.info(
        "ragas.start",
        judge_model=judge_model_id,
        embed_model=embed_model_id,
        region=region,
    )

    rows = _build_dataset()
    log.info("ragas.dataset_built", n=len(rows))

    judge = LangchainLLMWrapper(
        ChatBedrockConverse(
            model=judge_model_id,
            region_name=region,
            temperature=0.0,
            max_tokens=600,
        )
    )
    embed = LangchainEmbeddingsWrapper(
        BedrockEmbeddings(model_id=embed_model_id, region_name=region)
    )

    eval_ds = EvaluationDataset.from_list(rows)
    metrics = [
        Faithfulness(llm=judge),
        AnswerRelevancy(llm=judge, embeddings=embed),
        ContextPrecision(llm=judge),
        ContextRecall(llm=judge),
    ]

    results = evaluate(
        dataset=eval_ds,
        metrics=metrics,
        llm=judge,
        embeddings=embed,
    )

    summary = {
        "n": len(rows),
        "judge_model": judge_model_id,
        "metrics": {k: round(float(v), 3) for k, v in results.scores.items()}
        if hasattr(results, "scores")
        else dict(results),
    }
    out_path = write_results("ragas_eval", summary)

    banner("RAGAS")
    for metric, score in summary["metrics"].items():
        print(f"  {metric:25} {score}")
    print(f"\n  results → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
