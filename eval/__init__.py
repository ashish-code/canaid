"""Evaluation harness — RAGAS + TruLens + LangFuse + custom router/refusal evals.

Three external frameworks because each does a different job:

  * **RAGAS** (offline RAG metrics) — faithfulness, answer_relevancy,
    context_precision, context_recall on a curated Q&A dataset. Runs in
    CI; the gate before promoting a prompt change.
  * **TruLens** (online RAG triad) — groundedness + context-relevance +
    answer-relevance on sampled live traces. Uses Bedrock as the LLM
    judge.
  * **LangFuse** (datasets + tracing) — uploaded golden datasets become
    runnable experiments on LangFuse Cloud. Production traces flow into
    the same project for ongoing observation.

Plus first-party runners that don't fit any of the three:
  * `router_eval` — intent + route accuracy on the router_golden dataset.
  * `refusals_eval` — refusal-coverage on adversarial prompts.

Each runner emits a JSON summary into `eval/results/`.
"""
