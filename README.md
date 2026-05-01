# CanAID

> Multi-agent contact-center chatbot harness on AWS Bedrock — LangGraph
> orchestration, hybrid retrieval, three-layer guardrails, online +
> offline evaluation, per-turn cost & latency telemetry.

A reference implementation of the **harness around an LLM** rather than
the LLM itself. The agents and prompts are deliberately modest; the
interesting engineering is in observability, evaluation, guardrails,
caching, audit, and IaC. The domain — a B2B healthcare-supply-chain
contact center — is synthetic; nothing here is affiliated with any real
distributor, and no real client data is used.

## What's interesting in this repo

**Per-agent model registry.** Six agents, four model families across
three vendors. Routing-only traffic goes to a cheap fast model (Haiku
4.5); reasoning paths go to Sonnet 4.5; tool-using paths go to Llama
3.3 (Meta) for cross-vendor portability proof; cheap utility paths go
to Nova Lite (Amazon). All swappable via env vars — see
`src/canaid/llm/registry.py`.

**Two-node RAG with citations on the wire.** The `rag_retrieve` node
runs synchronously (Titan v2 embed → pgvector / FAISS top-k →
similarity threshold). The `rag_generate` node streams tokens. The API
emits a `citations` SSE frame the moment retrieval finishes — the UI
renders source cards before any token arrives. See
`src/canaid/agents/rag.py`.

**Three independent guardrail layers.**
- Bedrock Guardrails policy enforced inside the Converse call (topic
  deny-list, content filters, PII anonymize/block).
- AWS Comprehend PII detection on the audit / persistence path.
- Regex log scrubber as a structlog processor — every log line, every
  field, idempotent.

Plus templated refusals so the bot's "no" voice is consistent across
policy-side and routing-side rejections.

**Three independent evaluation frameworks.** RAGAS (offline batch on a
golden Q/A set, Bedrock as judge), TruLens (RAG triad — groundedness +
answer-relevance + context-relevance), LangFuse (online tracing +
datasets). Each catches different failure modes; none replaces the
others.

**Per-turn cost meter.** Every Converse response surfaces
`{input_tokens, output_tokens, cost_usd, latency_ms, by_model}` as a
final SSE frame. The `by_model` breakdown makes "different LLMs per
agent" measurable, not just configuration.

**Two transports, one chat protocol.** `canaid.api.local.run_chat` is
an async generator yielding plain frames. The FastAPI handler wraps
each in SSE for HTTP streaming; the Streamlit-embedded path consumes
the generator directly. Same protocol; the deploy mode chooses the
transport.

**Defense-in-depth audit.** Append-only `audit_events` table records
every turn (PII-redacted at write time even after Bedrock Guardrails
has already anonymized the LLM-bound copy). Postgres backend with a
log fallback that always succeeds.

## Stack

| Layer | Choice |
|---|---|
| Language | Python 3.12 |
| LLM gateway | AWS Bedrock (Converse / ConverseStream) |
| Models | Anthropic Claude Sonnet 4.5 + Haiku 4.5 · Meta Llama 3.3 70B · Amazon Nova Lite · Amazon Titan Embed v2 |
| Orchestration | LangGraph (`StateGraph` + checkpointer) |
| RAG | pgvector (Aurora) for prod / FAISS in-memory for single-process |
| Cache | Valkey / Redis (TTL) for prod / in-memory dict fallback |
| Guardrails | Bedrock Guardrails + AWS Comprehend + regex |
| Evaluation | RAGAS + TruLens + LangFuse Cloud |
| API | FastAPI + Server-Sent Events |
| UI | Streamlit |
| IaC | AWS CDK (Python) |
| Observability | structlog · LangFuse spans · audit log · cost calculator |

## Architecture

```
                       (HumanMessage in)
                              │
                              ▼
                       ┌──────────────┐
                       │   intent     │  Haiku 4.5 (Bedrock toolConfig → JSON)
                       └──────┬───────┘
                              │
                       supervisor_route()  ← deterministic mapping
                              │
   ┌──────────┬───────────┬───┴────────┬──────────┬──────────┐
   ▼          ▼           ▼            ▼          ▼          ▼
qualifier  rag_retrieve→generate    lookup    escalation   refusal    fallback
Sonnet 4.5  Sonnet 4.5              Llama 3.3 Nova Lite    template   Sonnet 4.5
                                                            (no LLM)
   └──────────┴───────────┴────────────┴──────────┴────────────┘
                              │
                              ▼
                            (END)
```

Two transport modes share the same graph + same frame protocol:

```
                    ┌────────────────────────────────────────┐
   Browser ─────▶  │ Streamlit (UI)                          │
                    └──────────────────┬─────────────────────┘
                                       │
        ┌──────────────────────────────┴────────────────────────────────┐
        │                                                                │
        ▼ embedded mode (Streamlit Cloud)                                ▼ API mode (AWS / ECS)
  run_chat() in-process                                          httpx.stream over SSE
        │                                                                │
        └──────────────────────────────┬─────────────────────────────────┘
                                       ▼
                              LangGraph StateGraph
                                       │
                                       ▼
                                  AWS Bedrock
```

## Quickstart — local single-process

```bash
# 1. Install deps
make install                       # base
uv sync --extra rag --extra embedded

# 2. Configure
cp .env.example .env
# fill in AWS_PROFILE / AWS_REGION; LangFuse keys optional
# add CANAID_USE_FAISS=true and CANAID_EMBEDDED=true to .env

# 3. Run
make ui                            # Streamlit on :8501
```

The first request builds a FAISS index from `data/catalog/skus.jsonl`
and `data/policies/*.md` (~38 chunks, ~5–10s on cold start). Each
subsequent turn is sub-3 seconds.

## Quickstart — local API + UI

```bash
make infra-up                      # postgres+pgvector and redis
psql ... < scripts/sql/01-rag.sql
psql ... < scripts/sql/02-audit.sql
make build-index
make api                           # FastAPI on :8000
make ui                            # Streamlit on :8501
```

## Quickstart — production AWS deploy

CDK stack in `infra/`:

```bash
cd infra
uv sync --extra infra
uv run cdk bootstrap
uv run cdk deploy
```

Provisions VPC + Aurora Serverless v2 (Postgres + pgvector) + Valkey
Serverless + ECS Fargate behind an ALB + Secrets Manager + IAM task
role with the necessary Bedrock + Comprehend permissions. Outputs the
ALB URL.

## Project layout

```
src/canaid/
├── agents/          intent · qualifier · rag · lookup · escalation · refusal · fallback
├── graph/           state · router · workflow (StateGraph + checkpointer)
├── llm/             BedrockClient + ChatBedrockConverse + per-agent registry
├── retrieval/       chunker · Titan embeddings · pgvector / FAISS stores · Retriever
├── tools/           mock CRM + LangChain @tool wrappers
├── guardrails/      Bedrock Guardrails + Comprehend PII + regex log scrubber + refusals
├── cache/           turn-level Redis (or in-memory dict) cache
├── memory/          LangGraph checkpointer factory (Memory / Postgres)
├── observability/   structlog · LangFuse · audit log · cost calculator
├── api/             FastAPI + SSE server + run_chat in-process protocol
└── ui/              Streamlit chat surface (dual-mode: API or embedded)

eval/                golden datasets + 5 runners (router, refusals, RAGAS, TruLens, LangFuse)
infra/               AWS CDK stack (Python)
scripts/             SQL migrations + index builder + Bedrock Guardrail bootstrap
tests/               unit tests, no AWS calls — sub-second
```

## Tests

```bash
make test
```

~80 tests covering router logic, chunker, mock CRM, cache, guardrail
templates + redaction, model registry, cost calculator, FAISS store.
No tests touch AWS — the eval runners (`make eval-*`) cover the
AWS-touching paths.

## Eval suite

```bash
uv sync --extra eval
make build-index
make eval-router       # intent + route accuracy on golden set
make eval-refusals     # safety-coverage on adversarial prompts
make eval-ragas        # faithfulness · answer_relevancy · context_*
make eval-trulens      # RAG triad — groundedness + answer + context relevance
make eval-langfuse     # upload golden datasets to LangFuse Cloud
```

Each runner writes a JSON summary into `eval/results/`.

## License

Proprietary. Demonstration project — no warranty, no support.
