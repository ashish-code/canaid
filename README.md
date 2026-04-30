# CanAID — Multi-Agent Contact Center Chatbot

A production-pattern multi-agent chatbot for a B2B healthcare supply-chain
contact center. The bot itself is intentionally modest; the **harness
around it** — observability, evaluation, guardrails, caching, audit,
IaC — is the point.

> **Why this exists.** Built as a portfolio piece to demonstrate
> end-to-end "harness engineering" for production LLM systems on AWS.
> The contact-center domain is realistic but synthetic — there is no
> real client data, no real integrations, and the bot does not give
> clinical advice.

## Stack at a glance

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.12 | Modern typing |
| LLM gateway | **AWS Bedrock** (Converse API) | One API across vendors |
| Models | Sonnet · Haiku · Llama 3.3 · Nova Lite · Titan | Different LLM per agent — see `docs/02-agents.md` and ADR-0003 |
| Orchestration | **LangGraph** StateGraph + checkpointer | Supervisor/sub-agent, durable resume |
| RAG | pgvector locally / Aurora-pgvector in prod | OpenSearch upgrade in ADR-0005 |
| Cache | **Valkey** (Redis-compatible) Serverless | Turn-level, exact-match, intent-gated |
| Guardrails | Bedrock Guardrails + Comprehend + regex | Three-layer defense in depth |
| Evaluation | **RAGAS** + **TruLens** + **LangFuse Cloud** | Each catches different failures — `docs/07-evaluation.md` |
| Observability | structlog (PII-scrubbed) + LangFuse spans + audit log | Every turn traceable |
| API | FastAPI + SSE | Streaming end-to-end |
| UI | Streamlit | Fastest believable demo surface |
| IaC | AWS CDK (Python) | ECS Fargate + Aurora + Valkey + ALB |

## Quickstart (local)

```bash
# 1. Install
make install                       # uv sync (base deps)
make install-rag                   # add psycopg + pgvector + opensearch-py + tiktoken

# 2. Configure
cp .env.example .env
# edit AWS_PROFILE / AWS_REGION; LangFuse keys optional

# 3. Local infra (Postgres + Valkey-compatible Redis)
make infra-up
psql ... < scripts/sql/01-rag.sql      # apply RAG schema (see infra/README.md)
psql ... < scripts/sql/02-audit.sql    # apply audit schema

# 4. Build the index
make build-index

# 5. Optional: create the Bedrock Guardrail (paste ID into .env)
uv run python scripts/setup_guardrail.py

# 6. Run
make api                           # FastAPI on :8000
make ui                            # Streamlit on :8501
```

## Deploy to AWS

See `docs/09-deployment.md` and `infra/README.md`. Short version:

```bash
cd infra
uv sync --extra infra
uv run cdk bootstrap     # one-time
uv run cdk deploy
```

CDK builds the Docker image, pushes to ECR, stands up the stack
(VPC + Aurora + Valkey + ECS Fargate + ALB), prints the API URL.
Streamlit Cloud hosts the UI pointing at that URL.

## Demo

`docs/demo-script.md` is a 5-minute scripted walkthrough that fires
each specialist once. The Streamlit sidebar has six sample-question
chips that mirror those steps; click any one to skip typing.

## Documentation

Read in order:

1. [`docs/00-overview.md`](docs/00-overview.md) — what CanAID is and why
2. [`docs/01-architecture.md`](docs/01-architecture.md) — process boundaries, state, failure modes
3. [`docs/02-agents.md`](docs/02-agents.md) — the LangGraph + per-agent design
4. [`docs/03-rag.md`](docs/03-rag.md) — chunker, embeddings, retrieve→generate split, citations
5. [`docs/04-tool-use.md`](docs/04-tool-use.md) — tool-use loop, mock CRM, PII at the tool boundary
6. [`docs/05-guardrails-and-pii.md`](docs/05-guardrails-and-pii.md) — three-layer guardrails
7. [`docs/06-caching-memory.md`](docs/06-caching-memory.md) — turn cache + LangGraph checkpointer
8. [`docs/07-evaluation.md`](docs/07-evaluation.md) — RAGAS + TruLens + LangFuse, golden datasets
9. [`docs/08-observability.md`](docs/08-observability.md) — logs · spans · audit · cost meter
10. [`docs/09-deployment.md`](docs/09-deployment.md) — CDK stack, deploy steps, cost shape
11. [`docs/10-runbook.md`](docs/10-runbook.md) — day-2 ops
12. [`docs/adr/`](docs/adr/) — architecture decisions
13. [`docs/demo-script.md`](docs/demo-script.md) — 5-minute demo walk-through

## Layout

```
src/canaid/
├── agents/         intent · qualifier · rag · lookup · escalation · refusal · fallback
├── graph/          state · router · workflow (StateGraph + checkpointer)
├── llm/            BedrockClient + ChatBedrockConverse + per-agent registry
├── retrieval/      chunker · Titan embeddings · pgvector store · Retriever facade
├── tools/          mock CRM + LangChain @tool wrappers
├── guardrails/     Bedrock Guardrails + Comprehend PII + regex log scrubber + refusal templates
├── cache/          turn-level Redis cache (intent-gated)
├── memory/         LangGraph checkpointer factory (MemorySaver / PostgresSaver)
├── observability/  structlog · LangFuse · audit log · cost calculator
├── api/            FastAPI + SSE
└── ui/             Streamlit chat surface

eval/               golden datasets + 5 runners (router, refusals, RAGAS, TruLens, LangFuse)
infra/              AWS CDK stack
scripts/            SQL migrations + index builder + guardrail setup
docs/               numbered explainers + ADRs + demo script
```

## Tests

```bash
make test          # ~70 tests, no AWS calls; sub-second
```

Tests cover routing, chunker, mock CRM, cache logic, guardrail
templates + redaction, model registry, cost calculator. AWS-touching
behavior is covered by the eval runners (which require Bedrock access).

## What this is *not*

- A real product. No real CRM, no real auth, no real ordering.
- A clinical assistant. The bot refuses clinical/medical questions.
- Optimized for absolute lowest latency. The harness trades milliseconds
  for traceability, evaluability, and audit.

## License

Proprietary — interview / portfolio use.
