.PHONY: help install dev api ui infra-up infra-down infra-reset test fmt lint clean
.DEFAULT_GOAL := help

help:  ## list targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## install deps via uv
	uv sync

install-rag:  ## install Phase 3 RAG extras
	uv sync --extra rag

install-eval:  ## install Phase 7 evaluation extras
	uv sync --extra eval

install-all:  ## install every optional group
	uv sync --extra rag --extra eval --extra guardrails

infra-up:  ## start local Postgres+pgvector and Redis
	docker compose up -d
	@printf "Waiting for Postgres "
	@until docker compose exec -T postgres pg_isready -U canaid >/dev/null 2>&1; do printf "."; sleep 1; done
	@echo " ok"

infra-down:  ## stop local stack (keep data)
	docker compose down

infra-reset:  ## stop local stack and wipe volumes
	docker compose down -v

api:  ## run FastAPI on :8000 with reload
	uv run uvicorn canaid.api.server:app --host 0.0.0.0 --port 8000 --reload

ui:  ## run Streamlit chat UI on :8501
	uv run streamlit run src/canaid/ui/streamlit_app.py

test:  ## run unit tests
	uv run pytest

fmt:  ## format code
	uv run ruff format src tests

lint:  ## lint code
	uv run ruff check src tests

eval-router:  ## intent + route accuracy on golden dataset
	uv run python -m eval.runners.router_eval

eval-refusals:  ## refusal-coverage on adversarial prompts
	uv run python -m eval.runners.refusals_eval

eval-ragas:  ## RAGAS suite (faithfulness, answer_relevancy, context_*)
	uv run python -m eval.runners.ragas_eval

eval-trulens:  ## TruLens RAG triad
	uv run python -m eval.runners.trulens_eval

eval-langfuse:  ## upload golden datasets to LangFuse Cloud
	uv run python -m eval.runners.langfuse_upload

eval-all:  ## run every eval (skips ones missing creds gracefully)
	-$(MAKE) eval-router
	-$(MAKE) eval-refusals
	-$(MAKE) eval-ragas
	-$(MAKE) eval-trulens
	-$(MAKE) eval-langfuse

build-index:  ## (re)build the RAG index from data/
	uv run python scripts/build_index.py --reset

clean:  ## clean caches
	rm -rf .ruff_cache .mypy_cache .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
