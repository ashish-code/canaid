"""Eval runners — each module is a CLI entrypoint.

Run individually:

    uv run python -m eval.runners.router_eval
    uv run python -m eval.runners.refusals_eval
    uv run python -m eval.runners.ragas_eval
    uv run python -m eval.runners.trulens_eval
    uv run python -m eval.runners.langfuse_upload

Or `make eval-all` to run them in sequence and aggregate results.
"""
