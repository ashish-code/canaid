"""Push the golden datasets to LangFuse Cloud as runnable datasets.

Once uploaded, you can:
  * Open the dataset in the LangFuse UI and inspect items.
  * Run experiments by linking traces to dataset items.
  * Use LangFuse's LLM-as-judge presets (toxicity, hallucination,
    helpfulness) on production traces sampled into the same dataset.

Skips silently if LANGFUSE_*_KEY env vars aren't set — useful in CI where
we don't want to fail on a missing optional dep.
"""

from __future__ import annotations

import sys

from canaid.config import get_settings
from canaid.observability.logging import configure_logging, get_logger
from eval.runners._io import banner, load_jsonl, write_results

DATASET_NAMES = {
    "router_golden.jsonl": "canaid-router-golden",
    "rag_golden.jsonl": "canaid-rag-golden",
    "refusals_golden.jsonl": "canaid-refusals-golden",
}


def main() -> int:
    configure_logging()
    log = get_logger(__name__)

    s = get_settings()
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        print(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set — skipping upload."
        )
        return 0

    try:
        from langfuse import Langfuse
    except ImportError as e:
        print(f"langfuse not installed: {e}", file=sys.stderr)
        return 1

    lf = Langfuse(
        public_key=s.langfuse_public_key,
        secret_key=s.langfuse_secret_key,
        host=s.langfuse_host,
    )

    summary: dict = {"datasets": []}
    for filename, dataset_name in DATASET_NAMES.items():
        rows = load_jsonl(filename)
        try:
            lf.create_dataset(name=dataset_name, description=f"CanAID — {filename}")
        except Exception as exc:
            log.info("langfuse.dataset_exists_or_failed", name=dataset_name, error=str(exc))

        for row in rows:
            try:
                lf.create_dataset_item(
                    dataset_name=dataset_name,
                    input={"message": row.get("message") or row.get("query")},
                    expected_output={
                        k: v
                        for k, v in row.items()
                        if k.startswith("expected_") or k in {"ground_truth"}
                    },
                    metadata={"id": row.get("id"), "note": row.get("note")},
                )
            except Exception as exc:
                log.warning(
                    "langfuse.item_failed", dataset=dataset_name, id=row.get("id"), error=str(exc)
                )
        summary["datasets"].append({"name": dataset_name, "items": len(rows)})

    lf.flush()
    out_path = write_results("langfuse_upload", summary)

    banner("LangFuse dataset upload")
    for ds in summary["datasets"]:
        print(f"  {ds['name']}  ({ds['items']} items)")
    print(f"\n  results → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
