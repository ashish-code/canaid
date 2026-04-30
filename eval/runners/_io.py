"""Shared helpers for runners — dataset loading, results writing."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DATASETS = REPO / "eval" / "datasets"
RESULTS = REPO / "eval" / "results"


def load_jsonl(name: str) -> list[dict[str, Any]]:
    path = DATASETS / name
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def write_results(runner_name: str, payload: dict[str, Any]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS / f"{runner_name}-{ts}.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    # Also overwrite a "latest" symlink-ish copy for convenience.
    (RESULTS / f"{runner_name}-latest.json").write_text(
        json.dumps(payload, indent=2, default=str)
    )
    return out


def banner(title: str) -> None:
    line = "─" * len(title)
    print(f"\n{line}\n{title}\n{line}")
