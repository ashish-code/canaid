"""Router accuracy on the golden dataset.

Runs the real intent_node against each labeled message and computes:

  * Overall intent accuracy.
  * Per-intent precision/recall (so we see *which* intents the router
    confuses).
  * Route accuracy (intent → route via the supervisor function).

Why this is the most-evaluated component: every other agent's eval starts
from "did we route here?" If the router misroutes 10% of the time, every
specialist's apparent quality is degraded by that headline rate.

Requires AWS Bedrock access for the intent model (Haiku 4.5).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from langchain_core.messages import HumanMessage

from canaid.agents.intent import intent_node
from canaid.graph.router import supervisor_route
from canaid.observability.logging import configure_logging, get_logger
from eval.runners._io import banner, load_jsonl, write_results


def _per_class_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    classes: set[str] = {r["expected"] for r in rows} | {r["actual"] for r in rows}
    out: dict[str, dict[str, float]] = {}
    for cls in classes:
        tp = sum(1 for r in rows if r["expected"] == cls and r["actual"] == cls)
        fp = sum(1 for r in rows if r["expected"] != cls and r["actual"] == cls)
        fn = sum(1 for r in rows if r["expected"] == cls and r["actual"] != cls)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[cls] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}
    return out


def main() -> int:
    configure_logging()
    log = get_logger(__name__)

    golden = load_jsonl("router_golden.jsonl")
    intent_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    user_type_rows: list[dict[str, Any]] = []
    confusion: Counter = Counter()
    by_class_count: dict[str, int] = defaultdict(int)

    for ex in golden:
        msg = ex["message"]
        state = {"messages": [HumanMessage(content=msg)]}
        try:
            out = intent_node(state)
        except Exception as exc:  # surface but continue
            log.warning("router_eval.intent_failed", id=ex["id"], error=str(exc))
            continue
        actual_intent = out.get("intent", "unknown")
        actual_user_type = out.get("user_type", "unknown")
        actual_route = supervisor_route(out)  # type: ignore[arg-type]

        intent_rows.append({"expected": ex["expected_intent"], "actual": actual_intent})
        route_rows.append({"expected": ex["expected_route"], "actual": actual_route})
        user_type_rows.append(
            {"expected": ex["expected_user_type"], "actual": actual_user_type}
        )

        if ex["expected_intent"] != actual_intent:
            confusion[(ex["expected_intent"], actual_intent)] += 1
        by_class_count[ex["expected_intent"]] += 1

    if not intent_rows:
        print("no rows scored — likely an AWS / model-access issue. exiting.")
        return 1

    intent_acc = sum(1 for r in intent_rows if r["expected"] == r["actual"]) / len(intent_rows)
    route_acc = sum(1 for r in route_rows if r["expected"] == r["actual"]) / len(route_rows)
    user_type_acc = (
        sum(1 for r in user_type_rows if r["expected"] == r["actual"]) / len(user_type_rows)
    )

    payload = {
        "n": len(intent_rows),
        "intent_accuracy": round(intent_acc, 3),
        "route_accuracy": round(route_acc, 3),
        "user_type_accuracy": round(user_type_acc, 3),
        "per_intent": _per_class_metrics(intent_rows, "expected"),
        "confusion_pairs_top": [
            {"expected": k[0], "predicted": k[1], "count": v}
            for k, v in confusion.most_common(5)
        ],
    }
    out_path = write_results("router_eval", payload)

    banner("Router accuracy")
    print(f"  intent:    {intent_acc:.1%}")
    print(f"  route:     {route_acc:.1%}")
    print(f"  user_type: {user_type_acc:.1%}")
    print(f"\n  results → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
