"""In-memory mock CRM for the demo.

Loads accounts and orders from `data/crm/*.jsonl` once, exposes pure
read-only query functions. The state is intentionally global-singleton
so multiple tool calls in one turn share the same view.

Phase 9 will swap this for DynamoDB tables. The ``MockCRM`` API is the
contract — ``lookup_tools.py`` and any other consumer should depend on
that, not on JSONL files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
ACCOUNTS_PATH = REPO_ROOT / "data" / "crm" / "accounts.jsonl"
ORDERS_PATH = REPO_ROOT / "data" / "crm" / "orders.jsonl"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


@dataclass
class MockCRM:
    accounts: list[dict[str, Any]] = field(default_factory=list)
    orders: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls) -> MockCRM:
        return cls(
            accounts=_load_jsonl(ACCOUNTS_PATH),
            orders=_load_jsonl(ORDERS_PATH),
        )

    # ---- account lookups ------------------------------------------
    def find_account(
        self,
        *,
        account_id: str | None = None,
        email: str | None = None,
        facility_name: str | None = None,
    ) -> dict[str, Any] | None:
        if account_id:
            return self._first(
                a for a in self.accounts if a["account_id"].lower() == account_id.lower()
            )
        if email:
            target = email.lower().strip()
            return self._first(
                a for a in self.accounts
                if a.get("primary_contact", {}).get("email", "").lower() == target
            )
        if facility_name:
            target = facility_name.lower().strip()
            for a in self.accounts:
                name = a.get("facility_name", "").lower()
                if target in name or name in target:
                    return a
        return None

    # ---- order lookups --------------------------------------------
    def get_orders(self, account_id: str, limit: int = 5) -> list[dict[str, Any]]:
        rows = [o for o in self.orders if o["account_id"] == account_id]
        rows.sort(key=lambda o: o.get("ordered_at", ""), reverse=True)
        return rows[:limit]

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        return self._first(
            o for o in self.orders if o["order_id"].lower() == order_id.lower()
        )

    @staticmethod
    def _first(it):
        try:
            return next(iter(it))
        except StopIteration:
            return None


@lru_cache(maxsize=1)
def get_crm() -> MockCRM:
    return MockCRM.load()
