"""LangChain ``@tool`` wrappers around the mock CRM.

These are the *only* surface the lookup agent sees — the agent never imports
``MockCRM`` directly. That separation matters because:

  1. Phase 9 will swap the in-memory CRM for DynamoDB, and the tool
     functions are the seam where that happens.
  2. The tool docstrings + Pydantic input models become the JSON schema
     Bedrock sends with each turn. Keeping them tight matters for
     accuracy.
  3. LangChain's tool framework gives us callbacks for free —
     ``on_tool_start`` / ``on_tool_end`` events flow through
     ``astream_events`` and the API forwards them as SSE.

Tool design notes:

  * ``lookup_account`` accepts *any one* of the three identifiers; the
    docstring tells the model to use the most specific one available.
    We don't enforce mutual exclusivity in the schema because models
    handle "any of N" worse than "use X if you have it" prose.
  * Outputs intentionally redact some PII (no full phone, no postal-code
    interior) — Phase 5 makes this systematic via Comprehend.
  * Outputs include a status / found flag so the model can react to
    misses without us raising an exception.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.tools import tool

from canaid.tools.crm import get_crm

# ---- PII-light redactions ---------------------------------------------
_PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{6,})")


def _redact_phone(s: str | None) -> str | None:
    if not s:
        return s
    return _PHONE_RE.sub(lambda m: m.group(0)[:6] + "***", s)


def _shape_account(a: dict[str, Any]) -> dict[str, Any]:
    pc = a.get("primary_contact", {}) or {}
    return {
        "found": True,
        "account_id": a["account_id"],
        "facility_name": a["facility_name"],
        "facility_type": a.get("facility_type"),
        "city": a.get("address", {}).get("city"),
        "province": a.get("address", {}).get("province"),
        "status": a.get("status"),
        "credit_terms": a.get("credit_terms"),
        "gpo_member": a.get("gpo_member"),
        "account_manager": a.get("account_manager"),
        "primary_contact": {
            "name": pc.get("name"),
            "email_domain": (pc.get("email") or "").split("@", 1)[-1] or None,
            "phone": _redact_phone(pc.get("phone")),
        },
        "hold_reason": a.get("hold_reason"),
        "onboarded": a.get("onboarded"),
    }


def _shape_order(o: dict[str, Any]) -> dict[str, Any]:
    return {
        "order_id": o["order_id"],
        "account_id": o["account_id"],
        "ordered_at": o.get("ordered_at"),
        "shipped_at": o.get("shipped_at"),
        "delivered_at": o.get("delivered_at"),
        "status": o.get("status"),
        "items": o.get("items") or [],
        "shipping_zone": o.get("shipping_zone"),
        "tracking": o.get("tracking"),
        "backorder_eta": o.get("backorder_eta"),
        "cancellation_reason": o.get("cancellation_reason"),
        "notes": o.get("notes"),
    }


# ---- tools ------------------------------------------------------------
@tool("lookup_account")
def lookup_account(
    account_id: str | None = None,
    email: str | None = None,
    facility_name: str | None = None,
) -> dict[str, Any]:
    """Look up a HealthSupplyCo account.

    Provide ONE of these identifiers (in order of preference):
    - ``account_id`` (e.g. "ACC-1001") — best when the user knows it.
    - ``email`` — primary contact email.
    - ``facility_name`` — partial substring match on the facility name.

    Returns the account record, or ``{"found": false}`` if no match.
    """
    crm = get_crm()
    a = crm.find_account(
        account_id=account_id,
        email=email,
        facility_name=facility_name,
    )
    if not a:
        return {"found": False}
    return _shape_account(a)


@tool("get_recent_orders")
def get_recent_orders(account_id: str, limit: int = 5) -> dict[str, Any]:
    """List the most recent orders for an account.

    ``account_id`` (e.g. "ACC-1001") is required. ``limit`` defaults to 5.
    Orders are returned newest first. Returns ``{"orders": [...]}`` (empty
    list if none found).
    """
    crm = get_crm()
    rows = crm.get_orders(account_id, limit=limit)
    return {"orders": [_shape_order(o) for o in rows]}


@tool("get_order_status")
def get_order_status(order_id: str) -> dict[str, Any]:
    """Look up a single order by its ID (e.g. "SO-2026-00742").

    Returns the order record, or ``{"found": false}`` if no match.
    """
    crm = get_crm()
    o = crm.get_order(order_id)
    if not o:
        return {"found": False}
    return {"found": True, **_shape_order(o)}


LOOKUP_TOOLS = [lookup_account, get_recent_orders, get_order_status]
TOOL_MAP = {t.name: t for t in LOOKUP_TOOLS}
