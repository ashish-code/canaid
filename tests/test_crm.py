"""Tests for the mock CRM and the LangChain tool wrappers."""

from __future__ import annotations

from canaid.tools.crm import get_crm
from canaid.tools.lookup_tools import (
    get_order_status,
    get_recent_orders,
    lookup_account,
)


def test_crm_loads_accounts_and_orders() -> None:
    crm = get_crm()
    assert len(crm.accounts) >= 5
    assert len(crm.orders) >= 10


def test_lookup_by_account_id() -> None:
    out = lookup_account.invoke({"account_id": "ACC-1001"})
    assert out["found"] is True
    assert out["account_id"] == "ACC-1001"
    assert out["facility_name"].startswith("Northbridge")


def test_lookup_by_email() -> None:
    out = lookup_account.invoke({"email": "marc.tremblay@riverdalegh.example"})
    assert out["found"] is True
    assert out["account_id"] == "ACC-1002"


def test_lookup_by_partial_facility_name() -> None:
    out = lookup_account.invoke({"facility_name": "lakeshore"})
    assert out["found"] is True
    assert out["account_id"] == "ACC-1005"


def test_lookup_misses_return_found_false() -> None:
    out = lookup_account.invoke({"account_id": "ACC-9999"})
    assert out == {"found": False}


def test_lookup_redacts_phone() -> None:
    out = lookup_account.invoke({"account_id": "ACC-1001"})
    phone = out["primary_contact"]["phone"]
    assert phone is not None
    assert "***" in phone
    # Should not contain the trailing 4 digits in clear.
    assert "0142" not in phone


def test_lookup_returns_email_domain_only() -> None:
    out = lookup_account.invoke({"account_id": "ACC-1001"})
    pc = out["primary_contact"]
    assert pc["email_domain"] == "northbridgefh.example"
    # Full email must NOT round-trip through the tool.
    assert "priya.desai" not in str(out)


def test_get_recent_orders_sorted_desc() -> None:
    out = get_recent_orders.invoke({"account_id": "ACC-1001", "limit": 10})
    orders = out["orders"]
    assert orders, "expected orders for ACC-1001"
    dates = [o["ordered_at"] for o in orders]
    assert dates == sorted(dates, reverse=True)


def test_get_recent_orders_limit_respected() -> None:
    out = get_recent_orders.invoke({"account_id": "ACC-1002", "limit": 2})
    assert len(out["orders"]) <= 2


def test_get_order_status_found() -> None:
    out = get_order_status.invoke({"order_id": "SO-2026-00742"})
    assert out["found"] is True
    assert out["status"] == "delivered"


def test_get_order_status_missing() -> None:
    out = get_order_status.invoke({"order_id": "SO-9999-99999"})
    assert out == {"found": False}


def test_account_on_hold_surfaces_reason() -> None:
    out = lookup_account.invoke({"account_id": "ACC-1004"})
    assert out["status"] == "on_hold"
    assert "MDEL" in (out.get("hold_reason") or "")
