"""Cost calculator tests — pure-Python."""

from __future__ import annotations

import pytest

from canaid.observability.cost import _normalize_model_key, cost_for, known_models


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", "anthropic.claude-sonnet-4-5"),
        ("anthropic.claude-haiku-4-5-20251001-v1:0", "anthropic.claude-haiku-4-5"),
        ("meta.llama3-3-70b-instruct-v1:0", "meta.llama3-3-70b-instruct"),
        ("us.amazon.nova-lite-v1:0", "amazon.nova-lite"),
        ("amazon.titan-embed-text-v2:0", "amazon.titan-embed-text"),
        ("us.anthropic.claude-opus-4-7-20260101-v1:0", "anthropic.claude-opus-4-7"),
    ],
)
def test_normalize_model_key(raw: str, expected: str) -> None:
    assert _normalize_model_key(raw) == expected


def test_cost_for_known_model() -> None:
    # 1M in + 1M out at Sonnet rates → $3 + $15 = $18
    assert cost_for("us.anthropic.claude-sonnet-4-5-20250929-v1:0", 1_000_000, 1_000_000) == pytest.approx(18.0)


def test_cost_for_haiku_is_cheaper() -> None:
    sonnet = cost_for("us.anthropic.claude-sonnet-4-5-20250929-v1:0", 1000, 1000)
    haiku = cost_for("us.anthropic.claude-haiku-4-5-20251001-v1:0", 1000, 1000)
    assert haiku < sonnet


def test_cost_for_zero_tokens() -> None:
    assert cost_for("us.anthropic.claude-sonnet-4-5-20250929-v1:0", 0, 0) == 0.0


def test_cost_for_unknown_model_returns_zero() -> None:
    assert cost_for("totally.fake-model-v9:0", 100, 100) == 0.0


def test_cost_for_negative_tokens_clamps_to_zero() -> None:
    assert cost_for("us.anthropic.claude-sonnet-4-5-20250929-v1:0", -1, 100) == 0.0


def test_known_models_includes_each_vendor() -> None:
    keys = known_models()
    vendors = {k.split(".", 1)[0] for k in keys}
    assert {"anthropic", "meta", "amazon"} <= vendors
