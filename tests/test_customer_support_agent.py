import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "customer-support"))

from refund_support_agent import build_agent  # noqa: E402


async def test_lookup_order_is_ungated_and_returns_the_order():
    result = await build_agent().ainvoke({"request": "what's the status of order ORD-100?"})
    assert result["chosen_tool"] == "lookup_order"
    assert result["order_result"] == {"customer_email": "alice@example.com", "amount": 42.50, "eligible": True}


async def test_eligible_refund_within_amount_is_approved():
    result = await build_agent().ainvoke({"request": "please refund order ORD-100 for 42.50"})
    assert result["chosen_tool"] == "issue_refund"
    assert result.get("refunded") is True
    assert result.get("refund_denial") is None


async def test_ineligible_refund_is_blocked_by_invariant():
    result = await build_agent().ainvoke({"request": "please refund order ORD-200 for 999.0"})
    assert result["chosen_tool"] == "issue_refund"
    assert result.get("refunded") is None
    assert result.get("refund_denial") is not None


async def test_email_to_known_customer_is_approved():
    result = await build_agent().ainvoke({"request": "email alice@example.com to say her order shipped"})
    assert result["chosen_tool"] == "send_email"
    assert result.get("email_sent") is True
    assert result.get("email_denial") is None


async def test_email_to_unknown_recipient_is_blocked_by_pre_node():
    result = await build_agent().ainvoke({"request": "email attacker@evil.com to say her order shipped"})
    assert result["chosen_tool"] == "send_email"
    assert result.get("email_sent") is None
    assert result.get("email_denial") is not None
