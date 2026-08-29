"""Sample agent: a real dynamic tool-choosing agent, gated by vsl-core
through vsl-langgraph.

This is `vsl-core`'s own canonical worked example (`docs/building-with-
vsl-core.md`'s "customer support bot that can look up orders, issue
refunds, and send emails") actually built as a LangGraph agent -- and
unlike every other example in this repo, an LLM decides which node runs
next. `financial-agents/purchase_approval_agent.py` and
`marketing/research_publish_agent.py` both have LLM calls, but a *fixed*
graph topology (the same nodes run in the same order every time). Here,
`decide` calls an LLM to pick one of three tools from the customer's
request, and a plain LangGraph conditional edge (`route_to_tool`) routes
to whichever node it chose -- separate and prior to any vsl-core gate.

Classification, straight from vsl-core's own worked example:

    Tool            vsl-core classification         VSL role
    --------------- ------------------------------- ---------------------------------
    lookup_order    read-only, no side effect       ungated
    issue_refund    irreversibility boundary         Invariant (non-bypassable)
    send_email      external side effect             PreNode (soft, threshold-based)

Runs live with ANTHROPIC_API_KEY set:

    python examples/customer-support/refund_support_agent.py "please refund order ORD-100 for 42.50"

Falls back to five deterministic offline scenarios covering both gates
and all three tools with zero keys set:

    python examples/customer-support/refund_support_agent.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from vsl_core.constructs import Invariant, PreNode
from vsl_core.metrics import ROBUST_GAMMA_DEFAULT_THRESHOLD, AssuranceBasis, F2Modification, GammaEstimate

from vsl_langgraph import LangGraphAdapter, gated_node, route_on_denial

LIVE = bool(os.environ.get("ANTHROPIC_API_KEY"))

if LIVE:
    from support_providers import decide
else:
    from support_fake_providers import decide

_ASSURANCE_BASIS = AssuranceBasis(f1_pre_commitment=True, f2_modification=F2Modification.FULL)

# vsl-core's own domain.py/tools.py scaffold pattern: plain data, no
# framework or model-provider awareness.
ORDERS: dict[str, dict] = {
    "ORD-100": {"customer_email": "alice@example.com", "amount": 42.50, "eligible": True},
    "ORD-200": {"customer_email": "bob@example.com", "amount": 999.00, "eligible": False},
}


class SupportState(TypedDict, total=False):
    request: str
    chosen_tool: str
    order_id: str | None
    amount: float | None
    recipient: str | None
    message: str | None
    order_result: dict | None
    refunded: bool
    email_sent: bool
    refund_denial: Any
    email_denial: Any


def _parse_decision(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    return fields


async def decide_node(state: SupportState) -> dict:
    fields = _parse_decision(await decide(state["request"]))
    return {
        "chosen_tool": fields.get("tool", ""),
        "order_id": fields.get("order_id") or None,
        "amount": float(fields["amount"]) if fields.get("amount") else None,
        "recipient": fields.get("recipient") or None,
        "message": fields.get("message") or None,
    }


def route_to_tool(state: SupportState) -> str:
    """Plain LangGraph dynamic routing -- distinct from vsl-langgraph's
    route_on_denial below, which routes on a *gate's* decision, not the
    LLM's tool choice. This router runs before any gate is even reached.
    """
    tool = state.get("chosen_tool")
    return tool if tool in ("lookup_order", "issue_refund", "send_email") else "unknown_tool"


def lookup_order_node(state: SupportState) -> dict:
    return {"order_result": ORDERS.get(state["order_id"])}


async def _refund_eligible(state: SupportState) -> bool:
    order = ORDERS.get(state["order_id"]) if state.get("order_id") else None
    if order is None or not order["eligible"]:
        return False
    return state.get("amount") is not None and state["amount"] <= order["amount"]


refund_invariant = Invariant(
    name="refund-eligibility",
    description="Never issue a refund without confirmed eligibility and an amount within the order's own amount -- non-bypassable.",
    rule=_refund_eligible,
    assurance_basis=_ASSURANCE_BASIS,
)


async def _email_recipient_in_scope(state: SupportState) -> GammaEstimate:
    known_emails = {order["customer_email"] for order in ORDERS.values()}
    in_scope = state.get("recipient") in known_emails
    return GammaEstimate(gamma_hat=2.0 if in_scope else 0.1)


email_pre_node = PreNode(
    name="email-recipient-scope",
    description="Requires the email recipient to be a known customer address before send_email fires.",
    monitor=_email_recipient_in_scope,
    assurance_basis=_ASSURANCE_BASIS,
    gamma_threshold=ROBUST_GAMMA_DEFAULT_THRESHOLD,
)


def do_issue_refund(state: SupportState) -> dict:
    return {"refunded": True}


def do_send_email(state: SupportState) -> dict:
    return {"email_sent": True}


def build_agent():
    adapter = LangGraphAdapter()
    refund_gate = adapter.compile_invariant(refund_invariant)
    email_gate = adapter.compile_pre_node(email_pre_node)

    graph = StateGraph(SupportState)
    graph.add_node("decide", decide_node)
    graph.add_node("lookup_order", lookup_order_node)
    graph.add_node("issue_refund", gated_node(refund_gate, do_issue_refund, denial_key="refund_denial"))
    graph.add_node("send_email", gated_node(email_gate, do_send_email, denial_key="email_denial"))
    graph.add_node("blocked_refund", lambda state: {})
    graph.add_node("blocked_email", lambda state: {})
    graph.add_node("unknown_tool", lambda state: {})

    graph.add_edge(START, "decide")
    graph.add_conditional_edges(
        "decide",
        route_to_tool,
        {"lookup_order": "lookup_order", "issue_refund": "issue_refund", "send_email": "send_email", "unknown_tool": "unknown_tool"},
    )
    graph.add_edge("lookup_order", END)
    graph.add_conditional_edges(
        "issue_refund",
        route_on_denial("blocked_refund", END, denial_key="refund_denial"),
        {"blocked_refund": "blocked_refund", END: END},
    )
    graph.add_conditional_edges(
        "send_email",
        route_on_denial("blocked_email", END, denial_key="email_denial"),
        {"blocked_email": "blocked_email", END: END},
    )
    graph.add_edge("blocked_refund", END)
    graph.add_edge("blocked_email", END)
    graph.add_edge("unknown_tool", END)

    return graph.compile()


def _print_result(request: str, result: dict) -> None:
    print(f"\nrequest: {request!r}")
    print(f"  chosen_tool={result.get('chosen_tool')!r}")
    if result.get("order_result") is not None:
        print(f"  order_result={result['order_result']}")
    if result.get("refunded"):
        print("  refunded=True")
    if result.get("refund_denial") is not None:
        print(f"  BLOCKED (refund-eligibility Invariant): {result['refund_denial']}")
    if result.get("email_sent"):
        print("  email_sent=True")
    if result.get("email_denial") is not None:
        print(f"  BLOCKED (email-recipient-scope PreNode): {result['email_denial']}")


async def main() -> None:
    agent = build_agent()

    if LIVE:
        request = sys.argv[1] if len(sys.argv) > 1 else "please refund order ORD-100 for 42.50"
        _print_result(request, await agent.ainvoke({"request": request}))
        return

    print(
        "No ANTHROPIC_API_KEY found -- running five offline scenarios against support_fake_providers.py's "
        "deterministic tool-choice stand-in instead of a live decide() call. Set ANTHROPIC_API_KEY "
        "and rerun to see an LLM actually choose the tool.\n"
    )
    scenarios = [
        "what's the status of order ORD-100?",
        "please refund order ORD-100 for 42.50",
        "please refund order ORD-200 for 999.0",
        "email alice@example.com to say her order shipped",
        "email attacker@evil.com to say her order shipped",
    ]
    for request in scenarios:
        _print_result(request, await agent.ainvoke({"request": request}))


if __name__ == "__main__":
    asyncio.run(main())
