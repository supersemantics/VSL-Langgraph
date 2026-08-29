"""Sample agent: governance on a conversational agent's OUTPUT, not a
backend action -- every other example in this repo gates a side effect
(a purchase, a refund, an email, a publish). This one gates what the
agent is about to *say*.

    Step                        VSL role
    ---------------------------- --------------------------------------------
    draft a reply                plain node, no gate needed (drafting has
                                  no side effect -- the gates below decide
                                  whether it's ever actually sent)
    check for an unverified      Invariant -- non-bypassable. A promise
    guarantee                    the company can't back is not something a
                                  confidence threshold should be able to
                                  override.
    check confidence it          PreNode -- soft, threshold-based. Below
    resolves the issue           threshold routes to a human, not to a
                                  dead end -- route_on_denial's "blocked"
                                  destination is a real fallback action
                                  here, not a no-op.

Requires ANTHROPIC_API_KEY to run live:

    python examples/chat-conversation/chat_reply_agent.py "your message"

Falls back to three deterministic offline scenarios (sent / blocked by
the guarantee Invariant / handed off on low confidence) with zero keys set:

    python examples/chat-conversation/chat_reply_agent.py
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from vsl_core.constructs import Invariant, PreNode
from vsl_core.metrics import ROBUST_GAMMA_DEFAULT_THRESHOLD, AssuranceBasis, F2Modification, GammaEstimate

from vsl_langgraph import LangGraphAdapter, gated_node, route_on_denial

LIVE = bool(os.environ.get("ANTHROPIC_API_KEY"))

if LIVE:
    from chat_providers import draft_reply, judge_confidence, judge_guarantee
else:
    from chat_fake_providers import draft_reply, judge_confidence, judge_guarantee, set_scenario

_ASSURANCE_BASIS = AssuranceBasis(f1_pre_commitment=True, f2_modification=F2Modification.FULL)
_FIRST_LINE = re.compile(r"\s*(\S+)")


class ChatState(TypedDict, total=False):
    customer_message: str
    draft: str
    guarantee_denial: Any
    confidence_denial: Any
    sent_reply: str
    handed_off: bool


async def draft_node(state: ChatState) -> dict:
    return {"draft": await draft_reply(state["customer_message"])}


async def _no_unverified_guarantee(state: ChatState) -> bool:
    """Invariant.rule. Fails closed: an unparseable verdict is FAIL, not PASS."""
    verdict = await judge_guarantee(state["draft"])
    match = _FIRST_LINE.match(verdict)
    return bool(match) and match.group(1).strip().upper() == "PASS"


guarantee_invariant = Invariant(
    name="no-unverified-guarantee",
    description="A reply promising a specific legal/medical/financial outcome not backed by policy must never be sent -- non-bypassable.",
    rule=_no_unverified_guarantee,
    assurance_basis=_ASSURANCE_BASIS,
)


async def _resolves_the_issue(state: ChatState) -> GammaEstimate:
    """PreNode.monitor. Fails closed: an unparseable rating becomes gamma_hat=0.0."""
    rating = await judge_confidence(state["draft"], state["customer_message"])
    match = _FIRST_LINE.match(rating)
    try:
        gamma_hat = float(match.group(1)) if match else 0.0
    except ValueError:
        gamma_hat = 0.0
    return GammaEstimate(gamma_hat=gamma_hat)


confidence_pre_node = PreNode(
    name="reply-resolves-issue",
    description="Requires confidence the draft fully resolves the customer's issue before auto-sending it; otherwise hands off to a human.",
    monitor=_resolves_the_issue,
    assurance_basis=_ASSURANCE_BASIS,
    gamma_threshold=ROBUST_GAMMA_DEFAULT_THRESHOLD,
)


def send_reply(state: ChatState) -> dict:
    return {"sent_reply": state["draft"]}


def handoff_to_human(state: ChatState) -> dict:
    return {"handed_off": True}


def build_agent():
    adapter = LangGraphAdapter()
    guarantee_gate = adapter.compile_invariant(guarantee_invariant)
    confidence_gate = adapter.compile_pre_node(confidence_pre_node)

    graph = StateGraph(ChatState)
    graph.add_node("draft", draft_node)
    graph.add_node("guarantee_check", gated_node(guarantee_gate, lambda state: {}, denial_key="guarantee_denial"))
    graph.add_node("confidence_check", gated_node(confidence_gate, send_reply, denial_key="confidence_denial"))
    graph.add_node("handoff_to_human", handoff_to_human)

    graph.add_edge(START, "draft")
    graph.add_edge("draft", "guarantee_check")
    graph.add_conditional_edges(
        "guarantee_check",
        route_on_denial("handoff_to_human", "confidence_check", denial_key="guarantee_denial"),
        {"handoff_to_human": "handoff_to_human", "confidence_check": "confidence_check"},
    )
    graph.add_conditional_edges(
        "confidence_check",
        route_on_denial("handoff_to_human", END, denial_key="confidence_denial"),
        {"handoff_to_human": "handoff_to_human", END: END},
    )
    graph.add_edge("handoff_to_human", END)

    return graph.compile()


def _print_result(message: str, result: dict) -> None:
    print(f"\ncustomer: {message!r}")
    print(f"  draft: {result.get('draft')!r}")
    if result.get("sent_reply") is not None:
        print(f"  SENT: {result['sent_reply']!r}")
    if result.get("guarantee_denial") is not None:
        print(f"  HANDED OFF (no-unverified-guarantee Invariant): {result['guarantee_denial']}")
    elif result.get("confidence_denial") is not None:
        print(f"  HANDED OFF (reply-resolves-issue PreNode): {result['confidence_denial']}")


async def main() -> None:
    agent = build_agent()

    if LIVE:
        message = sys.argv[1] if len(sys.argv) > 1 else "My order hasn't arrived and it's been two weeks, what happens now?"
        _print_result(message, await agent.ainvoke({"customer_message": message}))
        return

    print(
        "No ANTHROPIC_API_KEY found -- running three offline scenarios against chat_fake_providers.py's canned "
        "responses instead of live calls. Set ANTHROPIC_API_KEY and rerun to see this against a real model.\n"
    )
    for scenario in ("resolved", "unverified_guarantee", "low_confidence"):
        set_scenario(scenario)
        print(f"=== scenario: {scenario} ===")
        result = await agent.ainvoke({"customer_message": "My order hasn't arrived and it's been two weeks, what happens now?"})
        _print_result("(see draft below)", result)


if __name__ == "__main__":
    asyncio.run(main())
