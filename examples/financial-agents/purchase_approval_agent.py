"""Sample agent: purchase auto-approval, gated by vsl-core through vsl-langgraph.

Demonstrates the two governance primitives that actually differ in
behavior, chained in one graph:

- A PreNode ("purchase-confidence"): soft, threshold-based -- a monitor's
  confidence signal must clear the robust Gamma threshold before the
  purchase can even be considered.
- An Invariant ("purchase-hard-cap"): non-bypassable -- the amount must
  never exceed a fixed cap, regardless of how confident the monitor is.
  Per vsl-core's spec, an Invariant violation has no Fallback and goes
  straight to a terminal/blocked state.

Run directly to see three scenarios (approved, blocked by low confidence,
blocked by the hard cap) exercised against a real compiled StateGraph:

    python examples/purchase_approval_agent.py
"""

from __future__ import annotations

import asyncio
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from vsl_core.constructs import Invariant, PreNode
from vsl_core.metrics import ROBUST_GAMMA_DEFAULT_THRESHOLD, AssuranceBasis, F2Modification, GammaEstimate

from vsl_langgraph import LangGraphAdapter, gated_node, route_on_denial

HARD_CAP = 1000.0

_ASSURANCE_BASIS = AssuranceBasis(f1_pre_commitment=True, f2_modification=F2Modification.FULL)


class PurchaseState(TypedDict, total=False):
    amount: float
    confidence: float
    approved: bool
    confidence_denial: Any
    cap_denial: Any


async def _confidence_monitor(state: PurchaseState) -> GammaEstimate:
    return GammaEstimate(gamma_hat=state["confidence"])


async def _under_hard_cap(state: PurchaseState) -> bool:
    return state["amount"] <= HARD_CAP


def build_agent():
    adapter = LangGraphAdapter()

    confidence_pre_node = PreNode(
        name="purchase-confidence",
        description="Requires the monitor's confidence signal to clear the robust Gamma threshold before a purchase is auto-approved.",
        monitor=_confidence_monitor,
        assurance_basis=_ASSURANCE_BASIS,
        gamma_threshold=ROBUST_GAMMA_DEFAULT_THRESHOLD,
    )
    confidence_gate = adapter.compile_pre_node(confidence_pre_node)

    hard_cap_invariant = Invariant(
        name="purchase-hard-cap",
        description=f"Amount must never exceed the hard cap of {HARD_CAP}, regardless of confidence -- non-bypassable.",
        rule=_under_hard_cap,
        assurance_basis=_ASSURANCE_BASIS,
    )
    cap_gate = adapter.compile_invariant(hard_cap_invariant)

    def approve_purchase(state: PurchaseState) -> dict:
        return {"approved": True}

    graph = StateGraph(PurchaseState)
    graph.add_node("confidence_check", gated_node(confidence_gate, lambda state: {}, denial_key="confidence_denial"))
    graph.add_node("cap_check", gated_node(cap_gate, approve_purchase, denial_key="cap_denial"))
    graph.add_node("blocked_low_confidence", lambda state: {})
    graph.add_node("blocked_over_cap", lambda state: {})

    graph.add_edge(START, "confidence_check")
    graph.add_conditional_edges(
        "confidence_check",
        route_on_denial("blocked_low_confidence", "cap_check", denial_key="confidence_denial"),
        # path_map is optional -- route_on_denial's two args already are the
        # possible destinations -- but naming them explicitly here lets
        # get_graph().draw_mermaid() render both branches instead of guessing.
        {"blocked_low_confidence": "blocked_low_confidence", "cap_check": "cap_check"},
    )
    graph.add_conditional_edges(
        "cap_check",
        route_on_denial("blocked_over_cap", END, denial_key="cap_denial"),
        {"blocked_over_cap": "blocked_over_cap", END: END},
    )
    graph.add_edge("blocked_low_confidence", END)
    graph.add_edge("blocked_over_cap", END)

    return graph.compile()


async def _run_scenarios() -> None:
    agent = build_agent()

    scenarios = [
        ("approved (high confidence, under cap)", {"amount": 250.0, "confidence": 5.0}),
        ("blocked by low confidence (under cap)", {"amount": 250.0, "confidence": 0.2}),
        ("blocked by hard cap (high confidence, over cap)", {"amount": 5000.0, "confidence": 5.0}),
    ]

    for label, purchase in scenarios:
        result = await agent.ainvoke(purchase)
        print(f"\n{label}")
        print(f"  input:  {purchase}")
        print(f"  approved={result.get('approved')} confidence_denial={result.get('confidence_denial')!r} cap_denial={result.get('cap_denial')!r}")


if __name__ == "__main__":
    asyncio.run(_run_scenarios())
