from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from vsl_core.constructs import Invariant, PreNode
from vsl_core.exceptions import InvariantViolation
from vsl_core.metrics import GammaEstimate

from vsl_langgraph import LangGraphAdapter, gated_node, route_on_denial


class GraphState(TypedDict, total=False):
    gamma: float
    ran_side_effect: bool
    vsl_denial: Any


async def _monitor(candidate_input: GraphState) -> GammaEstimate:
    return GammaEstimate(gamma_hat=candidate_input["gamma"])


def _build_graph():
    pre_node = PreNode(
        name="integration-test-pre-node",
        description="Requires gamma above 1.0 before the guarded node runs.",
        monitor=_monitor,
        gamma_threshold=1.0,
    )
    gate = LangGraphAdapter().compile_pre_node(pre_node)

    def guarded(state: GraphState) -> dict:
        return {"ran_side_effect": True}

    graph = StateGraph(GraphState)
    graph.add_node("guarded", gated_node(gate, guarded))
    graph.add_node("blocked", lambda state: {})
    graph.add_node("allowed", lambda state: {})
    graph.add_edge(START, "guarded")
    graph.add_conditional_edges("guarded", route_on_denial("blocked", "allowed"))
    graph.add_edge("blocked", END)
    graph.add_edge("allowed", END)
    return graph.compile()


async def test_sufficient_gamma_runs_node_and_routes_to_allowed():
    result = await _build_graph().ainvoke({"gamma": 5.0})
    assert result.get("ran_side_effect") is True
    assert result.get("vsl_denial") is None


async def test_insufficient_gamma_skips_node_and_routes_to_blocked():
    result = await _build_graph().ainvoke({"gamma": 0.1})
    assert result.get("ran_side_effect") is None
    assert result.get("vsl_denial") is not None


class InvariantState(TypedDict, total=False):
    allowed: bool
    ran_side_effect: bool
    vsl_denial: Any


async def _rule(candidate_input: InvariantState) -> bool:
    return candidate_input["allowed"]


def _build_invariant_graph():
    invariant = Invariant(
        name="integration-test-invariant",
        description="Requires state['allowed'] to be True before the guarded node runs.",
        rule=_rule,
    )
    gate = LangGraphAdapter().compile_invariant(invariant)

    def guarded(state: InvariantState) -> dict:
        return {"ran_side_effect": True}

    graph = StateGraph(InvariantState)
    graph.add_node("guarded", gated_node(gate, guarded))
    graph.add_node("blocked", lambda state: {})
    graph.add_node("allowed", lambda state: {})
    graph.add_edge(START, "guarded")
    graph.add_conditional_edges("guarded", route_on_denial("blocked", "allowed"))
    graph.add_edge("blocked", END)
    graph.add_edge("allowed", END)
    return graph.compile()


async def test_passing_invariant_runs_node_and_routes_to_allowed():
    result = await _build_invariant_graph().ainvoke({"allowed": True})
    assert result.get("ran_side_effect") is True
    assert result.get("vsl_denial") is None


async def test_violated_invariant_skips_node_and_routes_to_blocked():
    result = await _build_invariant_graph().ainvoke({"allowed": False})
    assert result.get("ran_side_effect") is None
    denial = result.get("vsl_denial")
    assert isinstance(denial, InvariantViolation)


import pytest  # noqa: E402

from vsl_core.ledger import Enforcement, VerbaLedger  # noqa: E402


@pytest.mark.parametrize("allowed", [True, False])
async def test_ledger_shows_the_gate_held_before_the_node(allowed):
    ledger = VerbaLedger()

    async def rule(state: InvariantState) -> bool:
        ledger.write_gate_decision(identity_key="refund-agent", primitive_name="refund-ok",
                                   allowed=state["allowed"], decision_id="d1")
        return state["allowed"]

    def guarded(state: InvariantState) -> dict:
        ledger.write_action(identity_key="refund-agent", action_name="issue_refund", decision_id="d1")
        return {"ran_side_effect": True}

    gate = LangGraphAdapter().compile_invariant(Invariant(name="refund-ok", description="d", rule=rule))
    graph = StateGraph(InvariantState)
    graph.add_node("guarded", gated_node(gate, guarded))
    graph.add_node("blocked", lambda state: {})
    graph.add_node("allowed", lambda state: {})
    graph.add_edge(START, "guarded")
    graph.add_conditional_edges("guarded", route_on_denial("blocked", "allowed"))
    graph.add_edge("blocked", END)
    graph.add_edge("allowed", END)

    result = await graph.compile().ainvoke({"allowed": allowed})
    assert bool(result.get("ran_side_effect")) is allowed
    assert ledger.enforcement("d1") == Enforcement.BEFORE_ACTION


def test_compiling_gates_without_a_basis_warns_about_nothing():
    import warnings

    async def monitor(_state) -> GammaEstimate:
        return GammaEstimate(gamma_hat=2.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        LangGraphAdapter().compile_invariant(Invariant(name="i", description="d", rule=_rule))
        LangGraphAdapter().compile_pre_node(PreNode(name="p", description="d", monitor=monitor))
