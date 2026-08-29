"""Wiring helpers for calling a compiled VSL gate from inside a LangGraph
node and routing around a denial with `add_conditional_edges`.

vsl-core does not intercept anything on its own (see
vsl-core/docs/building-with-vsl-core.md, "know what's automatic and what
you write yourself"): a compiled gate only ever raises or returns, on
demand, when called. `gated_node` is that call site, made reusable for a
LangGraph StateGraph -- it calls the gate immediately before the wrapped
node would run (preserving F1 pre-commitment: the gate completes strictly
before the guarded node's side effect), and on denial returns a state
update instead of letting the exception propagate, so a conditional edge
can route to a blocked/terminal node in the normal LangGraph way rather
than crashing the graph run.
"""

from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable, Mapping

from vsl_core.conformance.protocol import CompiledGate
from vsl_core.exceptions import AutomationDeniedException

StateNode = Callable[[Mapping[str, Any]], Any]

DEFAULT_DENIAL_KEY = "vsl_denial"


def gated_node(
    gate: CompiledGate,
    node: StateNode,
    *,
    candidate_input_fn: Callable[[Mapping[str, Any]], Any] | None = None,
    denial_key: str = DEFAULT_DENIAL_KEY,
) -> Callable[[Mapping[str, Any]], Awaitable[dict[str, Any]]]:
    """Wrap `node` so `gate` is checked immediately before it runs.

    `candidate_input_fn` extracts whatever the gate should evaluate from
    graph state; defaults to passing the whole state through unchanged.

    On denial, `node` is never called and the returned state update sets
    `denial_key` to the caught exception -- pair with `route_on_denial`
    (or your own router reading `denial_key`) in `add_conditional_edges` to
    send the run to a blocked/terminal node.

    Does not consume `PreNode.fallback` -- a denial raises once,
    immediately, with no retry loop, regardless of `fallback.max_retries`.
    `LangGraphAdapter.compile_pre_node` warns if a non-default `Fallback`
    is set. See docs/dos-and-donts.md.
    """
    extract = candidate_input_fn or (lambda state: state)

    async def wrapped(state: Mapping[str, Any]) -> dict[str, Any]:
        try:
            await gate(extract(state))
        except AutomationDeniedException as exc:
            return {denial_key: exc}

        result = node(state)
        if inspect.isawaitable(result):
            result = await result
        return dict(result) if result else {}

    return wrapped


def route_on_denial(
    blocked_node: str,
    allowed_node: str,
    *,
    denial_key: str = DEFAULT_DENIAL_KEY,
) -> Callable[[Mapping[str, Any]], str]:
    """Build a router for `add_conditional_edges` matching `gated_node`'s
    convention: routes to `blocked_node` if `denial_key` is set in state
    (truthy), otherwise to `allowed_node`.
    """

    def router(state: Mapping[str, Any]) -> str:
        return blocked_node if state.get(denial_key) is not None else allowed_node

    return router
