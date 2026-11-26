"""LangGraphAdapter: a VSLAdapter conformant with vsl-core's conformance
contract, intended for use inside LangGraph nodes.

LangGraph has no dedicated guardrail/interception type -- a StateGraph node
is just a plain sync/async callable, and conditional edges are plain router
functions evaluated after a node returns. There is therefore nothing
LangGraph-specific to compile a PreNode/Invariant *into*: the compiled gate
is the same framework-agnostic async callable vsl-core's own
PlainPythonReferenceAdapter produces, called directly inside a node body
before the side effect it guards. What this package adds on top (see
`integration.py`) is the wiring convention for calling that gate from a
node and routing around a denial with `add_conditional_edges`, not a
different gate-compilation strategy.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

from vsl_core.constructs import Fallback, Invariant, PreNode
from vsl_core.conformance.protocol import CompiledGate
from vsl_core.exceptions import AutomationDeniedException, InvariantViolation

_DEFAULT_FALLBACK = Fallback()

class LangGraphAdapter:
    def compile_pre_node(self, pre_node: PreNode) -> CompiledGate:
        if pre_node.fallback != _DEFAULT_FALLBACK:
            warnings.warn(
                f"PreNode '{pre_node.name}' sets a non-default Fallback "
                f"({pre_node.fallback!r}), but gated_node does not consume "
                "Fallback -- a denial raises once, immediately, with no "
                "retry loop. See docs/dos-and-donts.md if you need retry "
                "behavior; you must implement it yourself around the "
                "compiled gate.",
                stacklevel=2,
            )

        async def gate(candidate_input: Any) -> None:
            estimate = await pre_node.monitor(candidate_input)
            if not estimate.sufficient(pre_node.gamma_threshold):
                raise AutomationDeniedException(
                    reason=f"Pre-node '{pre_node.name}' insufficient Gamma",
                    identity_key="vsl-langgraph",
                )

        return gate

    def compile_invariant(self, invariant: Invariant) -> CompiledGate:
        async def gate(candidate_input: Any) -> None:
            holds = await invariant.rule(candidate_input)
            if not holds:
                reason = f"Invariant '{invariant.name}' violated"
                terminal_state_name = invariant.on_violation.name if invariant.on_violation is not None else None
                if terminal_state_name is not None:
                    reason += f" -- entering terminal state '{terminal_state_name}'"
                raise InvariantViolation(
                    reason=reason,
                    identity_key="vsl-langgraph",
                    invariant_name=invariant.name,
                    terminal_state_name=terminal_state_name,
                )

        return gate
