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

import inspect
import warnings
from typing import Any, Callable

from vsl_core.constructs import Fallback, Invariant, PreNode
from vsl_core.conformance.protocol import CompiledGate
from vsl_core.exceptions import AutomationDeniedException, InvariantViolation
from vsl_core.metrics import F2Modification

_DEFAULT_FALLBACK = Fallback()

# Source-text markers for hosted, black-box model APIs. A gate whose
# monitor/rule reaches one of these cannot honestly claim F2Modification.FULL
# -- FULL requires direct access to the energy function itself (logits,
# weights, activations), which a hosted API call structurally cannot expose.
# The honest ceiling for a hosted-API-backed check is PARTIAL (MEDIUM), per
# vsl-core's own README example. This is a heuristic, not a proof: it scans
# source text -- the monitor/rule's OWN body (inspect.getsource(fn), not its
# containing module's entire source; see _reaches_hosted_api's docstring for
# why that distinction matters), plus one level of indirection through
# whatever names it directly calls -- catching the common "the actual API
# call lives in a helper module" shape, e.g. a monitor in agent.py calling
# judge_confidence() from a separate providers.py that imports anthropic --
# for these markers. It can still miss deeper indirection (two or more
# call-frames away, a callable reached only via a dict/attribute lookup,
# dynamic imports) and cannot confirm a genuine FULL claim is correct either
# -- it only catches the one concrete, previously-observed failure mode of
# claiming FULL while calling a hosted API.
_HOSTED_API_MARKERS = (
    "openai",
    "anthropic",
    "google.genai",
    "google.generativeai",
    "cohere",
    "mistralai",
)


def _reaches_hosted_api(fn: Callable[..., Any], _seen: set | None = None) -> bool:
    """Scoped to fn's OWN source (inspect.getsource(fn)), not its containing
    module's entire source. Scanning the whole module was a real false-
    positive bug found while building a companion prototype: an unrelated
    function's comment or docstring elsewhere in the same file could
    trigger a match that had nothing to do with fn itself. Function-level
    scoping fixes that without requiring every monitor/rule to live in its
    own file -- real hosted-API calls are actual code inside the relevant
    function (or a helper it calls, covered by the indirection walk below),
    not incidental mentions elsewhere in the file.
    """
    if _seen is None:
        _seen = set()
    if fn in _seen:
        return False
    _seen.add(fn)

    if inspect.getmodule(fn) is None:
        return False
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return False
    # Case-insensitive: real code capitalizes SDK class names (AsyncAnthropic,
    # AsyncOpenAI), which a case-sensitive check against lowercase "anthropic"/
    # "openai" markers would miss -- found and fixed after function-scoping
    # (above) removed an accidental crutch: the old whole-module scan happened
    # to also catch the lowercase `from anthropic import ...` line elsewhere
    # in the file, masking this case bug until scoping narrowed to fn itself.
    source_lower = source.lower()
    if any(marker in source_lower for marker in _HOSTED_API_MARKERS):
        return True

    # One level of indirection: follow names this function's body directly
    # references, in case the real hosted-API call lives in a helper it
    # calls rather than in its own module.
    code = getattr(fn, "__code__", None)
    globalns = getattr(fn, "__globals__", None)
    if code is None or globalns is None:
        return False
    for referenced_name in code.co_names:
        candidate = globalns.get(referenced_name)
        if callable(candidate) and candidate is not fn:
            if _reaches_hosted_api(candidate, _seen):
                return True
    return False


def _warn_if_full_claim_reaches_hosted_api(name: str, assurance_basis: Any, fn: Callable[..., Any]) -> None:
    if assurance_basis.f2_modification != F2Modification.FULL:
        return
    if not _reaches_hosted_api(fn):
        return
    warnings.warn(
        f"'{name}' claims F2Modification.FULL, but its monitor/rule appears to "
        "call a hosted model API (openai/anthropic/google.genai/cohere/mistralai "
        "detected in its module's source). A hosted, black-box API call cannot "
        "directly modify the energy function -- FULL is not achievable this way. "
        "The honest ceiling is F2Modification.PARTIAL (MEDIUM assurance), per "
        "vsl-core's own README F1/F2 example. This is a heuristic check, not "
        "proof -- it can miss indirect calls, and passing it does not confirm "
        "FULL is correct either.",
        stacklevel=3,
    )


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
        _warn_if_full_claim_reaches_hosted_api(f"PreNode '{pre_node.name}'", pre_node.assurance_basis, pre_node.monitor)

        async def gate(candidate_input: Any) -> None:
            estimate = await pre_node.monitor(candidate_input)
            if not estimate.sufficient(pre_node.gamma_threshold):
                raise AutomationDeniedException(
                    reason=f"Pre-node '{pre_node.name}' insufficient Gamma",
                    identity_key="vsl-langgraph",
                )

        return gate

    def compile_invariant(self, invariant: Invariant) -> CompiledGate:
        _warn_if_full_claim_reaches_hosted_api(f"Invariant '{invariant.name}'", invariant.assurance_basis, invariant.rule)

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
