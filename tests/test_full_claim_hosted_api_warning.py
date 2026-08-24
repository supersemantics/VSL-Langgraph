"""Covers the F2 fix in adapter.py: LangGraphAdapter must warn when a
PreNode/Invariant claims F2Modification.FULL but its monitor/rule reaches a
hosted, black-box model API -- FULL is not honestly achievable that way (see
adapter.py's _reaches_hosted_api / _warn_if_full_claim_reaches_hosted_api
docstrings for why). This is the exact pattern found in this repo's own
examples (chat_reply_agent.py, research_publish_agent.py): FULL claimed
next to a hosted Anthropic/OpenAI call.
"""

import warnings
from typing import Any

import pytest
from vsl_core.constructs import Invariant, PreNode
from vsl_core.metrics import AssuranceBasis, F2Modification, GammaEstimate

from vsl_langgraph import LangGraphAdapter
from tests.fixtures.hosted_api_monitor import hosted_monitor, hosted_rule
from tests.fixtures.mixed_hosted_and_local import mixed_hosted_monitor, mixed_local_monitor


def _compile_and_collect_warnings(build_gate) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        build_gate()
        return list(record)

_FULL_BASIS = AssuranceBasis(f1_pre_commitment=True, f2_modification=F2Modification.FULL)
_PARTIAL_BASIS = AssuranceBasis(f1_pre_commitment=True, f2_modification=F2Modification.PARTIAL)
_NONE_BASIS = AssuranceBasis(f1_pre_commitment=True, f2_modification=F2Modification.NONE)


async def _plain_monitor(candidate_input: Any) -> GammaEstimate:
    # No hosted-API marker anywhere in this module's source.
    return GammaEstimate(gamma_hat=2.0)


async def _plain_rule(candidate_input: Any) -> bool:
    return True


def test_full_claim_reaching_hosted_api_warns_on_pre_node():
    pre_node = PreNode(
        name="hosted-pre-node",
        description="Claims FULL but its monitor's module mentions a hosted API.",
        monitor=hosted_monitor,
        assurance_basis=_FULL_BASIS,
    )
    with pytest.warns(UserWarning, match="claims F2Modification.FULL"):
        LangGraphAdapter().compile_pre_node(pre_node)


def test_full_claim_reaching_hosted_api_warns_on_invariant():
    invariant = Invariant(
        name="hosted-invariant",
        description="Claims FULL but its rule's module mentions a hosted API.",
        rule=hosted_rule,
        assurance_basis=_FULL_BASIS,
    )
    with pytest.warns(UserWarning, match="claims F2Modification.FULL"):
        LangGraphAdapter().compile_invariant(invariant)


def test_full_claim_without_hosted_api_does_not_warn():
    pre_node = PreNode(
        name="local-pre-node",
        description="Claims FULL; nothing hosted-API-shaped in this module.",
        monitor=_plain_monitor,
        assurance_basis=_FULL_BASIS,
    )
    record = _compile_and_collect_warnings(lambda: LangGraphAdapter().compile_pre_node(pre_node))
    assert not any("claims F2Modification.FULL" in str(w.message) for w in record)


def test_partial_claim_reaching_hosted_api_does_not_trigger_full_warning():
    # PARTIAL is the honest rating for a hosted-API-backed check -- only a
    # FULL claim should ever trigger this warning.
    pre_node = PreNode(
        name="hosted-pre-node-partial",
        description="Honestly rated PARTIAL for a hosted-API-backed check.",
        monitor=hosted_monitor,
        assurance_basis=_PARTIAL_BASIS,
    )
    record = _compile_and_collect_warnings(lambda: LangGraphAdapter().compile_pre_node(pre_node))
    assert not any("claims F2Modification.FULL" in str(w.message) for w in record)


def test_none_claim_reaching_hosted_api_does_not_trigger_full_warning():
    pre_node = PreNode(
        name="hosted-pre-node-none",
        description="NONE claim next to a hosted-API-shaped module.",
        monitor=hosted_monitor,
        assurance_basis=_NONE_BASIS,
    )
    record = _compile_and_collect_warnings(lambda: LangGraphAdapter().compile_pre_node(pre_node))
    assert not any("claims F2Modification.FULL" in str(w.message) for w in record)


def test_same_file_sibling_function_does_not_cross_contaminate():
    # Regression test for the false positive found while building a
    # companion prototype: mixed_local_monitor shares a file with
    # mixed_hosted_monitor, which genuinely mentions "anthropic". Before
    # _reaches_hosted_api was scoped to each function's own source (rather
    # than its whole containing module), this would have wrongly flagged
    # mixed_local_monitor too, just because of its neighbor.
    pre_node = PreNode(
        name="mixed-local-pre-node",
        description="Claims FULL; genuinely local, but shares a file with a hosted-API monitor.",
        monitor=mixed_local_monitor,
        assurance_basis=_FULL_BASIS,
    )
    record = _compile_and_collect_warnings(lambda: LangGraphAdapter().compile_pre_node(pre_node))
    assert not any("claims F2Modification.FULL" in str(w.message) for w in record)


def test_same_file_sibling_that_genuinely_reaches_hosted_api_is_still_caught():
    # The other half of the same regression test: the fix must not become
    # so narrow it stops detecting real cases sitting in a shared file.
    pre_node = PreNode(
        name="mixed-hosted-pre-node",
        description="Claims FULL; genuinely reaches a hosted API.",
        monitor=mixed_hosted_monitor,
        assurance_basis=_FULL_BASIS,
    )
    with pytest.warns(UserWarning, match="claims F2Modification.FULL"):
        LangGraphAdapter().compile_pre_node(pre_node)
