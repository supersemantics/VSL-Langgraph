"""Sample agent: a research-to-publish pipeline with six real LLM tasks
across three providers, gated by vsl-core through vsl-langgraph.

    Task                      Provider   Model                       VSL role
    ------------------------- ---------- --------------------------- --------------------------------
    1. research                Gemini    gemini-3.5-flash            plain node (no gate needed --
                                                                       gathering info has no side effect)
    2. draft                   OpenAI    gpt-5.6-sol                 plain node (same reason)
    3. fact-check the draft    Claude    claude-opus-4-8             Invariant.rule -- non-bypassable
    4. brand-safety review     Claude    claude-sonnet-5             PreNode.monitor -- soft threshold
    5. compliance/tone review  Claude    claude-sonnet-5             PreNode.monitor -- soft threshold
    6. write final published   Claude    claude-haiku-4-5-20251001   the guarded action itself; only
       text                                                          runs if all three gates above passed

This is deliberately the same PreNode + Invariant chaining pattern as
examples/purchase_approval_agent.py -- the point of putting three unrelated
provider SDKs (OpenAI's Responses API, Gemini's generate_content, Anthropic's
Messages API) behind it is to make it obvious that vsl-core's gates don't
care what produced the signal they're checking. A monitor or an Invariant
rule is just an async callable returning a GammaEstimate or a bool --
here that callable happens to make a real API call to a different company
each time, and the gate wiring (compile_pre_node/compile_invariant,
gated_node, route_on_denial) doesn't change at all from the synthetic
version. That's the thing LangGraph alone doesn't give you: LangGraph
doesn't know or care about Gamma thresholds, Invariants, or F1
pre-commitment -- vsl-core defines that vocabulary once, and
vsl-langgraph is only the (small) glue making it callable from a node.

With OPENAI_API_KEY, GEMINI_API_KEY, and ANTHROPIC_API_KEY all set, this
makes real calls against a topic you supply:

    python examples/marketing/research_publish_agent.py "your topic here"

Without all three keys set, it automatically falls back to
marketing_fake_providers.py's deterministic stand-ins and runs all four
governance outcomes (approved / blocked-by-fact-check / blocked-by-brand-safety
/ blocked-by-compliance)
against canned data instead -- no keys, no cost, no SDKs required, so the
gate-wiring logic is still fully readable and runnable on a bare checkout:

    python examples/marketing/research_publish_agent.py
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

from marketing_model_ids import CLAUDE_COMPLIANCE_MODEL, CLAUDE_EDITORIAL_MODEL, CLAUDE_FACT_CHECK_MODEL

LIVE = all(os.environ.get(key) for key in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"))

if LIVE:
    from marketing_providers import call_claude, call_gemini, call_openai
else:
    from marketing_fake_providers import call_claude, call_gemini, call_openai, set_scenario

_ASSURANCE_BASIS = AssuranceBasis(f1_pre_commitment=True, f2_modification=F2Modification.FULL)

_FIRST_LINE = re.compile(r"\s*(\S+)")


class ArticleState(TypedDict, total=False):
    topic: str
    research: str
    draft: str
    fact_check_denial: Any
    compliance_denial: Any
    brand_safety_denial: Any
    final_text: str
    published: bool


async def research_node(state: ArticleState) -> dict:
    research = await call_gemini(
        instructions="You are a research assistant. Produce a concise, factual briefing (5-8 bullet points) on the given topic, suitable as source material for a short article. Stick to well-established facts.",
        input_text=state["topic"],
    )
    return {"research": research}


async def draft_node(state: ArticleState) -> dict:
    draft = await call_openai(
        instructions="You are a journalist. Write a short (150-250 word) article on the given topic, using ONLY the supplied research briefing as your factual source. Do not introduce claims that aren't supported by the briefing.",
        input_text=f"Topic: {state['topic']}\n\nResearch briefing:\n{state['research']}",
    )
    return {"draft": draft}


async def _fact_check_rule(state: ArticleState) -> bool:
    """Invariant.rule: Claude Opus checks the draft's claims against the
    research briefing. Fails closed -- if the model's answer can't be
    parsed as PASS/FAIL, this returns False (deny), never True.
    """
    verdict = await call_claude(
        CLAUDE_FACT_CHECK_MODEL,
        system="You are a fact-checker. Compare the DRAFT against the RESEARCH briefing it's supposed to be based on. Reply with exactly one word on the first line -- PASS if every factual claim in the draft is supported by the research, FAIL if the draft contains any claim the research doesn't support -- then a one-sentence reason on the next line.",
        input_text=f"RESEARCH:\n{state['research']}\n\nDRAFT:\n{state['draft']}",
    )
    match = _FIRST_LINE.match(verdict)
    return bool(match) and match.group(1).strip().upper() == "PASS"


async def _compliance_monitor(state: ArticleState) -> GammaEstimate:
    """PreNode.monitor: Claude Sonnet rates tone/policy compliance on a
    0.0-2.0 scale (ROBUST_GAMMA_DEFAULT_THRESHOLD, 1.1, is the passing
    line). Fails closed -- an unparseable response becomes gamma_hat=0.0.
    """
    rating = await call_claude(
        CLAUDE_COMPLIANCE_MODEL,
        system=(
            "You review articles for neutral tone and editorial policy compliance "
            "(no unattributed opinion stated as fact, no inflammatory language). "
            "Reply with exactly one number between 0.0 and 2.0 on the first line -- "
            f"{ROBUST_GAMMA_DEFAULT_THRESHOLD} or above means it passes, below means it doesn't -- "
            "then a one-sentence reason on the next line."
        ),
        input_text=state["draft"],
    )
    match = _FIRST_LINE.match(rating)
    try:
        gamma_hat = float(match.group(1)) if match else 0.0
    except ValueError:
        gamma_hat = 0.0
    return GammaEstimate(gamma_hat=gamma_hat)


async def _brand_safety_monitor(state: ArticleState) -> GammaEstimate:
    """PreNode.monitor: a second, independent editorial signal -- brand-
    safety screening is a real, distinct concern from the tone/policy
    compliance check above (a draft can be perfectly policy-compliant and
    still be brand-unsafe, e.g. adjacent to a sensitive current event).
    Same fail-closed shape as _compliance_monitor."""
    rating = await call_claude(
        CLAUDE_COMPLIANCE_MODEL,
        system=(
            "You screen articles for brand-safety risk (proximity to sensitive "
            "current events, controversial topics, or anything that could "
            "embarrass an advertiser). Reply with exactly one number between "
            f"0.0 and 2.0 on the first line -- {ROBUST_GAMMA_DEFAULT_THRESHOLD} "
            "or above means safe, below means it isn't -- then a one-sentence "
            "reason on the next line."
        ),
        input_text=state["draft"],
    )
    match = _FIRST_LINE.match(rating)
    try:
        gamma_hat = float(match.group(1)) if match else 0.0
    except ValueError:
        gamma_hat = 0.0
    return GammaEstimate(gamma_hat=gamma_hat)


async def publish_node(state: ArticleState) -> dict:
    """The actual guarded side effect -- only reached if the fact-check
    Invariant, the brand-safety PreNode and the compliance PreNode all
    passed. Claude Haiku
    writes the final editorial note; a real deployment would swap the
    print() in main() for an actual publish call here.
    """
    final_text = await call_claude(
        CLAUDE_EDITORIAL_MODEL,
        system="Write a one-paragraph editorial note confirming this article cleared fact-check and compliance review, suitable to prepend before publishing.",
        input_text=state["draft"],
        max_tokens=256,
    )
    return {"final_text": final_text, "published": True}


def build_agent():
    adapter = LangGraphAdapter()

    fact_check_invariant = Invariant(
        name="article-fact-check",
        description="Every factual claim in the draft must be supported by the research briefing -- non-bypassable.",
        rule=_fact_check_rule,
        assurance_basis=_ASSURANCE_BASIS,
    )
    fact_check_gate = adapter.compile_invariant(fact_check_invariant)

    compliance_pre_node = PreNode(
        name="article-compliance",
        description="Requires the compliance reviewer's confidence signal to clear the robust Gamma threshold before publishing.",
        monitor=_compliance_monitor,
        assurance_basis=_ASSURANCE_BASIS,
        gamma_threshold=ROBUST_GAMMA_DEFAULT_THRESHOLD,
    )
    compliance_gate = adapter.compile_pre_node(compliance_pre_node)

    brand_safety_pre_node = PreNode(
        name="article-brand-safety",
        description="Requires the brand-safety reviewer's confidence signal to clear the robust Gamma threshold before publishing -- independent of, and in addition to, the tone/policy compliance check.",
        monitor=_brand_safety_monitor,
        assurance_basis=_ASSURANCE_BASIS,
        gamma_threshold=ROBUST_GAMMA_DEFAULT_THRESHOLD,
    )
    brand_safety_gate = adapter.compile_pre_node(brand_safety_pre_node)

    graph = StateGraph(ArticleState)
    graph.add_node("research", research_node)
    graph.add_node("draft", draft_node)
    graph.add_node("fact_check", gated_node(fact_check_gate, lambda state: {}, denial_key="fact_check_denial"))
    graph.add_node("brand_safety_check", gated_node(brand_safety_gate, lambda state: {}, denial_key="brand_safety_denial"))
    graph.add_node("compliance_check", gated_node(compliance_gate, publish_node, denial_key="compliance_denial"))
    graph.add_node("blocked_unverified", lambda state: {})
    graph.add_node("blocked_unsafe", lambda state: {})
    graph.add_node("blocked_noncompliant", lambda state: {})

    graph.add_edge(START, "research")
    graph.add_edge("research", "draft")
    graph.add_edge("draft", "fact_check")
    graph.add_conditional_edges(
        "fact_check",
        route_on_denial("blocked_unverified", "brand_safety_check", denial_key="fact_check_denial"),
        # explicit path_map so get_graph().draw_mermaid() can render both branches
        {"blocked_unverified": "blocked_unverified", "brand_safety_check": "brand_safety_check"},
    )
    graph.add_conditional_edges(
        "brand_safety_check",
        route_on_denial("blocked_unsafe", "compliance_check", denial_key="brand_safety_denial"),
        {"blocked_unsafe": "blocked_unsafe", "compliance_check": "compliance_check"},
    )
    graph.add_conditional_edges(
        "compliance_check",
        route_on_denial("blocked_noncompliant", END, denial_key="compliance_denial"),
        {"blocked_noncompliant": "blocked_noncompliant", END: END},
    )
    graph.add_edge("blocked_unverified", END)
    graph.add_edge("blocked_unsafe", END)
    graph.add_edge("blocked_noncompliant", END)

    return graph.compile()


def _print_result(result: dict) -> None:
    print("--- research (Gemini) ---")
    print(result.get("research"))
    print("\n--- draft (OpenAI) ---")
    print(result.get("draft"))

    if result.get("fact_check_denial") is not None:
        print("\n--- BLOCKED: fact-check Invariant violated (claude-opus-4-8) ---")
        print(result["fact_check_denial"])
        return

    if result.get("brand_safety_denial") is not None:
        print("\n--- BLOCKED: brand-safety PreNode denied (claude-sonnet-5) ---")
        print(result["brand_safety_denial"])
        return

    if result.get("compliance_denial") is not None:
        print("\n--- BLOCKED: compliance PreNode denied (claude-sonnet-5) ---")
        print(result["compliance_denial"])
        return

    print("\n--- published (claude-haiku-4-5-20251001 editorial note) ---")
    print(result.get("final_text"))
    print(f"\npublished={result.get('published')}")


async def main() -> None:
    if LIVE:
        topic = sys.argv[1] if len(sys.argv) > 1 else "recent advances in solid-state EV battery technology"
        print(f"Topic: {topic}\n")
        result = await build_agent().ainvoke({"topic": topic})
        _print_result(result)
        return

    print(
        "No OPENAI_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY found -- running the four offline demo "
        "scenarios against marketing_fake_providers.py's canned responses instead of live calls. Set all "
        "three env vars and rerun to see this against real providers.\n"
    )
    for scenario in ("approved", "unverified", "unsafe", "noncompliant"):
        set_scenario(scenario)
        print(f"=== scenario: {scenario} ===")
        result = await build_agent().ainvoke({"topic": "(offline demo -- no live research/draft calls made)"})
        _print_result(result)
        print()


if __name__ == "__main__":
    asyncio.run(main())
