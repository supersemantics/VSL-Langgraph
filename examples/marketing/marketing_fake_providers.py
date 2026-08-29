"""Deterministic, zero-dependency stand-ins for marketing_providers.py's
three calls.

Used automatically when OPENAI_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY
aren't all set, so a developer can read and run the governance flow --
including all three of its outcomes -- with no API keys, no cost, and none
of the openai/anthropic/google-genai SDKs installed. Imports nothing but
marketing_model_ids (also dependency-free) for exactly that reason.
"""

from __future__ import annotations

from marketing_model_ids import CLAUDE_COMPLIANCE_MODEL, CLAUDE_FACT_CHECK_MODEL

_FAKE_RESEARCH = (
    "- Solid-state batteries replace the liquid electrolyte with a solid one, removing the main fire-risk pathway.\n"
    "- Several manufacturers have announced pilot production lines targeting the late-2020s.\n"
    "- Reported lab energy densities are meaningfully higher than current lithium-ion cells.\n"
    "- Manufacturing cost and cycle-life at scale remain the two open engineering problems.\n"
    "(offline demo data -- no live Gemini call was made)"
)

_FAKE_DRAFT = (
    "Solid-state EV batteries are moving from lab demos toward pilot production, with several manufacturers "
    "targeting the late 2020s. The core change -- a solid electrolyte instead of a liquid one -- removes the "
    "main fire-risk pathway of today's lithium-ion packs and reports meaningfully higher lab energy densities. "
    "Cost and long-term cycle life at manufacturing scale remain the open questions before wide adoption.\n"
    "(offline demo data -- no live OpenAI call was made)"
)

# Each entry controls what the two Claude gates decide for that run, so all
# three governance outcomes are reachable without a live model in the loop.
_SCENARIOS: dict[str, dict[str, str]] = {
    "approved": {"fact_check": "PASS", "compliance": "1.6"},
    "unverified": {"fact_check": "FAIL", "compliance": "1.6"},
    "noncompliant": {"fact_check": "PASS", "compliance": "0.4"},
}

_current_scenario = "approved"


def set_scenario(name: str) -> None:
    if name not in _SCENARIOS:
        raise ValueError(f"Unknown offline scenario {name!r}; expected one of {sorted(_SCENARIOS)}")
    global _current_scenario
    _current_scenario = name


async def call_gemini(instructions: str, input_text: str) -> str:
    return _FAKE_RESEARCH


async def call_openai(instructions: str, input_text: str) -> str:
    return _FAKE_DRAFT


async def call_claude(model: str, system: str, input_text: str, *, max_tokens: int = 1024) -> str:
    verdicts = _SCENARIOS[_current_scenario]
    if model == CLAUDE_FACT_CHECK_MODEL:
        return f"{verdicts['fact_check']}\nOffline stub verdict for the {_current_scenario!r} scenario -- no live Claude call was made."
    if model == CLAUDE_COMPLIANCE_MODEL:
        return f"{verdicts['compliance']}\nOffline stub rating for the {_current_scenario!r} scenario -- no live Claude call was made."
    return "This is an offline stub editorial note. Set OPENAI_API_KEY, GEMINI_API_KEY, and ANTHROPIC_API_KEY to see a live one."
