"""Deterministic offline stand-ins for chat_providers.py, used automatically
when ANTHROPIC_API_KEY isn't set -- same shape as
marketing/marketing_fake_providers.py: a module-level scenario switch so all
three governance outcomes are reachable with zero API keys and zero cost.
"""

from __future__ import annotations

_SCENARIOS: dict[str, dict[str, str]] = {
    "resolved": {
        "draft": "Thanks for reaching out! I've checked your account and the issue should now be resolved -- let us know if anything else comes up.",
        "guarantee": "PASS",
        "confidence": "1.6",
    },
    "unverified_guarantee": {
        "draft": "Don't worry, I guarantee you'll get a full refund and this will definitely never happen again.",
        "guarantee": "FAIL",
        "confidence": "1.6",
    },
    "low_confidence": {
        "draft": "I'm not entirely sure what's causing this, but you could try restarting the app?",
        "guarantee": "PASS",
        "confidence": "0.4",
    },
}

_current_scenario = "resolved"


def set_scenario(name: str) -> None:
    if name not in _SCENARIOS:
        raise ValueError(f"Unknown offline scenario {name!r}; expected one of {sorted(_SCENARIOS)}")
    global _current_scenario
    _current_scenario = name


async def draft_reply(customer_message: str) -> str:
    return _SCENARIOS[_current_scenario]["draft"]


async def judge_guarantee(draft: str) -> str:
    verdict = _SCENARIOS[_current_scenario]["guarantee"]
    return f"{verdict}\nOffline stub verdict for the {_current_scenario!r} scenario -- no live Claude call was made."


async def judge_confidence(draft: str, customer_message: str) -> str:
    rating = _SCENARIOS[_current_scenario]["confidence"]
    return f"{rating}\nOffline stub rating for the {_current_scenario!r} scenario -- no live Claude call was made."
