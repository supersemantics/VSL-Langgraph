"""Test fixture: a monitor that reaches a hosted model API through a helper
function -- the same shape as the real chat_reply_agent.py ->
chat_providers.judge_confidence relationship. The "anthropic" mention lives
INSIDE the helper's own body (a comment describing the real call), not just
in a module docstring -- required now that _reaches_hosted_api is scoped to
each function's own source rather than its whole containing module.
"""

from typing import Any

from vsl_core.metrics import GammaEstimate


async def _call_anthropic_judge(prompt: str) -> str:
    # Real integration: client = anthropic.AsyncAnthropic(api_key=...)
    #                   return await client.messages.create(...)
    return "PASS"


async def hosted_monitor(candidate_input: Any) -> GammaEstimate:
    verdict = await _call_anthropic_judge(str(candidate_input))
    return GammaEstimate(gamma_hat=2.0 if verdict == "PASS" else 0.0)


async def hosted_rule(candidate_input: Any) -> bool:
    verdict = await _call_anthropic_judge(str(candidate_input))
    return verdict == "PASS"
