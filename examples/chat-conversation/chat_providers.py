"""Real Claude calls for the chat agent: draft a reply, then judge it two
ways -- for an unverified guarantee (feeds the Invariant) and for
confidence that it actually resolves the issue (feeds the PreNode).

Kept separate from chat_reply_agent.py for the same reason as every other
example's providers.py: the point of that file is the gate wiring.
"""

from __future__ import annotations

import os

from anthropic import AsyncAnthropic

DRAFT_MODEL = "claude-sonnet-5"
GUARANTEE_JUDGE_MODEL = "claude-opus-4-8"
CONFIDENCE_JUDGE_MODEL = "claude-haiku-4-5-20251001"


def _require_env(var_name: str) -> str:
    value = os.environ.get(var_name)
    if not value:
        raise RuntimeError(f"{var_name} is not set. Export it before running this example.")
    return value


async def _call(model: str, system: str, input_text: str, *, max_tokens: int = 512) -> str:
    client = AsyncAnthropic(api_key=_require_env("ANTHROPIC_API_KEY"))
    message = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": input_text}],
    )
    return "".join(block.text for block in message.content if block.type == "text")


async def draft_reply(customer_message: str) -> str:
    return await _call(
        DRAFT_MODEL,
        system="You are a customer support agent. Draft a short, helpful reply to the customer's message.",
        input_text=customer_message,
    )


async def judge_guarantee(draft: str) -> str:
    """Reply with PASS on the first line if the draft makes no unverified
    legal/medical/financial guarantee, FAIL if it does, then a reason.
    """
    return await _call(
        GUARANTEE_JUDGE_MODEL,
        system=(
            "You review support replies for unverified guarantees -- promising a specific legal, medical, "
            "or financial outcome that isn't backed by stated policy (e.g. 'guaranteed full refund', "
            "'this will definitely cure it', 'you will win your case'). Reply with exactly one word on the "
            "first line -- PASS if the draft makes no such unverified guarantee, FAIL if it does -- then a "
            "one-sentence reason on the next line."
        ),
        input_text=draft,
    )


async def judge_confidence(draft: str, customer_message: str) -> str:
    """Reply with a number 0.0-2.0 on the first line rating confidence the
    draft fully and correctly resolves the customer's issue, then a reason.
    """
    return await _call(
        CONFIDENCE_JUDGE_MODEL,
        system=(
            "You rate whether a draft support reply fully and correctly resolves the customer's issue. "
            "Reply with exactly one number between 0.0 and 2.0 on the first line -- 1.1 or above means it "
            "resolves the issue, below means it doesn't (and should be handed off to a human) -- then a "
            "one-sentence reason on the next line."
        ),
        input_text=f"CUSTOMER MESSAGE:\n{customer_message}\n\nDRAFT REPLY:\n{draft}",
    )
