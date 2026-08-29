"""Real call to Claude for the `decide` step: given a customer's request and
the three available tools, pick exactly one and extract its parameters.

Kept separate from refund_support_agent.py for the same reason as
marketing/providers.py: the point of that file is the gate wiring, not one
more provider SDK's request shape.
"""

from __future__ import annotations

import os

from anthropic import AsyncAnthropic

DECIDE_MODEL = "claude-sonnet-5"

_SYSTEM_PROMPT = """You are a customer support agent's decision step. You have exactly three tools:

- lookup_order(order_id) -- read-only, look up an order's details.
- issue_refund(order_id, amount) -- refund an order.
- send_email(recipient, message) -- email the customer.

Given the customer's request, choose exactly ONE tool and extract its parameters.
Reply in this exact format, nothing else:

tool: <lookup_order|issue_refund|send_email>
order_id: <value, or blank if not applicable>
amount: <value, or blank if not applicable>
recipient: <value, or blank if not applicable>
message: <value, or blank if not applicable>
"""


def _require_env(var_name: str) -> str:
    value = os.environ.get(var_name)
    if not value:
        raise RuntimeError(f"{var_name} is not set. Export it before running this example.")
    return value


async def decide(request: str) -> str:
    client = AsyncAnthropic(api_key=_require_env("ANTHROPIC_API_KEY"))
    message = await client.messages.create(
        model=DECIDE_MODEL,
        max_tokens=256,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": request}],
    )
    return "".join(block.text for block in message.content if block.type == "text")
