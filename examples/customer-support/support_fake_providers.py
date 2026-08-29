"""Deterministic, zero-dependency stand-in for support_providers.decide(),
used automatically when ANTHROPIC_API_KEY isn't set -- same reasoning and
same output format as marketing/marketing_fake_providers.py, so the
graph-building code never has to know which one it's calling.

Pattern-matches a few fixed phrasings rather than doing real language
understanding -- this is a stand-in for a real LLM call, not a parser
meant to generalize.
"""

from __future__ import annotations

import re

_REFUND_RE = re.compile(r"refund order ([A-Za-z0-9-]+) for ([\d.]+)", re.IGNORECASE)
_LOOKUP_RE = re.compile(r"status of order ([A-Za-z0-9-]+)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"email (\S+@\S+) to say (.+)", re.IGNORECASE)


async def decide(request: str) -> str:
    if m := _REFUND_RE.search(request):
        return f"tool: issue_refund\norder_id: {m.group(1)}\namount: {m.group(2)}\nrecipient: \nmessage: "
    if m := _LOOKUP_RE.search(request):
        return f"tool: lookup_order\norder_id: {m.group(1)}\namount: \nrecipient: \nmessage: "
    if m := _EMAIL_RE.search(request):
        return f"tool: send_email\norder_id: \namount: \nrecipient: {m.group(1)}\nmessage: {m.group(2)}"
    return "tool: lookup_order\norder_id: UNKNOWN\namount: \nrecipient: \nmessage: "
