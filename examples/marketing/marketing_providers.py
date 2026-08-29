"""Thin async wrappers around three real LLM providers, each reading its own
API key from an environment variable explicitly (never hardcoded, never
passed through vsl-core or vsl-langgraph, which stay provider-agnostic).

Kept separate from research_publish_agent.py on purpose: the point of that
file is to show how VSL's gates wire around real LLM calls, and that's
easier to see when it isn't interleaved with three different SDKs' request
shapes.
"""

from __future__ import annotations

import os

from anthropic import AsyncAnthropic
from google import genai
from openai import AsyncOpenAI

from marketing_model_ids import GEMINI_MODEL, OPENAI_MODEL


def _require_env(var_name: str) -> str:
    value = os.environ.get(var_name)
    if not value:
        raise RuntimeError(f"{var_name} is not set. Export it before running this example.")
    return value


async def call_openai(instructions: str, input_text: str) -> str:
    client = AsyncOpenAI(api_key=_require_env("OPENAI_API_KEY"))
    response = await client.responses.create(
        model=OPENAI_MODEL,
        instructions=instructions,
        input=input_text,
    )
    return response.output_text


async def call_gemini(instructions: str, input_text: str) -> str:
    client = genai.Client(api_key=_require_env("GEMINI_API_KEY"))
    response = await client.aio.models.generate_content(
        model=GEMINI_MODEL,
        contents=input_text,
        config={"system_instruction": instructions},
    )
    return response.text


async def call_claude(model: str, system: str, input_text: str, *, max_tokens: int = 1024) -> str:
    client = AsyncAnthropic(api_key=_require_env("ANTHROPIC_API_KEY"))
    message = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": input_text}],
    )
    return "".join(block.text for block in message.content if block.type == "text")
