"""Model IDs shared between providers.py (real calls) and fake_providers.py
(offline stand-ins). Deliberately has zero third-party imports so the
offline path never needs the openai/anthropic/google-genai SDKs installed.
"""

OPENAI_MODEL = "gpt-5.6-sol"
GEMINI_MODEL = "gemini-3.5-flash"
CLAUDE_FACT_CHECK_MODEL = "claude-opus-4-8"
CLAUDE_COMPLIANCE_MODEL = "claude-sonnet-5"
CLAUDE_EDITORIAL_MODEL = "claude-haiku-4-5-20251001"
