# Sample agents

Each subfolder is a self-contained example of `vsl-langgraph`'s gate-wiring
(`compile_pre_node`/`compile_invariant`, `gated_node`, `route_on_denial`)
applied to a specific domain. The wiring code is identical across all of
them -- what changes is what the gates are checking and what they're
guarding, which is the point: the domain doesn't change how VSL governs it.

## Built

- **`financial-agents/`** -- `purchase_approval_agent.py`. Synthetic, no
  LLM calls: a spending cap approval chaining a `PreNode` (confidence
  threshold) and an `Invariant` (hard cap, non-bypassable). The fastest way
  to read the gate-wiring pattern with nothing else competing for
  attention.
- **`marketing/`** -- `research_publish_agent.py`. Real LLM calls across
  three providers (Gemini research, OpenAI draft, three different Claude
  models for fact-check / compliance / final editorial), gated the same
  way. Falls back to `marketing_fake_providers.py`'s canned responses
  automatically if `OPENAI_API_KEY` / `GEMINI_API_KEY` / `ANTHROPIC_API_KEY`
  aren't all set, so it's runnable with zero setup.
- **`customer-support/`** -- `refund_support_agent.py`. The one example
  with a *dynamic* graph topology: an LLM (`decide`) picks one of three
  tools from a customer's request, and a plain LangGraph conditional edge
  routes to whichever node it chose -- before any vsl-core gate is even
  reached. Straight from `vsl-core`'s own canonical worked example
  (`docs/building-with-vsl-core.md`'s support-bot prompt): `lookup_order`
  is read-only and ungated, `issue_refund` is gated by a non-bypassable
  `Invariant`, `send_email` is gated by a soft `PreNode`. Falls back to
  `support_fake_providers.py`'s deterministic tool-choice matching with no
  `ANTHROPIC_API_KEY` set.
- **`chat-conversation/`** -- `chat_reply_agent.py`. Governance on a
  conversational agent's *output*, not a backend action: an `Invariant`
  blocking any reply that contains an unverified legal/medical/refund
  guarantee, a `PreNode` gating hand-off-to-a-human when the agent's own
  confidence is low -- and both denial paths converge on the same
  `handoff_to_human` node, showing that `route_on_denial`'s "blocked"
  destination can be a real fallback action, not just a dead end. Falls
  back to `chat_fake_providers.py`'s canned responses with no
  `ANTHROPIC_API_KEY` set.

## Roadmap (not built yet)

- **`simple/`** -- a single-gate "hello world," shorter than
  `financial-agents/purchase_approval_agent.py`, once one exists that's
  worth the extra folder. (`docs/quickstart.md`'s Step 1 is this content
  inline already, if you want it today without a new folder.)

Have a domain you want an example for? A new subfolder should follow the
same shape as the four above:

- a `build_agent()` function returning a compiled `StateGraph`
- real or fake provider calls kept in their own file(s), **named with a
  domain prefix** (`yourdomain_providers.py`, not a bare `providers.py`) --
  Python's module cache is process-global, and a generic name will collide
  with another example's same-named module the moment both get imported in
  the same test run
- a test in `../tests/` that exercises every one of its gate outcomes (see
  [`../docs/testing-your-gates.md`](../docs/testing-your-gates.md))
- an entry in this file and in the root `README.md`'s Sample agents /
  Layout sections
