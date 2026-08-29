# Testing your gates

`run_conformance_suite(LangGraphAdapter())` proves the *adapter* is correct — it says nothing about whether *your* `PreNode`/`Invariant` actually blocks the case it should and passes the case it shouldn't for your own domain logic. That's your test suite to write. This page generalizes the pattern this repo's own five test files already use — direct links to all of them are at the bottom.

`vsl-core`'s own guidance for its generic project scaffold says it plainly: *"Prove the governance shape: each PreNode/Invariant actually blocks the case it should and passes the case it shouldn't."* Two things to prove, minimum, per gate — three if it's chained with another:

1. **The pass case** — a candidate input the gate should let through.
2. **The deny case** — a candidate input the gate should block, and that it raises the *right* exception type (`InvariantViolation` specifically for an `Invariant`, not just any `AutomationDeniedException`).
3. **For a chain of two or more gates** — that an earlier denial actually prevents a later gate from running, not just that each gate works in isolation. (See [dos-and-donts.md](dos-and-donts.md#testing-your-own-gates).)

## Pattern 1: test the compiled gate directly, no graph needed

The simplest, fastest option — call the `CompiledGate` the same way `gated_node` does, without building a `StateGraph` at all:

```python
import pytest
from vsl_core.constructs import Invariant
from vsl_core.exceptions import InvariantViolation
from vsl_core.metrics import AssuranceBasis, F2Modification
from vsl_langgraph import LangGraphAdapter

BASIS = AssuranceBasis(f1_pre_commitment=True, f2_modification=F2Modification.FULL)

async def eligible(candidate: dict) -> bool:
    return candidate["eligible"]

refund_invariant = Invariant(
    name="refund-eligibility", description="...", rule=eligible, assurance_basis=BASIS,
)
gate = LangGraphAdapter().compile_invariant(refund_invariant)

async def test_eligible_order_passes():
    await gate({"eligible": True})  # does not raise

async def test_ineligible_order_raises_invariant_violation():
    with pytest.raises(InvariantViolation):
        await gate({"eligible": False})
```

This is what `tests/test_conformance.py` and `tests/test_integration.py` do in this repo — good for proving a gate's *logic* is right, independent of how it ends up wired into a graph.

## Pattern 2: test through the compiled graph

Closer to what actually ships — build the real graph, `ainvoke()` it, and assert on the resulting state's denial keys, exactly like [`tests/test_purchase_approval_agent.py`](../tests/test_purchase_approval_agent.py):

```python
async def test_eligible_refund_is_approved():
    result = await build_agent().ainvoke({"order": {"eligible": True}})
    assert result.get("refunded") is True
    assert result.get("vsl_denial") is None

async def test_ineligible_refund_is_blocked():
    result = await build_agent().ainvoke({"order": {"eligible": False}})
    assert result.get("refunded") is None
    assert result.get("vsl_denial") is not None
```

Prefer this pattern once you have a real `StateGraph` — it also catches wiring mistakes Pattern 1 can't (a typo'd `route_on_denial` target, a reused `denial_key`), not just gate logic.

For a **chain of gates**, add the third case explicitly — don't just trust that testing each gate alone implies the chain is right:

```python
async def test_high_confidence_cannot_bypass_hard_cap():
    """The Invariant must still block even when the earlier PreNode passed."""
    result = await build_agent().ainvoke({"amount": 5000.0, "confidence": 5.0})
    assert result.get("approved") is None
    assert result.get("cap_denial") is not None
```

This exact test (from [`examples/financial-agents/purchase_approval_agent.py`](../examples/financial-agents/purchase_approval_agent.py) via its test file) is the one that would have caught a `denial_key` collision (see [troubleshooting.md#c](troubleshooting.md#c-reusing-the-default-denial_key-across-chained-gates)) — a stale denial from the wrong gate would make this assertion fail on the wrong field.

## Testing a gate whose monitor/rule calls a real LLM

You don't need real API keys or real spend to test this reliably. `examples/marketing/`, `examples/customer-support/`, and `examples/chat-conversation/` each ship a deterministic offline stand-in (`*_fake_providers.py`) that the agent module swaps in automatically when the relevant API key isn't set — the same mechanism that makes the examples runnable with zero setup also makes them testable with zero setup. [`tests/test_chat_reply_agent.py`](../tests/test_chat_reply_agent.py) drives this directly with a scenario switch:

```python
import chat_fake_providers as fake_providers
from chat_reply_agent import build_agent

async def test_unverified_guarantee_is_handed_off_not_sent():
    fake_providers.set_scenario("unverified_guarantee")
    result = await build_agent().ainvoke({"customer_message": "..."})
    assert result.get("sent_reply") is None
    assert result.get("guarantee_denial") is not None
```

If you're testing your own LLM-backed gate without this repo's scenario-switch convention, the underlying idea still applies: don't let your test suite's reliability depend on a live model's non-deterministic output. Inject a fake/stub for the monitor or rule callable itself in tests, and reserve real API calls for a manual or explicitly-opt-in check (exactly how `examples/marketing/research_publish_agent.py` behaves when you *do* set real keys).

## Async tests, no decorator needed

This repo's `pyproject.toml` sets `asyncio_mode = "auto"` under `[tool.pytest.ini_options]`, so `async def test_...` functions run without `@pytest.mark.asyncio` on each one. If your own project uses `pytest-asyncio`, set the same option (or add the marker per-test if you'd rather be explicit).

## Real examples in this repo

- [`tests/test_conformance.py`](../tests/test_conformance.py) — Pattern 1, at the adapter level.
- [`tests/test_integration.py`](../tests/test_integration.py) — Pattern 1 and 2 together, against a real `StateGraph`.
- [`tests/test_purchase_approval_agent.py`](../tests/test_purchase_approval_agent.py) — Pattern 2, including the chained-gate case.
- [`tests/test_customer_support_agent.py`](../tests/test_customer_support_agent.py) — Pattern 2, plus dynamic tool routing.
- [`tests/test_chat_reply_agent.py`](../tests/test_chat_reply_agent.py) — Pattern 2, plus the offline-scenario-switch pattern for LLM-backed gates.

## Next

- [dos-and-donts.md](dos-and-donts.md) — the compressed checklist this page's testing section points back to.
- [troubleshooting.md](troubleshooting.md) — failure modes a good test suite catches before you do, in production.
