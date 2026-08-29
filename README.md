# vsl-langgraph

A [LangGraph](https://github.com/langchain-ai/langgraph) adapter for
[`vsl-core`](https://github.com/supersemantics/VSL-Core): implements `vsl_core.conformance.protocol.VSLAdapter`
and provides the LangGraph-specific wiring to call a compiled gate from a
`StateGraph` node and route around a denial.

**Is:**
- A `VSLAdapter` implementation (`LangGraphAdapter`) that passes `vsl_core.conformance.suite.run_conformance_suite` with an empty result.
- A small set of wiring helpers (`gated_node`, `route_on_denial`) for calling a compiled gate immediately before a node's side effect and routing to a blocked/terminal node via `add_conditional_edges` on denial.

**Is not:**
- A replacement for `vsl-core` itself -- the PreNode/Invariant/Fallback/TerminalState constructs, the ledger, and the governance vocabulary all live in `vsl-core`. This package only compiles and wires those constructs into LangGraph.
- Automatic interception. LangGraph has no dedicated guardrail/hook type -- a node is a plain callable, so the gate is called explicitly inside `gated_node`, the same way `vsl-core`'s own docs describe calling a gate in plain Python.

## Documentation

- [docs/concepts.md](docs/concepts.md) -- the vocabulary (`PreNode`, `Invariant`, `GammaEstimate`, `AssuranceBasis`/F1-F2, `CompiledGate`), a diagram of the gate lifecycle, which gate shape fits which kind of LangGraph node, and a complete scope table for every `vsl-core` construct -- not just the two this adapter compiles.
- [docs/quickstart.md](docs/quickstart.md) -- install through a working chained-gate graph, building up from one gate to real LLM-backed gates, with real diagrams generated from the actual compiled graphs.
- [docs/api-reference.md](docs/api-reference.md) -- `LangGraphAdapter` / `gated_node` / `route_on_denial`, every parameter, scannable.
- [docs/beyond-the-gate.md](docs/beyond-the-gate.md) -- everything `vsl-core` defines that this adapter doesn't compile (`TerminalState`, `VerbaLedger`, re-enablement, `Cluster`, `Specification`/`Proofing`/`Binding`) but that still applies inside a LangGraph app, each with real, run worked examples.
- [docs/dos-and-donts.md](docs/dos-and-donts.md) -- a scannable checklist for writing or reviewing `vsl-langgraph` code, each rule traced back to a real reason.
- [docs/testing-your-gates.md](docs/testing-your-gates.md) -- how to prove your own `PreNode`/`Invariant` wiring is correct, generalized from this repo's own test suite.
- [docs/troubleshooting.md](docs/troubleshooting.md) -- real, reproduced failure modes (not guessed) with exact error text and fixes.

## Install

```
pip install git+https://github.com/supersemantics/VSL-Core.git
pip install -e .   # from a local clone of this repo
```

## Quick taste

```python
gate = LangGraphAdapter().compile_pre_node(pre_node)  # or .compile_invariant(invariant)

graph.add_node("do_the_thing", gated_node(gate, do_the_thing))
graph.add_conditional_edges("do_the_thing", route_on_denial("blocked", "allowed"))
```

`gated_node` calls `gate` immediately before `do_the_thing` runs; on denial, `do_the_thing` never runs and `route_on_denial` sends the graph to `"blocked"` instead. Full walkthrough with real output: [docs/quickstart.md](docs/quickstart.md).

## Sample agents

`examples/` is organized by domain -- see [examples/README.md](examples/README.md) for the full list and the roadmap for domains not built yet. Four so far:

- **`examples/financial-agents/purchase_approval_agent.py`** -- both governance primitives chained in one graph, synthetic and LLM-free. The fastest read of the gate-wiring pattern.
- **`examples/marketing/research_publish_agent.py`** -- the same pattern behind five real LLM calls across three providers (Gemini, OpenAI, three Claude models), with an automatic offline fallback when no API keys are set.
- **`examples/customer-support/refund_support_agent.py`** -- the one with a *dynamic* graph topology: an LLM picks which tool to call (`lookup_order` / `issue_refund` / `send_email`) and a plain LangGraph router sends the run to whichever node it chose, before any vsl-core gate is even reached. `issue_refund` is gated by an `Invariant`, `send_email` by a `PreNode`, `lookup_order` is ungated -- straight from `vsl-core`'s own canonical worked example.
- **`examples/chat-conversation/chat_reply_agent.py`** -- governance on what the agent is about to *say*, not a backend action: an `Invariant` blocking an unverified guarantee, a `PreNode` handing off to a human on low confidence.

```
python examples/financial-agents/purchase_approval_agent.py
python examples/marketing/research_publish_agent.py
python examples/customer-support/refund_support_agent.py
python examples/chat-conversation/chat_reply_agent.py
```

All four run with zero setup (deterministic offline fallbacks) or live against real providers with the right API keys set. Diagrammed and walked through step by step in [docs/quickstart.md](docs/quickstart.md).

## Layout

```
src/vsl_langgraph/
├── adapter.py       LangGraphAdapter (VSLAdapter conformance contract)
└── integration.py   gated_node, route_on_denial (StateGraph wiring)
docs/
├── concepts.md            vocabulary, gate-lifecycle diagram, full construct scope table
├── quickstart.md          install -> working graph, with real diagrams
├── api-reference.md       every export, scannable
├── beyond-the-gate.md     TerminalState/ledger/escalation, Cluster, Specification -- real worked examples
├── dos-and-donts.md       scannable checklist for writing/reviewing this code
├── testing-your-gates.md  how to prove your own gate wiring is correct
└── troubleshooting.md     real, reproduced failure modes
examples/
├── README.md                 domain taxonomy and roadmap
├── financial-agents/
│   └── purchase_approval_agent.py   PreNode + Invariant, no LLM calls
├── marketing/
│   ├── research_publish_agent.py    same pattern, 5 real LLM calls / 3 providers
│   └── marketing_providers.py, marketing_fake_providers.py, marketing_model_ids.py
├── customer-support/
│   ├── refund_support_agent.py      LLM picks the tool; dynamic routing, not fixed topology
│   └── support_providers.py, support_fake_providers.py
└── chat-conversation/
    ├── chat_reply_agent.py          governs the agent's own reply, not a backend action
    └── chat_providers.py, chat_fake_providers.py
tests/
├── test_conformance.py               run_conformance_suite(LangGraphAdapter()) == []
├── test_integration.py               exercises a real compiled StateGraph
├── test_purchase_approval_agent.py   exercises the financial-agents sample's three scenarios
├── test_customer_support_agent.py    exercises dynamic routing + both gate types
└── test_chat_reply_agent.py          exercises the chat sample's three scenarios
```

Every example domain's provider modules are named with a domain prefix (`marketing_*`, `support_*`, `chat_*`) rather than a generic `providers.py` -- several examples import a same-named module in the same test run, and Python's module cache is process-global, so generic names collide across examples.

## Status

Alpha. Conformance-suite green; integration tested against a real
`StateGraph` for both the `PreNode` and `Invariant` paths (pass-through and
denial routing), including chained multi-gate graphs, real LLM-backed
gates across three providers, and -- as of `customer-support/refund_support_agent.py`
-- a graph where an LLM's own output decides which node runs next, not
just a fixed topology with LLM-backed gates. `gated_node` doesn't implement
`Fallback`'s retry/intervention fields (`delta_factor`, `max_retries`,
`intervention`) -- those aren't consumed by `vsl-core`'s own
`PlainPythonReferenceAdapter` either, so this mirrors the reference
adapter's actual behavior rather than leaving out something `vsl-core`
itself defines a contract for.
