# Dos and don'ts

A checklist for writing or reviewing `vsl-langgraph` code — scannable while you work, not something you read once. Every rule here traces back to a real reason established elsewhere in these docs (mostly [troubleshooting.md](troubleshooting.md), where most of these were actually triggered, not guessed). This is not a replacement for [concepts.md](concepts.md) or [api-reference.md](api-reference.md) — it's the compressed, imperative version of both.

## Writing a `PreNode`/`Invariant`

- **DO** write `monitor`/`rule` as `async def`, never plain `def`. A sync function breaks with `TypeError: object ... can't be used in 'await' expression`. → [troubleshooting.md#a](troubleshooting.md#a-sync-monitor-or-rule-instead-of-async-typeerror-on-await)
- **DON'T** treat `GammaEstimate.gamma_hat` as a 0-1 confidence score. It's compared directly against `gamma_threshold` (default `1.1`) — a monitor returning `0.95` meaning "95% confident" always fails. → [troubleshooting.md#f](troubleshooting.md#f-gamma_hat-mistaken-for-a-0-1-confidence-score)
- **DO** justify every `AssuranceBasis` with real F1 (does the check run before the side effect?) and F2 (`FULL`/`PARTIAL`/`INDIRECT`/`NONE` — does it touch the actual mechanism or just a prompt/parameter?) facts. Never default to `F2Modification.FULL` because `HIGH` sounds better — a gate in front of a hosted LLM API can only touch the prompt, which caps it at `MEDIUM`. → [concepts.md](concepts.md#glossary)
- **DO** reach for `Invariant` on irreversibility boundaries (refund, delete, transfer — can't be cleanly undone) and `PreNode` on soft, threshold-based checks. Not the reverse. → [concepts.md](concepts.md#which-gate-for-which-langgraph-node)
- **DO** give every `PreNode`/`Invariant` a real, specific `name` — it ends up in the raised exception's `reason` string and in any ledger entries you write, and is how you'll actually distinguish denials when reading logs.

## Wiring a gate into a `StateGraph`

- **DON'T** call a `CompiledGate` directly without `await`. It's `async def gate(...) -> None` — calling it without awaiting just constructs a coroutine and discards it silently, with **no exception**. The guarded action runs as if the gate had passed. Always go through `gated_node`, which awaits it correctly. → [troubleshooting.md#b](troubleshooting.md#b-forgetting-await-on-a-manually-called-gate-silent-bypass) — this is the one to actually lose sleep over; every other mistake on this page at least fails loudly.
- **DO** give every `gated_node` in a chain its own `denial_key`. Reusing the default lets an earlier gate's denial silently survive in state after a later, unrelated gate actually passes. → [troubleshooting.md#c](troubleshooting.md#c-reusing-the-default-denial_key-across-chained-gates)
- **DO** double-check that the two node names passed to `route_on_denial` exactly match real `graph.add_node(...)` calls. A typo doesn't raise — LangGraph logs a one-line stderr warning and silently drops the transition, leaving the run stalled with partial state. → [troubleshooting.md#d](troubleshooting.md#d-route_on_denial-targeting-a-node-name-that-was-never-add_noded)
- **DO** pass an explicit `path_map` to `add_conditional_edges` alongside `route_on_denial` if you want `get_graph().draw_mermaid()` to render the branches — without one, LangGraph can't infer the router's possible outputs and silently omits them from the diagram. → [quickstart.md](quickstart.md#visualizing-your-own-graph)
- **DON'T** assume a `Cluster`'s `invariants` need special handling. They're plain `Invariant` objects — compile and wire each one through `LangGraphAdapter`/`gated_node` exactly like any other. → [beyond-the-gate.md](beyond-the-gate.md#multi-agent-governance-cluster)

## Ledger and escalation

- **DON'T** assume `gated_node` writes to `VerbaLedger`. It never does — only the gate's raise/return is automatic. If you need an audit trail, write it explicitly, typically from the node reached via `route_on_denial`. → [concepts.md](concepts.md#gated_node-doesnt-write-to-the-ledger-either), full worked example in [beyond-the-gate.md](beyond-the-gate.md#the-full-lifecycle-log-and-escalate)
- **DON'T** assume `Fallback`'s `max_retries`/`delta_factor` do anything automatically. `gated_node` (and `vsl-core`'s own `PlainPythonReferenceAdapter`) raise once, immediately, with no retry loop. If you want retry behavior, you write the loop yourself, using `Fallback`'s fields as configuration. → [concepts.md](concepts.md#glossary)
- **DON'T** call `Instance._re_enable()` directly. The leading underscore is deliberate — it's the bare mechanism with no authority check. Always go through `governance.request_re_enablement()`, which pairs it with the `HumanAuthorisedTransition` audit record.
- **DO** populate `caused_by` on ledger writes once more than one decision can be in flight for the same `identity_key`/`instance_id`. Without it, `audit()` falls back to a timestamp-ordering heuristic that can mismatch an unrelated entry as resolving the wrong one.

## Testing your own gates

- **DO** test both outcomes for every gate: the case that should pass and the case that should deny. A gate that never denies in your test suite hasn't proven anything.
- **DO**, for a chain of two or more gates, test that an earlier denial actually prevents a later gate from running — not just that each gate works in isolation. See [testing-your-gates.md](testing-your-gates.md) for the full pattern, generalized from this repo's own test suite.

## Next

- [concepts.md](concepts.md) — the vocabulary and reasoning behind these rules.
- [troubleshooting.md](troubleshooting.md) — the real, reproduced failures most of these rules come from.
- [testing-your-gates.md](testing-your-gates.md) — how to prove your own gates are correct, not just written.
