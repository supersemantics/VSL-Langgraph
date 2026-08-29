# Troubleshooting

Every entry below was actually triggered against the real, installed `vsl-langgraph` (0.1.0) — not reasoned about from reading the code. The exact error text (or, in the more dangerous cases, exact *lack* of error text) is quoted as-is.

| Symptom | Jump to |
|---|---|
| `TypeError: ... can't be used in 'await' expression` from a gate call | [A](#a-sync-monitor-or-rule-instead-of-async-typeerror-on-await) |
| A gate that should have denied didn't — no error at all | [B](#b-forgetting-await-on-a-manually-called-gate-silent-bypass) |
| A later gate passed, but an earlier denial is still showing in state | [C](#c-reusing-the-default-denial_key-across-chained-gates) |
| Graph run finishes with partial state, nothing in your blocked/allowed node ran | [D](#d-route_on_denial-targeting-a-node-name-that-was-never-add_noded) |
| `ConformanceError` from `run_conformance_suite` | [E](#e-conformanceerror-from-a-broken-adapter) |
| A `PreNode` always denies even though your monitor looks "confident" | [F](#f-gamma_hat-mistaken-for-a-0-1-confidence-score) |

---

## A: sync monitor or rule instead of async (TypeError on await)

**Symptom:**
```
TypeError: object GammaEstimate can't be used in 'await' expression
```

**Cause:** `PreNode.monitor` and `Invariant.rule` both have to be `async def`. A plain `def` returns its value immediately (not a coroutine); `LangGraphAdapter`'s compiled gate does `await pre_node.monitor(candidate_input)`, and `await`ing a non-awaitable value is exactly this `TypeError`.

**Fix:** add `async` to the function definition.

**Reproduced:**
```python
def sync_monitor(candidate_input):  # missing `async`
    return GammaEstimate(gamma_hat=5.0)
```
```
TypeError: object GammaEstimate can't be used in 'await' expression
```

---

## B: forgetting `await` on a manually-called gate (silent bypass)

The most dangerous entry on this page — nothing fails loudly.

**Symptom:** the guarded action runs even though the gate should have denied it. Nothing raises. If you're watching stderr closely at exactly the right moment you might catch a `RuntimeWarning`, but it fires at garbage-collection time (not at the call site), so it's easy to miss entirely in production logs.

**Cause:** a compiled gate is `async def gate(candidate_input) -> None`. Calling `gate(state)` without `await` — most likely because a node was hand-written instead of built with `gated_node` — just constructs a coroutine object and discards it. None of the gate's body runs, so it never gets the chance to raise, regardless of whether it should have.

**Fix:** always go through `gated_node` — it awaits the gate correctly. If you're calling a compiled gate manually for some reason outside `gated_node`, that code deserves extra scrutiny in review specifically because this failure mode is silent and permissive, not loud.

**Reproduced** (a `PreNode` that should always deny, called without `await` inside a node):
```python
async def node_missing_await(state):
    gate_b(state)  # no `await`
    return {"ran": True}
```
```
node result despite gate that should have denied: {'ran': True}
warning captured: RuntimeWarning: coroutine 'LangGraphAdapter.compile_pre_node.<locals>.gate' was never awaited
```

---

## C: reusing the default `denial_key` across chained gates

**Symptom:** a later gate in a chain passes, its node's result merges into state fine — but `state["vsl_denial"]` still shows an *earlier* gate's denial, even though the most recent check actually passed.

**Cause:** `gated_node` only ever *sets* `denial_key` on the except branch. On the pass branch it returns the wrapped node's own result, which says nothing about `denial_key` at all. LangGraph merges state per-key; a key a later return value doesn't mention is left untouched. An earlier denial under the same key survives straight through a later, unrelated, actually-passing gate.

**Fix:** give every `gated_node` in a chain its own `denial_key` — `examples/financial-agents/purchase_approval_agent.py` uses `confidence_denial` and `cap_denial`, never the shared default, for exactly this reason.

**Reproduced** (two gates sharing the default `"vsl_denial"` key, gate 1 denies, gate 2 passes):
```
final state after gate1 DENIES then gate2 PASSES: {'vsl_denial': AutomationDeniedException(reason="Pre-node 'gate-1-denies' insufficient Gamma", identity_key='vsl-langgraph', ...), 'step': 2}
state['vsl_denial'] is still set: True  <- stale, gate2 actually passed
```

---

## D: `route_on_denial` targeting a node name that was never `add_node`'d

**Symptom:** no exception. `ainvoke()` returns normally, but with only partial state — as if the graph simply stopped after the gated node, never reaching the blocked or allowed node you expected. A one-line warning appears on **stderr**, easy to miss in production logging that only captures stdout or structured logs.

**Cause:** LangGraph doesn't validate a router's possible return values ahead of time, and doesn't raise when one turns out not to match a real node — it logs a warning and drops the transition, leaving the run stalled wherever it was.

**Fix:** double-check the node name strings passed to `route_on_denial` exactly match `graph.add_node(...)` calls (a typo is the most likely cause), and treat that stderr line as load-bearing — it's the only signal you get.

**Reproduced** (`route_on_denial("blocked_TYPO", "allowed")`, but only `"allowed"` was ever `add_node`'d):
```
stderr: Task gate_node with path ('__pregel_pull', 'gate_node') wrote to unknown channel branch:to:blocked_TYPO, ignoring it.
result: {'vsl_denial': AutomationDeniedException(reason="Pre-node 'gate-d' insufficient Gamma", identity_key='vsl-langgraph', ...)}
```
The graph never reached `"allowed"` or any blocked node — it just stopped.

---

## E: `ConformanceError` from a broken adapter

Only relevant if you're writing your own `VSLAdapter` rather than using `LangGraphAdapter` as-is.

**Symptom:**
```
ConformanceError: BrokenAdapter.compile_pre_node did not return a callable gate (got None); the conformance suite cannot run against it.
```

**Cause:** `compile_pre_node`/`compile_invariant` returned something that isn't callable — usually a missing `return` statement. This is a suite-can't-even-run error, distinct from an ordinary conformance failure (those come back as strings in the suite's returned list, never raised).

**Fix:** make sure both methods actually `return gate`.

---

## F: `gamma_hat` mistaken for a 0-1 confidence score

**Symptom:** a `PreNode` always denies even though your monitor looks like it's returning "high confidence" values.

**Cause:** `GammaEstimate.gamma_hat` is not a probability. It's compared directly against `gamma_threshold` (default `1.1`, `vsl_core.metrics.ROBUST_GAMMA_DEFAULT_THRESHOLD`). A monitor returning `0.95` meaning "95% confident" is really comparing `0.95 >= 1.1` — always `False`.

**Fix:** pick a `gamma_hat` scale that's meaningfully compared against your `gamma_threshold` — either rescale a true 0-1 confidence score so "confident" clears the threshold, or set `gamma_threshold` to something reachable on your actual scale, rather than assuming the default `1.1` fits every monitor.

**Reproduced:**
```
ROBUST_GAMMA_DEFAULT_THRESHOLD = 1.1
GammaEstimate(gamma_hat=0.95).sufficient() = False  <- '95% confident' still FAILS, Gamma isn't a probability
GammaEstimate(gamma_hat=1.5).sufficient() = True
```

---

## Still stuck?

- [concepts.md](concepts.md) — what `PreNode`/`Invariant`/`GammaEstimate`/`AssuranceBasis` actually mean.
- [api-reference.md](api-reference.md) — exact signatures and behavior for everything this package exports.
- [`vsl-core`'s own README](https://github.com/supersemantics/VSL-Core#readme) and [`vsl-core-primitives.json`](https://github.com/supersemantics/VSL-Core/blob/main/docs/vsl-core-primitives.json) — the canonical reference for anything not specific to LangGraph wiring (`Fallback`, `TerminalState`, `VerbaLedger`, identity/re-enablement).
