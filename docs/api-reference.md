# API reference

Everything `vsl-langgraph` exports, from `src/vsl_langgraph/`. Three functions/classes total — this is meant to be scannable in one pass, because that's genuinely everything this package's own code surface is. Constructs it doesn't touch at all (`VerbaLedger`, `Cluster`, `TerminalState`, and the rest of `vsl-core`) are documented in [concepts.md](concepts.md)'s scope table and [beyond-the-gate.md](beyond-the-gate.md), not here.

```python
from vsl_langgraph import LangGraphAdapter, gated_node, route_on_denial, DEFAULT_DENIAL_KEY
```

---

## `LangGraphAdapter`

`src/vsl_langgraph/adapter.py`. Implements `vsl_core.conformance.protocol.VSLAdapter` — proven conformant by `run_conformance_suite(LangGraphAdapter())` returning `[]` (`tests/test_conformance.py`). No constructor arguments.

### `compile_pre_node(pre_node: PreNode) -> CompiledGate`

Returns an async gate. Calling it (`await gate(candidate_input)`):

1. Calls `pre_node.monitor(candidate_input)` (must be `async def` — see [troubleshooting.md#a-sync-monitor-or-rule-typeerror-on-await](troubleshooting.md#a-sync-monitor-or-rule-instead-of-async-typeerror-on-await)), getting back a `GammaEstimate`.
2. If `estimate.sufficient(pre_node.gamma_threshold)` is `False`, raises `AutomationDeniedException(reason=f"Pre-node '{pre_node.name}' insufficient Gamma", identity_key="vsl-langgraph")`.
3. Otherwise returns `None`.

### `compile_invariant(invariant: Invariant) -> CompiledGate`

Same shape, for `Invariant`:

1. Calls `invariant.rule(candidate_input)` (must be `async def`), getting back a `bool`.
2. If `False`, raises `InvariantViolation` — `reason` includes `invariant.name`, and if `invariant.on_violation` names a `TerminalState`, both `reason` and the exception's `terminal_state_name` field record it.
3. Otherwise returns `None`.

Both methods mirror `vsl_core.conformance.reference_adapter.PlainPythonReferenceAdapter` exactly — there's no LangGraph-specific compilation step, because LangGraph has no dedicated guardrail hook for one to target. The gate itself is framework-agnostic; only `gated_node`/`route_on_denial` below are LangGraph-specific.

---

## `gated_node`

```python
def gated_node(
    gate: CompiledGate,
    node: Callable[[Mapping[str, Any]], Any],
    *,
    candidate_input_fn: Callable[[Mapping[str, Any]], Any] | None = None,
    denial_key: str = DEFAULT_DENIAL_KEY,
) -> Callable[[Mapping[str, Any]], Awaitable[dict[str, Any]]]
```

`src/vsl_langgraph/integration.py`. Wraps `node` (any state-in/partial-update-out callable — sync or async, matching LangGraph's own node shape) so `gate` is checked immediately before it runs. Returns a new async callable suitable for `graph.add_node(name, gated_node(...))`.

| Param | Type | Default | Behavior |
|---|---|---|---|
| `gate` | `CompiledGate` | required | The compiled gate from `compile_pre_node`/`compile_invariant`. |
| `node` | state → partial update (sync or async) | required | Only called if `gate` doesn't raise. Its return value is passed through `dict(...)` and merged into state as-is. |
| `candidate_input_fn` | `state -> Any` | `None` (passes the whole state through unchanged) | Extracts whatever `gate` should actually evaluate from graph state — use this when the gate's `monitor`/`rule` expects something narrower than the full state dict. |
| `denial_key` | `str` | `"vsl_denial"` | The state key set to the caught exception on denial. **Give each `gated_node` in a chain its own `denial_key`** — see [troubleshooting.md#c-reusing-the-default-denial_key-across-chained-gates](troubleshooting.md#c-reusing-the-default-denial_key-across-chained-gates). |

**Behavior on call:**

- Calls `await gate(candidate_input_fn(state))`. Catches `AutomationDeniedException` (which `InvariantViolation` subclasses, so both are caught) and returns `{denial_key: exc}` — `node` is never invoked.
- Otherwise calls `node(state)`, awaiting the result if it's awaitable, and returns `dict(result) if result else {}`.
- **F1 pre-commitment** holds by construction: the function does not — cannot — reach the `node(state)` line without `gate(...)` having already completed without raising.

---

## `route_on_denial`

```python
def route_on_denial(
    blocked_node: str,
    allowed_node: str,
    *,
    denial_key: str = DEFAULT_DENIAL_KEY,
) -> Callable[[Mapping[str, Any]], str]
```

`src/vsl_langgraph/integration.py`. Builds a router function for `graph.add_conditional_edges(source, route_on_denial(...))`, matching `gated_node`'s convention.

| Param | Type | Default | Behavior |
|---|---|---|---|
| `blocked_node` | `str` | required | Node name to route to if `state[denial_key]` is not `None`. |
| `allowed_node` | `str` | required | Node name to route to otherwise. Can be `langgraph.graph.END`. |
| `denial_key` | `str` | `"vsl_denial"` | Must match the `denial_key` used on the corresponding `gated_node`. |

Returns `blocked_node` if `state.get(denial_key) is not None`, else `allowed_node`. Both `blocked_node` and `allowed_node` must already exist via `graph.add_node(...)` (or be `END`) — see [troubleshooting.md#d-route_on_denial-targeting-a-node-name-that-was-never-add_noded](troubleshooting.md#d-route_on_denial-targeting-a-node-name-that-was-never-add_noded) for what happens if one doesn't.

For `get_graph().draw_mermaid()` to render both branches (rather than silently showing neither), pass an explicit `path_map` too: `graph.add_conditional_edges(source, route_on_denial(...), {blocked_node: blocked_node, allowed_node: allowed_node})`. See [quickstart.md#visualizing-your-own-graph](quickstart.md#visualizing-your-own-graph).

---

## `DEFAULT_DENIAL_KEY`

`str = "vsl_denial"`. The default value of `gated_node`'s and `route_on_denial`'s `denial_key` parameter.
