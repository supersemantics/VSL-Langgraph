# Beyond the gate

[concepts.md](concepts.md)'s scope table lists what `LangGraphAdapter` actually compiles: `PreNode` and `Invariant`, nothing else. Everything on this page is the rest of `vsl-core`'s vocabulary — used inside a LangGraph app exactly the same way it's used in any plain Python `vsl-core` app, because `vsl-langgraph` adds no wrapper, no special method, and no LangGraph-specific behavior for any of it. That's not an oversight; there's nothing to compile for a `Cluster` or a `VerbaLedger` the way there is for a `PreNode` — but "nothing to compile" still needs to be shown working, not just asserted, so both examples below are real, run code with real captured output.

## The full lifecycle: log and escalate

`vsl-core`'s own five-step lifecycle is **construct → compile → call → log → escalate**. [quickstart.md](quickstart.md) and every sample agent in `examples/` cover the first three. This section is the last two, wired into an actual LangGraph node — because `gated_node` deliberately doesn't do either of them (see [concepts.md#gated_node-doesnt-write-to-the-ledger-either](concepts.md#gated_node-doesnt-write-to-the-ledger-either)).

The scenario: a refund agent with a non-bypassable `Invariant` (`refund-eligibility`) that names a `TerminalState` (`governance_halt`) to enter on violation. When it violates, a second node — reached via the same `route_on_denial` pattern as every other example — writes the `TERMINAL` ledger entry. Later, a human reviews it and calls `request_re_enablement()`.

```python
from typing import Any, TypedDict
from langgraph.graph import END, START, StateGraph
from vsl_core.constructs import Invariant, TerminalState
from vsl_core.exceptions import InvariantViolation
from vsl_core.governance import GovernanceAuthority, request_re_enablement
from vsl_core.identity import Evidence, IdentityKey, Instance
from vsl_core.ledger import LedgerEntryType, VerbaLedger
from vsl_core.metrics import AssuranceBasis, F2Modification
from vsl_langgraph import LangGraphAdapter, gated_node, route_on_denial

BASIS = AssuranceBasis(f1_pre_commitment=True, f2_modification=F2Modification.FULL)

# 1. CONSTRUCT
governance_halt = TerminalState(
    name="governance_halt",
    description="Halts pending human review after the refund-eligibility invariant is violated.",
    entry_conditions=("invariant_violated",),
)

async def eligible_for_refund(state: dict) -> bool:
    return state["order"]["eligible"]

refund_invariant = Invariant(
    name="refund-eligibility",
    description="Never issue a refund without confirmed eligibility -- non-bypassable.",
    rule=eligible_for_refund,
    assurance_basis=BASIS,
    on_violation=governance_halt,
)

# 2. COMPILE
gate = LangGraphAdapter().compile_invariant(refund_invariant)

ledger = VerbaLedger()
identity_key = IdentityKey.generate("refund-agent")
instance = Instance.new(identity_key=identity_key)

class RefundState(TypedDict, total=False):
    order: dict
    refunded: bool
    vsl_denial: Any
    ledger_entry_id: str

def issue_refund(state: RefundState) -> dict:
    return {"refunded": True}

async def governance_halt_node(state: RefundState) -> dict:
    denial: InvariantViolation = state["vsl_denial"]
    # 4. LOG -- gated_node never does this; the node reached via
    # route_on_denial is where the calling application writes it.
    entry = ledger.write(
        LedgerEntryType.TERMINAL,
        identity_key=str(identity_key),
        instance_id=instance.instance_id,
        payload={"reason": denial.reason, "terminal_state": denial.terminal_state_name},
    )
    return {"ledger_entry_id": entry.entry_id}

graph = StateGraph(RefundState)
# 3. CALL -- via gated_node, same as every other example in this repo
graph.add_node("refund_check", gated_node(gate, issue_refund))
graph.add_node("governance_halt", governance_halt_node)
graph.add_edge(START, "refund_check")
graph.add_conditional_edges(
    "refund_check",
    route_on_denial("governance_halt", END),
    {"governance_halt": "governance_halt", END: END},
)
graph.add_edge("governance_halt", END)
app = graph.compile()
```

Eligible order — passes straight through, nothing to log:
```python
>>> await app.ainvoke({"order": {"eligible": True}})
{'order': {'eligible': True}, 'refunded': True}
```

Ineligible order — `Invariant` violates, `governance_halt` node writes the `TERMINAL` entry:
```python
>>> result = await app.ainvoke({"order": {"eligible": False}})
>>> result
{'order': {'eligible': False}, 'vsl_denial': InvariantViolation(reason="Invariant 'refund-eligibility' violated -- entering terminal state 'governance_halt'", identity_key='vsl-langgraph', instance_id=None, timestamp=..., invariant_name='refund-eligibility', terminal_state_name='governance_halt'), 'ledger_entry_id': '9f41e913-ef70-4bf7-977e-b9e1aba773c4'}
```

**5. ESCALATE** — later, once a human has reviewed it, outside the graph run entirely:

```python
authority = GovernanceAuthority(name="Risk Committee", escalation_contact="risk@example.com")
evidence = Evidence(
    root_cause_analysis="Order was flagged ineligible due to a stale eligibility cache; manual review confirmed the customer is actually eligible.",
    specification_update_proposal="Refresh the eligibility cache every 5 minutes instead of hourly.",
)
new_instance, transition = request_re_enablement(instance, authority, authorised_by="j.doe", evidence=evidence)

hat_entry = ledger.write(
    LedgerEntryType.HUMAN_AUTHORISED_TRANSITION,
    identity_key=str(identity_key),
    instance_id=instance.instance_id,
    caused_by=result["ledger_entry_id"],  # explicit causal link to the TERMINAL entry it resolves
    payload={"authorised_by": transition.authorised_by, "authority": transition.authority.name},
)
ledger.write(
    LedgerEntryType.RE_ENABLEMENT,
    identity_key=str(new_instance.identity_key),
    instance_id=new_instance.instance_id,
    caused_by=hat_entry.entry_id,
    payload={"predecessor_instance_id": new_instance.predecessor_instance_id},
)
```
```python
>>> new_instance.generation, new_instance.predecessor_instance_id
(1, 'cf201c18-ccc3-4ef1-9647-ebc58744b4af')
>>> transition.authorised_by, transition.authority.name
('j.doe', 'Risk Committee')

>>> report = ledger.audit()
>>> report.all_passed
True
>>> report.checks_passed
('no_monitoring_gaps', 'drift_flagged_monitor_has_pre_node', 'pre_node_has_verification', 'insufficient_verification_has_specification_update', 'terminal_has_human_authorised_transition')

>>> ledger.issue_certificate()
VerbaCertificate(entries_covered=3, all_passed=True) -- This certificate does not guarantee no Inadmissible State was ever reached. It certifies that every detected Drift event was addressed, Terminal States were properly escalated, and governance was continuously monitored -- it certifies the governance process, not the underlying system, model, or domain.

>>> ledger.verify_integrity()
True
```

`caused_by` is what makes `terminal_has_human_authorised_transition` pass deterministically rather than by timestamp-ordering luck — see `vsl-core`'s own note on this in [`vsl-core-primitives.json`](https://github.com/supersemantics/VSL-Core/blob/main/docs/vsl-core-primitives.json)'s `VerbaLedger.important_gotchas`.

## Multi-agent governance: `Cluster`

LangGraph is specifically a multi-agent framework, so this is the construct most worth not skipping. `Cluster` aggregates by `IdentityKey` — "multi-agent" here means identity-scoped, not necessarily separate processes, so two nodes in one `StateGraph` already qualify.

The scenario: a `book` node and a `pay` node, each representing its own sub-agent's independent work. A `checkout_consistency` `Invariant` checks something neither agent's own local view can see on its own — that the amounts they each independently committed to actually match. This is `vsl-core`'s own Flash Crash motivation, concretely: each agent can look individually fine while the pair together isn't.

```python
from vsl_core.cluster import Cluster
from vsl_core.constructs import Invariant
from vsl_core.identity import ClusterKey, IdentityKey

booking_identity = IdentityKey.generate("booking")
payments_identity = IdentityKey.generate("payments")
cluster_key = ClusterKey.from_constituents((booking_identity, payments_identity))

async def amounts_match(state: dict) -> bool:
    return state["booking"]["reserved_amount"] == state["payment"]["charge_amount"]

checkout_consistency = Invariant(
    name="checkout-amount-consistency",
    description="The payment agent's charge must match the booking agent's reserved amount.",
    rule=amounts_match,
    assurance_basis=BASIS,
)

checkout_cluster = Cluster(
    name="checkout_flow",
    cluster_key=cluster_key,
    constituents=(booking_identity, payments_identity),
    invariants=(checkout_consistency,),
)

# The interesting part: NO special Cluster-aware method exists on
# LangGraphAdapter. A Cluster's Invariant is a plain Invariant -- compile
# and wire it exactly like any other.
gate = LangGraphAdapter().compile_invariant(checkout_cluster.invariants[0])
```

Wired into a graph the normal way (`book` → `pay` → `checkout_check` (gated) → confirm or block), both scenarios run for real:

```python
>>> consistent = {"booking": {"reserved_amount": 250.0}, "payment": {"charge_amount": 250.0}}
>>> await app.ainvoke(consistent)
{'booking': {'reserved_amount': 250.0}, 'payment': {'charge_amount': 250.0}, 'confirmed': True}

>>> inconsistent = {"booking": {"reserved_amount": 250.0}, "payment": {"charge_amount": 999.0}}
>>> await app.ainvoke(inconsistent)
{'booking': {'reserved_amount': 250.0}, 'payment': {'charge_amount': 999.0}, 'vsl_denial': InvariantViolation(reason="Invariant 'checkout-amount-consistency' violated", identity_key='vsl-langgraph', instance_id=None, timestamp=..., invariant_name='checkout-amount-consistency', terminal_state_name=None)}
```

`Cluster`'s own safety check — its `__post_init__` refuses to construct at all if a declared constituent isn't covered by its `ClusterKey`, so a Cluster can't silently omit a member the way the real Flash Crash's governance did:

```python
>>> stray_identity = IdentityKey.generate("stray")
>>> Cluster(name="broken_cluster", cluster_key=cluster_key, constituents=(booking_identity, payments_identity, stray_identity))
ValueError: Cluster 'broken_cluster' declares constituents not covered by its own cluster_key: [IdentityKey(value='stray-722028cd-...')]. A Cluster's ClusterKey must be derived from (at least) all of its declared constituents.
```

If you want ledger entries scoped at the cluster level rather than per-agent, `VerbaLedger.write(...)` takes a `cluster_key` kwarg (`str(cluster_key.value)`) alongside `identity_key` — same `write`/`write_monitor`/`write_verification` methods as the lifecycle example above, just with one more field populated.

## Policy-as-Code: `Specification`, `Proofing`, `Binding`

For bundling a versioned, machine-readable set of `PreNode`/`Invariant`/`TerminalState` objects that requires sign-off before it's live — not a runtime gating mechanism, so there's nothing LangGraph-specific about it either. A `Specification` needs a `GovernanceAuthority` (`approved_by`) at construction and refuses to `activate()` without a `Binding`, which itself refuses to construct from an unverified `Proofing`:

```python
from vsl_core.governance import Binding, GovernanceAuthority, Proofing, Specification

authority = GovernanceAuthority(name="Risk Committee", escalation_contact="risk@example.com")
spec = Specification(name="refund-policy", version="1.0", approved_by=authority, invariants=(refund_invariant,))

proofing = Proofing(verified=True, verified_against="policy-doc-v1.0.pdf#refund-eligibility")
binding = Binding(proofing=proofing, escalation_path="risk@example.com")
active_spec = spec.with_binding(binding).activate()
```

Whatever `PreNode`/`Invariant` objects end up in an active `Specification` still get compiled and wired into your `StateGraph` exactly the same way as any other — `Specification` is a bundling/approval layer sitting *above* the objects you're already constructing, not a different object to compile.

## Descriptive-only: `AllowedState`, `InadmissibleState`

Neither has a compile step or a runtime check — they're classification/documentation types for describing outcomes (`AllowedState`: met all prerequisites; `InadmissibleState`: must not occur, described by a `signature` string), not objects a `VSLAdapter` does anything with. The actual runtime enforcement of "don't reach this `InadmissibleState`" is whatever `PreNode`/`Invariant` you've gated the relevant LangGraph node with — these two types just let you name and document the state your gates are protecting, for a reader or an auditor, not for the graph itself.

## Next

- [concepts.md](concepts.md) — the scope table this page starts from, and the core PreNode/Invariant vocabulary.
- [`vsl-core-primitives.json`](https://github.com/supersemantics/VSL-Core/blob/main/docs/vsl-core-primitives.json) — the canonical, field-exact reference for every construct on this page.
