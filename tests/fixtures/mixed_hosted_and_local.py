"""Deliberately hostile fixture: two UNRELATED monitors in the SAME file,
one that genuinely reaches a hosted API, one that doesn't -- regression
test for the false-positive bug found while building a companion
prototype (module-wide source scan let one function's mention contaminate
its neighbor's result).

Note for whoever edits this file: keep hosted-API marker words out of
mixed_local_monitor's own body, including its comments -- inspect.getsource(fn)
includes a function's own comments, so putting a marker there would defeat
the point of this fixture (this exact mistake was made once while building
the companion prototype and had to be fixed).
"""

from typing import Any

from vsl_core.metrics import GammaEstimate


async def mixed_hosted_monitor(candidate_input: Any) -> GammaEstimate:
    # Real integration: client = anthropic.AsyncAnthropic(api_key=...)
    return GammaEstimate(gamma_hat=2.0)


async def mixed_local_monitor(candidate_input: Any) -> GammaEstimate:
    # Plain arithmetic. No model access of any kind.
    amount = candidate_input.get("amount", 0.0) if isinstance(candidate_input, dict) else 0.0
    return GammaEstimate(gamma_hat=2.0 if amount <= 1000.0 else 0.0)
