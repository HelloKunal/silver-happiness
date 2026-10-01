"""Fault injection for resilience testing.

A FaultPlan describes *what* breaks (LLM calls per model, carrier API, payment API) and how often.
The eval harness runs the same tickets under different plans and measures whether cases still reach
a safe terminal state, and that no customer is ever refunded twice.
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass, field

from support_agent.config import LLMConfig, ModelSpec
from support_agent.llm.base import LLMError, LLMRequest, LLMResponse, RateLimitedError, TransientLLMError


@dataclass(frozen=True)
class FaultPlan:
    name: str = "none"
    # model id (e.g. "claude-opus-5-5") or "*" -> {"rate_limit": p, "server_error": p, "timeout": p}
    llm: dict[str, dict[str, float]] = field(default_factory=dict)
    carrier_error: float = 0.0
    payment_timeout_before_commit: float = 0.0
    payment_timeout_after_commit: float = 0.0  # the nasty one: money moved, response lost
    rate_limit_retry_after_s: float = 1.0

    @property
    def active(self) -> bool:
        return bool(self.llm) or any(
            (self.carrier_error, self.payment_timeout_before_commit, self.payment_timeout_after_commit)
        )


def build_fault_plan(name: str, llm_config: LLMConfig) -> FaultPlan:
    primary = llm_config.chain("specialist")[0].model
    presets = {
        "none": FaultPlan(),
        # Everything is a bit unreliable, including the "timeout after the refund committed" case.
        "flaky": FaultPlan(
            name="flaky",
            llm={"*": {"rate_limit": 0.15, "server_error": 0.10, "timeout": 0.05}},
            carrier_error=0.3,
            payment_timeout_before_commit=0.15,
            payment_timeout_after_commit=0.25,
        ),
        # Primary model hard down -> fallback chain + circuit breaker must carry the load.
        "primary-outage": FaultPlan(name="primary-outage", llm={primary: {"server_error": 1.0}}),
        # Every model down -> cases must end in a clean human hand-off, never a crash or silence.
        "total-outage": FaultPlan(name="total-outage", llm={"*": {"server_error": 1.0}}),
    }
    if name not in presets:
        raise ValueError(f"unknown fault plan {name!r}; choose from {sorted(presets)}")
    return presets[name]


class FaultInjector:
    def __init__(self, plan: FaultPlan, seed: int = 7):
        self.plan = plan
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self.injected: dict[str, int] = {}

    def roll(self, probability: float, kind: str) -> bool:
        if probability <= 0:
            return False
        with self._lock:
            hit = self._rng.random() < probability
            if hit:
                self.injected[kind] = self.injected.get(kind, 0) + 1
        return hit

    def llm_fault(self, model: str) -> LLMError | None:
        rates = self.plan.llm.get(model) or self.plan.llm.get("*") or {}
        if self.roll(rates.get("rate_limit", 0.0), "llm_rate_limit"):
            return RateLimitedError(f"[chaos] 429 ({model})", retry_after_s=self.plan.rate_limit_retry_after_s)
        if self.roll(rates.get("server_error", 0.0), "llm_server_error"):
            return TransientLLMError(f"[chaos] 529 overloaded ({model})")
        if self.roll(rates.get("timeout", 0.0), "llm_timeout"):
            return TransientLLMError(f"[chaos] request timeout ({model})")
        return None


class ChaosProvider:
    """Wraps a real provider and fails some calls before they reach the network."""

    def __init__(self, inner, injector: FaultInjector):
        self._inner = inner
        self._injector = injector
        self.name = inner.name

    def complete(self, request: LLMRequest, model: ModelSpec) -> LLMResponse:
        fault = self._injector.llm_fault(model.model)
        if fault is not None:
            raise fault
        return self._inner.complete(request, model)
