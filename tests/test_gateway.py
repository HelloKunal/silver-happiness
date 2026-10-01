"""LLM gateway: retry, rate-limit handling, fallback chain, circuit breaker."""

from __future__ import annotations

import random

import pytest

from support_agent.config import BreakerSettings, LLMConfig, ModelSpec, RetrySettings, RouteSpec
from support_agent.llm.base import (
    AllModelsFailedError,
    InvalidOutputError,
    LLMRequest,
    ModelUnavailableError,
    RateLimitedError,
    RefusalError,
    TransientLLMError,
)
from support_agent.llm.gateway import LLMGateway

from .fakes import text


def config(**retry) -> LLMConfig:
    models = {k: ModelSpec(key=k, provider="p", model=f"model-{k}") for k in ("a", "b")}
    return LLMConfig(
        models=models,
        routes={"r": RouteSpec(name="r", chain=("a", "b"))},
        retry=RetrySettings(**{"max_attempts_per_model": 3, "base_delay_s": 1.0, "max_delay_s": 8.0, "max_retry_after_s": 10.0, **retry}),
        breaker=BreakerSettings(failure_threshold=2, cooldown_s=30.0),
    )


class Scripted:
    name = "p"

    def __init__(self, outcomes: dict[str, list]):
        self.outcomes = outcomes  # model -> list of Exception|"ok" consumed in order; last one repeats
        self.calls: list[str] = []

    def complete(self, request, model):
        self.calls.append(model.model)
        queue = self.outcomes[model.model]
        outcome = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(outcome, Exception):
            raise outcome
        return text("ok", model=model.model)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def gateway(provider, clock=None, **retry):
    sleeps: list[float] = []
    clock = clock or Clock()

    def sleep(s):
        sleeps.append(s)
        clock.t += s

    gw = LLMGateway(config(**retry), {"p": provider}, sleep=sleep, clock=clock, rng=random.Random(0))
    return gw, sleeps


REQ = LLMRequest(system="s", messages=[{"role": "user", "text": "hi"}])


def test_retries_transient_then_succeeds_on_same_model():
    p = Scripted({"model-a": [TransientLLMError("529"), TransientLLMError("529"), "ok"], "model-b": ["ok"]})
    gw, sleeps = gateway(p)
    resp, _ = gw.complete("r", REQ)
    assert resp.model == "model-a" and resp.attempts == 3 and resp.fallback_depth == 0
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0] * 0.9  # exponential-ish backoff


def test_honours_retry_after_on_429():
    p = Scripted({"model-a": [RateLimitedError("429", retry_after_s=4.0), "ok"], "model-b": ["ok"]})
    gw, sleeps = gateway(p)
    gw.complete("r", REQ)
    assert sleeps == [4.0]


def test_long_retry_after_falls_back_immediately():
    p = Scripted({"model-a": [RateLimitedError("429", retry_after_s=120.0)], "model-b": ["ok"]})
    gw, sleeps = gateway(p)
    resp, _ = gw.complete("r", REQ)
    assert resp.model == "model-b" and resp.fallback_depth == 1
    assert sleeps == [] and p.calls == ["model-a", "model-b"]


def test_non_retryable_error_skips_to_next_model_without_retry():
    p = Scripted({"model-a": [ModelUnavailableError("404")], "model-b": ["ok"]})
    gw, _ = gateway(p)
    resp, _ = gw.complete("r", REQ)
    assert p.calls == ["model-a", "model-b"] and resp.model == "model-b"


def test_invalid_output_tries_next_model():
    p = Scripted({"model-a": ["ok"], "model-b": ["ok"]})
    gw, _ = gateway(p)

    def validate(resp):
        if resp.model == "model-a":
            raise InvalidOutputError("bad json")
        return "parsed"

    resp, parsed = gw.complete("r", REQ, validate=validate)
    assert resp.model == "model-b" and parsed == "parsed"
    assert gw.breakers["a"].state == "closed"  # a malformed answer is not an outage


def test_refusal_is_not_model_shopped():
    p = Scripted({"model-a": [RefusalError("declined")], "model-b": ["ok"]})
    gw, _ = gateway(p)
    with pytest.raises(RefusalError):
        gw.complete("r", REQ)
    assert p.calls == ["model-a"]


def test_all_models_failed_carries_attempt_log():
    p = Scripted({"model-a": [TransientLLMError("500")], "model-b": [TransientLLMError("500")]})
    gw, _ = gateway(p)
    with pytest.raises(AllModelsFailedError) as exc:
        gw.complete("r", REQ)
    assert len(exc.value.attempts) == 6


def test_circuit_breaker_opens_then_half_opens():
    clock = Clock()
    p = Scripted({"model-a": [TransientLLMError("503")], "model-b": ["ok"]})
    gw, _ = gateway(p, clock=clock, max_attempts_per_model=1)
    gw.complete("r", REQ)
    gw.complete("r", REQ)  # second consecutive failure -> circuit opens
    assert gw.breakers["a"].state == "open"

    p.calls.clear()
    gw.complete("r", REQ)
    assert p.calls == ["model-b"]  # primary skipped entirely while open

    clock.t += 31  # cooldown elapsed -> one probe allowed
    p.outcomes["model-a"] = ["ok"]
    p.calls.clear()
    resp, _ = gw.complete("r", REQ)
    assert p.calls == ["model-a"] and resp.model == "model-a"
    assert gw.breakers["a"].state == "closed"
