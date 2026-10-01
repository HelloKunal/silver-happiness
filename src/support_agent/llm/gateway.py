"""LLM gateway: one entry point for every model call in the workflow.

Recovery ladder for a single call on route R with chain [m1, m2, m3]:

  1. Circuit breaker open for m1?  -> skip straight to m2 (fail fast during an outage).
  2. Call m1. 429 -> wait `retry-after` (if it is short) or backoff, retry m1.
              5xx / overloaded / timeout -> jittered exponential backoff, retry m1.
              model unavailable / bad request / truncated / malformed output -> no retry, next model.
              refusal -> stop: a policy decline goes to a human, not to model-shopping.
  3. m1 exhausted -> record breaker failure -> m2 -> m3.
  4. Everything failed -> AllModelsFailedError with the full attempt log. The graph's error handler
     turns that into a human hand-off, so the customer is never left without a response.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from support_agent.config import LLMConfig, ModelSpec
from support_agent.llm.base import (
    AllModelsFailedError,
    InvalidOutputError,
    LLMError,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    RateLimitedError,
    RefusalError,
)
from support_agent.resilience import CircuitBreaker, backoff_delay

log = logging.getLogger(__name__)

Validator = Callable[[LLMResponse], Any]


class LLMGateway:
    def __init__(
        self,
        config: LLMConfig,
        providers: dict[str, LLMProvider],
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ):
        self.config = config
        self.providers = providers
        self._sleep = sleep
        self._clock = clock
        self._rng = rng or random.Random()
        self.breakers = {
            key: CircuitBreaker(
                name=key,
                failure_threshold=config.breaker.failure_threshold,
                cooldown_s=config.breaker.cooldown_s,
                clock=clock,
            )
            for key in config.models
        }

    def complete(self, route: str, request: LLMRequest, validate: Validator | None = None) -> tuple[LLMResponse, Any]:
        """Run `request` on the route's model chain. Returns (response, validate(response))."""
        route_spec = self.config.routes[route]
        request = _with_route_defaults(request, route_spec.max_tokens, route_spec.effort)
        retry = self.config.retry
        attempts_log: list[dict[str, Any]] = []
        total_attempts = 0
        started = self._clock()

        for depth, spec in enumerate(self.config.chain(route)):
            breaker = self.breakers[spec.key]
            if not breaker.allow():
                attempts_log.append({"model": spec.model, "error": "circuit open - skipped"})
                continue
            provider = self.providers[spec.provider]
            model_is_healthy = False  # did the model answer at all (even if unusably)?

            for attempt in range(1, retry.max_attempts_per_model + 1):
                total_attempts += 1
                try:
                    response = provider.complete(request, spec)
                    parsed = validate(response) if validate else None
                except RefusalError:
                    breaker.record_success()  # the model is healthy; it just declined
                    raise
                except InvalidOutputError as e:
                    attempts_log.append({"model": spec.model, "attempt": attempt, "error": str(e)})
                    model_is_healthy = True
                    break  # same model, same prompt -> likely same bad output; try the next model
                except LLMError as e:
                    attempts_log.append({"model": spec.model, "attempt": attempt, "error": str(e)})
                    if not e.retryable or attempt == retry.max_attempts_per_model:
                        break
                    delay = self._retry_delay(e, attempt)
                    if delay is None:
                        log.warning("%s asked for a long wait; falling back instead", spec.model)
                        break
                    log.info("retrying %s in %.1fs after: %s", spec.model, delay, e)
                    self._sleep(delay)
                    continue
                else:
                    breaker.record_success()
                    response.route = route
                    response.model_key = spec.key
                    response.attempts = total_attempts
                    response.fallback_depth = depth
                    response.latency_s = self._clock() - started
                    response.cost_usd = _cost(spec, response)
                    if depth:
                        log.warning("route %s served by fallback model %s", route, spec.model)
                    return response, parsed
            if model_is_healthy:
                breaker.record_success()
            else:
                breaker.record_failure()

        raise AllModelsFailedError(route, attempts_log)

    def _retry_delay(self, error: LLMError, attempt: int) -> float | None:
        retry = self.config.retry
        if isinstance(error, RateLimitedError) and error.retry_after_s is not None:
            if error.retry_after_s > retry.max_retry_after_s:
                return None  # waiting that long is worse than using the next model
            return error.retry_after_s
        return backoff_delay(attempt, retry.base_delay_s, retry.max_delay_s, self._rng)


def _with_route_defaults(request: LLMRequest, max_tokens: int, effort: str | None) -> LLMRequest:
    return replace(request, max_tokens=max_tokens, effort=request.effort or effort)


def _cost(spec: ModelSpec, response: LLMResponse) -> float:
    u = response.usage
    # Cache reads bill at a fraction of input; approximate at 10% (5% on newest models) - fine for estimates.
    billable_in = u.input_tokens + u.cache_write_tokens * 1.25 + u.cache_read_tokens * 0.1
    return (billable_in * spec.input_cost_per_mtok + u.output_tokens * spec.output_cost_per_mtok) / 1_000_000
