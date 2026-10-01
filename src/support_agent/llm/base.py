"""Provider-neutral LLM contract.

Everything above this layer (graph nodes, agents, evals) speaks these types only, so a new provider
is one adapter class. Conversation messages are plain JSON-able dicts so they can live in LangGraph's
checkpointed state:

    {"role": "user", "text": "..."}
    {"role": "assistant", "text": "...", "tool_calls": [{"id", "name", "arguments"}],
     "native": {"provider": "anthropic", "content": [...]}}     # verbatim provider blocks, see below
    {"role": "tool_results", "results": [{"tool_call_id", "content", "is_error"}]}

`native` keeps the provider's exact assistant blocks (e.g. Claude's thinking blocks) so the same
provider can replay them unchanged; other providers rebuild the turn from `text` + `tool_calls`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from support_agent.config import ModelSpec

Message = dict[str, Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMRequest:
    system: str
    messages: list[Message]
    max_tokens: int = 8000
    tools: list[ToolSpec] = field(default_factory=list)
    output_schema: dict[str, Any] | None = None  # JSON schema -> structured output
    effort: str | None = None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str
    model: str
    provider: str
    usage: Usage = field(default_factory=Usage)
    native_content: list[dict[str, Any]] | None = None
    served_by_provider_fallback: bool = False
    # Filled in by the gateway:
    route: str = ""
    model_key: str = ""
    attempts: int = 1
    fallback_depth: int = 0  # 0 = primary model served the call
    latency_s: float = 0.0
    cost_usd: float = 0.0

    def assistant_message(self) -> Message:
        return {
            "role": "assistant",
            "text": self.text,
            "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in self.tool_calls],
            "native": {"provider": self.provider, "content": self.native_content} if self.native_content else None,
        }

    def call_record(self) -> dict[str, Any]:
        """Compact telemetry row stored in case state (drives eval metrics)."""
        return {
            "route": self.route,
            "model": self.model,
            "model_key": self.model_key,
            "attempts": self.attempts,
            "fallback_depth": self.fallback_depth,
            "provider_fallback": self.served_by_provider_fallback,
            "latency_s": round(self.latency_s, 3),
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "cache_read_tokens": self.usage.cache_read_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


class LLMProvider(Protocol):
    name: str

    def complete(self, request: LLMRequest, model: ModelSpec) -> LLMResponse: ...


# --------------------------------------------------------------------------- errors
# Adapters translate SDK exceptions into these, so resilience logic never depends on a vendor SDK.


class LLMError(Exception):
    retryable: bool = False  # worth retrying on the same model
    fallback_ok: bool = True  # worth trying the next model in the chain


class RateLimitedError(LLMError):
    retryable = True

    def __init__(self, message: str, retry_after_s: float | None = None):
        super().__init__(message)
        self.retry_after_s = retry_after_s


class TransientLLMError(LLMError):
    """5xx, overloaded, timeouts, dropped connections."""

    retryable = True


class ModelUnavailableError(LLMError):
    """Model missing or not permitted for this key - retrying the same model is pointless."""


class NonRetryableLLMError(LLMError):
    """Bad request / auth / truncated output. Another model or provider may still succeed."""


class InvalidOutputError(LLMError):
    """The model answered, but not in the shape the caller required."""


class RefusalError(LLMError):
    """The model (and any server-side fallback) declined. Route to a human rather than shopping models."""

    fallback_ok = False


class AllModelsFailedError(LLMError):
    """Every model in the chain failed. Carries the attempt log for the human hand-off."""

    fallback_ok = False

    def __init__(self, route: str, attempts: list[dict[str, Any]]):
        self.route = route
        self.attempts = attempts
        summary = "; ".join(f"{a['model']}#{a.get('attempt', '-')}: {a['error']}" for a in attempts[-6:])
        super().__init__(f"all models failed for route '{route}': {summary}")
