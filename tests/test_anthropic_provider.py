"""Claude adapter: request shape, append-only replay, response parsing, error translation."""

from __future__ import annotations

from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from support_agent.config import ModelSpec
from support_agent.llm.anthropic_provider import SERVER_SIDE_FALLBACK_BETA, AnthropicProvider
from support_agent.llm.base import (
    LLMRequest,
    ModelUnavailableError,
    NonRetryableLLMError,
    RateLimitedError,
    RefusalError,
    ToolSpec,
    TransientLLMError,
)
from support_agent.models import TriageResult

OPUS = ModelSpec(key="opus", provider="anthropic", model="claude-opus-5-5", supports_effort=True, server_side_fallback=True)
HAIKU = ModelSpec(key="haiku", provider="anthropic", model="claude-haiku-4-5", supports_effort=False)

RAW = {
    "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
    "content": [
        {"type": "thinking", "thinking": "", "signature": "sig-abc"},
        {"type": "text", "text": "Looking that up."},
        {"type": "tool_use", "id": "toolu_1", "name": "get_order", "input": {"order_id": "A1001"}},
    ],
    "stop_reason": "tool_use", "stop_sequence": None,
    "usage": {"input_tokens": 120, "output_tokens": 30, "cache_read_input_tokens": 80, "cache_creation_input_tokens": 0},
}


class FakeEndpoint:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.params = response, error, None

    def create(self, **params):
        self.params = params
        if self.error:
            raise self.error
        return self.response


def provider(response=None, error=None):
    std = FakeEndpoint(anthropic.types.Message.model_validate(response or RAW), error)
    beta = FakeEndpoint(anthropic.types.beta.BetaMessage.model_validate(response or RAW), error)
    client = SimpleNamespace(messages=std, beta=SimpleNamespace(messages=beta))
    return AnthropicProvider(client=client), std, beta


def status_error(cls, code, headers=None):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx2.Response(code, headers=headers or {}, request=request), body=None)


def test_request_uses_strict_tools_structured_output_effort_and_server_fallback():
    p, std, beta = provider()
    req = LLMRequest(system="sys", messages=[{"role": "user", "text": "hi"}], effort="low",
                     tools=[ToolSpec("get_order", "d", {"type": "object", "properties": {"order_id": {"type": "string"}},
                                                        "required": ["order_id"]})],
                     output_schema=TriageResult.model_json_schema())
    p.complete(req, OPUS)
    params = beta.params
    assert std.params is None
    assert params["betas"] == [SERVER_SIDE_FALLBACK_BETA] and params["fallbacks"] == "default"
    assert params["tools"][0]["strict"] is True
    assert params["tools"][0]["input_schema"]["additionalProperties"] is False
    assert params["output_config"]["effort"] == "low"
    assert params["output_config"]["format"]["type"] == "json_schema"
    assert "tool_choice" not in params  # forced tool choice is rejected by current models
    assert params["cache_control"] == {"type": "ephemeral"}


def test_effort_omitted_for_models_without_it():
    p, std, _ = provider()
    p.complete(LLMRequest(system="s", messages=[{"role": "user", "text": "hi"}], effort="low"), HAIKU)
    assert "output_config" not in std.params


def test_parses_text_tool_calls_usage_and_keeps_native_blocks():
    p, _, _ = provider()
    resp = p.complete(LLMRequest(system="s", messages=[{"role": "user", "text": "hi"}]), OPUS)
    assert resp.text == "Looking that up."
    assert resp.tool_calls[0].name == "get_order" and resp.tool_calls[0].arguments == {"order_id": "A1001"}
    assert resp.usage.cache_read_tokens == 80
    assert resp.native_content[0]["type"] == "thinking" and resp.native_content[0]["signature"] == "sig-abc"


def test_assistant_turns_replay_verbatim_for_same_provider():
    p, _, _ = provider()
    resp = p.complete(LLMRequest(system="s", messages=[{"role": "user", "text": "hi"}]), OPUS)
    history = [
        {"role": "user", "text": "hi"},
        resp.assistant_message(),
        {"role": "tool_results", "results": [{"tool_call_id": "toolu_1", "content": "{}", "is_error": False}]},
    ]
    native = p.to_native_messages(history)
    assert native[1]["content"] == RAW["content"]  # thinking block + signature untouched (append-only)
    assert native[2]["content"][0] == {"type": "tool_result", "tool_use_id": "toolu_1", "content": "{}", "is_error": False}


def test_foreign_assistant_turns_are_rebuilt_from_neutral_fields():
    p, _, _ = provider()
    msg = {"role": "assistant", "text": "ok", "tool_calls": [{"id": "x", "name": "t", "arguments": {"a": 1}}],
           "native": {"provider": "other", "content": ["opaque"]}}
    [native] = p.to_native_messages([msg])
    assert native["content"] == [{"type": "text", "text": "ok"}, {"type": "tool_use", "id": "x", "name": "t", "input": {"a": 1}}]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (status_error(anthropic.RateLimitError, 429, {"retry-after": "7"}), RateLimitedError),
        (status_error(anthropic.OverloadedError, 529), TransientLLMError),
        (status_error(anthropic.InternalServerError, 500), TransientLLMError),
        (status_error(anthropic.NotFoundError, 404), ModelUnavailableError),
        (status_error(anthropic.BadRequestError, 400), NonRetryableLLMError),
        (anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x")), TransientLLMError),
    ],
)
def test_sdk_errors_map_to_neutral_errors(error, expected):
    p, _, _ = provider(error=error)
    with pytest.raises(expected) as exc:
        p.complete(LLMRequest(system="s", messages=[{"role": "user", "text": "hi"}]), OPUS)
    if expected is RateLimitedError:
        assert exc.value.retry_after_s == 7.0


def test_refusal_and_truncation():
    p, _, _ = provider(response={**RAW, "stop_reason": "refusal", "content": []})
    with pytest.raises(RefusalError):
        p.complete(LLMRequest(system="s", messages=[{"role": "user", "text": "hi"}]), OPUS)
    p, _, _ = provider(response={**RAW, "stop_reason": "max_tokens"})
    with pytest.raises(NonRetryableLLMError):
        p.complete(LLMRequest(system="s", messages=[{"role": "user", "text": "hi"}]), OPUS)
