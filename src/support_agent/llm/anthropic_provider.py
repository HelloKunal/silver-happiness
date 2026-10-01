"""Claude adapter: neutral request -> Messages API -> neutral response.

Notes on the API surface used here (anthropic SDK 1.x):
- SDK retries are disabled (`max_retries=0`); the gateway owns retry/fallback so there is one policy.
- Tools are declared `strict` so tool arguments always match the schema; structured output uses
  `output_config.format`. Forced `tool_choice` is not used (the current models reject it).
- For models that support it, `fallbacks="default"` asks the API to re-run a policy refusal on
  Anthropic's recommended fallback model inside the same call. Rate limits and outages are NOT
  covered by that - those are the gateway's job.
- Assistant turns are replayed verbatim from `native` content, keeping the transcript append-only
  so thinking blocks stay valid across turns of the tool loop.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic

from support_agent.config import ModelSpec
from support_agent.llm.base import (
    LLMRequest,
    LLMResponse,
    Message,
    ModelUnavailableError,
    NonRetryableLLMError,
    RateLimitedError,
    RefusalError,
    ToolCall,
    TransientLLMError,
    Usage,
)

SERVER_SIDE_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, timeout_s: float = 90.0, client: anthropic.Anthropic | None = None):
        self._client = client or anthropic.Anthropic(max_retries=0, timeout=timeout_s)

    def complete(self, request: LLMRequest, model: ModelSpec) -> LLMResponse:
        params = self.build_params(request, model)
        try:
            if model.server_side_fallback:
                message = self._client.beta.messages.create(
                    betas=[SERVER_SIDE_FALLBACK_BETA], fallbacks="default", **params
                )
            else:
                message = self._client.messages.create(**params)
        except anthropic.RateLimitError as e:
            raise RateLimitedError(f"429 rate limited ({model.model})", retry_after_s=_retry_after(e)) from e
        except (
            anthropic.OverloadedError,
            anthropic.ServiceUnavailableError,
            anthropic.DeadlineExceededError,
            anthropic.InternalServerError,
        ) as e:
            raise TransientLLMError(f"{e.status_code} {e.type or 'server error'} ({model.model})") from e
        except (anthropic.NotFoundError, anthropic.PermissionDeniedError) as e:
            raise ModelUnavailableError(f"{e.status_code} model unavailable ({model.model}): {e.message}") from e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise TransientLLMError(f"{e.status_code} ({model.model})") from e
            raise NonRetryableLLMError(f"{e.status_code} {e.type} ({model.model}): {e.message}") from e
        except anthropic.APIConnectionError as e:  # includes APITimeoutError
            raise TransientLLMError(f"connection/timeout ({model.model}): {e}") from e
        return self.parse_message(message, model)

    # ------------------------------------------------------------------ request

    def build_params(self, request: LLMRequest, model: ModelSpec) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": model.model,
            "max_tokens": request.max_tokens,
            "system": request.system,
            # `fallback` marker blocks only exist on the beta (server-side fallback) surface.
            "messages": self.to_native_messages(request.messages, keep_fallback_markers=model.server_side_fallback),
            # Auto-caching: the system prompt + tools are identical across tickets, and inside a
            # tool loop each turn re-reads the previous turn's prefix from cache.
            "cache_control": {"type": "ephemeral"},
        }
        if request.tools:
            params["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": anthropic.transform_schema(t.input_schema),
                    "strict": True,
                }
                for t in request.tools
            ]
        output_config: dict[str, Any] = {}
        if request.effort and model.supports_effort:
            output_config["effort"] = request.effort
        if request.output_schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": anthropic.transform_schema(request.output_schema),
            }
        if output_config:
            params["output_config"] = output_config
        return params

    def to_native_messages(self, messages: list[Message], keep_fallback_markers: bool = True) -> list[dict[str, Any]]:
        native: list[dict[str, Any]] = []
        for m in messages:
            role = m["role"]
            if role == "user":
                native.append({"role": "user", "content": m["text"]})
            elif role == "assistant":
                raw = m.get("native")
                if raw and raw.get("provider") == self.name:
                    content = raw["content"]  # verbatim replay (thinking blocks included)
                    if not keep_fallback_markers:
                        content = [b for b in content if b.get("type") != "fallback"]
                else:
                    content = []
                    if m.get("text"):
                        content.append({"type": "text", "text": m["text"]})
                    for c in m.get("tool_calls", []):
                        content.append({"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]})
                native.append({"role": "assistant", "content": content})
            elif role == "tool_results":
                native.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": r["tool_call_id"],
                                "content": r["content"],
                                "is_error": bool(r.get("is_error")),
                            }
                            for r in m["results"]
                        ],
                    }
                )
            else:
                raise ValueError(f"unknown message role {role!r}")
        return native

    # ------------------------------------------------------------------ response

    def parse_message(self, message: Any, model: ModelSpec) -> LLMResponse:
        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise RefusalError(f"model declined (category={category})")
        if message.stop_reason == "max_tokens":
            raise NonRetryableLLMError(f"output truncated at max_tokens ({model.model})")

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in message.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                args = block.input if isinstance(block.input, dict) else json.loads(block.input)
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=args))

        usage = message.usage
        iterations = getattr(usage, "iterations", None) or []
        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=message.stop_reason or "end_turn",
            model=message.model,
            provider=self.name,
            usage=Usage(
                input_tokens=usage.input_tokens or 0,
                output_tokens=usage.output_tokens or 0,
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            ),
            native_content=[block.to_dict() for block in message.content],
            served_by_provider_fallback=any(getattr(i, "type", None) == "fallback_message" for i in iterations),
        )


def _retry_after(error: anthropic.APIStatusError) -> float | None:
    try:
        value = error.response.headers.get("retry-after")
        return float(value) if value is not None else None
    except (AttributeError, ValueError):
        return None
