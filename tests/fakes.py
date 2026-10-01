"""Scripted LLM for deterministic workflow tests (no API key, no network)."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from support_agent.agents.prompts import REPLY_SYSTEM, TRIAGE_SYSTEM
from support_agent.config import ModelSpec
from support_agent.llm.base import LLMRequest, LLMResponse, ToolCall, Usage

Step = list[tuple[str, dict[str, Any]]]  # tool calls the fake specialist makes in one turn


def text(content: str, model: str = "fake-model") -> LLMResponse:
    return LLMResponse(text=content, tool_calls=[], stop_reason="end_turn", model=model, provider="anthropic",
                       usage=Usage(input_tokens=100, output_tokens=20))


def tools(calls: Step, model: str = "fake-model", turn: int = 0) -> LLMResponse:
    return LLMResponse(
        text="",
        tool_calls=[ToolCall(id=f"call_{turn}_{i}", name=n, arguments=a) for i, (n, a) in enumerate(calls)],
        stop_reason="tool_use", model=model, provider="anthropic", usage=Usage(input_tokens=200, output_tokens=40),
    )


def triage(intent: str, order_ids: list[str] | None = None, risk: list[str] | None = None, confidence: float = 0.95) -> dict:
    return {
        "intent": intent, "confidence": confidence, "order_ids": order_ids or [], "summary": "test summary",
        "sentiment": "neutral", "urgency": "normal", "risk_signals": risk or [],
    }


def resolution(outcome: str = "resolve", actions: list[dict] | None = None, transfer_to: str | None = None,
               reply_points: list[str] | None = None, handoff_reason: str | None = None) -> dict:
    full_actions = [
        {"order_id": None, "amount": None, "new_value": None, "reason": "test", **a} for a in (actions or [])
    ]
    return {
        "outcome": outcome, "case_notes": "test notes", "reply_points": reply_points or ["We looked into it."],
        "handoff_reason": handoff_reason, "actions": full_actions, "transfer_to": transfer_to,
    }


def auto_reply(request: LLMRequest) -> LLMResponse:
    """Writes a reply that mentions every completed refund amount (so the reply guard passes)."""
    outcome = json.loads(re.search(r"<case_outcome>\n(.*?)\n</case_outcome>", request.messages[0]["text"], re.S).group(1))
    amounts = [f"${float(a['amount']):.2f}" for a in outcome["completed_actions"] if a.get("amount")]
    body = "Hi,\n\n" + " ".join(outcome["reply_points"])
    if amounts:
        body += f"\nWe've refunded {', '.join(amounts)}."
    return text(json.dumps({"subject": "Re: your request", "body": body + "\n\nThe Harbor & Pine Support Team"}))


class FakeLLM:
    """Plays triage, one specialist script per specialist name, and the reply writer.

    specialist_scripts: {"refunds": [step, step, ...]} - one step per agent turn.
    fail: optional hook (request, model) -> Exception | None to inject failures.
    """

    name = "anthropic"

    def __init__(
        self,
        *,
        triage_result: dict,
        specialist_scripts: dict[str, list[Step]] | None = None,
        reply: Callable[[LLMRequest], LLMResponse] = auto_reply,
        fail: Callable[[LLMRequest, ModelSpec], Exception | None] | None = None,
    ):
        self.triage_result = triage_result
        self.scripts = specialist_scripts or {}
        self.reply = reply
        self.fail = fail
        self.calls: list[tuple[str, str]] = []  # (role, model)

    @staticmethod
    def role(request: LLMRequest) -> str:
        if request.system == TRIAGE_SYSTEM:
            return "triage"
        if request.system == REPLY_SYSTEM:
            return "reply"
        for name in ("returns & refunds", "shipping & delivery", "account & billing", "generalist"):
            if name in request.system:
                return {"returns & refunds": "refunds", "shipping & delivery": "shipping",
                        "account & billing": "account", "generalist": "general"}[name]
        raise AssertionError("unrecognised request")

    def complete(self, request: LLMRequest, model: ModelSpec) -> LLMResponse:
        role = self.role(request)
        self.calls.append((role, model.model))
        if self.fail:
            error = self.fail(request, model)
            if error is not None:
                raise error
        if role == "triage":
            return text(json.dumps(self.triage_result), model=model.model)
        if role == "reply":
            response = self.reply(request)
            response.model = model.model
            return response
        turn = sum(1 for m in request.messages if m["role"] == "assistant")
        script = self.scripts[role]
        return tools(script[min(turn, len(script) - 1)], model=model.model, turn=turn)
