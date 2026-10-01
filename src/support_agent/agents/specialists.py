"""Specialist agents. Each is a LangGraph subgraph running a ReAct-style tool loop:

    START -> agent --(tool calls)--> tools --(no resolution yet)--> agent ...
                   '--(text only)--> nudge ---------------------------^
    ...until the agent calls `submit_resolution` (or runs out of turns) -> END

Every LLM turn and every tool execution is a checkpointed step, so a crash mid-investigation
resumes where it stopped instead of re-paying for earlier LLM calls.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from support_agent.agents.prompts import SPECIALIST_BASE, SPECIALIST_INSTRUCTIONS, case_brief
from support_agent.agents.tools import TOOLS, ToolContext
from support_agent.graph.state import SpecialistState
from support_agent.llm.base import LLMRequest, ToolSpec
from support_agent.models import SPECIALISTS, Resolution, submit_resolution_model
from support_agent.services import Services
from support_agent.util import iso

log = logging.getLogger(__name__)

SUBMIT = "submit_resolution"


@dataclass(frozen=True)
class SpecialistSpec:
    name: str
    title: str
    tools: tuple[str, ...]
    allowed_actions: tuple[str, ...]

    @property
    def transfer_targets(self) -> tuple[str, ...]:
        return tuple(s for s in SPECIALISTS if s != self.name)

    def submit_model(self) -> type[BaseModel]:
        return submit_resolution_model(self.allowed_actions, self.transfer_targets)

    def system_prompt(self) -> str:
        return SPECIALIST_BASE.format(
            title=self.title,
            instructions=SPECIALIST_INSTRUCTIONS[self.name],
            allowed=", ".join(self.allowed_actions) or "none (information only)",
        )

    def tool_specs(self) -> list[ToolSpec]:
        specs = [TOOLS[name].spec() for name in self.tools]
        specs.append(
            ToolSpec(
                SUBMIT,
                "Submit your final decision for this case. Call exactly once, after investigating.",
                self.submit_model().model_json_schema(),
            )
        )
        return specs


_COMMON = ("get_customer_profile", "list_orders", "get_order", "search_help_center")

SPECS: dict[str, SpecialistSpec] = {
    "refunds": SpecialistSpec(
        "refunds", "returns & refunds specialist",
        _COMMON + ("check_refund_eligibility", "track_shipment"),
        ("refund", "reship", "store_credit"),
    ),
    "shipping": SpecialistSpec(
        "shipping", "shipping & delivery specialist",
        _COMMON + ("track_shipment",),
        ("reship", "refund", "cancel_order", "update_shipping_address"),
    ),
    "account": SpecialistSpec(
        "account", "account & billing specialist",
        _COMMON,
        ("send_password_reset", "update_email", "refund"),
    ),
    "general": SpecialistSpec(
        "general", "customer support generalist",
        ("search_help_center", "get_customer_profile", "list_orders"),
        (),
    ),
}


def handoff_resolution(reason: str) -> dict[str, Any]:
    return Resolution(outcome="handoff", handoff_reason=reason, case_notes=reason).model_dump()


def _event(services: Services, node: str, message: str, **data: Any) -> dict[str, Any]:
    return {"at": iso(services.clock()), "node": node, "message": message, "data": data}


def build_specialist_subgraph(spec: SpecialistSpec, services: Services):
    system = spec.system_prompt()
    tool_specs = spec.tool_specs()
    submit_model = spec.submit_model()
    max_turns = services.policy_config.max_agent_turns
    node_name = f"{spec.name}_specialist"

    def agent(state: SpecialistState) -> dict[str, Any]:
        messages = state.get("messages") or [{"role": "user", "text": case_brief(state)}]
        turns = state.get("turns", 0)
        if turns >= max_turns:
            reason = f"{spec.name} specialist used its {max_turns}-turn budget without resolving"
            return {"resolution": handoff_resolution(reason), "timeline": [_event(services, node_name, reason)]}
        response, _ = services.gateway.complete(
            "specialist", LLMRequest(system=system, messages=messages, tools=tool_specs)
        )
        return {
            "messages": messages + [response.assistant_message()],
            "turns": turns + 1,
            "llm_calls": [response.call_record()],
        }

    def tools(state: SpecialistState) -> dict[str, Any]:
        ctx = ToolContext(services=services, customer=state.get("customer"), case_id=state["case_id"])
        last = state["messages"][-1]
        results, events, resolution = [], [], None
        for call in last["tool_calls"]:
            name, args = call["name"], call["arguments"]
            is_error = False
            if name == SUBMIT:
                try:
                    parsed = submit_model.model_validate(args)
                    resolution = Resolution.model_validate(parsed.model_dump()).model_dump()
                    content = "Resolution recorded."
                    events.append(_event(services, node_name, f"submitted resolution: {resolution['outcome']}",
                                         actions=resolution["actions"], transfer_to=resolution["transfer_to"]))
                except ValidationError as e:
                    content, is_error = f"Invalid resolution, fix and resubmit: {e.errors(include_url=False)}", True
                    events.append(_event(services, node_name, "rejected invalid resolution", errors=e.error_count()))
            elif name in spec.tools:
                try:
                    output = TOOLS[name].run(ctx, args)
                    content = json.dumps(output, default=str)
                    is_error = "error" in output
                except ValidationError as e:
                    content, is_error = f"Invalid arguments: {e.errors(include_url=False)}", True
                except Exception as e:  # a broken tool must not kill the case; the agent can adapt
                    log.exception("tool %s failed", name)
                    content, is_error = f"Tool failed unexpectedly: {type(e).__name__}", True
                events.append(_event(services, node_name, f"tool {name}", args=args, error=is_error))
            else:
                content, is_error = f"Unknown tool '{name}'. Available: {', '.join(spec.tools)}, {SUBMIT}.", True
            results.append({"tool_call_id": call["id"], "content": content, "is_error": is_error})

        update: dict[str, Any] = {
            "messages": state["messages"] + [{"role": "tool_results", "results": results}],
            "timeline": events,
        }
        if resolution is not None:
            update["resolution"] = resolution
        return update

    def nudge(state: SpecialistState) -> dict[str, Any]:
        if state.get("nudges", 0) >= 1:
            reason = f"{spec.name} specialist stopped without submitting a resolution"
            return {"resolution": handoff_resolution(reason), "timeline": [_event(services, node_name, reason)]}
        return {
            "messages": state["messages"] + [{"role": "user", "text": f"Finish by calling {SUBMIT} with your decision."}],
            "nudges": state.get("nudges", 0) + 1,
        }

    def after_agent(state: SpecialistState) -> str:
        if state.get("resolution"):
            return END
        return "tools" if state["messages"][-1].get("tool_calls") else "nudge"

    def after_tools(state: SpecialistState) -> str:
        return END if state.get("resolution") else "agent"

    g = StateGraph(SpecialistState)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_node("nudge", nudge)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", after_agent, {"tools": "tools", "nudge": "nudge", END: END})
    g.add_conditional_edges("tools", after_tools, {"agent": "agent", END: END})
    g.add_conditional_edges("nudge", after_tools, {"agent": "agent", END: END})
    return g.compile()
