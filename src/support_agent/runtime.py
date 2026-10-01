"""Running cases: submit a ticket, resume after a human decision, inspect state."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from langgraph.types import Command

from support_agent.services import Services
from support_agent.util import iso


@dataclass
class CaseResult:
    case_id: str
    state: dict[str, Any]
    interrupt: dict[str, Any] | None

    @property
    def status(self) -> str:
        return self.state.get("status", "unknown")


def new_case_id() -> str:
    return "case-" + uuid.uuid4().hex[:10]


def _config(case_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": case_id}, "recursion_limit": 80}


def _result(graph, case_id: str, output: dict[str, Any]) -> CaseResult:
    interrupts = output.get("__interrupt__") or []
    state = graph.get_state(_config(case_id)).values if graph.checkpointer else {
        k: v for k, v in output.items() if k != "__interrupt__"
    }
    return CaseResult(case_id, state, interrupts[0].value if interrupts else None)


def submit_ticket(
    graph, services: Services, *, sender: str, subject: str, body: str, case_id: str | None = None
) -> CaseResult:
    case_id = case_id or new_case_id()
    ticket = {"id": case_id, "sender": sender, "subject": subject, "body": body, "received_at": iso(services.clock())}
    # durability="sync": the checkpoint is written before the next step starts, so a crash can
    # never lose a step that already had side effects.
    output = graph.invoke({"case_id": case_id, "ticket": ticket}, _config(case_id), durability="sync")
    return _result(graph, case_id, output)


def resume_case(graph, case_id: str, decision: dict[str, Any]) -> CaseResult:
    output = graph.invoke(Command(resume=decision), _config(case_id), durability="sync")
    return _result(graph, case_id, output)


def case_state(graph, case_id: str) -> tuple[dict[str, Any], tuple[str, ...]]:
    snapshot = graph.get_state(_config(case_id))
    return snapshot.values, snapshot.next
