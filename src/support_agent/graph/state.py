"""The single shared state object that carries a support case through the workflow.

It is checkpointed after every step (SQLite), which is what makes a case resumable: a case paused
for human approval on Monday resumes on Tuesday in a different process with full context. List
fields use an append reducer, so every node adds to the audit trail instead of overwriting it.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class CaseState(TypedDict, total=False):
    case_id: str
    ticket: dict[str, Any]  # {id, sender, subject, body, received_at}
    customer: dict[str, Any] | None  # verified by sender email; None = unknown sender

    triage: dict[str, Any] | None  # TriageResult
    route: str  # next hop decided by the last routing node
    specialist: str | None  # specialist that produced the current resolution
    transfers: int
    transfer_note: str | None

    resolution: dict[str, Any] | None  # Resolution proposed by the specialist
    policy: dict[str, Any] | None  # PolicyReport: per-action verdicts + outcome
    approval: dict[str, Any] | None  # {status, nonce, decision, reviewer, note}
    reply: dict[str, Any] | None  # {subject, body, source: llm|template}

    status: str  # open | awaiting_approval | resolved | needs_info | declined | escalated
    handoff_reason: str | None

    executed_actions: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[dict[str, Any]], operator.add]
    llm_calls: Annotated[list[dict[str, Any]], operator.add]
    timeline: Annotated[list[dict[str, Any]], operator.add]


class SpecialistState(TypedDict, total=False):
    """Private state of a specialist's tool loop (a subgraph). Only the resolution and telemetry
    flow back to the case; the scratch conversation stays inside the subgraph checkpoint."""

    case_id: str
    ticket: dict[str, Any]
    customer: dict[str, Any] | None
    triage: dict[str, Any] | None
    specialist: str
    transfer_note: str | None

    messages: list[dict[str, Any]]
    turns: int
    nudges: int
    resolution: dict[str, Any] | None

    llm_calls: Annotated[list[dict[str, Any]], operator.add]
    timeline: Annotated[list[dict[str, Any]], operator.add]
