"""Approval channel: turns a human decision (CLI or a tapped push-notification button) into a
resumed workflow.

Order of operations matters: the decision is first recorded atomically in the DB
(pending -> decided, exactly once), and only the caller that wins that transition resumes the
graph. A double tap, a replayed ntfy message or two reviewers answering at once can therefore
never execute the actions twice.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from support_agent.backend.notifier import DECISIONS, NtfyNotifier, verify
from support_agent.runtime import CaseResult, resume_case
from support_agent.services import Services

log = logging.getLogger(__name__)


@dataclass
class DecisionOutcome:
    accepted: bool
    message: str
    result: CaseResult | None = None


def decide(graph, services: Services, *, case_id: str, decision: str, reviewer: str, note: str = "") -> DecisionOutcome:
    if decision not in DECISIONS:
        return DecisionOutcome(False, f"decision must be one of {DECISIONS}")
    approval = services.repo.approval(case_id)
    if approval is None:
        return DecisionOutcome(False, f"no approval request for {case_id}")
    if not services.repo.decide_approval(case_id=case_id, decision=decision, reviewer=reviewer, note=note, at=services.clock()):
        return DecisionOutcome(False, f"{case_id} was already decided ({approval['status']})")
    result = resume_case(graph, case_id, {"decision": decision, "reviewer": reviewer, "note": note})
    return DecisionOutcome(True, f"{case_id}: {decision} -> {result.status}", result)


def handle_ntfy_message(graph, services: Services, secret: str, message: dict[str, Any]) -> DecisionOutcome:
    try:
        payload = json.loads(message.get("message", ""))
        case_id, decision, nonce, sig = payload["case_id"], payload["decision"], payload["nonce"], payload["sig"]
    except (ValueError, KeyError, TypeError):
        return DecisionOutcome(False, "ignored malformed reply")
    approval = services.repo.approval(case_id)
    if approval is None or approval["nonce"] != nonce:
        return DecisionOutcome(False, f"ignored reply for unknown/stale approval {case_id}")
    if not verify(secret, case_id, decision, nonce, sig):
        log.warning("rejected forged approval for %s", case_id)
        return DecisionOutcome(False, f"rejected reply with bad signature for {case_id}")
    return decide(graph, services, case_id=case_id, decision=decision, reviewer="ntfy")


def listen(graph, services: Services, notifier: NtfyNotifier, *, interval_s: float = 3.0, once: bool = False) -> None:
    """Poll the ntfy reply topic and apply button presses. Cursor is persisted, so restarts don't
    re-process old messages (and re-processing would be harmless anyway - see module docstring)."""
    cursor_key = f"ntfy_cursor:{notifier.reply_topic}"
    while True:
        since = services.repo.kv_get(cursor_key) or "all"
        try:
            messages = notifier.poll_replies(since)
        except Exception as e:  # network blips shouldn't kill the listener
            log.warning("ntfy poll failed: %s", e)
            messages = []
        for m in messages:
            outcome = handle_ntfy_message(graph, services, notifier.secret, m)
            log.info(outcome.message)
            services.repo.kv_set(cursor_key, m["id"])
        if once:
            return
        time.sleep(interval_s)
