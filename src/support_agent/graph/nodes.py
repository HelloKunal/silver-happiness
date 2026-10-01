"""Nodes of the top-level case workflow. Each node reads the shared CaseState and returns a partial
update; routing decisions are written to `route` and read by conditional edges."""

from __future__ import annotations

import json
import logging
import re
import secrets
import traceback
from typing import Any

from langgraph.errors import NodeError
from langgraph.graph import END
from langgraph.types import Command, interrupt

from support_agent.agents.prompts import REPLY_SYSTEM, TRIAGE_SYSTEM, reply_input, triage_input
from support_agent.agents.specialists import SPECS, build_specialist_subgraph, handoff_resolution
from support_agent.backend.notifier import DECISIONS
from support_agent.executor import ActionFailedError
from support_agent.graph.state import CaseState
from support_agent.llm.base import InvalidOutputError, LLMError, LLMRequest, LLMResponse
from support_agent.models import CustomerReply, TriageResult
from support_agent.resilience import TransientError
from support_agent.services import Services
from support_agent.util import iso, money, to_cents

log = logging.getLogger(__name__)

COMPANY_SIGNOFF = "The Harbor & Pine Support Team"


def _structured(model_cls):
    """Validator for gateway.complete: parse the JSON text into a pydantic model."""

    def validate(response: LLMResponse):
        try:
            return model_cls.model_validate_json(response.text)
        except ValueError as e:
            raise InvalidOutputError(f"{model_cls.__name__} did not validate: {e}") from e

    return validate


class CaseNodes:
    def __init__(self, services: Services):
        self.s = services
        self.cfg = services.policy_config
        self.specialist_graphs = {name: build_specialist_subgraph(spec, services) for name, spec in SPECS.items()}

    def event(self, node: str, message: str, **data: Any) -> dict[str, Any]:
        return {"at": iso(self.s.clock()), "node": node, "message": message, "data": data}

    # ------------------------------------------------------------------ intake / triage

    def intake(self, state: CaseState) -> dict[str, Any]:
        ticket = state["ticket"]
        customer = self.s.repo.customer_by_email(ticket["sender"])
        self.s.repo.create_ticket(
            ticket_id=ticket["id"], sender=ticket["sender"], subject=ticket["subject"], body=ticket["body"], at=self.s.clock()
        )
        who = f"{customer['name']} ({customer['id']})" if customer else "unknown sender"
        return {
            "customer": customer,
            "status": "open",
            "transfers": 0,
            "timeline": [self.event("intake", f"ticket received from {who}")],
        }

    def triage(self, state: CaseState) -> dict[str, Any]:
        request = LLMRequest(
            system=TRIAGE_SYSTEM,
            messages=[{"role": "user", "text": triage_input(state["ticket"])}],
            output_schema=TriageResult.model_json_schema(),
        )
        response, result = self.s.gateway.complete("triage", request, validate=_structured(TriageResult))
        triage = result.model_dump()
        route, reason = self._route(triage)
        self.s.repo.update_ticket(state["ticket"]["id"], at=self.s.clock(), route=route)
        update: dict[str, Any] = {
            "triage": triage,
            "route": route,
            "llm_calls": [response.call_record()],
            "timeline": [self.event("triage", f"intent={triage['intent']} ({triage['confidence']:.2f}) -> {route}",
                                    reason=reason, risk_signals=triage["risk_signals"])],
        }
        if route == "human_handoff":
            update["handoff_reason"] = reason
        return update

    def _route(self, triage: dict[str, Any]) -> tuple[str, str]:
        """Deterministic routing on top of the LLM's classification."""
        blocking = set(triage["risk_signals"]) & set(self.cfg.handoff_signals)
        if blocking:
            return "human_handoff", f"risk signals require a human: {', '.join(sorted(blocking))}"
        if triage["confidence"] < self.cfg.min_triage_confidence:
            return "human_handoff", f"triage confidence {triage['confidence']:.2f} below threshold"
        specialist = triage["intent"] if triage["intent"] in SPECS else "general"
        return f"{specialist}_specialist", f"intent {triage['intent']}"

    # ------------------------------------------------------------------ specialists

    def specialist(self, name: str):
        graph = self.specialist_graphs[name]
        max_transfers = self.cfg.max_transfers

        def run(state: CaseState) -> dict[str, Any]:
            out = graph.invoke(
                {
                    "case_id": state["case_id"],
                    "ticket": state["ticket"],
                    "customer": state.get("customer"),
                    "triage": state.get("triage"),
                    "specialist": name,
                    "transfer_note": state.get("transfer_note"),
                    "messages": [],
                    "turns": 0,
                    "nudges": 0,
                    "resolution": None,
                }
            )
            resolution = out.get("resolution") or handoff_resolution(f"{name} specialist produced no resolution")
            update: dict[str, Any] = {
                "specialist": name,
                "resolution": resolution,
                "llm_calls": out.get("llm_calls", []),
                "timeline": out.get("timeline", []),
                "route": "policy_gate",
            }
            target = resolution.get("transfer_to")
            if target and target != name:
                if state.get("transfers", 0) < max_transfers:
                    update.update(
                        route=f"{target}_specialist",
                        transfers=state.get("transfers", 0) + 1,
                        transfer_note=f"from the {name} specialist: {resolution.get('case_notes', '')}",
                        resolution=None,
                    )
                else:
                    update["resolution"] = handoff_resolution(
                        f"{name} specialist wants to transfer to {target}, but the transfer limit was reached"
                    )
            return update

        run.__name__ = f"{name}_specialist"
        return run

    # ------------------------------------------------------------------ policy gate & approval

    def policy_gate(self, state: CaseState) -> dict[str, Any]:
        spec = SPECS[state["specialist"]]
        report = self.s.policy.evaluate(
            resolution=state["resolution"],
            allowed_actions=spec.allowed_actions,
            customer=state.get("customer"),
            triage=state.get("triage"),
        )
        route = {
            "execute": "execute_actions",
            "approval": "notify_approver",
            "handoff": "human_handoff",
            "reply_only": "respond",
        }[report.outcome]
        update: dict[str, Any] = {
            "policy": report.to_state(),
            "route": route,
            "timeline": [self.event("policy_gate", f"{report.outcome}: {report.reason}",
                                    verdicts=[(d.action["type"], d.verdict) for d in report.decisions])],
        }
        if report.outcome == "handoff":
            update["handoff_reason"] = report.reason
        return update

    def notify_approver(self, state: CaseState) -> dict[str, Any]:
        case_id = state["case_id"]
        existing = self.s.repo.approval(case_id)
        title, message = self._approval_message(state)
        if existing is None:
            nonce = secrets.token_hex(8)
            self.s.repo.create_approval(case_id=case_id, nonce=nonce, summary=f"{title}\n{message}", at=self.s.clock())
            self.s.notifier.approval_requested(case_id=case_id, nonce=nonce, title=title, message=message)
        else:
            nonce = existing["nonce"]  # node retried: don't notify twice
        self.s.repo.update_ticket(case_id, at=self.s.clock(), status="awaiting_approval")
        return {
            "status": "awaiting_approval",
            "approval": {"status": "pending", "nonce": nonce},
            "timeline": [self.event("notify_approver", f"approval requested: {title}")],
        }

    def _approval_message(self, state: CaseState) -> tuple[str, str]:
        customer = state.get("customer") or {}
        pending = [d for d in state["policy"]["decisions"] if d["verdict"] == "approval"]
        first = pending[0]["action"] if pending else {}
        amount = f" {money(to_cents(first['amount']))}" if first.get("amount") else ""
        title = f"Approve {first.get('type', 'action')}{amount} for {first.get('order_id') or customer.get('name', 'customer')}?"
        lines = [f"{customer.get('name', 'Unknown')} - \"{state['ticket']['subject']}\""]
        for d in state["policy"]["decisions"]:
            a = d["action"]
            amt = f" {money(to_cents(a['amount']))}" if a.get("amount") else ""
            why = "; ".join(r.split("] ", 1)[-1] for r in d["reasons"]) or "within limits"
            lines.append(f"- {a['type']}{amt} {a.get('order_id') or ''} [{d['verdict']}]: {why}")
        notes = (state["resolution"] or {}).get("case_notes", "")
        if notes:
            lines.append(f"Agent notes: {notes[:400]}")
        lines.append(f"Case {state['case_id']}")
        return title, "\n".join(lines)

    def await_approval(self, state: CaseState) -> dict[str, Any]:
        decision = interrupt(
            {
                "type": "approval_request",
                "case_id": state["case_id"],
                "decisions": state["policy"]["decisions"],
                "case_notes": (state.get("resolution") or {}).get("case_notes"),
            }
        )
        if not isinstance(decision, dict) or decision.get("decision") not in DECISIONS:
            raise ValueError(f"invalid approval decision: {decision!r}")
        verdict = decision["decision"]
        route = {"approve": "execute_actions", "reject": "respond", "takeover": "human_handoff"}[verdict]
        update: dict[str, Any] = {
            "approval": {**(state.get("approval") or {}), "status": verdict,
                         "reviewer": decision.get("reviewer", "unknown"), "note": decision.get("note", "")},
            "route": route,
            "timeline": [self.event("await_approval", f"human decision: {verdict}", reviewer=decision.get("reviewer"))],
        }
        if verdict == "takeover":
            update["handoff_reason"] = f"reviewer {decision.get('reviewer', '')} took over the case"
        return update

    # ------------------------------------------------------------------ execution

    def execute_actions(self, state: CaseState) -> dict[str, Any]:
        approved = (state.get("approval") or {}).get("status") == "approve"
        runnable = [
            d["action"] for d in state["policy"]["decisions"]
            if d["verdict"] == "auto" or (d["verdict"] == "approval" and approved)
        ]
        results, failures = [], []
        for action in runnable:
            try:
                results.append(self.s.executor.execute(case_id=state["case_id"], action=action, customer=state["customer"]))
            except ActionFailedError as e:
                failures.append(f"{action['type']} {action.get('order_id') or ''}: {e}")
                results.append({**action, "status": "failed", "error": str(e)})
            except TransientError as e:
                failures.append(f"{action['type']} {action.get('order_id') or ''}: outcome unknown after retries ({e})")
                results.append({**action, "status": "unknown", "error": str(e)})
        update: dict[str, Any] = {
            "executed_actions": results,
            "route": "human_handoff" if failures else "respond",
            "timeline": [self.event("execute_actions", f"{len(results) - len(failures)}/{len(results)} actions succeeded",
                                    replayed=[r.get("replayed") for r in results])],
        }
        if failures:
            update["handoff_reason"] = "action execution failed: " + "; ".join(failures)
        return update

    # ------------------------------------------------------------------ reply

    def respond(self, state: CaseState) -> dict[str, Any]:
        ticket = state["ticket"]
        resolution = state.get("resolution") or {}
        approval = state.get("approval") or {}
        if approval.get("status") == "reject":
            status = "declined"
        elif resolution.get("outcome") == "needs_info":
            status = "needs_info"
        else:
            status = "resolved"
        completed = [a for a in state.get("executed_actions", []) if a.get("status") == "succeeded"]
        outcome = {
            "status": status,
            "customer_first_name": ((state.get("customer") or {}).get("name") or "").split(" ")[0] or None,
            "completed_actions": [
                {"type": a["type"], "order_id": a.get("order_id"), "amount": a.get("amount"), "details": a.get("result")}
                for a in completed
            ],
            "reply_points": resolution.get("reply_points", []),
            "declined_reason": (approval.get("note") or "the request falls outside our policy") if status == "declined" else None,
        }
        llm_calls, source, problem = [], "llm", None
        prompt = reply_input(outcome, ticket)
        try:
            response, reply = self.s.gateway.complete(
                "reply",
                LLMRequest(system=REPLY_SYSTEM, messages=[{"role": "user", "text": prompt}],
                           output_schema=CustomerReply.model_json_schema()),
                validate=_structured(CustomerReply),
            )
            llm_calls.append(response.call_record())
            subject, body = reply.subject, reply.body
            problem = reply_guard(body, completed, grounding=prompt)
        except LLMError as e:  # includes AllModelsFailedError
            problem = f"reply model unavailable: {e}"
        if problem:
            # Graceful degradation: the actions already happened, so the customer must still hear
            # about them. A deterministic template is less friendly but never wrong.
            source = "template"
            subject, body = template_reply(outcome, ticket)
        self.s.repo.send_email(
            ticket_id=ticket["id"], to=ticket["sender"], subject=subject, body=body, kind="reply",
            idempotency_key=f"{state['case_id']}:reply", at=self.s.clock(),
        )
        self.s.repo.update_ticket(ticket["id"], at=self.s.clock(), status=status)
        return {
            "reply": {"subject": subject, "body": body, "source": source},
            "status": status,
            "llm_calls": llm_calls,
            "timeline": [self.event("respond", f"reply sent ({source}), status={status}",
                                    **({"fallback_reason": problem} if problem else {}))],
        }

    # ------------------------------------------------------------------ human hand-off

    def human_handoff(self, state: CaseState) -> dict[str, Any]:
        ticket = state["ticket"]
        reason = state.get("handoff_reason") or "unspecified"
        package = {
            "reason": reason,
            "triage": state.get("triage"),
            "specialist": state.get("specialist"),
            "resolution": state.get("resolution"),
            "policy": state.get("policy"),
            "approval": state.get("approval"),
            "executed_actions": state.get("executed_actions", []),
            "ledger": self.s.ledger.for_case(state["case_id"]),
            "errors": state.get("errors", []),
        }
        self.s.repo.update_ticket(
            ticket["id"], at=self.s.clock(), status="escalated", queue="tier2",
            internal_note=json.dumps(package, default=str, indent=2),
        )
        first = ((state.get("customer") or {}).get("name") or "").split(" ")[0]
        self.s.repo.send_email(
            ticket_id=ticket["id"], to=ticket["sender"], subject=f"Re: {ticket['subject']}",
            body=(
                f"Hi{(' ' + first) if first else ''},\n\nThanks for reaching out. Your request needs a closer look from "
                "one of our specialists, who will reply within 24 hours. You don't need to do anything in the meantime.\n\n"
                f"{COMPANY_SIGNOFF}"
            ),
            kind="holding", idempotency_key=f"{state['case_id']}:holding", at=self.s.clock(),
        )
        try:
            self.s.notifier.escalated(
                case_id=state["case_id"],
                title=f"Escalated: {ticket['subject'][:60]}",
                message=f"{reason}\nFrom {ticket['sender']}\nCase {state['case_id']}",
            )
        except Exception:  # notification is best effort; the ticket queue is the source of truth
            log.exception("escalation notification failed")
        return {
            "status": "escalated",
            "timeline": [self.event("human_handoff", f"escalated to tier2: {reason}")],
        }

    # ------------------------------------------------------------------ error handling

    def on_node_error(self, state: CaseState, error: NodeError) -> Command:
        """Graph-level error boundary: any node that still fails after its retry policy lands here.
        The case is handed to a human with the error attached, never dropped."""
        err = error.error
        record = {
            "at": iso(self.s.clock()),
            "node": error.node,
            "type": type(err).__name__,
            "message": str(err)[:2000],
            "attempts": getattr(err, "attempts", None),
            "trace": "".join(traceback.format_exception_only(type(err), err)).strip(),
        }
        log.error("node %s failed: %s", error.node, err)
        return Command(
            update={
                "errors": [record],
                "handoff_reason": f"system failure in {error.node}: {type(err).__name__}",
                "timeline": [self.event(error.node, f"failed: {type(err).__name__} - routing to human")],
            },
            goto="human_handoff",
        )

    def on_handoff_error(self, state: CaseState, error: NodeError) -> Command:
        """Last resort if the hand-off itself fails (e.g. DB down): record and stop, don't loop."""
        log.critical("human_handoff failed for %s: %s", state.get("case_id"), error.error)
        return Command(
            update={
                "status": "handoff_failed",
                "errors": [{"at": iso(self.s.clock()), "node": error.node, "type": type(error.error).__name__,
                            "message": str(error.error)[:2000]}],
            },
            goto=END,
        )


# ---------------------------------------------------------------------- reply helpers

# "we've issued a refund", "your refund has been processed" - a claim that money is moving *now*.
# ("your refund was already issued last week" is a statement about the past and is allowed.)
_REFUND_CLAIM = re.compile(
    r"\b(we|we've|we have|i've|i have)\s+(just\s+)?(issued|processed|initiated|sent)\b[^.]{0,40}\brefund"
    r"|\brefund\b[^.]{0,30}\b(has been|is being|will be)\s+(issued|processed|initiated)",
    re.I,
)
_DOLLARS = re.compile(r"\$\s?(\d[\d,]*(?:\.\d{1,2})?)")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _num(text: str) -> float:
    return round(float(text.replace(",", "")), 2)


def reply_guard(body: str, completed: list[dict[str, Any]], grounding: str = "") -> str | None:
    """Output guardrail on the customer email. Returns a problem description or None.

    - no claim that a refund is being issued unless one was actually executed
    - every executed refund amount is stated exactly
    - every dollar amount mentioned is grounded in the case data (no invented numbers)
    """
    money_moved = [a for a in completed if a["type"] in ("refund", "cancel_order")]
    if not money_moved and _REFUND_CLAIM.search(body):
        return "reply claims a refund was issued, but no refund was executed"
    for a in completed:
        if a["type"] == "refund" and f"{float(a['amount']):,.2f}" not in body.replace("$", ""):
            return f"reply does not state the refunded amount {a['amount']}"
    if grounding:
        known = {_num(n) for n in _NUMBER.findall(grounding)}
        invented = [m for m in _DOLLARS.findall(body) if _num(m) not in known]
        if invented:
            return f"reply mentions amounts not in the case data: {', '.join('$' + m for m in invented)}"
    return None


def template_reply(outcome: dict[str, Any], ticket: dict[str, Any]) -> tuple[str, str]:
    name = outcome.get("customer_first_name")
    lines = [f"Hi {name}," if name else "Hello,", "", "Thanks for contacting Harbor & Pine support."]
    for a in outcome["completed_actions"]:
        d, order = a.get("details") or {}, a.get("order_id")
        if a["type"] == "refund":
            lines.append(f"We've issued a refund of ${d.get('amount', a.get('amount'))} for order {order}. "
                         "It should appear on your statement within 5-10 business days.")
        elif a["type"] == "reship":
            lines.append(f"We're sending a free replacement for order {order} "
                         f"(tracking {d.get('replacement_tracking_number', 'to follow')}).")
        elif a["type"] == "cancel_order":
            lines.append(f"Order {order} has been cancelled and your payment refunded.")
        elif a["type"] == "update_shipping_address":
            lines.append(f"The shipping address for order {order} is now: {d.get('shipping_address')}.")
        elif a["type"] == "send_password_reset":
            lines.append("We've sent a password reset link to the email address on your account.")
        elif a["type"] == "update_email":
            lines.append("Your account email address has been updated.")
        elif a["type"] == "store_credit":
            lines.append(f"We've added ${d.get('store_credit')} in store credit to your account.")
    if outcome["status"] == "declined":
        lines.append(f"After review we're unable to complete this request: {outcome.get('declined_reason')}.")
    for point in outcome.get("reply_points", []):
        lines.append(point if point.endswith((".", "?", "!")) else point + ".")
    lines += ["", COMPANY_SIGNOFF]
    return f"Re: {ticket['subject']}", "\n".join(lines)
