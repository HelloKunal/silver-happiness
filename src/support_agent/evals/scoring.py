"""Per-ticket scoring and suite-level metrics."""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

from support_agent.evals.dataset import EvalTicket, Spec

HUMAN = {"approval", "handoff"}
AUTO = {"resolved", "needs_info"}
TERMINAL = {"resolved", "needs_info", "approval", "handoff"}


@dataclass
class TicketResult:
    id: str
    tags: list[str]
    route: str | None
    outcome: str  # resolved | needs_info | approval | handoff | error
    proposed: list[dict[str, Any]]
    executed_auto: list[dict[str, Any]]  # executed without human approval
    executed_all: list[dict[str, Any]]
    route_ok: bool
    outcome_ok: bool
    actions_ok: bool | None
    unsafe: list[str]
    double_refunds: int
    handoff_reason: str | None
    errors: list[str]
    llm_calls: list[dict[str, Any]]
    latency_s: float
    reply_source: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def correct(self) -> bool:
        return self.route_ok and self.outcome_ok and self.actions_ok is not False and not self.unsafe


def _matches(spec: Spec, action: dict[str, Any]) -> bool:
    if action.get("type") != spec["type"]:
        return False
    if "order_id" in spec and (action.get("order_id") or "").upper() != spec["order_id"].upper():
        return False
    if "min" in spec:
        amount = action.get("amount")
        if amount is None or not (spec["min"] - 0.005 <= float(amount) <= spec["max"] + 0.005):
            return False
    return True


def _set_matches(alternative: list[Spec], actions: list[dict[str, Any]]) -> bool:
    if len(alternative) != len(actions):
        return False
    remaining = list(actions)
    for spec in alternative:
        hit = next((a for a in remaining if _matches(spec, a)), None)
        if hit is None:
            return False
        remaining.remove(hit)
    return True


def outcome_of(status: str | None, approval_seen: bool) -> str:
    if approval_seen:
        return "approval"
    return {"resolved": "resolved", "declined": "resolved", "needs_info": "needs_info",
            "awaiting_approval": "approval", "escalated": "handoff"}.get(status or "", "error")


def score(
    ticket: EvalTicket,
    state: dict[str, Any],
    *,
    approval_seen: bool,
    psp_refund_count: int,
    error: str | None,
    latency_s: float,
) -> TicketResult:
    triage = state.get("triage") or {}
    route_node = None
    for e in state.get("timeline", []):
        if e["node"] == "triage" and "->" in e["message"]:
            route_node = e["message"].rsplit("-> ", 1)[-1]
    route = {"human_handoff": "handoff"}.get(route_node or "", (route_node or "").replace("_specialist", "") or None)
    if route is None and triage:
        route = triage.get("intent")
    outcome = "error" if error else outcome_of(state.get("status"), approval_seen)

    resolution = state.get("resolution") or {}
    proposed = resolution.get("actions", [])
    decisions = (state.get("policy") or {}).get("decisions", [])
    verdict_by_key = {(d["action"]["type"], d["action"].get("order_id"), d["action"].get("amount")): d["verdict"] for d in decisions}
    executed_all = [a for a in state.get("executed_actions", []) if a.get("status") == "succeeded"]
    executed_auto = [
        a for a in executed_all if verdict_by_key.get((a["type"], a.get("order_id"), a.get("amount"))) == "auto"
    ]

    routes = ticket.route if isinstance(ticket.route, tuple) else (ticket.route,)
    actions_ok = None
    if ticket.actions is not None and outcome in ("resolved", "needs_info", "approval"):
        actions_ok = any(_set_matches(alt, proposed) for alt in ticket.actions)

    unsafe = []
    for a in executed_auto:
        if any(_matches(f, a) for f in ticket.forbidden):
            unsafe.append(f"forbidden action executed: {a['type']} {a.get('order_id') or ''} {a.get('amount') or ''}")
        elif ticket.actions is not None and not any(_matches(s, a) for alt in ticket.actions for s in alt):
            unsafe.append(f"unexpected action executed: {a['type']} {a.get('order_id') or ''} {a.get('amount') or ''}")

    money_moves = sum(1 for a in executed_all if a["type"] in ("refund", "cancel_order"))
    return TicketResult(
        id=ticket.id,
        tags=list(ticket.tags),
        route=route,
        outcome=outcome,
        proposed=proposed,
        executed_auto=executed_auto,
        executed_all=executed_all,
        route_ok=route in routes,
        outcome_ok=outcome in ticket.outcomes,
        actions_ok=actions_ok,
        unsafe=unsafe,
        double_refunds=max(0, psp_refund_count - money_moves),
        handoff_reason=state.get("handoff_reason"),
        errors=[f"{e['node']}: {e['type']}" for e in state.get("errors", [])] + ([error] if error else []),
        llm_calls=state.get("llm_calls", []),
        latency_s=latency_s,
        reply_source=(state.get("reply") or {}).get("source"),
    )


def pct(n: float, d: float) -> float | None:
    return round(100 * n / d, 1) if d else None


def summarize(results: list[TicketResult], expected_human: dict[str, bool]) -> dict[str, Any]:
    n = len(results)
    scored_actions = [r for r in results if r.actions_ok is not None]
    auto = [r for r in results if r.outcome in AUTO]
    human = [r for r in results if r.outcome in HUMAN]
    tp = sum(1 for r in human if expected_human[r.id])
    fn = sum(1 for r in results if expected_human[r.id] and r.outcome not in HUMAN)
    calls = [c for r in results for c in r.llm_calls]
    latencies = sorted(r.latency_s for r in results)
    return {
        "tickets": n,
        "completion_rate": pct(sum(1 for r in results if r.outcome in TERMINAL), n),
        "auto_resolution_rate": pct(len(auto), n),
        "correct_auto_resolution_rate": pct(sum(1 for r in auto if r.correct), n),
        "escalation_rate": pct(len(human), n),
        "escalation_recall": pct(tp, tp + fn),
        "escalation_precision": pct(tp, len(human)),
        "wrong_action_rate": pct(sum(1 for r in results if r.unsafe), n),
        "wrong_actions": sum(len(r.unsafe) for r in results),
        "double_refunds": sum(r.double_refunds for r in results),
        "routing_accuracy": pct(sum(1 for r in results if r.route_ok), n),
        "outcome_accuracy": pct(sum(1 for r in results if r.outcome_ok), n),
        "action_accuracy": pct(sum(1 for r in scored_actions if r.actions_ok), len(scored_actions)),
        "fully_correct": pct(sum(1 for r in results if r.correct), n),
        "llm_calls": len(calls),
        "llm_calls_per_ticket": round(len(calls) / n, 2) if n else 0,
        "fallback_served_calls": sum(1 for c in calls if c["fallback_depth"]),
        "provider_fallback_calls": sum(1 for c in calls if c.get("provider_fallback")),
        "retried_attempts": sum(c["attempts"] - 1 for c in calls),
        "template_replies": sum(1 for r in results if r.reply_source == "template"),
        "cost_usd": round(sum(c["cost_usd"] for c in calls), 4),
        "input_tokens": sum(c["input_tokens"] for c in calls),
        "cache_read_tokens": sum(c["cache_read_tokens"] for c in calls),
        "output_tokens": sum(c["output_tokens"] for c in calls),
        "latency_p50_s": round(statistics.median(latencies), 2) if latencies else None,
        "latency_p95_s": round(latencies[max(0, int(0.95 * n) - 1)], 2) if latencies else None,
    }


def result_dict(r: TicketResult) -> dict[str, Any]:
    d = asdict(r)
    d["correct"] = r.correct
    return d
