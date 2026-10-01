"""Deterministic policy engine: decides what the agent may do on its own.

The LLM proposes; this code disposes. Each proposed action gets a verdict:
  auto      - safe to execute without a human
  approval  - execute only after a human approves (push notification)
  deny      - never execute; the case goes to a human with the reason

Guardrails that must hold even if the model is wrong or manipulated (e.g. by prompt injection)
live here, not in prompts: ownership checks, refundable balances, thresholds, risk flags.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

from support_agent.backend.repository import Repository
from support_agent.config import PolicyConfig
from support_agent.models import MONEY_ACTIONS, ORDER_ACTIONS
from support_agent.util import dollars, money, parse_iso, to_cents

Verdict = Literal["auto", "approval", "deny"]
_RANK = {"auto": 0, "approval": 1, "deny": 2}


@dataclass
class Decision:
    action: dict[str, Any]
    verdict: Verdict = "auto"
    reasons: list[str] = field(default_factory=list)

    def escalate(self, verdict: Verdict, reason: str) -> None:
        if _RANK[verdict] > _RANK[self.verdict]:
            self.verdict = verdict
        self.reasons.append(f"[{verdict}] {reason}")


@dataclass
class PolicyReport:
    decisions: list[Decision]
    outcome: Literal["execute", "approval", "handoff", "reply_only"]
    reason: str

    def to_state(self) -> dict[str, Any]:
        return {"outcome": self.outcome, "reason": self.reason, "decisions": [asdict(d) for d in self.decisions]}


class PolicyEngine:
    def __init__(self, config: PolicyConfig, repo: Repository, clock: Callable[[], datetime]):
        self.cfg = config
        self.repo = repo
        self.clock = clock

    # ------------------------------------------------------------------ shared with the agent's tool

    def refund_eligibility(self, order: dict[str, Any]) -> dict[str, Any]:
        """Facts about refundability. The specialist sees this via a tool; the gate uses the same code."""
        now = self.clock()
        days = (now - parse_iso(order["delivered_at"])).days if order["delivered_at"] else None
        refundable = order["paid_cents"] - order["refunded_cents"]
        final_sale = [i["name"] for i in order["items"] if i["final_sale"]]
        notes = []
        if days is None:
            notes.append("Order has not been delivered (return window not started).")
        elif days > self.cfg.return_window_days:
            notes.append(f"Delivered {days} days ago - outside the {self.cfg.return_window_days}-day return window.")
        if final_sale:
            notes.append("Final-sale items are not refundable for change of mind; damaged/defective is still covered.")
        if order["refunded_cents"]:
            notes.append(f"{money(order['refunded_cents'])} has already been refunded on this order.")
        if refundable <= 0:
            notes.append("Nothing left to refund on this order.")
        return {
            "order_id": order["id"],
            "delivered": days is not None,
            "days_since_delivery": days,
            "return_window_days": self.cfg.return_window_days,
            "within_return_window": days is None or days <= self.cfg.return_window_days,
            "paid": dollars(order["paid_cents"]),
            "already_refunded": dollars(order["refunded_cents"]),
            "refundable_remaining": dollars(max(refundable, 0)),
            "final_sale_items": final_sale,
            "notes": notes,
        }

    # ------------------------------------------------------------------ gate

    def evaluate(
        self,
        *,
        resolution: dict[str, Any],
        allowed_actions: tuple[str, ...],
        customer: dict[str, Any] | None,
        triage: dict[str, Any] | None,
    ) -> PolicyReport:
        now = self.clock()
        decisions: list[Decision] = []
        pending_refund_by_order: dict[str, int] = {}
        pending_money = 0
        signals = set((triage or {}).get("risk_signals", []))

        for action in resolution.get("actions", []):
            d = Decision(action=dict(action))
            decisions.append(d)
            kind = action["type"]

            if kind not in allowed_actions:
                d.escalate("deny", f"'{kind}' is outside this specialist's permissions")
                continue
            if customer is None:
                d.escalate("deny", "sender is not a verified customer; account actions are not allowed")
                continue

            order = None
            if kind in ORDER_ACTIONS:
                order_id = action.get("order_id") or ""
                order = self.repo.order(order_id) if order_id else None
                if order is None or order["customer_id"] != customer["id"]:
                    d.escalate("deny", f"order '{order_id}' does not belong to the requesting customer")
                    continue

            if kind in MONEY_ACTIONS:
                self._risk_rules(d, customer, signals, now)

            if kind == "refund":
                amount = self._amount(d, action)
                if amount is None:
                    continue
                refundable = order["paid_cents"] - order["refunded_cents"] - pending_refund_by_order.get(order["id"], 0)
                if amount > refundable:
                    d.escalate("deny", f"{money(amount)} exceeds the refundable balance {money(max(refundable, 0))}")
                    continue
                pending_refund_by_order[order["id"]] = pending_refund_by_order.get(order["id"], 0) + amount
                if amount > self.cfg.refund_auto_max_cents:
                    d.escalate("approval", f"refund {money(amount)} is above the auto-approve limit")
                elig = self.refund_eligibility(order)
                if not elig["within_return_window"]:
                    d.escalate("approval", f"delivered {elig['days_since_delivery']} days ago - outside return window")
                if order["items"] and all(i["final_sale"] for i in order["items"]):
                    d.escalate("approval", "order contains only final-sale items")
                pending_money += amount
                self._budget_rule(d, customer, pending_money, now)

            elif kind == "reship":
                original = [s for s in order["shipments"] if not s["is_replacement"]]
                if any(s["is_replacement"] for s in order["shipments"]):
                    d.escalate("deny", "a replacement has already been shipped for this order")
                elif not original or original[0]["status"] not in ("delivered", "lost", "exception"):
                    d.escalate("deny", "original shipment is not delivered or lost - nothing to replace yet")
                if order["subtotal_cents"] > self.cfg.reship_auto_max_cents:
                    d.escalate("approval", f"replacement value {money(order['subtotal_cents'])} is above the auto limit")

            elif kind == "cancel_order":
                if order["status"] != "processing":
                    d.escalate("deny", f"order is '{order['status']}' - only processing orders can be cancelled")

            elif kind == "update_shipping_address":
                if order["status"] != "processing":
                    d.escalate("deny", f"order is '{order['status']}' - address can only change before shipping")
                if not (action.get("new_value") or "").strip():
                    d.escalate("deny", "no new address provided")

            elif kind == "store_credit":
                amount = self._amount(d, action)
                if amount is None:
                    continue
                if amount > self.cfg.store_credit_auto_max_cents:
                    d.escalate("approval", f"store credit {money(amount)} is above the auto limit")
                pending_money += amount
                self._budget_rule(d, customer, pending_money, now)

            elif kind == "update_email":
                if "@" not in (action.get("new_value") or ""):
                    d.escalate("deny", "no valid new email address provided")
                else:
                    d.escalate("approval", "email changes always need identity verification by a human")

            # send_password_reset: always auto - the link only ever goes to the email on file.

        return self._outcome(resolution, decisions)

    def _amount(self, d: Decision, action: dict[str, Any]) -> int | None:
        amount = action.get("amount")
        if amount is None or to_cents(amount) <= 0:
            d.escalate("deny", "missing or non-positive amount")
            return None
        return to_cents(amount)

    def _risk_rules(self, d: Decision, customer: dict[str, Any], signals: set[str], now: datetime) -> None:
        flagged = set(customer["risk_flags"]) & set(self.cfg.risk_flags_requiring_approval)
        if flagged:
            d.escalate("approval", f"account flagged: {', '.join(sorted(flagged))}")
        age = self.repo.account_age_days(customer, now)
        if age < self.cfg.new_account_days:
            d.escalate("approval", f"account is only {age} days old")
        suspicious = signals & set(self.cfg.ticket_signals_requiring_approval)
        if suspicious:
            d.escalate("approval", f"ticket shows risk signals: {', '.join(sorted(suspicious))}")

    def _budget_rule(self, d: Decision, customer: dict[str, Any], pending_cents: int, now: datetime) -> None:
        since = now - timedelta(days=self.cfg.lookback_days)
        prior = self.repo.refunds_and_credits_since(customer["id"], since)
        if prior + pending_cents > self.cfg.customer_auto_budget_cents:
            d.escalate(
                "approval",
                f"customer would receive {money(prior + pending_cents)} in refunds/credits within "
                f"{self.cfg.lookback_days} days (auto budget {money(self.cfg.customer_auto_budget_cents)})",
            )

    @staticmethod
    def _outcome(resolution: dict[str, Any], decisions: list[Decision]) -> PolicyReport:
        if resolution.get("outcome") == "handoff":
            return PolicyReport(decisions, "handoff", resolution.get("handoff_reason") or "agent requested a human")
        denied = [d for d in decisions if d.verdict == "deny"]
        if denied:
            why = "; ".join(r for d in denied for r in d.reasons if r.startswith("[deny]"))
            return PolicyReport(decisions, "handoff", f"proposed action blocked by policy: {why}")
        if any(d.verdict == "approval" for d in decisions):
            return PolicyReport(decisions, "approval", "one or more actions need human approval")
        if decisions:
            return PolicyReport(decisions, "execute", "all actions within automatic limits")
        return PolicyReport(decisions, "reply_only", "no actions proposed")
