"""Executes approved actions exactly once.

Two layers of idempotency, because "retry" and "move money" are a dangerous combination:

1. Action ledger (ours). Every action gets a deterministic key derived from (case, action). Before
   running, the ledger is checked: an action that already succeeded is returned from the ledger,
   not re-run. This covers graph re-execution (node retries, resume after a crash/approval).

2. Provider idempotency key (theirs). The same key is sent to the payment provider on every
   attempt. If a refund succeeds but the response is lost (timeout after commit), the retry returns
   the original refund instead of creating a second one. This covers network ambiguity.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

from support_agent.backend.payments import PaymentDeclinedError, PaymentGateway
from support_agent.backend.repository import Repository
from support_agent.resilience import TransientError, retry_call
from support_agent.util import dollars, dumps, iso, to_cents

log = logging.getLogger(__name__)


class ActionFailedError(Exception):
    """Permanent failure for one action (e.g. provider declined)."""


def idempotency_key(case_id: str, action: dict[str, Any]) -> str:
    amount = to_cents(action["amount"]) if action.get("amount") is not None else None
    material = dumps([case_id, action["type"], action.get("order_id"), amount, action.get("new_value")])
    return "act_" + hashlib.sha256(material.encode()).hexdigest()[:24]


class IdempotencyLedger:
    def __init__(self, repo: Repository, clock: Callable[[], datetime]):
        self.repo = repo
        self.clock = clock

    def get(self, key: str) -> dict[str, Any] | None:
        return self.repo.fetch_one("SELECT * FROM action_ledger WHERE idempotency_key = ?", key)

    def for_case(self, case_id: str) -> list[dict[str, Any]]:
        return self.repo.fetch_all("SELECT * FROM action_ledger WHERE case_id = ? ORDER BY updated_at", case_id)

    def run_once(
        self, *, key: str, case_id: str, action: dict[str, Any], fn: Callable[[], dict[str, Any]]
    ) -> tuple[dict[str, Any], bool]:
        """Returns (result, replayed). Raises if fn fails; the row records what we know."""
        row = self.get(key)
        if row and row["status"] == "succeeded":
            return json.loads(row["result"]), True
        now = iso(self.clock())
        self.repo.run(
            "INSERT INTO action_ledger (idempotency_key, case_id, action_type, payload, status, attempts, updated_at)"
            " VALUES (?, ?, ?, ?, 'in_progress', 1, ?) ON CONFLICT(idempotency_key) DO UPDATE SET"
            " status = 'in_progress', attempts = attempts + 1, updated_at = excluded.updated_at",
            key, case_id, action["type"], dumps(action), now,
        )
        try:
            result = fn()
        except TransientError as e:
            # Outcome unknown (e.g. timeout). Leave 'in_progress': a later run retries with the same
            # provider key, which is safe; a human sees exactly which action is unresolved.
            self._mark(key, "in_progress", error=f"outcome unknown: {e}")
            raise
        except Exception as e:
            self._mark(key, "failed", error=str(e))
            raise
        self._mark(key, "succeeded", result=result)
        return result, False

    def _mark(self, key: str, status: str, *, result: dict[str, Any] | None = None, error: str | None = None) -> None:
        self.repo.run(
            "UPDATE action_ledger SET status = ?, result = ?, error = ?, updated_at = ? WHERE idempotency_key = ?",
            status, dumps(result) if result is not None else None, error, iso(self.clock()), key,
        )


class ActionExecutor:
    def __init__(
        self,
        repo: Repository,
        payments: PaymentGateway,
        ledger: IdempotencyLedger,
        clock: Callable[[], datetime],
        *,
        retry_attempts: int = 4,
        retry_base_delay_s: float = 0.25,
        sleep: Callable[[float], None] | None = None,
    ):
        self.repo = repo
        self.payments = payments
        self.ledger = ledger
        self.clock = clock
        self.retry_attempts = retry_attempts
        self.retry_base_delay_s = retry_base_delay_s
        self._sleep_kwargs = {"sleep": sleep} if sleep else {}

    def execute(self, *, case_id: str, action: dict[str, Any], customer: dict[str, Any]) -> dict[str, Any]:
        key = idempotency_key(case_id, action)

        def attempt() -> dict[str, Any]:
            return retry_call(
                lambda: self._dispatch(action, customer, key, case_id),
                retry_on=(TransientError,),
                attempts=self.retry_attempts,
                base_delay_s=self.retry_base_delay_s,
                on_retry=lambda n, e: log.warning("retrying %s (%s) after: %s", action["type"], key, e),
                **self._sleep_kwargs,
            )

        result, replayed = self.ledger.run_once(key=key, case_id=case_id, action=action, fn=attempt)
        return {**action, "status": "succeeded", "result": result, "idempotency_key": key, "replayed": replayed}

    # ------------------------------------------------------------------ handlers

    def _dispatch(self, action: dict[str, Any], customer: dict[str, Any], key: str, case_id: str) -> dict[str, Any]:
        kind = action["type"]
        now = self.clock()
        if kind == "refund":
            return self._refund(action["order_id"], to_cents(action["amount"]), action.get("reason", ""), key, case_id)
        if kind == "cancel_order":
            order = self.repo.order(action["order_id"])
            remaining = order["paid_cents"] - order["refunded_cents"]
            refund = self._refund(order["id"], remaining, "order cancelled", key, case_id) if remaining > 0 else None
            self.repo.set_order_status(order["id"], "cancelled")
            return {"cancelled": order["id"], "refund": refund}
        if kind == "reship":
            shipment = self.repo.create_replacement_shipment(order_id=action["order_id"], idempotency_key=key, at=now)
            return {"replacement_tracking_number": shipment.get("tracking_number"), "carrier": shipment.get("carrier")}
        if kind == "update_shipping_address":
            self.repo.set_shipping_address(action["order_id"], action["new_value"])
            return {"order_id": action["order_id"], "shipping_address": action["new_value"]}
        if kind == "store_credit":
            amount = to_cents(action["amount"])
            self.repo.add_store_credit(
                customer_id=customer["id"], amount_cents=amount, reason=action.get("reason", ""), idempotency_key=key, at=now
            )
            return {"store_credit": dollars(amount)}
        if kind == "send_password_reset":
            self.repo.add_account_event(
                customer_id=customer["id"], kind="password_reset_sent", detail=customer["email"], idempotency_key=key, at=now
            )
            return {"reset_link_sent_to": customer["email"]}
        if kind == "update_email":
            self.repo.add_account_event(
                customer_id=customer["id"], kind="email_changed",
                detail=f"{customer['email']} -> {action['new_value']}", idempotency_key=key, at=now,
            )
            self.repo.set_customer_email(customer["id"], action["new_value"])
            return {"email": action["new_value"]}
        raise ActionFailedError(f"unknown action type {kind!r}")

    def _refund(self, order_id: str, amount_cents: int, reason: str, key: str, case_id: str) -> dict[str, Any]:
        recorded = self.repo.refund_by_key(key)
        if recorded is not None:  # crashed after recording but before the ledger was marked
            return {"refund_id": recorded["provider_ref"], "amount": dollars(recorded["amount_cents"]), "provider_replayed": True}
        order = self.repo.order(order_id)
        # Choose the payment to refund: newest first (so a duplicate charge refunds the duplicate).
        # Selection depends only on our own records, so a retry picks the same payment + same key.
        refunded_by_payment: dict[str, int] = {}
        for r in order["refunds"]:
            refunded_by_payment[r["payment_id"]] = refunded_by_payment.get(r["payment_id"], 0) + r["amount_cents"]
        payment = next(
            (p for p in reversed(order["payments"]) if p["amount_cents"] - refunded_by_payment.get(p["id"], 0) >= amount_cents),
            None,
        )
        if payment is None:
            raise ActionFailedError(f"no single payment on {order_id} can cover a {dollars(amount_cents)} refund")
        try:
            receipt = self.payments.refund(
                payment_ref=payment["provider_ref"],
                amount_cents=amount_cents,
                idempotency_key=key,
                metadata={"case_id": case_id, "order_id": order_id},
            )
        except PaymentDeclinedError as e:
            raise ActionFailedError(str(e)) from e
        self.repo.record_refund(
            order_id=order_id, payment_id=payment["id"], amount_cents=amount_cents, reason=reason,
            provider_ref=receipt.provider_ref, case_id=case_id, idempotency_key=key, at=self.clock(),
        )
        return {"refund_id": receipt.provider_ref, "amount": dollars(amount_cents), "provider_replayed": receipt.replayed}
