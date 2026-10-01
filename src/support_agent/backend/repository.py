"""Data access for the mock backend. Thin on purpose - the interesting logic lives in the workflow."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from support_agent.util import iso, new_id, parse_iso


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


class Repository:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._lock = threading.RLock()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield self.conn
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            else:
                self.conn.execute("COMMIT")

    def fetch_one(self, sql: str, *params: Any) -> dict[str, Any] | None:
        with self._lock:
            return _row(self.conn.execute(sql, params).fetchone())

    def fetch_all(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def run(self, sql: str, *params: Any) -> int:
        with self._lock:
            return self.conn.execute(sql, params).rowcount

    # ------------------------------------------------------------------ customers

    def customer_by_email(self, email: str) -> dict[str, Any] | None:
        c = self.fetch_one("SELECT * FROM customers WHERE lower(email) = lower(?)", email.strip())
        if c:
            c["risk_flags"] = json.loads(c["risk_flags"])
        return c

    def customer(self, customer_id: str) -> dict[str, Any] | None:
        c = self.fetch_one("SELECT * FROM customers WHERE id = ?", customer_id)
        if c:
            c["risk_flags"] = json.loads(c["risk_flags"])
        return c

    # ------------------------------------------------------------------ orders

    def list_orders(self, customer_id: str, limit: int = 10) -> list[dict[str, Any]]:
        return self.fetch_all(
            "SELECT id, placed_at, status, total_cents FROM orders WHERE customer_id = ? "
            "ORDER BY placed_at DESC LIMIT ?",
            customer_id,
            limit,
        )

    def order(self, order_id: str) -> dict[str, Any] | None:
        """Order with items, payments, refunds and shipments, plus derived money fields."""
        order = self.fetch_one("SELECT * FROM orders WHERE upper(id) = upper(?)", order_id.strip())
        if order is None:
            return None
        oid = order["id"]
        order["items"] = self.fetch_all("SELECT * FROM order_items WHERE order_id = ? ORDER BY sku", oid)
        order["payments"] = self.fetch_all("SELECT * FROM payments WHERE order_id = ? ORDER BY created_at", oid)
        order["refunds"] = self.fetch_all("SELECT * FROM refunds WHERE order_id = ? ORDER BY created_at", oid)
        order["shipments"] = self.fetch_all("SELECT * FROM shipments WHERE order_id = ? ORDER BY shipped_at", oid)
        order["paid_cents"] = sum(p["amount_cents"] for p in order["payments"])
        order["refunded_cents"] = sum(r["amount_cents"] for r in order["refunds"])
        original = [s for s in order["shipments"] if not s["is_replacement"]]
        order["delivered_at"] = next((s["delivered_at"] for s in original if s["delivered_at"]), None)
        return order

    def refunds_and_credits_since(self, customer_id: str, since: datetime) -> int:
        with self._lock:
            refunds = self.conn.execute(
                "SELECT COALESCE(SUM(r.amount_cents), 0) FROM refunds r JOIN orders o ON o.id = r.order_id "
                "WHERE o.customer_id = ? AND r.created_at >= ?",
                (customer_id, iso(since)),
            ).fetchone()[0]
            credits = self.conn.execute(
                "SELECT COALESCE(SUM(amount_cents), 0) FROM store_credits WHERE customer_id = ? AND created_at >= ?",
                (customer_id, iso(since)),
            ).fetchone()[0]
        return int(refunds) + int(credits)

    def record_refund(
        self, *, order_id: str, payment_id: str, amount_cents: int, reason: str, provider_ref: str,
        case_id: str, idempotency_key: str, at: datetime,
    ) -> None:
        self.run(
            "INSERT OR IGNORE INTO refunds (id, order_id, payment_id, amount_cents, reason, provider_ref, case_id,"
            " idempotency_key, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            new_id("re"), order_id, payment_id, amount_cents, reason, provider_ref, case_id, idempotency_key, iso(at),
        )

    def refund_by_key(self, idempotency_key: str) -> dict[str, Any] | None:
        return self.fetch_one("SELECT * FROM refunds WHERE idempotency_key = ?", idempotency_key)

    def shipment_by_tracking(self, tracking_number: str) -> dict[str, Any] | None:
        return self.fetch_one("SELECT * FROM shipments WHERE tracking_number = ?", tracking_number)

    def create_replacement_shipment(self, *, order_id: str, idempotency_key: str, at: datetime) -> dict[str, Any]:
        self.run(
            "INSERT OR IGNORE INTO shipments (id, order_id, carrier, tracking_number, status, shipped_at, eta,"
            " last_event, is_replacement, idempotency_key) VALUES (?, ?, 'UPS', ?, 'label_created', NULL, NULL,"
            " 'Replacement label created', 1, ?)",
            new_id("sh"), order_id, "1Z" + new_id("R")[2:14].upper(), idempotency_key,
        )
        return self.fetch_one("SELECT * FROM shipments WHERE idempotency_key = ?", idempotency_key) or {}

    def set_order_status(self, order_id: str, status: str) -> None:
        self.run("UPDATE orders SET status = ? WHERE id = ?", status, order_id)

    def set_shipping_address(self, order_id: str, address: str) -> None:
        self.run("UPDATE orders SET shipping_address = ? WHERE id = ?", address, order_id)

    def add_store_credit(self, *, customer_id: str, amount_cents: int, reason: str, idempotency_key: str, at: datetime) -> None:
        self.run(
            "INSERT OR IGNORE INTO store_credits (id, customer_id, amount_cents, reason, idempotency_key, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            new_id("sc"), customer_id, amount_cents, reason, idempotency_key, iso(at),
        )

    def add_account_event(self, *, customer_id: str, kind: str, detail: str, idempotency_key: str, at: datetime) -> None:
        self.run(
            "INSERT OR IGNORE INTO account_events (id, customer_id, kind, detail, idempotency_key, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            new_id("ae"), customer_id, kind, detail, idempotency_key, iso(at),
        )

    def set_customer_email(self, customer_id: str, email: str) -> None:
        self.run("UPDATE customers SET email = ? WHERE id = ?", email, customer_id)

    # ------------------------------------------------------------------ tickets & email

    def create_ticket(self, *, ticket_id: str, sender: str, subject: str, body: str, at: datetime) -> None:
        self.run(
            "INSERT OR IGNORE INTO tickets (id, sender, subject, body, status, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, 'open', ?, ?)",
            ticket_id, sender, subject, body, iso(at), iso(at),
        )

    def ticket(self, ticket_id: str) -> dict[str, Any] | None:
        return self.fetch_one("SELECT * FROM tickets WHERE id = ?", ticket_id)

    def update_ticket(self, ticket_id: str, *, at: datetime, **fields: Any) -> None:
        allowed = {"status", "route", "queue", "internal_note"}
        if not fields.keys() <= allowed:
            raise ValueError(f"cannot update ticket fields {set(fields) - allowed}")
        sets = ", ".join(f"{k} = ?" for k in fields)
        self.run(f"UPDATE tickets SET {sets}, updated_at = ? WHERE id = ?", *fields.values(), iso(at), ticket_id)

    def send_email(self, *, ticket_id: str, to: str, subject: str, body: str, kind: str, idempotency_key: str, at: datetime) -> bool:
        """Queue an outbound email. Returns False if this exact email was already sent (replay)."""
        return (
            self.run(
                "INSERT OR IGNORE INTO outbox (id, ticket_id, to_addr, subject, body, kind, idempotency_key, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                new_id("em"), ticket_id, to, subject, body, kind, idempotency_key, iso(at),
            )
            == 1
        )

    def emails_for(self, ticket_id: str) -> list[dict[str, Any]]:
        return self.fetch_all("SELECT * FROM outbox WHERE ticket_id = ? ORDER BY created_at", ticket_id)

    # ------------------------------------------------------------------ approvals

    def create_approval(self, *, case_id: str, nonce: str, summary: str, at: datetime) -> None:
        self.run(
            "INSERT OR IGNORE INTO approvals (case_id, nonce, status, summary, requested_at) VALUES (?, ?, 'pending', ?, ?)",
            case_id, nonce, summary, iso(at),
        )

    def approval(self, case_id: str) -> dict[str, Any] | None:
        return self.fetch_one("SELECT * FROM approvals WHERE case_id = ?", case_id)

    def pending_approvals(self) -> list[dict[str, Any]]:
        return self.fetch_all("SELECT * FROM approvals WHERE status = 'pending' ORDER BY requested_at")

    def decide_approval(self, *, case_id: str, decision: str, reviewer: str, note: str, at: datetime) -> bool:
        """Atomically move pending -> decided. Returns False if already decided (double tap, replay)."""
        return (
            self.run(
                "UPDATE approvals SET status = ?, reviewer = ?, note = ?, decided_at = ? "
                "WHERE case_id = ? AND status = 'pending'",
                decision, reviewer, note, iso(at), case_id,
            )
            == 1
        )

    # ------------------------------------------------------------------ misc

    def kv_get(self, key: str) -> str | None:
        row = self.fetch_one("SELECT value FROM kv WHERE key = ?", key)
        return row["value"] if row else None

    def kv_set(self, key: str, value: str) -> None:
        self.run("INSERT INTO kv (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", key, value)

    def account_age_days(self, customer: dict[str, Any], now: datetime) -> int:
        return (now - parse_iso(customer["created_at"])).days
