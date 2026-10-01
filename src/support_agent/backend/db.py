"""SQLite schema for the mock company backend (orders, payments, shipments, tickets...).

Money is stored as integer cents. Every side-effecting table has an `idempotency_key UNIQUE` column,
so replaying an action is a no-op at the storage layer too.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    tier TEXT NOT NULL,
    created_at TEXT NOT NULL,
    risk_flags TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'active'
);
CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(id),
    placed_at TEXT NOT NULL,
    status TEXT NOT NULL,                 -- processing | shipped | delivered | cancelled
    subtotal_cents INTEGER NOT NULL,
    shipping_cents INTEGER NOT NULL,
    total_cents INTEGER NOT NULL,
    shipping_address TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS order_items (
    order_id TEXT NOT NULL REFERENCES orders(id),
    sku TEXT NOT NULL,
    name TEXT NOT NULL,
    qty INTEGER NOT NULL,
    unit_price_cents INTEGER NOT NULL,
    final_sale INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (order_id, sku)
);
CREATE TABLE IF NOT EXISTS payments (
    id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES orders(id),
    amount_cents INTEGER NOT NULL,
    provider_ref TEXT NOT NULL,           -- mock: pi_mock_*, stripe: pi_* (test mode)
    method TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS refunds (
    id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    payment_id TEXT NOT NULL,
    amount_cents INTEGER NOT NULL,
    reason TEXT,
    provider_ref TEXT,
    case_id TEXT,
    idempotency_key TEXT UNIQUE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS shipments (
    id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    carrier TEXT NOT NULL,
    tracking_number TEXT NOT NULL,
    status TEXT NOT NULL,                 -- label_created | in_transit | delivered | lost | exception
    shipped_at TEXT,
    delivered_at TEXT,
    eta TEXT,
    last_event TEXT,
    is_replacement INTEGER NOT NULL DEFAULT 0,
    idempotency_key TEXT UNIQUE
);
CREATE TABLE IF NOT EXISTS store_credits (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    amount_cents INTEGER NOT NULL,
    reason TEXT,
    idempotency_key TEXT UNIQUE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS account_events (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    detail TEXT,
    idempotency_key TEXT UNIQUE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tickets (
    id TEXT PRIMARY KEY,
    sender TEXT NOT NULL,
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    status TEXT NOT NULL,                 -- open | resolved | needs_info | awaiting_approval | escalated
    route TEXT,
    queue TEXT,
    internal_note TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outbox (
    id TEXT PRIMARY KEY,
    ticket_id TEXT NOT NULL,
    to_addr TEXT NOT NULL,
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    kind TEXT NOT NULL,                   -- reply | holding
    idempotency_key TEXT UNIQUE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS action_ledger (
    idempotency_key TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL,                 -- in_progress | succeeded | failed
    attempts INTEGER NOT NULL DEFAULT 0,
    result TEXT,
    error TEXT,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS approvals (
    case_id TEXT PRIMARY KEY,
    nonce TEXT NOT NULL,
    status TEXT NOT NULL,                 -- pending | approve | reject | takeover
    summary TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    decided_at TEXT,
    reviewer TEXT,
    note TEXT
);
-- The payment provider's own ledger (what the mock "Stripe" actually did with money).
CREATE TABLE IF NOT EXISTS psp_refunds (
    id TEXT PRIMARY KEY,
    payment_ref TEXT NOT NULL,
    amount_cents INTEGER NOT NULL,
    idempotency_key TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def connect(path: Path | str) -> sqlite3.Connection:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
