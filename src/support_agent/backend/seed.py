"""Deterministic seed data: named scenario fixtures (used by the eval set) + random filler customers.

All dates are relative to `now`, so return windows and delivery ages are the same on every run.
"""

from __future__ import annotations

import json
import random
import sqlite3
import zlib
from dataclasses import dataclass
from datetime import datetime, timedelta

from support_agent.util import iso, to_cents

ADDRESSES = [
    "12 Harbor St, Portland, ME 04101",
    "480 Elm Ave, Austin, TX 78701",
    "77 Lakeview Dr, Madison, WI 53703",
    "9 Cedar Ct, Boulder, CO 80302",
    "215 Pine Rd, Raleigh, NC 27601",
    "1400 Mission St, San Francisco, CA 94103",
]

CATALOG = [
    ("KIT-MUG", "Stoneware Mug Set", 24.00),
    ("KIT-PAN", "Cast Iron Skillet 10in", 45.00),
    ("HOME-RUG", "Wool Area Rug 5x7", 189.00),
    ("HOME-VASE", "Glass Bud Vase", 19.50),
    ("APP-SOCK", "Merino Socks 3-Pack", 27.00),
    ("FIT-BAND", "Resistance Band Set", 29.99),
    ("AUD-EAR", "Wireless Earbuds", 79.00),
    ("OFF-LAMP", "LED Desk Lamp", 54.00),
]


@dataclass
class _Builder:
    conn: sqlite3.Connection
    now: datetime

    def ago(self, days: float) -> str:
        return iso(self.now - timedelta(days=days))

    def customer(self, cid: str, email: str, name: str, tier: str, age_days: int, flags: tuple[str, ...] = ()) -> None:
        self.conn.execute(
            "INSERT INTO customers (id, email, name, tier, created_at, risk_flags) VALUES (?, ?, ?, ?, ?, ?)",
            (cid, email, name, tier, self.ago(age_days), json.dumps(list(flags))),
        )

    def order(
        self,
        oid: str,
        cid: str,
        *,
        placed: float,
        status: str,
        items: list[tuple[str, str, int, float] | tuple[str, str, int, float, bool]],
        shipping: float = 5.00,
        address: str = ADDRESSES[0],
        shipment: dict | None = None,
        payments: list[float] | None = None,
    ) -> None:
        subtotal = sum(to_cents(i[3]) * i[2] for i in items)
        total = subtotal + to_cents(shipping)
        self.conn.execute(
            "INSERT INTO orders (id, customer_id, placed_at, status, subtotal_cents, shipping_cents, total_cents,"
            " shipping_address) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (oid, cid, self.ago(placed), status, subtotal, to_cents(shipping), total, address),
        )
        for item in items:
            sku, name, qty, price = item[:4]
            final_sale = bool(item[4]) if len(item) > 4 else False
            self.conn.execute(
                "INSERT INTO order_items (order_id, sku, name, qty, unit_price_cents, final_sale) VALUES (?, ?, ?, ?, ?, ?)",
                (oid, sku, name, qty, to_cents(price), int(final_sale)),
            )
        for n, amount in enumerate(payments or [total / 100], start=1):
            self.conn.execute(
                "INSERT INTO payments (id, order_id, amount_cents, provider_ref, method, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (f"pay_{oid}_{n}", oid, to_cents(amount), f"pi_mock_{oid}_{n}", "visa", self.ago(placed)),
            )
        if shipment:
            s = shipment
            self.conn.execute(
                "INSERT INTO shipments (id, order_id, carrier, tracking_number, status, shipped_at, delivered_at, eta,"
                " last_event) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"sh_{oid}",
                    oid,
                    s.get("carrier", "UPS"),
                    f"1Z{oid}{zlib.crc32(oid.encode()) % 10_000:04d}",
                    s["status"],
                    self.ago(s["shipped"]) if "shipped" in s else None,
                    self.ago(s["delivered"]) if "delivered" in s else None,
                    self.ago(-s["eta_in"]) if "eta_in" in s else None,
                    s.get("event"),
                ),
            )

    def past_refund(self, oid: str, amount: float, days: float, reason: str) -> None:
        key = f"seed:{oid}:{amount}"
        self.conn.execute(
            "INSERT INTO refunds (id, order_id, payment_id, amount_cents, reason, provider_ref, case_id, idempotency_key,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)",
            (f"re_seed_{oid}", oid, f"pay_{oid}_1", to_cents(amount), reason, f"re_seed_{oid}", key, self.ago(days)),
        )
        self.conn.execute(
            "INSERT INTO psp_refunds (id, payment_ref, amount_cents, idempotency_key, created_at) VALUES (?, ?, ?, ?, ?)",
            (f"re_seed_{oid}", f"pi_mock_{oid}_1", to_cents(amount), key, self.ago(days)),
        )


def seed(conn: sqlite3.Connection, now: datetime, filler_customers: int = 25) -> None:
    b = _Builder(conn, now)
    conn.execute("BEGIN")
    try:
        _fixtures(b)
        _filler(b, filler_customers)
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def _fixtures(b: _Builder) -> None:
    delivered = lambda shipped, arrived: {  # noqa: E731
        "status": "delivered", "shipped": shipped, "delivered": arrived, "event": "Delivered - left at front door"
    }

    # Maria: loyal gold customer. Damaged item, a package in transit, an order still processing.
    b.customer("CUST-001", "maria.gomez@example.com", "Maria Gomez", "gold", 800)
    b.order("A1001", "CUST-001", placed=9, status="delivered",
            items=[("KIT-POUR", "Ceramic Pour-Over Coffee Set", 1, 39.99)], shipment=delivered(7, 4))
    b.order("A1002", "CUST-001", placed=7, status="shipped", shipping=0,
            items=[("HOME-LAMP", "Arc Floor Lamp", 1, 129.00)],
            shipment={"status": "in_transit", "shipped": 5, "eta_in": 2, "event": "In transit - departed Memphis hub"})
    b.order("A1003", "CUST-001", placed=0.1, status="processing", shipping=0,
            items=[("APP-HOOD", "Merino Hoodie", 1, 89.00)])

    # James: expensive damaged headphones (over auto-approve limit) + a lost package.
    b.customer("CUST-002", "james.oconnor@example.com", "James O'Connor", "standard", 400)
    b.order("A2001", "CUST-002", placed=6, status="delivered", shipping=0, address=ADDRESSES[1],
            items=[("AUD-NC700", "NC700 Noise-Cancelling Headphones", 1, 249.00)], shipment=delivered(5, 3))
    b.order("A2002", "CUST-002", placed=15, status="shipped", address=ADDRESSES[1],
            items=[("KIT-KNIFE", "Chef Knife 8in", 1, 59.50)],
            shipment={"status": "lost", "shipped": 14, "event": "Exception - package lost in transit, carrier claim opened"})

    # Priya: return outside the window, wrong colour received, final-sale change of mind.
    b.customer("CUST-003", "priya.shah@example.com", "Priya Shah", "silver", 600)
    b.order("A3001", "CUST-003", placed=50, status="delivered", address=ADDRESSES[2],
            items=[("FIT-MAT", "Cork Yoga Mat", 1, 35.00)], shipment=delivered(48, 45))
    b.order("A3002", "CUST-003", placed=5, status="delivered", address=ADDRESSES[2],
            items=[("APP-TEE-BLU", "Organic Tee - Blue (M)", 2, 14.00)], shipment=delivered(4, 2))
    b.order("A3003", "CUST-003", placed=9, status="delivered", address=ADDRESSES[2],
            items=[("HOME-CNDL", "Clearance Candle Set", 1, 18.00, True)], shipment=delivered(8, 6))

    # Derek: account under fraud review - any money movement needs a human.
    b.customer("CUST-004", "derek.miller@example.com", "Derek Miller", "standard", 200, flags=("fraud_review",))
    b.order("A4001", "CUST-004", placed=8, status="delivered", address=ADDRESSES[3],
            items=[("AUD-SPK", "Bluetooth Speaker Mini", 1, 65.00)], shipment=delivered(7, 5))

    # Sam: 3-day-old account, expensive item "never arrived" though the carrier says delivered.
    b.customer("CUST-005", "sam.lee@example.com", "Sam Lee", "standard", 3)
    b.order("A5001", "CUST-005", placed=3, status="delivered", shipping=0, address=ADDRESSES[4],
            items=[("WEAR-SW2", "Smart Watch S2", 1, 180.00)], shipment=delivered(2, 1))

    # Olivia: already-refunded order, shipped order (address can't change), processing order (can change).
    b.customer("CUST-006", "olivia.brown@example.com", "Olivia Brown", "gold", 1000)
    b.order("A6001", "CUST-006", placed=15, status="delivered", shipping=0, address=ADDRESSES[5],
            items=[("KIT-GRND", "Burr Espresso Grinder", 1, 120.00)], shipment=delivered(14, 12))
    b.past_refund("A6001", 120.00, 8, "damaged on arrival")
    b.order("A6002", "CUST-006", placed=2, status="shipped", address=ADDRESSES[5],
            items=[("HOME-BLNK", "Chunky Knit Throw Blanket", 1, 49.00)],
            shipment={"status": "in_transit", "shipped": 1, "eta_in": 3, "event": "In transit - label scanned"})
    b.order("A6003", "CUST-006", placed=0.2, status="processing", address=ADDRESSES[5],
            items=[("HOME-PLNT", "Ceramic Planter Set", 1, 42.00)])

    # Noah: charged twice for one order; account-access requests.
    b.customer("CUST-007", "noah.wilson@example.com", "Noah Wilson", "standard", 300)
    b.order("A7001", "CUST-007", placed=10, status="delivered", shipping=0, address=ADDRESSES[1],
            items=[("FIT-SHOE", "Trail Running Shoes", 1, 95.00)], shipment=delivered(9, 8), payments=[95.00, 95.00])

    # Emma: multi-item order with one broken item; pricey blanket (reship is auto, refund needs approval).
    b.customer("CUST-008", "emma.davis@example.com", "Emma Davis", "silver", 500)
    b.order("A8001", "CUST-008", placed=9, status="delivered", shipping=0, address=ADDRESSES[3],
            items=[("KIT-GLAS", "Glass Food Containers (set of 5)", 1, 32.00),
                   ("KIT-BORD", "Bamboo Cutting Board", 1, 24.00),
                   ("KIT-SPAT", "Silicone Spatula Set", 1, 16.00)],
            shipment=delivered(8, 6))
    b.order("A8002", "CUST-008", placed=24, status="delivered", shipping=0, address=ADDRESSES[3],
            items=[("HOME-WBLK", "Weighted Blanket 15lb", 1, 139.00)], shipment=delivered(22, 20))

    # Liam: already received $120 in refunds this month - another refund exceeds the auto budget.
    b.customer("CUST-009", "liam.chen@example.com", "Liam Chen", "standard", 700)
    b.order("A9000", "CUST-009", placed=25, status="delivered", shipping=0, address=ADDRESSES[4],
            items=[("OFF-MAT", "Standing Desk Mat", 1, 120.00)], shipment=delivered(23, 20))
    b.past_refund("A9000", 120.00, 10, "defective")
    b.order("A9001", "CUST-009", placed=6, status="delivered", shipping=0, address=ADDRESSES[4],
            items=[("OFF-ORG", "Walnut Desk Organizer", 1, 45.00)], shipment=delivered(5, 3))

    # Ava: two orders delivered the same day (a vague "one item is broken" needs a clarifying question).
    b.customer("CUST-010", "ava.martinez@example.com", "Ava Martinez", "silver", 450)
    b.order("A10001", "CUST-010", placed=8, status="delivered", address=ADDRESSES[2],
            items=[("HOME-BLMP", "Ceramic Bedside Lamp", 1, 60.00)], shipment=delivered(7, 5))
    b.order("A10002", "CUST-010", placed=8, status="delivered", address=ADDRESSES[2],
            items=[("HOME-FRM", "Oak Picture Frame 8x10", 1, 22.00)], shipment=delivered(7, 5))


def _filler(b: _Builder, count: int) -> None:
    rng = random.Random(42)
    first = ["alex", "jordan", "casey", "riley", "morgan", "taylor", "quinn", "avery", "rowan", "sky"]
    last = ["nguyen", "patel", "kim", "garcia", "novak", "okafor", "silva", "berg", "haddad", "ito"]
    for n in range(count):
        f, l = rng.choice(first), rng.choice(last)
        cid = f"CUST-{100 + n}"
        b.customer(cid, f"{f}.{l}{n}@example.com", f"{f.title()} {l.title()}", rng.choice(["standard", "silver", "gold"]),
                   rng.randint(30, 1500))
        for k in range(rng.randint(1, 3)):
            sku, name, price = rng.choice(CATALOG)
            placed = rng.randint(3, 90)
            b.order(f"B{100 + n}{k}", cid, placed=placed, status="delivered", address=rng.choice(ADDRESSES),
                    items=[(sku, name, rng.randint(1, 2), price)],
                    shipment={"status": "delivered", "shipped": placed - 1, "delivered": max(placed - 4, 0.5),
                              "event": "Delivered"})
