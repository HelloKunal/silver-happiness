from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Any


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def to_cents(amount: float | int | str | Decimal) -> int:
    """Dollars -> integer cents. Money is stored and compared in cents everywhere."""
    return int((Decimal(str(amount)) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def dollars(cents: int) -> str:
    return f"{cents / 100:.2f}"


def money(cents: int) -> str:
    return f"${cents / 100:,.2f}"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def dumps(value: Any) -> str:
    """Deterministic JSON (sorted keys) so hashes and cached prompts stay stable."""
    return json.dumps(value, sort_keys=True, default=str)
