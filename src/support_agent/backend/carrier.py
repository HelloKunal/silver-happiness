"""Mock shipping-carrier tracking API (the kind of flaky third-party API agents must tolerate)."""

from __future__ import annotations

from typing import Any

from support_agent.backend.repository import Repository
from support_agent.chaos import FaultInjector
from support_agent.resilience import TransientError


class CarrierUnavailableError(TransientError):
    pass


class CarrierAPI:
    def __init__(self, repo: Repository, injector: FaultInjector | None = None):
        self.repo = repo
        self.injector = injector

    def track(self, tracking_number: str) -> dict[str, Any]:
        if self.injector and self.injector.roll(self.injector.plan.carrier_error, "carrier_error"):
            raise CarrierUnavailableError("carrier tracking API returned 503")
        s = self.repo.shipment_by_tracking(tracking_number)
        if s is None:
            return {"tracking_number": tracking_number, "status": "unknown", "events": []}
        events = []
        if s["shipped_at"]:
            events.append({"at": s["shipped_at"], "event": "Picked up by carrier"})
        if s["last_event"]:
            events.append({"at": s["delivered_at"] or s["eta"] or s["shipped_at"], "event": s["last_event"]})
        return {
            "carrier": s["carrier"],
            "tracking_number": s["tracking_number"],
            "status": s["status"],
            "shipped_at": s["shipped_at"],
            "delivered_at": s["delivered_at"],
            "estimated_delivery": s["eta"],
            "events": events,
        }
