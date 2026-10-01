"""Human-in-the-loop notifications.

NtfyNotifier pushes to a phone via https://ntfy.sh. Approval requests carry Approve / Reject /
Take over buttons; each button POSTs a signed payload to a *reply topic* that `support approvals
listen` consumes, so no public web server is needed. The HMAC signature (over case id, decision
and a per-request nonce) stops anyone who guesses the topic name from approving refunds.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

log = logging.getLogger(__name__)

DECISIONS = ("approve", "reject", "takeover")


def sign(secret: str, case_id: str, decision: str, nonce: str) -> str:
    msg = f"{case_id}:{decision}:{nonce}".encode()
    return hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()[:32]


def verify(secret: str, case_id: str, decision: str, nonce: str, signature: str) -> bool:
    return hmac.compare_digest(sign(secret, case_id, decision, nonce), signature)


class Notifier(Protocol):
    def approval_requested(self, *, case_id: str, nonce: str, title: str, message: str) -> None: ...

    def escalated(self, *, case_id: str, title: str, message: str) -> None: ...


@dataclass
class ConsoleNotifier:
    """Default when ntfy isn't configured (and in tests): records notifications, logs them."""

    sent: list[dict[str, Any]] = field(default_factory=list)

    def approval_requested(self, *, case_id: str, nonce: str, title: str, message: str) -> None:
        self.sent.append({"kind": "approval", "case_id": case_id, "title": title, "message": message})
        log.info("APPROVAL NEEDED %s: %s - decide with `support approvals decide %s approve|reject`", case_id, title, case_id)

    def escalated(self, *, case_id: str, title: str, message: str) -> None:
        self.sent.append({"kind": "escalation", "case_id": case_id, "title": title, "message": message})
        log.info("ESCALATED %s: %s", case_id, title)


class NtfyNotifier:
    def __init__(self, *, server: str, topic: str, reply_topic: str, secret: str, token: str | None = None):
        self.server = server.rstrip("/")
        self.topic = topic
        self.reply_topic = reply_topic
        self.secret = secret
        self.token = token
        self._http = httpx.Client(timeout=10)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _publish(self, payload: dict[str, Any]) -> None:
        # Notification failures must never fail the workflow: the approval is also persisted in the DB
        # and visible via `support approvals list`.
        try:
            r = self._http.post(self.server, json={"topic": self.topic, **payload}, headers=self._headers())
            r.raise_for_status()
        except httpx.HTTPError as e:
            log.error("ntfy publish failed (case still recorded in DB): %s", e)

    def approval_requested(self, *, case_id: str, nonce: str, title: str, message: str) -> None:
        labels = {"approve": "Approve", "reject": "Reject", "takeover": "Take over"}
        actions = []
        for decision in DECISIONS:
            body = json.dumps(
                {"case_id": case_id, "decision": decision, "nonce": nonce, "sig": sign(self.secret, case_id, decision, nonce)}
            )
            action = {
                "action": "http",
                "label": labels[decision],
                "url": f"{self.server}/{self.reply_topic}",
                "method": "POST",
                "body": body,
                "clear": True,
            }
            if self.token:
                action["headers"] = self._headers()
            actions.append(action)
        self._publish({"title": title, "message": message, "priority": 4, "tags": ["moneybag"], "actions": actions})

    def escalated(self, *, case_id: str, title: str, message: str) -> None:
        self._publish({"title": title, "message": message, "priority": 4, "tags": ["rotating_light"]})

    def poll_replies(self, since: str) -> list[dict[str, Any]]:
        """Fetch button presses published to the reply topic since `since` (message id or 'all')."""
        r = self._http.get(
            f"{self.server}/{self.reply_topic}/json", params={"poll": "1", "since": since}, headers=self._headers()
        )
        r.raise_for_status()
        events = [json.loads(line) for line in r.text.splitlines() if line.strip()]
        return [e for e in events if e.get("event") == "message"]
