"""Payment provider adapters. Both honour idempotency keys the way Stripe does:
the same key + same params returns the original result instead of moving money again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from support_agent.backend.repository import Repository
from support_agent.chaos import FaultInjector
from support_agent.resilience import TransientError
from support_agent.util import iso, new_id, utcnow


class PaymentError(Exception):
    pass


class PaymentTransientError(PaymentError, TransientError):
    """Timeout / 5xx / rate limit. Outcome unknown - retry with the SAME idempotency key."""


class PaymentDeclinedError(PaymentError):
    """Permanent: amount exceeds refundable balance, invalid payment, key reused with other params."""


@dataclass(frozen=True)
class RefundReceipt:
    provider_ref: str
    amount_cents: int
    replayed: bool = False  # True when the provider returned an earlier result for this key


class PaymentGateway(Protocol):
    name: str

    def refund(self, *, payment_ref: str, amount_cents: int, idempotency_key: str, metadata: dict[str, str]) -> RefundReceipt: ...


class MockPaymentGateway:
    """In-process stand-in for Stripe, with its own ledger table (psp_refunds) and fault injection."""

    name = "mock"

    def __init__(self, repo: Repository, injector: FaultInjector | None = None):
        self.repo = repo
        self.injector = injector

    def _fault(self, p_attr: str, kind: str) -> bool:
        return bool(self.injector and self.injector.roll(getattr(self.injector.plan, p_attr), kind))

    def refund(self, *, payment_ref: str, amount_cents: int, idempotency_key: str, metadata: dict[str, str]) -> RefundReceipt:
        if self._fault("payment_timeout_before_commit", "payment_timeout_before_commit"):
            raise PaymentTransientError("payment API timeout (request never reached the provider)")

        with self.repo.transaction() as conn:
            existing = conn.execute("SELECT * FROM psp_refunds WHERE idempotency_key = ?", (idempotency_key,)).fetchone()
            if existing is not None:
                if existing["amount_cents"] != amount_cents or existing["payment_ref"] != payment_ref:
                    raise PaymentDeclinedError("idempotency key reused with different parameters")
                return RefundReceipt(provider_ref=existing["id"], amount_cents=amount_cents, replayed=True)
            payment = conn.execute("SELECT amount_cents FROM payments WHERE provider_ref = ?", (payment_ref,)).fetchone()
            if payment is None:
                raise PaymentDeclinedError(f"no such payment {payment_ref}")
            refunded = conn.execute(
                "SELECT COALESCE(SUM(amount_cents), 0) FROM psp_refunds WHERE payment_ref = ?", (payment_ref,)
            ).fetchone()[0]
            if amount_cents <= 0 or refunded + amount_cents > payment["amount_cents"]:
                raise PaymentDeclinedError(
                    f"refund {amount_cents} exceeds refundable balance {payment['amount_cents'] - refunded}"
                )
            refund_id = new_id("re_mock")
            conn.execute(
                "INSERT INTO psp_refunds (id, payment_ref, amount_cents, idempotency_key, created_at) VALUES (?, ?, ?, ?, ?)",
                (refund_id, payment_ref, amount_cents, idempotency_key, iso(utcnow())),
            )

        if self._fault("payment_timeout_after_commit", "payment_timeout_after_commit"):
            # Money moved, but the caller never hears about it. Only an idempotent retry is safe now.
            raise PaymentTransientError("payment API timeout (response lost after the refund was created)")
        return RefundReceipt(provider_ref=refund_id, amount_cents=amount_cents)


class StripePaymentGateway:
    """Stripe TEST mode (real API, fake money). Requires `pip install stripe` and a sk_test_ key."""

    name = "stripe"

    def __init__(self, api_key: str):
        import stripe

        if not api_key.startswith(("sk_test_", "rk_test_")):
            raise ValueError("refusing to run against a live Stripe key - use a sk_test_ key")
        self._stripe = stripe
        # Our executor owns retries (with the same idempotency key), so keep Stripe's off.
        self._client = stripe.StripeClient(api_key, max_network_retries=0)

    def refund(self, *, payment_ref: str, amount_cents: int, idempotency_key: str, metadata: dict[str, str]) -> RefundReceipt:
        s = self._stripe
        try:
            refund = self._client.v1.refunds.create(
                params={
                    "payment_intent": payment_ref,
                    "amount": amount_cents,
                    "reason": "requested_by_customer",
                    "metadata": metadata,
                },
                options={"idempotency_key": idempotency_key},
            )
        except (s.APIConnectionError, s.RateLimitError, s.APIError) as e:
            raise PaymentTransientError(f"stripe: {e.user_message or e}") from e
        except (s.InvalidRequestError, s.CardError, s.IdempotencyError) as e:
            raise PaymentDeclinedError(f"stripe: {e.user_message or e}") from e
        return RefundReceipt(provider_ref=refund.id, amount_cents=refund.amount)

    def create_test_payment(self, amount_cents: int, metadata: dict[str, Any]) -> str:
        """Create + confirm a PaymentIntent with Stripe's test Visa. Used by `support seed-stripe`."""
        intent = self._client.v1.payment_intents.create(
            params={
                "amount": amount_cents,
                "currency": "usd",
                "payment_method": "pm_card_visa",
                "confirm": True,
                "automatic_payment_methods": {"enabled": True, "allow_redirects": "never"},
                "metadata": metadata,
            }
        )
        return intent.id
