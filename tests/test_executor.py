"""Exactly-once execution under retries, replays and crashes."""

from __future__ import annotations

import pytest

from support_agent.executor import idempotency_key

from .fakes import FakeLLM, triage

REFUND = {"type": "refund", "order_id": "A1001", "amount": 44.99, "new_value": None, "reason": "damaged"}


@pytest.fixture
def h(make_harness):
    return make_harness(FakeLLM(triage_result=triage("general")))


def maria(h):
    return h.services.repo.customer_by_email("maria.gomez@example.com")


def test_replaying_an_action_does_not_repeat_it(h):
    first = h.services.executor.execute(case_id="c1", action=REFUND, customer=maria(h))
    second = h.services.executor.execute(case_id="c1", action=REFUND, customer=maria(h))
    assert first["replayed"] is False and second["replayed"] is True
    assert len(h.psp_refunds()) == 1


def test_crash_between_refund_and_ledger_update_is_recovered(h, monkeypatch):
    ledger = h.services.ledger
    real_mark = ledger._mark

    def crash_on_success(key, status, **kw):
        if status == "succeeded":
            raise KeyboardInterrupt("process killed")  # money moved + recorded, ledger never marked
        return real_mark(key, status, **kw)

    monkeypatch.setattr(ledger, "_mark", crash_on_success)
    with pytest.raises(KeyboardInterrupt):
        h.services.executor.execute(case_id="c1", action=REFUND, customer=maria(h))
    assert ledger.get(idempotency_key("c1", REFUND))["status"] == "in_progress"

    monkeypatch.setattr(ledger, "_mark", real_mark)  # "restart"
    result = h.services.executor.execute(case_id="c1", action=REFUND, customer=maria(h))
    assert result["result"]["provider_replayed"] is True
    assert len(h.psp_refunds()) == 1


def test_duplicate_charge_refunds_the_second_payment(h):
    noah = h.services.repo.customer_by_email("noah.wilson@example.com")
    action = {"type": "refund", "order_id": "A7001", "amount": 95.0, "new_value": None, "reason": "duplicate"}
    h.services.executor.execute(case_id="c2", action=action, customer=noah)
    [refund] = h.psp_refunds()
    assert refund["payment_ref"] == "pi_mock_A7001_2"


def test_keys_are_stable_and_distinct():
    assert idempotency_key("c1", REFUND) == idempotency_key("c1", dict(REFUND))
    assert idempotency_key("c1", REFUND) != idempotency_key("c2", REFUND)
    assert idempotency_key("c1", REFUND) != idempotency_key("c1", {**REFUND, "amount": 10})
