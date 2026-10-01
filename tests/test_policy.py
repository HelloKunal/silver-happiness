"""Policy engine: the guardrails that hold regardless of what the model proposes."""

from __future__ import annotations

import pytest

from .fakes import FakeLLM, triage


@pytest.fixture
def engine(make_harness):
    h = make_harness(FakeLLM(triage_result=triage("general")))
    return h.services.policy, h.services.repo


def evaluate(engine, email, actions, allowed=("refund", "reship", "cancel_order", "update_shipping_address",
                                               "store_credit", "update_email", "send_password_reset"), signals=()):
    policy, repo = engine
    full = [{"order_id": None, "amount": None, "new_value": None, "reason": "", **a} for a in actions]
    return policy.evaluate(
        resolution={"outcome": "resolve", "actions": full},
        allowed_actions=allowed,
        customer=repo.customer_by_email(email),
        triage={"risk_signals": list(signals)},
    )


def verdicts(report):
    return [d.verdict for d in report.decisions]


def test_small_refund_is_automatic(engine):
    r = evaluate(engine, "maria.gomez@example.com", [{"type": "refund", "order_id": "A1001", "amount": 44.99}])
    assert verdicts(r) == ["auto"] and r.outcome == "execute"


def test_refund_over_limit_needs_approval(engine):
    r = evaluate(engine, "james.oconnor@example.com", [{"type": "refund", "order_id": "A2001", "amount": 249}])
    assert verdicts(r) == ["approval"] and r.outcome == "approval"


def test_refund_exceeding_paid_amount_is_denied(engine):
    r = evaluate(engine, "maria.gomez@example.com", [{"type": "refund", "order_id": "A1001", "amount": 60}])
    assert verdicts(r) == ["deny"] and r.outcome == "handoff"


def test_two_refunds_cannot_jointly_exceed_balance(engine):
    r = evaluate(engine, "maria.gomez@example.com", [{"type": "refund", "order_id": "A1001", "amount": 30},
                                                     {"type": "refund", "order_id": "A1001", "amount": 30}])
    assert verdicts(r) == ["auto", "deny"]


def test_already_refunded_order_is_denied(engine):
    r = evaluate(engine, "olivia.brown@example.com", [{"type": "refund", "order_id": "A6001", "amount": 120}])
    assert verdicts(r) == ["deny"]


def test_other_customers_order_is_denied(engine):
    r = evaluate(engine, "james.oconnor@example.com", [{"type": "refund", "order_id": "A1001", "amount": 10}])
    assert verdicts(r) == ["deny"]


def test_outside_return_window_needs_approval(engine):
    r = evaluate(engine, "priya.shah@example.com", [{"type": "refund", "order_id": "A3001", "amount": 35}])
    assert verdicts(r) == ["approval"]
    assert any("outside return window" in reason for reason in r.decisions[0].reasons)


def test_final_sale_refund_needs_approval(engine):
    r = evaluate(engine, "priya.shah@example.com", [{"type": "refund", "order_id": "A3003", "amount": 18}])
    assert verdicts(r) == ["approval"]


def test_fraud_flag_needs_approval(engine):
    r = evaluate(engine, "derek.miller@example.com", [{"type": "refund", "order_id": "A4001", "amount": 20}])
    assert verdicts(r) == ["approval"]


def test_new_account_needs_approval(engine):
    r = evaluate(engine, "sam.lee@example.com", [{"type": "reship", "order_id": "A5001"}])
    assert verdicts(r) == ["approval"]


def test_customer_budget_needs_approval(engine):
    r = evaluate(engine, "liam.chen@example.com", [{"type": "refund", "order_id": "A9001", "amount": 45}])
    assert verdicts(r) == ["approval"]


def test_prompt_injection_signal_needs_approval(engine):
    r = evaluate(engine, "maria.gomez@example.com", [{"type": "refund", "order_id": "A1001", "amount": 10}],
                 signals=["prompt_injection"])
    assert verdicts(r) == ["approval"]


def test_reship_while_in_transit_is_denied(engine):
    r = evaluate(engine, "maria.gomez@example.com", [{"type": "reship", "order_id": "A1002"}])
    assert verdicts(r) == ["deny"]


def test_reship_lost_package_is_automatic(engine):
    r = evaluate(engine, "james.oconnor@example.com", [{"type": "reship", "order_id": "A2002"}])
    assert verdicts(r) == ["auto"]


def test_cancel_only_while_processing(engine):
    assert verdicts(evaluate(engine, "maria.gomez@example.com", [{"type": "cancel_order", "order_id": "A1003"}])) == ["auto"]
    assert verdicts(evaluate(engine, "maria.gomez@example.com", [{"type": "cancel_order", "order_id": "A1002"}])) == ["deny"]


def test_address_change_only_while_processing(engine):
    ok = evaluate(engine, "olivia.brown@example.com",
                  [{"type": "update_shipping_address", "order_id": "A6003", "new_value": "1 New St, Boston, MA"}])
    shipped = evaluate(engine, "olivia.brown@example.com",
                       [{"type": "update_shipping_address", "order_id": "A6002", "new_value": "1 New St, Boston, MA"}])
    assert verdicts(ok) == ["auto"] and verdicts(shipped) == ["deny"]


def test_email_change_always_needs_approval(engine):
    r = evaluate(engine, "noah.wilson@example.com", [{"type": "update_email", "new_value": "noah@new.example"}])
    assert verdicts(r) == ["approval"]


def test_action_outside_specialist_scope_is_denied(engine):
    r = evaluate(engine, "maria.gomez@example.com", [{"type": "refund", "order_id": "A1001", "amount": 5}],
                 allowed=("send_password_reset",))
    assert verdicts(r) == ["deny"]


def test_unknown_sender_cannot_act(engine):
    r = evaluate(engine, "stranger@example.net", [{"type": "send_password_reset"}])
    assert verdicts(r) == ["deny"]
