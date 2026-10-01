"""Push-approval channel security, eval harness smoke test, and tool-schema compatibility."""

from __future__ import annotations

import json
from dataclasses import replace

from rich.console import Console

from support_agent.agents.specialists import SPECS
from support_agent.approvals import decide, handle_ntfy_message
from support_agent.backend.notifier import sign
from support_agent.config import ModelSpec
from support_agent.evals.runner import run_suite
from support_agent.llm.anthropic_provider import AnthropicProvider
from support_agent.llm.base import LLMRequest
from support_agent.runtime import submit_ticket

from .conftest import settings
from .fakes import FakeLLM, resolution, triage
from .test_workflow import refund_script

SECRET = "test-secret"


def pending_case(make_harness):
    llm = FakeLLM(triage_result=triage("refunds", ["A2001"]), specialist_scripts={"refunds": refund_script("A2001", 249.00)})
    h = make_harness(llm)
    result = submit_ticket(h.graph, h.services, sender="james.oconnor@example.com", subject="Broken", body="A2001 broken")
    nonce = h.services.repo.approval(result.case_id)["nonce"]
    return h, result.case_id, nonce


def ntfy_message(case_id, decision, nonce, sig):
    return {"id": "m1", "event": "message",
            "message": json.dumps({"case_id": case_id, "decision": decision, "nonce": nonce, "sig": sig})}


def test_signed_button_press_approves_and_executes(make_harness):
    h, case_id, nonce = pending_case(make_harness)
    outcome = handle_ntfy_message(h.graph, h.services, SECRET, ntfy_message(case_id, "approve", nonce, sign(SECRET, case_id, "approve", nonce)))
    assert outcome.accepted and outcome.result.status == "resolved"
    assert len(h.psp_refunds()) == 1


def test_forged_signature_is_rejected(make_harness):
    h, case_id, nonce = pending_case(make_harness)
    forged = sign("attacker-guess", case_id, "approve", nonce)
    outcome = handle_ntfy_message(h.graph, h.services, SECRET, ntfy_message(case_id, "approve", nonce, forged))
    assert not outcome.accepted and h.psp_refunds() == []
    assert h.services.repo.approval(case_id)["status"] == "pending"


def test_signature_for_reject_cannot_be_replayed_as_approve(make_harness):
    h, case_id, nonce = pending_case(make_harness)
    reject_sig = sign(SECRET, case_id, "reject", nonce)
    outcome = handle_ntfy_message(h.graph, h.services, SECRET, ntfy_message(case_id, "approve", nonce, reject_sig))
    assert not outcome.accepted


def test_stale_nonce_is_ignored(make_harness):
    h, case_id, _ = pending_case(make_harness)
    outcome = handle_ntfy_message(h.graph, h.services, SECRET,
                                  ntfy_message(case_id, "approve", "old-nonce", sign(SECRET, case_id, "approve", "old-nonce")))
    assert not outcome.accepted


def test_double_tap_executes_once(make_harness):
    h, case_id, nonce = pending_case(make_harness)
    msg = ntfy_message(case_id, "approve", nonce, sign(SECRET, case_id, "approve", nonce))
    first = handle_ntfy_message(h.graph, h.services, SECRET, msg)
    second = handle_ntfy_message(h.graph, h.services, SECRET, msg)
    third = decide(h.graph, h.services, case_id=case_id, decision="reject", reviewer="late-reviewer")
    assert first.accepted and not second.accepted and not third.accepted
    assert len(h.psp_refunds()) == 1


def test_every_specialist_tool_schema_is_valid_for_claude():
    provider = AnthropicProvider(client=object())
    model = ModelSpec(key="opus", provider="anthropic", model="claude-opus-5-5")
    for spec in SPECS.values():
        params = provider.build_params(
            LLMRequest(system=spec.system_prompt(), messages=[{"role": "user", "text": "x"}], tools=spec.tool_specs()), model
        )
        for tool in params["tools"]:
            schema = tool["input_schema"]
            assert schema["type"] == "object" and schema["additionalProperties"] is False, tool["name"]
        submit = next(t for t in params["tools"] if t["name"] == "submit_resolution")
        props = submit["input_schema"]["properties"]
        if spec.allowed_actions:
            action_schema = submit["input_schema"]["$defs"]["ProposedAction"]
            assert action_schema["properties"]["type"]["enum"] == list(spec.allowed_actions)
        else:
            assert "actions" not in props  # the generalist cannot even express an action


def test_eval_harness_runs_and_writes_report(tmp_path):
    fake = FakeLLM(triage_result=triage("general"),
                   specialist_scripts={"general": [[("submit_resolution", resolution(reply_points=["Here's our policy."]))]]})
    s = replace(settings(), root=tmp_path)
    report = run_suite(s, limit=4, concurrency=2, providers={"anthropic": fake}, console=Console(quiet=True))
    data = json.loads(report.with_suffix(".json").read_text(encoding="utf-8"))
    assert data["metrics"]["tickets"] == 4
    assert data["metrics"]["completion_rate"] == 100.0
    assert data["metrics"]["double_refunds"] == 0
    assert "## Headline" in report.read_text(encoding="utf-8")
