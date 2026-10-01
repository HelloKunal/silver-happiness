"""Eval harness: run the labelled tickets through the real workflow (real LLM, mock backend),
optionally under injected faults, and write a JSON + Markdown report."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console
from rich.progress import Progress

from support_agent.approvals import decide
from support_agent.backend.db import connect, create_schema
from support_agent.backend.notifier import ConsoleNotifier
from support_agent.backend.seed import seed
from support_agent.chaos import FaultInjector, build_fault_plan
from support_agent.config import Settings, load_llm_config
from support_agent.evals.dataset import TICKETS, EvalTicket
from support_agent.evals.scoring import HUMAN, TicketResult, result_dict, score, summarize
from support_agent.graph.builder import build_graph
from support_agent.llm.gateway import LLMGateway
from support_agent.llm.registry import build_gateway
from support_agent.runtime import case_state, submit_ticket
from support_agent.services import build_services
from support_agent.util import utcnow


def run_one(ticket: EvalTicket, *, settings: Settings, template_db: Path, workdir: Path, gateway: LLMGateway,
            injector: FaultInjector, now, auto_approve: bool) -> TicketResult:
    db = workdir / f"{ticket.id}.db"
    shutil.copy(template_db, db)
    services = build_services(settings, db_path=db, gateway=gateway, notifier=ConsoleNotifier(), injector=injector,
                              clock=lambda: now)
    graph = build_graph(services, checkpointer=InMemorySaver())
    case_id = f"eval-{ticket.id}"
    started = time.monotonic()
    state: dict[str, Any] = {}
    approval_seen, error = False, None
    try:
        result = submit_ticket(graph, services, sender=ticket.sender, subject=ticket.subject, body=ticket.body,
                               case_id=case_id)
        approval_seen = result.interrupt is not None
        if approval_seen and auto_approve:
            decide(graph, services, case_id=case_id, decision="approve", reviewer="eval-auto")
        state, _ = case_state(graph, case_id)
    except Exception as e:  # a crash is a scored failure, not a harness failure
        error = f"{type(e).__name__}: {e}"
        try:
            state, _ = case_state(graph, case_id)
        except Exception:
            state = {}
    psp = services.repo.fetch_all("SELECT * FROM psp_refunds WHERE id NOT LIKE 're_seed_%'")
    services.repo.conn.close()
    return score(ticket, state, approval_seen=approval_seen, psp_refund_count=len(psp), error=error,
                 latency_s=time.monotonic() - started)


def run_suite(settings: Settings, *, chaos: str = "none", limit: int = 0, concurrency: int = 4,
              auto_approve: bool = False, only: list[str] | None = None, console: Console | None = None,
              providers: dict | None = None) -> Path:
    console = console or Console()
    # The eval never touches real payment providers or phones.
    settings = replace(settings, payments_backend="mock", ntfy_topic=None, ntfy_reply_topic=None)
    llm_config = load_llm_config(settings.llm_config_path)
    plan = build_fault_plan(chaos, llm_config)
    injector = FaultInjector(plan, seed=11)
    # One gateway for the whole run, so circuit breakers see every ticket (as in production).
    gateway = build_gateway(llm_config, providers=providers, injector=injector)

    tickets = [t for t in TICKETS if not only or t.id in only]
    if limit:
        tickets = tickets[:limit]
    now = utcnow()
    workdir = Path(tempfile.mkdtemp(prefix="support-eval-"))
    template_db = workdir / "template.db"
    conn = connect(template_db)
    create_schema(conn)
    seed(conn, now)
    conn.close()

    results: list[TicketResult] = []
    started = time.monotonic()
    with Progress(console=console, transient=True) as progress:
        task = progress.add_task(f"eval ({plan.name})", total=len(tickets))
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(run_one, t, settings=settings, template_db=template_db, workdir=workdir, gateway=gateway,
                            injector=injector, now=now, auto_approve=auto_approve)
                for t in tickets
            ]
            for f in as_completed(futures):
                results.append(f.result())
                progress.advance(task)
    results.sort(key=lambda r: r.id)
    shutil.rmtree(workdir, ignore_errors=True)

    expected_human = {t.id: set(t.outcomes) <= HUMAN for t in tickets}
    metrics = summarize(results, expected_human)
    metrics["wall_clock_s"] = round(time.monotonic() - started, 1)
    meta = {
        "timestamp": now.isoformat(timespec="seconds"),
        "chaos": plan.name,
        "auto_approve": auto_approve,
        "model_chains": {r: [llm_config.models[k].model for k in spec.chain] for r, spec in llm_config.routes.items()},
        "injected_faults": dict(injector.injected),
    }

    out_dir = settings.root / "evals" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{now.strftime('%Y%m%d-%H%M%S')}-{plan.name}"
    (out_dir / f"{stem}.json").write_text(
        json.dumps({"meta": meta, "metrics": metrics, "results": [result_dict(r) for r in results]}, indent=2, default=str),
        encoding="utf-8",
    )
    md = out_dir / f"{stem}.md"
    md.write_text(render_markdown(meta, metrics, results, {t.id: t for t in tickets}), encoding="utf-8")
    _print_summary(console, meta, metrics)
    return md


def _fmt(v: Any, suffix: str = "%") -> str:
    return "n/a" if v is None else f"{v}{suffix}"


def render_markdown(meta: dict, m: dict, results: list[TicketResult], tickets: dict[str, EvalTicket]) -> str:
    lines = [
        f"# Eval report - {meta['timestamp']}",
        "",
        f"Fault injection: **{meta['chaos']}** | tickets: **{m['tickets']}** | auto-approve: {meta['auto_approve']} | "
        f"specialist chain: {' -> '.join(meta['model_chains']['specialist'])}",
        "",
        "## Headline",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Completion rate (case reached a safe end state) | {_fmt(m['completion_rate'])} |",
        f"| Resolved autonomously (no human) | {_fmt(m['auto_resolution_rate'])} |",
        f"| Resolved autonomously **and correctly** | {_fmt(m['correct_auto_resolution_rate'])} |",
        f"| Escalation recall (needed a human -> got one) | {_fmt(m['escalation_recall'])} |",
        f"| Escalation precision (sent to a human -> needed one) | {_fmt(m['escalation_precision'])} |",
        f"| Wrong actions executed without approval | {_fmt(m['wrong_action_rate'])} ({m['wrong_actions']}) |",
        f"| Double refunds | {m['double_refunds']} |",
        f"| Routing accuracy | {_fmt(m['routing_accuracy'])} |",
        f"| Outcome accuracy | {_fmt(m['outcome_accuracy'])} |",
        f"| Action accuracy (proposed actions match label) | {_fmt(m['action_accuracy'])} |",
        f"| Fully correct tickets | {_fmt(m['fully_correct'])} |",
        "",
        "## Resilience & cost",
        "",
        "| LLM calls | per ticket | served by fallback model | retried attempts | template replies | est. cost | p50 / p95 latency |",
        "|---|---|---|---|---|---|---|",
        f"| {m['llm_calls']} | {m['llm_calls_per_ticket']} | {m['fallback_served_calls']} | {m['retried_attempts']} | "
        f"{m['template_replies']} | ${m['cost_usd']} | {m['latency_p50_s']}s / {m['latency_p95_s']}s |",
        "",
        f"Injected faults: {json.dumps(meta['injected_faults']) if meta['injected_faults'] else 'none'}",
        "",
        "## Tickets that were not fully correct",
        "",
        "| id | tags | route (exp -> got) | outcome (exp -> got) | proposed actions | problems |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        if r.correct:
            continue
        t = tickets[r.id]
        exp_route = "/".join(t.route) if isinstance(t.route, tuple) else t.route
        proposed = ", ".join(f"{a['type']} {a.get('order_id') or ''} {a.get('amount') or ''}".strip() for a in r.proposed) or "-"
        problems = r.unsafe + r.errors + (["actions don't match label"] if r.actions_ok is False else [])
        if r.handoff_reason and r.outcome == "handoff":
            problems.append(f"handoff: {r.handoff_reason[:80]}")
        lines.append(f"| {r.id} | {', '.join(r.tags)} | {exp_route} -> {r.route} | {'/'.join(t.outcomes)} -> {r.outcome} | "
                     f"{proposed} | {'; '.join(problems) or '-'} |")
    return "\n".join(lines) + "\n"


def _print_summary(console: Console, meta: dict, m: dict) -> None:
    console.rule(f"eval: {meta['chaos']}")
    for key in ("completion_rate", "auto_resolution_rate", "correct_auto_resolution_rate", "escalation_recall",
                "escalation_precision", "wrong_action_rate", "double_refunds", "routing_accuracy", "action_accuracy",
                "fully_correct", "llm_calls", "fallback_served_calls", "cost_usd", "wall_clock_s"):
        console.print(f"  {key:<30} {m[key]}")
