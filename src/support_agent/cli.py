"""Command-line interface.

    support init-db [--reset]          create + seed the mock shop database
    support run --from EMAIL -s SUBJ "message"   run one ticket through the workflow
    support approvals list | decide CASE approve|reject|takeover | listen
    support case CASE_ID               show a case's state and audit timeline
    support graph                      print the workflow as a Mermaid diagram
    support eval [--chaos flaky] ...   run the labelled ticket set and write a report
    support seed-stripe                attach Stripe TEST-mode payments to the seeded orders
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Annotated

import typer
from langgraph.checkpoint.sqlite import SqliteSaver
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from support_agent.approvals import decide, listen
from support_agent.backend.notifier import NtfyNotifier
from support_agent.backend.seed import seed
from support_agent.config import Settings
from support_agent.graph.builder import build_graph
from support_agent.runtime import case_state, submit_ticket
from support_agent.services import Services, build_services
from support_agent.util import utcnow

app = typer.Typer(add_completion=False, help="Autonomous support resolution agent (LangGraph).")
approvals_app = typer.Typer(help="Human-in-the-loop approvals.")
app.add_typer(approvals_app, name="approvals")
console = Console()


def _check_llm_credentials(settings: Settings) -> None:
    """Fail fast on a missing key: without it every call fails and every case would be escalated."""
    from support_agent.config import load_llm_config

    providers = {m.provider for m in load_llm_config(settings.llm_config_path).models.values()}
    if "anthropic" in providers and not (
        any(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"))
        or (Path.home() / ".config" / "anthropic").exists()
    ):
        console.print("[red]No Claude credentials found. Put ANTHROPIC_API_KEY in .env (see .env.example).[/red]")
        raise typer.Exit(1)


def _setup(verbose: bool = False) -> tuple[Settings, Services, object]:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    if not settings.db_path.exists():
        console.print("[red]No database yet - run `support init-db` first.[/red]")
        raise typer.Exit(1)
    _check_llm_credentials(settings)
    services = build_services(settings)
    settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    saver = SqliteSaver(sqlite3.connect(str(settings.checkpoint_path), check_same_thread=False))
    return settings, services, build_graph(services, checkpointer=saver)


@app.command("init-db")
def init_db(reset: Annotated[bool, typer.Option(help="Delete existing data first.")] = False) -> None:
    """Create and seed the mock shop database."""
    settings = Settings.from_env()
    if settings.db_path.exists():
        if not reset:
            console.print(f"{settings.db_path} already exists (use --reset to recreate).")
            raise typer.Exit(0)
        for path in (settings.db_path, settings.checkpoint_path):
            for suffix in ("", "-wal", "-shm"):
                p = path.with_name(path.name + suffix)
                if p.exists():
                    p.unlink()
    from support_agent.backend.db import connect, create_schema

    conn = connect(settings.db_path)
    create_schema(conn)
    seed(conn, utcnow())
    n = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    console.print(f"[green]Seeded {settings.db_path} with {n} orders.[/green] Try:")
    console.print('  support run --from maria.gomez@example.com -s "Broken coffee set" '
                  '"My pour-over set from order A1001 arrived shattered. Refund please."')


@app.command()
def run(
    message: Annotated[str, typer.Argument(help="Email body.")],
    sender: Annotated[str, typer.Option("--from", help="Sender email (identifies the customer).")],
    subject: Annotated[str, typer.Option("--subject", "-s")] = "Support request",
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run one ticket through the workflow and show what happened."""
    _, services, graph = _setup(verbose)
    with console.status("Working the case..."):
        result = submit_ticket(graph, services, sender=sender, subject=subject, body=message)
    _print_case(result.state, services)
    if result.interrupt:
        console.print(Panel(
            f"Waiting for a human. Approve from the push notification, or:\n"
            f"  support approvals decide {result.case_id} approve\n"
            f"  support approvals decide {result.case_id} reject --note \"reason\"",
            title="Approval required", border_style="yellow"))


@app.command("case")
def show_case(case_id: str) -> None:
    """Show a case's current state and full audit timeline."""
    _, services, graph = _setup()
    state, pending = case_state(graph, case_id)
    if not state:
        console.print(f"[red]No case {case_id}[/red]")
        raise typer.Exit(1)
    _print_case(state, services)
    if pending:
        console.print(f"Paused before: {', '.join(pending)}")


@app.command("graph")
def show_graph() -> None:
    """Print the workflow graph as Mermaid (paste into mermaid.live or a README)."""
    from support_agent.config import load_llm_config
    from support_agent.llm.gateway import LLMGateway

    settings = Settings.from_env()
    gw = LLMGateway(load_llm_config(settings.llm_config_path), providers={})
    services = build_services(settings, db_path=":memory:", gateway=gw)
    typer.echo(build_graph(services).get_graph().draw_mermaid())


@approvals_app.command("list")
def approvals_list() -> None:
    """Cases waiting for a human decision."""
    _, services, _ = _setup()
    table = Table("case", "requested", "summary")
    for a in services.repo.pending_approvals():
        table.add_row(a["case_id"], a["requested_at"], a["summary"])
    console.print(table)


@approvals_app.command("decide")
def approvals_decide(
    case_id: str,
    decision: Annotated[str, typer.Argument(help="approve | reject | takeover")],
    note: Annotated[str, typer.Option(help="Reason (shown to the customer on reject).")] = "",
    reviewer: Annotated[str, typer.Option()] = "cli",
) -> None:
    """Approve / reject / take over a pending case and resume the workflow."""
    _, services, graph = _setup()
    outcome = decide(graph, services, case_id=case_id, decision=decision, reviewer=reviewer, note=note)
    console.print(outcome.message, style="green" if outcome.accepted else "red")
    if outcome.result:
        _print_case(outcome.result.state, services)


@approvals_app.command("listen")
def approvals_listen(once: Annotated[bool, typer.Option(help="Poll once and exit.")] = False) -> None:
    """Apply Approve/Reject button presses from the ntfy app (runs until Ctrl+C)."""
    _, services, graph = _setup(verbose=True)
    if not isinstance(services.notifier, NtfyNotifier):
        console.print("[red]Set NTFY_TOPIC and NTFY_REPLY_TOPIC in .env to use push approvals.[/red]")
        raise typer.Exit(1)
    console.print(f"Listening for decisions on {services.notifier.server}/{services.notifier.reply_topic} (Ctrl+C to stop)")
    try:
        listen(graph, services, services.notifier, once=once)
    except KeyboardInterrupt:
        console.print("stopped.")


@app.command("eval")
def run_eval(
    chaos: Annotated[str, typer.Option(help="none | flaky | primary-outage | total-outage")] = "none",
    limit: Annotated[int, typer.Option(help="Only the first N tickets (0 = all).")] = 0,
    concurrency: Annotated[int, typer.Option()] = 4,
    auto_approve: Annotated[bool, typer.Option(help="Simulate a reviewer approving every approval request.")] = False,
    ids: Annotated[str, typer.Option(help="Comma-separated ticket ids to run.")] = "",
) -> None:
    """Run the labelled ticket set; writes evals/results/<timestamp>.{json,md}."""
    from support_agent.evals.runner import run_suite

    logging.basicConfig(level=logging.ERROR)
    settings = Settings.from_env()
    _check_llm_credentials(settings)
    report_path = run_suite(settings, chaos=chaos, limit=limit, concurrency=concurrency, auto_approve=auto_approve,
                            only=[i.strip() for i in ids.split(",") if i.strip()], console=console)
    console.print(f"[green]Report written to {report_path}[/green]")


@app.command("seed-stripe")
def seed_stripe() -> None:
    """Create a Stripe TEST-mode PaymentIntent for every seeded payment (PAYMENTS_BACKEND=stripe)."""
    from support_agent.backend.payments import StripePaymentGateway

    settings = Settings.from_env()
    if not settings.stripe_api_key:
        console.print("[red]STRIPE_API_KEY is not set.[/red]")
        raise typer.Exit(1)
    gateway = StripePaymentGateway(settings.stripe_api_key)
    services = build_services(settings)
    payments = services.repo.fetch_all("SELECT * FROM payments WHERE provider_ref LIKE 'pi_mock_%'")
    with console.status(f"Creating {len(payments)} test payments..."):
        for p in payments:
            ref = gateway.create_test_payment(p["amount_cents"], {"order_id": p["order_id"], "payment_id": p["id"]})
            services.repo.run("UPDATE payments SET provider_ref = ? WHERE id = ?", ref, p["id"])
    console.print(f"[green]Linked {len(payments)} payments to Stripe test mode.[/green] Set PAYMENTS_BACKEND=stripe.")


# ---------------------------------------------------------------------- output


def _print_case(state: dict, services: Services) -> None:
    triage = state.get("triage") or {}
    console.rule(f"[bold]{state.get('case_id')}[/bold]  status: [bold]{state.get('status')}[/bold]")
    if triage:
        console.print(f"Triage: [cyan]{triage['intent']}[/cyan] ({triage['confidence']:.2f}) - {triage['summary']}")
        if triage.get("risk_signals"):
            console.print(f"Risk signals: [red]{', '.join(triage['risk_signals'])}[/red]")

    timeline = Table("time", "node", "event", box=None, show_header=True, header_style="dim")
    for e in state.get("timeline", []):
        timeline.add_row(e["at"][11:19], e["node"], e["message"])
    console.print(timeline)

    if state.get("policy"):
        for d in state["policy"]["decisions"]:
            a = d["action"]
            color = {"auto": "green", "approval": "yellow", "deny": "red"}[d["verdict"]]
            amount = f"${float(a['amount']):.2f}" if a.get("amount") else ""
            console.print(f"  [{color}]{d['verdict']:>8}[/{color}] {a['type']} {a.get('order_id') or ''} {amount}  "
                          f"{'; '.join(d['reasons'])}")
    for a in state.get("executed_actions", []):
        console.print(f"  executed: {a['type']} {a.get('order_id') or ''} -> {a['status']} "
                      f"{json.dumps(a.get('result') or a.get('error'), default=str)}")
    if state.get("handoff_reason"):
        console.print(f"Hand-off reason: [yellow]{state['handoff_reason']}[/yellow]")
    for err in state.get("errors", []):
        console.print(f"[red]error in {err['node']}: {err['type']}: {err['message'][:200]}[/red]")

    calls = state.get("llm_calls", [])
    if calls:
        fallback = sum(1 for c in calls if c["fallback_depth"])
        console.print(f"LLM calls: {len(calls)}  (fallbacks: {fallback}, retries: {sum(c['attempts'] - 1 for c in calls)}, "
                      f"cost ~${sum(c['cost_usd'] for c in calls):.4f})")
    for email in services.repo.emails_for(state["case_id"]):
        console.print(Panel(email["body"], title=f"email to {email['to_addr']} ({email['kind']}): {email['subject']}"))


if __name__ == "__main__":
    app()
