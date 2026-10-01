"""Dependency container handed to the graph builder. Swapping a dependency (fake LLM, chaos-injected
payments, a different notifier) is how tests and the eval harness exercise the same workflow."""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from support_agent.backend.carrier import CarrierAPI
from support_agent.backend.db import connect, create_schema
from support_agent.backend.kb import KnowledgeBase
from support_agent.backend.notifier import ConsoleNotifier, Notifier, NtfyNotifier
from support_agent.backend.payments import MockPaymentGateway, PaymentGateway, StripePaymentGateway
from support_agent.backend.repository import Repository
from support_agent.chaos import FaultInjector
from support_agent.config import LLMConfig, PolicyConfig, Settings, load_llm_config, load_policy_config
from support_agent.executor import ActionExecutor, IdempotencyLedger
from support_agent.llm.gateway import LLMGateway
from support_agent.llm.registry import build_gateway
from support_agent.policy import PolicyEngine
from support_agent.util import utcnow


@dataclass
class Services:
    repo: Repository
    gateway: LLMGateway
    policy: PolicyEngine
    policy_config: PolicyConfig
    executor: ActionExecutor
    ledger: IdempotencyLedger
    payments: PaymentGateway
    carrier: CarrierAPI
    kb: KnowledgeBase
    notifier: Notifier
    clock: Callable[[], datetime]
    sleep: Callable[[float], None] = time.sleep


def approval_secret(settings: Settings, repo: Repository) -> str:
    if settings.approval_secret:
        return settings.approval_secret
    secret = repo.kv_get("approval_secret")
    if secret is None:
        secret = secrets.token_hex(32)
        repo.kv_set("approval_secret", secret)
    return secret


def build_services(
    settings: Settings,
    *,
    db_path: Path | str | None = None,
    llm_config: LLMConfig | None = None,
    policy_config: PolicyConfig | None = None,
    gateway: LLMGateway | None = None,
    notifier: Notifier | None = None,
    injector: FaultInjector | None = None,
    clock: Callable[[], datetime] = utcnow,
    sleep: Callable[[float], None] = time.sleep,
) -> Services:
    conn = connect(db_path or settings.db_path)
    create_schema(conn)
    repo = Repository(conn)
    policy_config = policy_config or load_policy_config(settings.policy_config_path)
    if gateway is None:
        gateway = build_gateway(llm_config or load_llm_config(settings.llm_config_path), injector=injector)

    payments: PaymentGateway
    if settings.payments_backend == "stripe":
        if not settings.stripe_api_key:
            raise ValueError("PAYMENTS_BACKEND=stripe requires STRIPE_API_KEY")
        payments = StripePaymentGateway(settings.stripe_api_key)
    else:
        payments = MockPaymentGateway(repo, injector)

    if notifier is None:
        if settings.ntfy_topic and settings.ntfy_reply_topic:
            notifier = NtfyNotifier(
                server=settings.ntfy_server,
                topic=settings.ntfy_topic,
                reply_topic=settings.ntfy_reply_topic,
                secret=approval_secret(settings, repo),
                token=settings.ntfy_token,
            )
        else:
            notifier = ConsoleNotifier()

    ledger = IdempotencyLedger(repo, clock)
    return Services(
        repo=repo,
        gateway=gateway,
        policy=PolicyEngine(policy_config, repo, clock),
        policy_config=policy_config,
        executor=ActionExecutor(repo, payments, ledger, clock, sleep=sleep),
        ledger=ledger,
        payments=payments,
        carrier=CarrierAPI(repo, injector),
        kb=KnowledgeBase.load_default(),
        notifier=notifier,
        clock=clock,
        sleep=sleep,
    )
