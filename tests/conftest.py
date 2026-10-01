from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from support_agent.backend.notifier import ConsoleNotifier
from support_agent.backend.seed import seed
from support_agent.chaos import FaultInjector, FaultPlan
from support_agent.config import Settings, load_llm_config
from support_agent.graph.builder import build_graph
from support_agent.llm.gateway import LLMGateway
from support_agent.services import build_services

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)


def settings() -> Settings:
    return Settings(
        root=ROOT,
        db_path=Path(":memory:"),
        checkpoint_path=Path(":memory:"),
        llm_config_path=ROOT / "config" / "llm.yaml",
        policy_config_path=ROOT / "config" / "policy.yaml",
    )


class Harness:
    def __init__(self, provider, plan: FaultPlan | None = None):
        self.provider = provider
        self.llm_config = load_llm_config(ROOT / "config" / "llm.yaml")
        self.gateway = LLMGateway(
            self.llm_config, {"anthropic": provider}, sleep=lambda s: None, rng=random.Random(0)
        )
        self.notifier = ConsoleNotifier()
        injector = FaultInjector(plan) if plan else None
        self.services = build_services(
            settings(), db_path=":memory:", gateway=self.gateway, notifier=self.notifier,
            injector=injector, clock=lambda: NOW, sleep=lambda s: None,
        )
        seed(self.services.repo.conn, NOW)
        self.graph = build_graph(self.services, checkpointer=InMemorySaver())

    def psp_refunds(self) -> list[dict]:
        return self.services.repo.fetch_all("SELECT * FROM psp_refunds WHERE id NOT LIKE 're_seed_%'")

    def emails(self, case_id: str) -> list[dict]:
        return self.services.repo.emails_for(case_id)


@pytest.fixture
def make_harness():
    return Harness
