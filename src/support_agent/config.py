from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


def project_root() -> Path:
    return Path(os.environ.get("SUPPORT_AGENT_HOME", Path.cwd())).resolve()


@dataclass(frozen=True)
class Settings:
    root: Path
    db_path: Path
    checkpoint_path: Path
    llm_config_path: Path
    policy_config_path: Path
    payments_backend: str = "mock"
    stripe_api_key: str | None = None
    ntfy_server: str = "https://ntfy.sh"
    ntfy_topic: str | None = None
    ntfy_reply_topic: str | None = None
    ntfy_token: str | None = None
    approval_secret: str | None = None

    @classmethod
    def from_env(cls) -> Settings:
        root = project_root()
        load_dotenv(root / ".env")
        env = os.environ.get
        return cls(
            root=root,
            db_path=Path(env("SUPPORT_DB_PATH", root / "data" / "shop.db")),
            checkpoint_path=Path(env("SUPPORT_CHECKPOINT_PATH", root / "data" / "checkpoints.db")),
            llm_config_path=Path(env("SUPPORT_LLM_CONFIG", root / "config" / "llm.yaml")),
            policy_config_path=Path(env("SUPPORT_POLICY_CONFIG", root / "config" / "policy.yaml")),
            payments_backend=env("PAYMENTS_BACKEND", "mock") or "mock",
            stripe_api_key=env("STRIPE_API_KEY") or None,
            ntfy_server=(env("NTFY_SERVER") or "https://ntfy.sh").rstrip("/"),
            ntfy_topic=env("NTFY_TOPIC") or None,
            ntfy_reply_topic=env("NTFY_REPLY_TOPIC") or None,
            ntfy_token=env("NTFY_TOKEN") or None,
            approval_secret=env("APPROVAL_SIGNING_SECRET") or None,
        )


# --------------------------------------------------------------------------- policy


@dataclass(frozen=True)
class PolicyConfig:
    refund_auto_max_cents: int = 10_000
    return_window_days: int = 30
    lookback_days: int = 30
    customer_auto_budget_cents: int = 15_000
    reship_auto_max_cents: int = 15_000
    store_credit_auto_max_cents: int = 5_000
    risk_flags_requiring_approval: tuple[str, ...] = ("fraud_review", "chargeback_history")
    new_account_days: int = 14
    ticket_signals_requiring_approval: tuple[str, ...] = ("prompt_injection", "suspicious_identity")
    min_triage_confidence: float = 0.6
    handoff_signals: tuple[str, ...] = ("legal_threat", "chargeback_threat", "safety_issue", "press_inquiry")
    max_agent_turns: int = 10
    max_transfers: int = 1


def load_policy_config(path: Path) -> PolicyConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    refunds = raw.get("refunds", {})
    risk = raw.get("risk", {})
    triage = raw.get("triage", {})
    agent = raw.get("agent", {})
    d = PolicyConfig()
    return PolicyConfig(
        refund_auto_max_cents=round(refunds.get("auto_approve_max", d.refund_auto_max_cents / 100) * 100),
        return_window_days=refunds.get("return_window_days", d.return_window_days),
        lookback_days=refunds.get("lookback_days", d.lookback_days),
        customer_auto_budget_cents=round(refunds.get("customer_auto_budget", d.customer_auto_budget_cents / 100) * 100),
        reship_auto_max_cents=round(
            raw.get("reshipments", {}).get("auto_approve_max_value", d.reship_auto_max_cents / 100) * 100
        ),
        store_credit_auto_max_cents=round(
            raw.get("store_credit", {}).get("auto_approve_max", d.store_credit_auto_max_cents / 100) * 100
        ),
        risk_flags_requiring_approval=tuple(risk.get("flags_requiring_approval", d.risk_flags_requiring_approval)),
        new_account_days=risk.get("new_account_days", d.new_account_days),
        ticket_signals_requiring_approval=tuple(
            risk.get("ticket_signals_requiring_approval", d.ticket_signals_requiring_approval)
        ),
        min_triage_confidence=triage.get("min_confidence", d.min_triage_confidence),
        handoff_signals=tuple(triage.get("handoff_signals", d.handoff_signals)),
        max_agent_turns=agent.get("max_turns", d.max_agent_turns),
        max_transfers=agent.get("max_transfers", d.max_transfers),
    )


# --------------------------------------------------------------------------- llm


@dataclass(frozen=True)
class ModelSpec:
    key: str
    provider: str
    model: str
    supports_effort: bool = True
    server_side_fallback: bool = False
    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0


@dataclass(frozen=True)
class RouteSpec:
    name: str
    chain: tuple[str, ...]
    max_tokens: int = 8000
    effort: str | None = None


@dataclass(frozen=True)
class RetrySettings:
    max_attempts_per_model: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 16.0
    max_retry_after_s: float = 20.0


@dataclass(frozen=True)
class BreakerSettings:
    failure_threshold: int = 3
    cooldown_s: float = 60.0


@dataclass(frozen=True)
class LLMConfig:
    models: dict[str, ModelSpec]
    routes: dict[str, RouteSpec]
    retry: RetrySettings = field(default_factory=RetrySettings)
    breaker: BreakerSettings = field(default_factory=BreakerSettings)
    request_timeout_s: float = 90.0

    def chain(self, route: str) -> list[ModelSpec]:
        return [self.models[key] for key in self.routes[route].chain]


def load_llm_config(path: Path) -> LLMConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models = {}
    for key, m in raw["models"].items():
        cost = m.get("cost_per_mtok", {})
        models[key] = ModelSpec(
            key=key,
            provider=m["provider"],
            model=m["model"],
            supports_effort=m.get("supports_effort", True),
            server_side_fallback=m.get("server_side_fallback", False),
            input_cost_per_mtok=float(cost.get("input", 0.0)),
            output_cost_per_mtok=float(cost.get("output", 0.0)),
        )
    routes = {}
    for name, r in raw["routes"].items():
        unknown = [k for k in r["chain"] if k not in models]
        if unknown:
            raise ValueError(f"route {name!r} references unknown models {unknown}")
        routes[name] = RouteSpec(
            name=name, chain=tuple(r["chain"]), max_tokens=r.get("max_tokens", 8000), effort=r.get("effort")
        )
    return LLMConfig(
        models=models,
        routes=routes,
        retry=RetrySettings(**raw.get("retry", {})),
        breaker=BreakerSettings(**raw.get("circuit_breaker", {})),
        request_timeout_s=float(raw.get("request_timeout_s", 90)),
    )
