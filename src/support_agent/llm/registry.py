from __future__ import annotations

from collections.abc import Callable

from support_agent.chaos import ChaosProvider, FaultInjector
from support_agent.config import LLMConfig
from support_agent.llm.base import LLMProvider
from support_agent.llm.gateway import LLMGateway


def _anthropic(config: LLMConfig) -> LLMProvider:
    from support_agent.llm.anthropic_provider import AnthropicProvider

    return AnthropicProvider(timeout_s=config.request_timeout_s)


# Register new providers here, e.g. "openai": _openai. Each factory returns an LLMProvider.
PROVIDER_FACTORIES: dict[str, Callable[[LLMConfig], LLMProvider]] = {
    "anthropic": _anthropic,
}


def build_gateway(
    config: LLMConfig,
    *,
    providers: dict[str, LLMProvider] | None = None,
    injector: FaultInjector | None = None,
    **gateway_kwargs,
) -> LLMGateway:
    if providers is None:
        needed = {m.provider for m in config.models.values()}
        unknown = needed - PROVIDER_FACTORIES.keys()
        if unknown:
            raise ValueError(f"no provider factory registered for {sorted(unknown)}")
        providers = {name: PROVIDER_FACTORIES[name](config) for name in needed}
    if injector is not None and injector.plan.llm:
        providers = {name: ChaosProvider(p, injector) for name, p in providers.items()}
    return LLMGateway(config, providers, **gateway_kwargs)
