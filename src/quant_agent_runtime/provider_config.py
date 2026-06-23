from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from quant_agent_runtime.models import ProviderMode, ProviderRuntimeStatus


SUPPORTED_RUNTIME_PROVIDER_MODES = {
    ProviderMode.fake_provider.value,
    ProviderMode.disabled_or_local_fallback.value,
    ProviderMode.openai.value,
    ProviderMode.ollama.value,
}

SHARED_LLM_PROVIDER_MODES = {
    ProviderMode.openai.value,
    ProviderMode.ollama.value,
    "disabled",
}
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
DEFAULT_OLLAMA_MODEL = "gemma4:e2b"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT_SECONDS = 45
DEFAULT_MAX_CONTEXT_CHARS = 18000


def internal_provider_status(config_source: str = "internal_default") -> ProviderRuntimeStatus:
    return ProviderRuntimeStatus(
        config_source=config_source,
        configured_provider_mode=ProviderMode.fake_provider.value,
        effective_provider_mode=ProviderMode.fake_provider,
        provider_identifier="fake",
        model_profile="deterministic-plan-fixture",
        allowed_model_roles=["planner"],
        configured=True,
        fallback_reason=None,
        health_check_enabled=False,
        provider_reachable=False,
        configuration_errors=[],
        timeout_seconds=0,
        max_context_chars=18000,
        retention_policy_label="not_applicable",
    )


def runtime_provider_status(
    *,
    base_status: ProviderRuntimeStatus | None = None,
) -> ProviderRuntimeStatus:
    """Return the effective planner provider status from safe env diagnostics.

    Agent-specific env vars override the shared drawer/documentation LLM env. If
    no planner provider env is set, the canonical contract/internal fallback is
    preserved.
    """

    base_status = base_status or internal_provider_status()
    raw_provider, provider_source = _first_env(
        "QUANT_AGENT_LLM_PROVIDER",
        "QUANT_LLM_PROVIDER",
    )
    if raw_provider is None:
        return base_status

    provider = raw_provider.strip().lower()
    config_source = provider_source
    configuration_errors: list[str] = []
    timeout_seconds = _env_int(
        "QUANT_LLM_TIMEOUT_SECONDS",
        DEFAULT_TIMEOUT_SECONDS,
        minimum=1,
        maximum=300,
    )
    max_context_chars = _env_int(
        "QUANT_LLM_MAX_CONTEXT_CHARS",
        DEFAULT_MAX_CONTEXT_CHARS,
        minimum=2000,
        maximum=60000,
    )

    if provider == "disabled":
        return ProviderRuntimeStatus(
            config_source=config_source,
            configured_provider_mode="disabled",
            effective_provider_mode=ProviderMode.disabled_or_local_fallback,
            provider_identifier="disabled",
            model_profile=_model_value(provider),
            allowed_model_roles=["planner"],
            configured=True,
            secret_reference_present=False,
            fallback_reason="QUANT_LLM_PROVIDER disables model-backed planning; using deterministic plan fixtures.",
            health_check_enabled=False,
            provider_reachable=False,
            configuration_errors=[],
            timeout_seconds=timeout_seconds,
            max_context_chars=max_context_chars,
            retention_policy_label="not_applicable",
        )

    if provider not in SHARED_LLM_PROVIDER_MODES:
        configuration_errors.append("Unsupported LLM provider for agent planning.")
        return ProviderRuntimeStatus(
            config_source=config_source,
            configured_provider_mode="unsupported",
            effective_provider_mode=ProviderMode.disabled_or_local_fallback,
            provider_identifier="unsupported",
            model_profile=_model_value("disabled"),
            allowed_model_roles=["planner"],
            configured=False,
            secret_reference_present=False,
            fallback_reason="Unsupported provider configuration; using deterministic plan fixtures.",
            health_check_enabled=False,
            provider_reachable=False,
            configuration_errors=configuration_errors,
            timeout_seconds=timeout_seconds,
            max_context_chars=max_context_chars,
            retention_policy_label="not_applicable",
        )

    model = _model_value(provider)
    openai_key = _env_value("QUANT_AGENT_OPENAI_API_KEY") or _env_value("OPENAI_API_KEY")
    openai_base_url = (
        _env_value("QUANT_AGENT_OPENAI_BASE_URL")
        or _env_value("OPENAI_BASE_URL")
        or DEFAULT_OPENAI_BASE_URL
    )
    ollama_base_url = (
        _env_value("QUANT_AGENT_OLLAMA_BASE_URL")
        or _env_value("QUANT_OLLAMA_BASE_URL")
        or DEFAULT_OLLAMA_BASE_URL
    )

    if provider == ProviderMode.openai.value:
        if not openai_key:
            configuration_errors.append("OPENAI_API_KEY is required when agent planning uses OpenAI.")
        if not _is_http_url(openai_base_url):
            configuration_errors.append("OPENAI_BASE_URL must be an HTTP URL when set.")
    if provider == ProviderMode.ollama.value:
        if not _is_http_url(ollama_base_url):
            configuration_errors.append("QUANT_OLLAMA_BASE_URL must be an HTTP URL when set.")

    effective_mode = ProviderMode(provider) if not configuration_errors else ProviderMode.disabled_or_local_fallback
    return ProviderRuntimeStatus(
        config_source=config_source,
        configured_provider_mode=provider,
        effective_provider_mode=effective_mode,
        provider_identifier=provider,
        model_profile=model,
        allowed_model_roles=["planner"],
        configured=not configuration_errors,
        secret_reference_present=provider == ProviderMode.openai.value and bool(openai_key),
        fallback_reason=(
            None
            if not configuration_errors
            else "Provider configuration is incomplete or invalid; using deterministic plan fixtures."
        ),
        health_check_enabled=False,
        provider_reachable=False,
        configuration_errors=configuration_errors,
        timeout_seconds=timeout_seconds,
        max_context_chars=max_context_chars,
        retention_policy_label="not_applicable",
        hosted_provider_enabled=provider == ProviderMode.openai.value and not configuration_errors,
    )


def provider_status_from_contract_payload(
    payload: Mapping[str, Any],
    *,
    config_source: str,
    load_errors: list[str] | None = None,
) -> ProviderRuntimeStatus:
    provider_mode = _string_value(
        payload.get("provider_mode"),
        default=ProviderMode.disabled_or_local_fallback.value,
    )
    configuration_errors = list(load_errors or [])
    health_check = payload.get("health_check")
    if not isinstance(health_check, Mapping):
        health_check = {}

    for item in health_check.get("configuration_errors", []):
        if isinstance(item, str) and item:
            configuration_errors.append(item)

    if provider_mode in SUPPORTED_RUNTIME_PROVIDER_MODES:
        effective_mode = ProviderMode(provider_mode)
        configured = not configuration_errors
    else:
        effective_mode = ProviderMode.disabled_or_local_fallback
        configured = False
        configuration_errors.append(
            f"Provider mode '{provider_mode}' is not supported by this plan-only runtime slice."
        )

    fallback_reason = _fallback_reason(payload, effective_mode)

    return ProviderRuntimeStatus(
        config_source=config_source,
        configured_provider_mode=provider_mode,
        effective_provider_mode=effective_mode,
        provider_identifier=_string_value(payload.get("provider_identifier"), default="disabled"),
        model_profile=_string_value(payload.get("model_profile"), default="deterministic_plan_fixture"),
        allowed_model_roles=_string_list(payload.get("allowed_model_roles")),
        configured=configured,
        secret_reference_present=bool(payload.get("secret_reference_present", False)),
        fallback_reason=fallback_reason,
        health_check_enabled=bool(health_check.get("enabled", False)),
        provider_reachable=bool(health_check.get("provider_reachable", False)),
        configuration_errors=configuration_errors,
        timeout_seconds=_integer_value(payload.get("timeout_seconds"), default=0),
        max_context_chars=_integer_value(payload.get("max_context_chars"), default=18000),
        retention_policy_label=_string_value(
            payload.get("retention_policy_label"),
            default="not_applicable",
        ),
    )


def _fallback_reason(payload: Mapping[str, Any], effective_mode: ProviderMode) -> str | None:
    if effective_mode != ProviderMode.disabled_or_local_fallback:
        return None
    configured_reason = payload.get("disabled_mode_fallback")
    if isinstance(configured_reason, str) and configured_reason.strip():
        return configured_reason.strip()
    return "Use deterministic plan fixtures and do not call a hosted model provider."


def _string_value(value: object, *, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return ["planner"]
    return [item for item in value if isinstance(item, str) and item]


def _integer_value(value: object, *, default: int) -> int:
    if isinstance(value, int):
        return value
    return default


def _first_env(*names: str) -> tuple[str | None, str]:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value, name
    return None, "canonical_provider_config"


def _env_value(name: str) -> str:
    return os.getenv(name, "").strip()


def _model_value(provider: str) -> str:
    raw_model = _env_value("QUANT_AGENT_LLM_MODEL") or _env_value("QUANT_LLM_MODEL")
    if raw_model:
        return raw_model
    if provider == ProviderMode.openai.value:
        return DEFAULT_OPENAI_MODEL
    if provider == ProviderMode.ollama.value:
        return DEFAULT_OLLAMA_MODEL
    return "deterministic-plan-fixture"


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def _is_http_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))
