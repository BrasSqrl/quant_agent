from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from quant_agent_runtime.llm_profiles import resolve_llm_profile
from quant_agent_runtime.models import ProviderMode, ProviderRuntimeStatus


SUPPORTED_RUNTIME_PROVIDER_MODES = {
    ProviderMode.fake_provider.value,
    ProviderMode.disabled_or_local_fallback.value,
    ProviderMode.openai.value,
    ProviderMode.ollama.value,
}
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"


def internal_provider_status(config_source: str = "internal_default") -> ProviderRuntimeStatus:
    return ProviderRuntimeStatus(
        config_source=config_source,
        configured_provider_mode=ProviderMode.fake_provider.value,
        effective_provider_mode=ProviderMode.fake_provider,
        provider_identifier="fake",
        model_profile="deterministic-plan-fixture",
        model="deterministic-plan-fixture",
        llm_mode="fake_provider",
        role="agent_planning",
        context_window_tokens=0,
        configuration_source=config_source,
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
    """Return the effective planner provider status from safe env diagnostics."""

    base_status = base_status or internal_provider_status()
    resolved = resolve_llm_profile("agent_planning")
    if (
        resolved.configuration_source == "legacy_env"
        and resolved.provider == "disabled"
        and not any(
            _env_value(name)
            for name in (
                "QUANT_LLM_MODEL_PROFILE",
                "QUANT_AGENT_LLM_PROVIDER",
                "QUANT_AGENT_LLM_MODEL",
                "QUANT_LLM_PROVIDER",
                "QUANT_LLM_MODEL",
            )
        )
    ):
        return base_status

    provider = resolved.provider
    effective_mode = (
        ProviderMode(provider)
        if provider in {ProviderMode.openai.value, ProviderMode.ollama.value} and not resolved.errors
        else ProviderMode.disabled_or_local_fallback
    )
    fallback_reason = None
    if provider == "disabled":
        fallback_reason = "Selected LLM profile disables model-backed planning; using deterministic plan fixtures."
    elif resolved.errors:
        fallback_reason = "Provider configuration is incomplete or invalid; using deterministic plan fixtures."
    return ProviderRuntimeStatus(
        config_source=resolved.configuration_source,
        configured_provider_mode=provider,
        effective_provider_mode=effective_mode,
        provider_identifier=provider,
        model_profile=resolved.model_profile,
        model=resolved.model,
        llm_mode=resolved.llm_mode,
        role=resolved.role,
        context_window_tokens=resolved.context_window_tokens,
        configuration_source=resolved.configuration_source,
        allowed_model_roles=["agent_planning"],
        configured=resolved.enabled,
        secret_reference_present=resolved.secret_reference_present,
        fallback_reason=fallback_reason,
        health_check_enabled=False,
        provider_reachable=False,
        configuration_errors=list(resolved.errors),
        configuration_warnings=list(resolved.warnings),
        timeout_seconds=resolved.timeout_seconds,
        max_context_chars=resolved.max_context_chars,
        max_output_tokens=resolved.max_output_tokens,
        openai_settings=dict(resolved.openai_settings),
        ollama_settings=dict(resolved.ollama_settings),
        retention_policy_label="not_applicable",
        hosted_provider_enabled=provider == ProviderMode.openai.value and not resolved.errors,
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
        model=_string_value(payload.get("model"), default=_string_value(payload.get("model_profile"), default="deterministic_plan_fixture")),
        llm_mode=_string_value(payload.get("llm_mode"), default="disabled"),
        role=_string_value(payload.get("role"), default="agent_planning"),
        context_window_tokens=_integer_value(payload.get("context_window_tokens"), default=0),
        configuration_source=_string_value(payload.get("configuration_source"), default=config_source),
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


def _env_value(name: str) -> str:
    return os.getenv(name, "").strip()
