from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from quant_agent_runtime.models import ProviderMode, ProviderRuntimeStatus


SUPPORTED_RUNTIME_PROVIDER_MODES = {
    ProviderMode.fake_provider.value,
    ProviderMode.disabled_or_local_fallback.value,
}


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
