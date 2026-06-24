"""Suite LLM profile resolution for Quant Agent planning."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MAX_CONTEXT_CHARS = 18_000
DEFAULT_TIMEOUT_SECONDS = 45


@dataclass(frozen=True)
class ResolvedLlmProfile:
    provider: str
    model: str
    enabled: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    model_profile: str
    display_name: str
    llm_mode: str
    role: str
    context_window_tokens: int
    max_context_chars: int
    timeout_seconds: int
    configuration_source: str
    secret_reference_present: bool
    openai_settings: dict[str, Any] = field(default_factory=dict)
    ollama_settings: dict[str, Any] = field(default_factory=dict)
    max_output_tokens: int = 0


def resolve_llm_profile(role: str) -> ResolvedLlmProfile:
    catalog = _load_catalog()
    explicit_profile_id = os.getenv("QUANT_LLM_MODEL_PROFILE", "").strip()
    legacy_provider = os.getenv("QUANT_AGENT_LLM_PROVIDER", "").strip().lower() or os.getenv("QUANT_LLM_PROVIDER", "").strip().lower()
    legacy_model = os.getenv("QUANT_AGENT_LLM_MODEL", "").strip() or os.getenv("QUANT_LLM_MODEL", "").strip()
    agent_override_warnings = _agent_override_warnings()
    if explicit_profile_id and catalog:
        resolved = _from_catalog(catalog, explicit_profile_id, role)
        return _with_warnings(resolved, agent_override_warnings)
    if catalog and not (legacy_provider or legacy_model):
        return _from_catalog(catalog, str(catalog.get("default_profile_id") or ""), role)
    if legacy_provider or legacy_model:
        return _legacy_profile(role, legacy_provider, legacy_model)
    if catalog:
        return _from_catalog(catalog, str(catalog.get("default_profile_id") or ""), role)
    return _legacy_profile(role, "disabled", "")


def _from_catalog(catalog: dict[str, Any], profile_id: str, role: str) -> ResolvedLlmProfile:
    profiles = catalog.get("profiles") if isinstance(catalog.get("profiles"), list) else []
    profile = next((item for item in profiles if isinstance(item, dict) and item.get("profile_id") == profile_id), None)
    if not profile:
        return _disabled_with_error(role, f"QUANT_LLM_MODEL_PROFILE '{profile_id}' was not found.")
    roles = profile.get("roles") if isinstance(profile.get("roles"), dict) else {}
    role_settings = roles.get(role) if isinstance(roles.get(role), dict) else {}
    provider = str(profile.get("provider") or "disabled").strip().lower()
    model = str(profile.get("model") or "").strip()
    errors: list[str] = []
    if provider not in {"disabled", "ollama", "openai"}:
        errors.append(f"Unsupported QUANT_LLM_MODEL_PROFILE provider: {provider}")
        provider = "disabled"
    enabled = bool(profile.get("enabled")) and provider in {"ollama", "openai"}
    api_key_env = str(profile.get("api_key_env") or "").strip()
    secret_reference_present = bool(api_key_env and os.getenv(api_key_env, "").strip())
    if enabled and profile.get("requires_secret") is True and not secret_reference_present:
        errors.append(f"{api_key_env} is required by QUANT_LLM_MODEL_PROFILE={profile_id}.")
    if enabled and not model:
        errors.append(f"Model is required by QUANT_LLM_MODEL_PROFILE={profile_id}.")
    return ResolvedLlmProfile(
        provider=provider,
        model=model if provider != "disabled" else "",
        enabled=enabled and not errors,
        errors=tuple(errors),
        warnings=(),
        model_profile=str(profile.get("profile_id") or profile_id),
        display_name=str(profile.get("display_name") or profile_id),
        llm_mode=str(profile.get("llm_mode") or provider),
        role=role,
        context_window_tokens=_positive_int(profile.get("context_window_tokens"), 0),
        max_context_chars=_positive_int(role_settings.get("max_context_chars"), _positive_int(profile.get("max_context_chars"), DEFAULT_MAX_CONTEXT_CHARS)),
        timeout_seconds=_positive_int(profile.get("timeout_seconds"), DEFAULT_TIMEOUT_SECONDS),
        configuration_source="llm_model_profiles.v1.json",
        secret_reference_present=secret_reference_present,
        openai_settings=dict(role_settings.get("openai") or {}),
        ollama_settings=dict(role_settings.get("ollama") or {}),
        max_output_tokens=_positive_int(role_settings.get("max_output_tokens"), 0),
    )


def _legacy_profile(role: str, legacy_provider: str, legacy_model: str) -> ResolvedLlmProfile:
    provider = legacy_provider or "disabled"
    errors: list[str] = []
    config_source = (
        "QUANT_AGENT_LLM_PROVIDER"
        if os.getenv("QUANT_AGENT_LLM_PROVIDER", "").strip()
        else "QUANT_LLM_PROVIDER"
        if os.getenv("QUANT_LLM_PROVIDER", "").strip()
        else "legacy_env"
    )
    if provider not in {"disabled", "ollama", "openai"}:
        errors.append("Unsupported LLM provider for agent planning.")
        provider = "unsupported"
    model = legacy_model or ("gpt-5.4-mini" if provider == "openai" else "gemma4:e2b" if provider == "ollama" else "deterministic-plan-fixture")
    secret_reference_present = bool(os.getenv("QUANT_AGENT_OPENAI_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip())
    if provider == "openai" and not secret_reference_present:
        errors.append("OPENAI_API_KEY is required when agent planning uses OpenAI.")
    timeout_seconds = _positive_int(os.getenv("QUANT_LLM_TIMEOUT_SECONDS"), DEFAULT_TIMEOUT_SECONDS)
    max_context_chars = _positive_int(os.getenv("QUANT_LLM_MAX_CONTEXT_CHARS"), DEFAULT_MAX_CONTEXT_CHARS)
    return ResolvedLlmProfile(
        provider=provider,
        model=model if provider != "disabled" else "",
        enabled=provider in {"ollama", "bedrock", "openai"} and not errors,
        errors=tuple(errors),
        warnings=("Using legacy agent/shared LLM provider configuration.",),
        model_profile=model if provider in {"ollama", "openai"} else os.getenv("QUANT_LLM_PROFILE", "legacy_env").strip() or "legacy_env",
        display_name="Legacy environment configuration",
        llm_mode="legacy_env",
        role=role,
        context_window_tokens=0,
        max_context_chars=max_context_chars,
        timeout_seconds=timeout_seconds,
        configuration_source=config_source,
        secret_reference_present=provider == "openai" and secret_reference_present,
        openai_settings={"max_output_tokens": 1800, "text": {"verbosity": "low"}, "json_schema": True},
        ollama_settings={"temperature": 0.0, "format": "json"},
        max_output_tokens=1800,
    )


def _load_catalog() -> dict[str, Any] | None:
    for path in _catalog_candidates():
        if not path.is_file():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _catalog_candidates() -> list[Path]:
    candidates: list[Path] = []
    explicit = os.getenv("QUANT_LLM_PROFILE_CATALOG", "").strip()
    if explicit:
        candidates.append(Path(explicit))
    suite_root = os.getenv("QUANT_SUITE_ROOT", "").strip()
    if suite_root:
        candidates.append(Path(suite_root) / "llm_model_profiles.v1.json")
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidates.append(parent / "llm_model_profiles.v1.json")
        candidates.append(parent / "quant_suite" / "llm_model_profiles.v1.json")
    return candidates


def _disabled_with_error(role: str, message: str) -> ResolvedLlmProfile:
    return ResolvedLlmProfile(
        provider="disabled",
        model="",
        enabled=False,
        errors=(message,),
        warnings=(),
        model_profile=os.getenv("QUANT_LLM_MODEL_PROFILE", "disabled_deterministic") or "disabled_deterministic",
        display_name="Disabled deterministic fallback",
        llm_mode="disabled",
        role=role,
        context_window_tokens=0,
        max_context_chars=DEFAULT_MAX_CONTEXT_CHARS,
        timeout_seconds=0,
        configuration_source="llm_model_profiles.v1.json",
        secret_reference_present=False,
    )


def _with_warnings(profile: ResolvedLlmProfile, warnings: tuple[str, ...]) -> ResolvedLlmProfile:
    if not warnings:
        return profile
    return ResolvedLlmProfile(
        **{**profile.__dict__, "warnings": (*profile.warnings, *warnings)}
    )


def _agent_override_warnings() -> tuple[str, ...]:
    ignored = [
        name
        for name in ("QUANT_AGENT_LLM_PROVIDER", "QUANT_AGENT_LLM_MODEL")
        if os.getenv(name, "").strip()
    ]
    if not ignored:
        return ()
    return (f"Ignored agent-specific model selector while QUANT_LLM_MODEL_PROFILE is set: {', '.join(ignored)}.",)


def _positive_int(value: object, default: int) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
