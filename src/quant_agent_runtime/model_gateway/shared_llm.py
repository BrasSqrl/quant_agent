from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

from quant_agent_runtime.model_gateway.fake import FakePlanProvider
from quant_agent_runtime.model_gateway.provider import (
    ModelProvider,
    ProviderPlanRequest,
    ProviderResult,
)
from quant_agent_runtime.models import (
    CapabilityDefinition,
    ProviderMetadata,
    ProviderMode,
    ProviderRuntimeStatus,
)
from quant_agent_runtime.provider_config import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OPENAI_BASE_URL,
)

_PLAN_OUTPUT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "user_goal_summary": {"type": "string"},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "missing_inputs": {"type": "array", "items": {"type": "string"}},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "step_id": {"type": "string"},
                    "title": {"type": "string"},
                    "capability_id": {"type": "string"},
                    "app_id": {"type": "string"},
                    "risk_tier": {"type": "string"},
                    "operation": {"const": "plan"},
                    "preflight_required": {"type": "boolean"},
                    "requires_confirmation": {"type": "boolean"},
                    "action_input": {"type": "object"},
                    "expected_artifacts": {"type": "array", "items": {"type": "string"}},
                    "validation_checks": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "step_id",
                    "title",
                    "capability_id",
                    "app_id",
                    "risk_tier",
                    "operation",
                    "preflight_required",
                    "requires_confirmation",
                    "action_input",
                    "expected_artifacts",
                    "validation_checks",
                ],
            },
        },
    },
    "required": ["user_goal_summary", "assumptions", "missing_inputs", "steps"],
}


class SharedLlmPlanProvider(ModelProvider):
    """Server-side OpenAI/Ollama planner with deterministic fake fallback."""

    def __init__(self, provider_status: ProviderRuntimeStatus) -> None:
        self._provider_status = provider_status

    def generate_plan(self, request: ProviderPlanRequest) -> ProviderResult:
        if request.policy.provider_mode == ProviderMode.disabled_or_local_fallback:
            return self._fallback(
                request,
                "Provider disabled by request policy; using deterministic plan fixtures.",
            )
        if self._provider_status.effective_provider_mode not in {
            ProviderMode.openai,
            ProviderMode.ollama,
        }:
            return self._fallback(
                request,
                self._provider_status.fallback_reason
                or "Model-backed planning is not configured; using deterministic plan fixtures.",
            )
        if self._provider_status.configuration_errors:
            return self._fallback(
                request,
                "Provider configuration errors prevent model-backed planning.",
            )

        prompt_packet = _prompt_packet(
            request,
            max_context_chars=self._provider_status.max_context_chars,
        )
        try:
            if self._provider_status.effective_provider_mode == ProviderMode.openai:
                text = _call_openai(self._provider_status, prompt_packet)
            else:
                text = _call_ollama(self._provider_status, prompt_packet)
        except _ProviderUnavailable as exc:
            return self._fallback(request, str(exc))

        return ProviderResult(
            raw_output=_parse_json_object(text),
            metadata=_metadata(self._provider_status),
        )

    def _fallback(self, request: ProviderPlanRequest, reason: str) -> ProviderResult:
        fallback_status = self._provider_status.model_copy(
            update={
                "effective_provider_mode": ProviderMode.disabled_or_local_fallback,
                "provider_identifier": "disabled",
                "configured": False,
                "hosted_provider_enabled": False,
                "provider_reachable": False,
                "fallback_reason": reason,
            }
        )
        return FakePlanProvider(provider_status=fallback_status).generate_plan(request)


class _ProviderUnavailable(Exception):
    pass


def _metadata(status: ProviderRuntimeStatus) -> ProviderMetadata:
    return ProviderMetadata(
        provider=status.provider_identifier,
        model=status.model_profile,
        provider_mode=status.effective_provider_mode,
        config_source=status.config_source,
        configured_provider_mode=status.configured_provider_mode,
        fallback_reason=status.fallback_reason,
        configuration_errors=status.configuration_errors,
        request_purpose="plan_generation",
        supports_execution=False,
    )


def _prompt_packet(
    request: ProviderPlanRequest,
    *,
    max_context_chars: int,
) -> dict[str, str]:
    context = {
        "user_goal": request.user_goal,
        "context_summary": request.context_summary,
        "capabilities": [_capability_payload(item) for item in request.capabilities],
        "policy": {
            "plan_only": request.policy.plan_only,
            "allowed_risk_tiers": [item.value for item in request.policy.allowed_risk_tiers],
            "confirmation_required_tiers": [
                item.value for item in request.policy.confirmation_required_tiers
            ],
            "forbidden_action_ids": request.policy.forbidden_action_ids,
        },
    }
    context_json = json.dumps(context, sort_keys=True, separators=(",", ":"))
    if len(context_json) > max_context_chars:
        context_json = context_json[:max_context_chars]
    system = (
        "You are the Quant Suite governed agent planner. Return exactly one JSON object "
        "that satisfies the requested schema. Use only sanitized summaries and capability "
        "metadata. Do not include raw rows, raw paths, URLs, bucket names, secrets, "
        "credentials, hidden commands, raw prompts, provider responses, execution commands, "
        "or app mutations."
    )
    user = (
        "Create a plan JSON object with exactly these top-level keys: "
        "{\"user_goal_summary\": string, \"assumptions\": string[], "
        "\"missing_inputs\": string[], \"steps\": step[]}. Each step must have "
        "step_id, title, capability_id, app_id, risk_tier, operation='plan', "
        "preflight_required, requires_confirmation, action_input, expected_artifacts, "
        "and validation_checks. Use only capability_ids in the supplied capabilities. "
        "Set action_input only from available summaries; if required context is missing, "
        "put a concise item in missing_inputs and use \"[missing]\" for that field. "
        f"Sanitized planning context JSON:\n{context_json}"
    )
    return {"system": system, "user": user}


def _capability_payload(capability: CapabilityDefinition) -> dict[str, Any]:
    return {
        "capability_id": capability.capability_id,
        "app_id": capability.app_id,
        "display_name": capability.display_name,
        "risk_tier": capability.risk_tier.value,
        "enabled": capability.enabled,
        "required_fields": capability.required_fields,
        "preflight_required": capability.preflight_required,
        "confirmation_required": capability.confirmation_required,
    }


def _call_openai(status: ProviderRuntimeStatus, prompt_packet: Mapping[str, str]) -> str:
    api_key = _env_value("QUANT_AGENT_OPENAI_API_KEY") or _env_value("OPENAI_API_KEY")
    if not api_key:
        raise _ProviderUnavailable("OpenAI planning provider is missing a server-side API key.")
    base_url = (
        _env_value("QUANT_AGENT_OPENAI_BASE_URL")
        or _env_value("OPENAI_BASE_URL")
        or DEFAULT_OPENAI_BASE_URL
    )
    url = f"{base_url.rstrip('/')}/responses"
    payload = {
        "model": status.model_profile,
        "input": [
            {"role": "system", "content": prompt_packet["system"]},
            {"role": "user", "content": prompt_packet["user"]},
        ],
        "max_output_tokens": 1800,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "quant_agent_provider_plan",
                "schema": _PLAN_OUTPUT_JSON_SCHEMA,
                "strict": False,
            },
            "verbosity": "low",
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=status.timeout_seconds or 45) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise _ProviderUnavailable(f"OpenAI planning provider returned HTTP {exc.code}.") from exc
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise _ProviderUnavailable(
            f"OpenAI planning provider unavailable: {type(exc).__name__}."
        ) from exc
    text = _openai_response_text(body)
    if not text:
        raise _ProviderUnavailable("OpenAI planning provider returned an empty response.")
    return text


def _call_ollama(status: ProviderRuntimeStatus, prompt_packet: Mapping[str, str]) -> str:
    base_url = (
        _env_value("QUANT_AGENT_OLLAMA_BASE_URL")
        or _env_value("QUANT_OLLAMA_BASE_URL")
        or DEFAULT_OLLAMA_BASE_URL
    )
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": status.model_profile,
        "prompt": f"{prompt_packet['system']}\n\n{prompt_packet['user']}",
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=status.timeout_seconds or 45) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise _ProviderUnavailable(f"Ollama planning provider returned HTTP {exc.code}.") from exc
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise _ProviderUnavailable(
            f"Ollama planning provider unavailable: {type(exc).__name__}."
        ) from exc
    if isinstance(body, Mapping) and isinstance(body.get("response"), str):
        text = str(body["response"]).strip()
        if text:
            return text
    message = body.get("message") if isinstance(body, Mapping) else None
    if isinstance(message, Mapping) and isinstance(message.get("content"), str):
        text = str(message["content"]).strip()
        if text:
            return text
    raise _ProviderUnavailable("Ollama planning provider returned an empty response.")


def _openai_response_text(body: object) -> str:
    if isinstance(body, Mapping) and isinstance(body.get("output_text"), str):
        text = str(body["output_text"]).strip()
        if text:
            return text
    if isinstance(body, Mapping):
        parts: list[str] = []
        for item in body.get("output", []):
            if not isinstance(item, Mapping):
                continue
            content_items = item.get("content", [])
            if isinstance(content_items, str):
                content_items = [{"text": content_items}]
            for content in content_items:
                if isinstance(content, Mapping) and isinstance(content.get("text"), str):
                    text = str(content["text"]).strip()
                    if text:
                        parts.append(text)
                elif isinstance(content, Mapping) and isinstance(content.get("json"), Mapping):
                    parts.append(json.dumps(content["json"]))
        if parts:
            return "\n".join(parts)
    return ""


def _parse_json_object(text: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return payload if isinstance(payload, Mapping) else {}


def _env_value(name: str) -> str:
    return os.getenv(name, "").strip()
