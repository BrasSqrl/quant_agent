from __future__ import annotations

from quant_agent_runtime.model_gateway.provider import (
    ModelProvider,
    ProviderPlanRequest,
    ProviderResult,
)
from quant_agent_runtime.models import (
    ProviderMetadata,
    ProviderMode,
    ProviderRuntimeStatus,
    RiskTier,
)
from quant_agent_runtime.provider_config import internal_provider_status


class FakePlanProvider(ModelProvider):
    def __init__(self, provider_status: ProviderRuntimeStatus | None = None) -> None:
        self._provider_status = provider_status or internal_provider_status()

    def generate_plan(self, request: ProviderPlanRequest) -> ProviderResult:
        steps: list[dict[str, object]] = []
        missing_inputs: list[str] = []

        for index, capability in enumerate(request.capabilities, start=1):
            if not capability.enabled or capability.risk_tier == RiskTier.forbidden:
                continue
            action_input: dict[str, str] = {}
            for field in capability.required_fields:
                field_summary = self._safe_field_summary(field, request.context_summary)
                if field_summary is None:
                    missing_inputs.append(
                        f"{capability.capability_id} requires {field}."
                    )
                    action_input[field] = "[missing]"
                else:
                    action_input[field] = field_summary
            requires_confirmation = (
                capability.confirmation_required
                or capability.risk_tier in request.policy.confirmation_required_tiers
            )
            steps.append(
                {
                    "step_id": f"step_{index}",
                    "title": capability.display_name,
                    "capability_id": capability.capability_id,
                    "app_id": capability.app_id,
                    "risk_tier": capability.risk_tier.value,
                    "operation": "plan",
                    "preflight_required": capability.preflight_required,
                    "requires_confirmation": requires_confirmation,
                    "action_input": action_input,
                    "expected_artifacts": [],
                    "validation_checks": [
                        "capability_known",
                        "policy_allowed",
                        "plan_only",
                    ],
                }
            )

        if not steps:
            missing_inputs.append("No enabled capabilities are available.")
        raw_output = {
            "user_goal_summary": self._summarize_goal(request.user_goal),
            "assumptions": [
                "Planning uses sanitized summaries and allowed capability metadata only.",
                "No app workflow execution is permitted in this runtime slice.",
            ],
            "missing_inputs": missing_inputs,
            "steps": steps,
        }
        metadata = self._provider_metadata_for(request)
        return ProviderResult(raw_output=raw_output, metadata=metadata)

    def _summarize_goal(self, user_goal: str) -> str:
        return " ".join(user_goal.strip().split())[:240]

    def _safe_field_summary(self, field: str, context_summary: dict[str, object]) -> str | None:
        value = context_summary.get(field)
        if isinstance(value, str) and value.strip():
            return value[:240]
        return None

    def _provider_metadata_for(self, request: ProviderPlanRequest) -> ProviderMetadata:
        effective_mode = self._provider_status.effective_provider_mode
        fallback_reason = self._provider_status.fallback_reason
        if request.policy.provider_mode == ProviderMode.disabled_or_local_fallback:
            effective_mode = ProviderMode.disabled_or_local_fallback
            fallback_reason = (
                fallback_reason
                or "Provider disabled by request policy; using deterministic plan fixtures."
            )

        provider = self._provider_status.provider_identifier
        if effective_mode == ProviderMode.disabled_or_local_fallback:
            provider = "disabled"

        return ProviderMetadata(
            provider=provider,
            model=self._provider_status.model_profile,
            provider_mode=effective_mode,
            config_source=self._provider_status.config_source,
            configured_provider_mode=self._provider_status.configured_provider_mode,
            fallback_reason=fallback_reason,
            configuration_errors=self._provider_status.configuration_errors,
            supports_execution=False,
        )
