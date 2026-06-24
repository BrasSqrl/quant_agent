from __future__ import annotations

import copy

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

_BASELINE_PLAN_CAPABILITY_ORDER = [
    "quant_data.run_source_preflight",
    "quant_studio.prepare_model_config_draft",
    "quant_documentation.inspect_package",
    "quant_documentation.create_draft_workspace",
    "quant_monitoring.validate_bundle",
]


class FakePlanProvider(ModelProvider):
    def __init__(self, provider_status: ProviderRuntimeStatus | None = None) -> None:
        self._provider_status = provider_status or internal_provider_status()

    def generate_plan(self, request: ProviderPlanRequest) -> ProviderResult:
        steps: list[dict[str, object]] = []
        missing_inputs: list[str] = []
        capabilities = self._capabilities_for_plan(request)

        for index, capability in enumerate(capabilities, start=1):
            if not capability.enabled or capability.risk_tier == RiskTier.forbidden:
                continue
            action_input: dict[str, object] = {}
            for field in capability.required_fields:
                field_summary = self._safe_field_summary(field, request.context_summary)
                if field_summary is None:
                    if self._is_deferred_full_lifecycle_field(field, request.context_summary):
                        action_input[field] = "Pending upstream workflow handoff."
                    else:
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
                "Only confirmed app-owned review-draft actions can execute in this runtime slice.",
            ],
            "missing_inputs": missing_inputs,
            "steps": steps,
        }
        metadata = self._provider_metadata_for(request)
        return ProviderResult(raw_output=raw_output, metadata=metadata)

    def _capabilities_for_plan(self, request: ProviderPlanRequest) -> list:
        capabilities = request.capabilities
        if isinstance(request.context_summary.get("workflow_scope"), dict):
            return capabilities
        by_id = {capability.capability_id: capability for capability in capabilities}
        if (
            len(capabilities) > len(_BASELINE_PLAN_CAPABILITY_ORDER)
            and all(capability_id in by_id for capability_id in _BASELINE_PLAN_CAPABILITY_ORDER)
        ):
            return [by_id[capability_id] for capability_id in _BASELINE_PLAN_CAPABILITY_ORDER]
        return capabilities

    def _summarize_goal(self, user_goal: str) -> str:
        return " ".join(user_goal.strip().split())[:240]

    def _safe_field_summary(self, field: str, context_summary: dict[str, object]) -> object | None:
        value = context_summary.get(field)
        if isinstance(value, str) and value.strip():
            return value[:240]
        if isinstance(value, dict) and value:
            return copy.deepcopy(value)
        if isinstance(value, list) and value:
            return copy.deepcopy(value[:20])
        return None

    def _is_deferred_full_lifecycle_field(self, field: str, context_summary: dict[str, object]) -> bool:
        scope = context_summary.get("workflow_scope")
        return (
            isinstance(scope, dict)
            and scope.get("workflow_scope") == "full_lifecycle"
            and field in {"target_summary", "package_summary", "bundle_summary"}
        )

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
