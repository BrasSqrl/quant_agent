from __future__ import annotations

from quant_agent_runtime.model_gateway.provider import (
    ModelProvider,
    ProviderPlanRequest,
    ProviderResult,
)
from quant_agent_runtime.models import ProviderMetadata, RiskTier


class FakePlanProvider(ModelProvider):
    def generate_plan(self, request: ProviderPlanRequest) -> ProviderResult:
        steps: list[dict[str, object]] = []

        for index, capability in enumerate(request.capabilities, start=1):
            if not capability.enabled or capability.risk_tier == RiskTier.forbidden:
                continue
            action_input = {
                field: self._safe_field_summary(field, request.context_summary)
                for field in capability.required_fields
            }
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

        missing_inputs = [] if steps else ["No enabled capabilities are available."]
        raw_output = {
            "user_goal_summary": self._summarize_goal(request.user_goal),
            "assumptions": [
                "Planning uses sanitized summaries and allowed capability metadata only.",
                "No app workflow execution is permitted in this runtime slice.",
            ],
            "missing_inputs": missing_inputs,
            "steps": steps,
        }
        metadata = ProviderMetadata(
            provider="fake",
            model="deterministic-plan-fixture",
            provider_mode=request.policy.provider_mode,
            supports_execution=False,
        )
        return ProviderResult(raw_output=raw_output, metadata=metadata)

    def _summarize_goal(self, user_goal: str) -> str:
        return " ".join(user_goal.strip().split())[:240]

    def _safe_field_summary(self, field: str, context_summary: dict[str, object]) -> str:
        value = context_summary.get(field)
        if isinstance(value, str) and value.strip():
            return value[:240]
        return f"safe_{field}"
