from __future__ import annotations

from quant_agent_runtime.capabilities import CapabilityRegistry
from quant_agent_runtime.models import PlanValidationResult, PolicySettings, ProviderPlanOutput, ValidationIssue
from quant_agent_runtime.policy import PolicyEngine
from quant_agent_runtime.redaction import find_unsafe_payload_issues


class PlanValidator:
    def __init__(self, policy_engine: PolicyEngine | None = None) -> None:
        self._policy_engine = policy_engine or PolicyEngine()

    def validate(
        self,
        provider_plan: ProviderPlanOutput,
        registry: CapabilityRegistry,
        policy: PolicySettings,
    ) -> PlanValidationResult:
        errors: list[ValidationIssue] = []
        errors.extend(self._policy_engine.validate_settings(policy))
        for issue in find_unsafe_payload_issues(
            provider_plan.model_dump(mode="json"),
            root="provider_plan",
        ):
            errors.append(issue.model_copy(update={"code": "unsafe_provider_plan_payload"}))

        for step in provider_plan.steps:
            capability = registry.get(step.capability_id)
            if capability is None:
                errors.append(
                    ValidationIssue(
                        code="unknown_capability",
                        message="The plan references an unknown capability.",
                        step_id=step.step_id,
                        capability_id=step.capability_id,
                    )
                )
                continue

            if not capability.enabled:
                errors.append(
                    ValidationIssue(
                        code="capability_disabled",
                        message="The plan references a disabled capability.",
                        step_id=step.step_id,
                        capability_id=step.capability_id,
                    )
                )
            if step.app_id != capability.app_id:
                errors.append(
                    ValidationIssue(
                        code="app_id_mismatch",
                        message="The step app_id does not match the capability owner.",
                        step_id=step.step_id,
                        capability_id=step.capability_id,
                    )
                )
            if step.risk_tier != capability.risk_tier:
                errors.append(
                    ValidationIssue(
                        code="risk_tier_mismatch",
                        message="The step risk tier does not match the capability registry.",
                        step_id=step.step_id,
                        capability_id=step.capability_id,
                    )
                )
            if step.preflight_required != capability.preflight_required:
                errors.append(
                    ValidationIssue(
                        code="preflight_requirement_mismatch",
                        message="The step preflight gate does not match the capability registry.",
                        step_id=step.step_id,
                        capability_id=step.capability_id,
                    )
                )

            missing_fields = [
                field for field in capability.required_fields if field not in step.action_input
            ]
            for field in missing_fields:
                errors.append(
                    ValidationIssue(
                        code="missing_required_action_field",
                        message=f"The action input is missing required field '{field}'.",
                        step_id=step.step_id,
                        capability_id=step.capability_id,
                    )
                )

            errors.extend(
                self._policy_engine.validate_step(step=step, capability=capability, policy=policy)
            )
        status = "rejected" if errors else "valid"
        return PlanValidationResult(status=status, errors=errors)
