from __future__ import annotations

from quant_agent_runtime.models import (
    CapabilityDefinition,
    PlanStep,
    PolicySettings,
    ProviderMode,
    RiskTier,
    StepOperation,
    ValidationIssue,
)


SUPPORTED_PROVIDER_MODES = {
    ProviderMode.fake_provider,
    ProviderMode.disabled_or_local_fallback,
}


class PolicyEngine:
    def validate_settings(self, policy: PolicySettings) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if policy.provider_mode not in SUPPORTED_PROVIDER_MODES:
            issues.append(
                ValidationIssue(
                    code="provider_mode_not_supported",
                    message="Only fake-provider and disabled/local fallback modes are supported.",
                )
            )
        if not policy.plan_only:
            issues.append(
                ValidationIssue(
                    code="plan_only_required",
                    message="This runtime slice requires plan-only mode.",
                )
            )
        return issues

    def validate_step(
        self,
        step: PlanStep,
        capability: CapabilityDefinition,
        policy: PolicySettings,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if step.operation != StepOperation.plan:
            issues.append(
                ValidationIssue(
                    code="execution_not_allowed",
                    message="Plans must not request execution in this runtime slice.",
                    step_id=step.step_id,
                    capability_id=step.capability_id,
                )
            )
        if step.capability_id in policy.forbidden_action_ids:
            issues.append(
                ValidationIssue(
                    code="forbidden_action",
                    message="The capability is forbidden by policy.",
                    step_id=step.step_id,
                    capability_id=step.capability_id,
                )
            )
        if capability.risk_tier == RiskTier.forbidden:
            issues.append(
                ValidationIssue(
                    code="forbidden_risk_tier",
                    message="Capabilities with forbidden risk tier cannot be planned.",
                    step_id=step.step_id,
                    capability_id=step.capability_id,
                )
            )
        if capability.risk_tier not in policy.allowed_risk_tiers:
            issues.append(
                ValidationIssue(
                    code="risk_tier_not_allowed",
                    message="The capability risk tier is not allowed by policy.",
                    step_id=step.step_id,
                    capability_id=step.capability_id,
                )
            )
        requires_confirmation = (
            capability.confirmation_required
            or capability.risk_tier in policy.confirmation_required_tiers
        )
        if requires_confirmation and not step.requires_confirmation:
            issues.append(
                ValidationIssue(
                    code="missing_confirmation_requirement",
                    message="The step must declare that confirmation is required.",
                    step_id=step.step_id,
                    capability_id=step.capability_id,
                )
            )

        return issues
