from __future__ import annotations

from uuid import uuid4

from pydantic import ValidationError

from quant_agent_runtime.capabilities import CapabilityRegistry
from quant_agent_runtime.context_preview import build_context_preview
from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.model_gateway import ModelProvider, ProviderPlanRequest
from quant_agent_runtime.models import (
    CapabilityDefinition,
    ConfirmationRequirement,
    ContextPreview,
    LedgerEntry,
    PlanRequest,
    PlanResult,
    PlanValidationResult,
    ProviderMetadata,
    ProviderMode,
    ProviderPlanOutput,
    RedactionSummary,
    StructuredPlan,
    ValidationIssue,
)
from quant_agent_runtime.redaction import (
    merge_redaction_summaries,
    redact_text,
    sanitize_value,
)
from quant_agent_runtime.run_state import run_state_for_entry
from quant_agent_runtime.validation import PlanValidator
from quant_agent_runtime.validation.errors import MalformedProviderOutputError, RuntimeValidationError


class PlannerService:
    def __init__(
        self,
        provider: ModelProvider,
        ledger: InMemoryLedger | None = None,
        validator: PlanValidator | None = None,
        default_capabilities: list[CapabilityDefinition] | None = None,
    ) -> None:
        self._provider = provider
        self._ledger = ledger or InMemoryLedger()
        self._validator = validator or PlanValidator()
        self._default_capabilities = default_capabilities

    @property
    def ledger(self) -> InMemoryLedger:
        return self._ledger

    def create_plan(self, request: PlanRequest) -> PlanResult:
        run_id = f"run_{uuid4().hex[:12]}"
        plan_id = f"plan_{uuid4().hex[:12]}"

        safe_goal, goal_redacted = redact_text(request.user_goal)
        goal_redaction = RedactionSummary(
            redacted=goal_redacted,
            redacted_fields=["user_goal"] if goal_redacted else [],
        )
        safe_context, context_redaction = sanitize_value(
            request.context_summary,
            path="context_summary",
        )
        if not isinstance(safe_context, dict):
            safe_context = {}
        context_preview = build_context_preview(
            safe_context,
            redaction_summary=context_redaction,
        )
        redaction_summary = merge_redaction_summaries(goal_redaction, context_redaction)

        registry = CapabilityRegistry.from_request(
            request.capabilities,
            default_registry=self._default_capabilities,
        )
        provider_request = ProviderPlanRequest(
            user_goal=safe_goal,
            context_summary=safe_context,
            capabilities=registry.enabled(),
            policy=request.policy,
        )
        capability_snapshot = [
            capability.model_dump(mode="json") for capability in registry.enabled()
        ]
        provider_result = self._provider.generate_plan(provider_request)
        provider_metadata = provider_result.metadata

        try:
            provider_plan = ProviderPlanOutput.model_validate(provider_result.raw_output)
        except ValidationError as exc:
            message = self._summarize_validation_error(exc)
            error = MalformedProviderOutputError.from_message(message)
            self._record_rejection(
                run_id=run_id,
                safe_goal=safe_goal,
                provider_metadata=provider_metadata,
                redaction_summary=redaction_summary,
                context_preview=context_preview,
                validation=error.validation,
            )
            raise error from exc

        validation = self._validator.validate(provider_plan, registry, request.policy)
        if validation.status == "rejected":
            self._record_rejection(
                run_id=run_id,
                safe_goal=safe_goal,
                provider_metadata=provider_metadata,
                redaction_summary=redaction_summary,
                context_preview=context_preview,
                validation=validation,
            )
            raise RuntimeValidationError(validation)

        structured_plan = self._to_structured_plan(plan_id, provider_plan)
        ledger_entry = LedgerEntry(
            run_id=run_id,
            user_goal_summary=safe_goal[:240],
            provider_mode=provider_metadata.provider_mode,
            provider_metadata=provider_metadata,
            redaction_summary=redaction_summary,
            context_preview=context_preview,
            plan_snapshot=structured_plan.model_dump(mode="json"),
            capability_snapshot=capability_snapshot,
            validation_results=validation,
            policy_rejections=[],
        )
        recorded_entry = self._ledger.append(ledger_entry)

        return PlanResult(
            run_id=run_id,
            run_state=run_state_for_entry(recorded_entry),
            provider_metadata=provider_metadata,
            redaction_summary=redaction_summary,
            context_preview=context_preview,
            plan=structured_plan,
            validation=validation,
            ledger_recorded=True,
        )

    def _to_structured_plan(
        self,
        plan_id: str,
        provider_plan: ProviderPlanOutput,
    ) -> StructuredPlan:
        risk_tiers = sorted({step.risk_tier for step in provider_plan.steps}, key=lambda item: item.value)
        required_confirmations = [
            ConfirmationRequirement(
                step_id=step.step_id,
                capability_id=step.capability_id,
                risk_tier=step.risk_tier,
                reason="Policy requires explicit confirmation before this step can execute.",
            )
            for step in provider_plan.steps
            if step.requires_confirmation
        ]
        status = "blocked" if provider_plan.missing_inputs else "valid"
        return StructuredPlan(
            plan_id=plan_id,
            user_goal_summary=provider_plan.user_goal_summary,
            assumptions=provider_plan.assumptions,
            missing_inputs=provider_plan.missing_inputs,
            proposed_steps=provider_plan.steps,
            risk_tiers=risk_tiers,
            required_confirmations=required_confirmations,
            status=status,
            execution_permitted=False,
        )

    def _record_rejection(
        self,
        run_id: str,
        safe_goal: str,
        provider_metadata: ProviderMetadata | None,
        redaction_summary: RedactionSummary,
        context_preview: ContextPreview,
        validation: PlanValidationResult,
    ) -> None:
        provider_mode = (
            provider_metadata.provider_mode
            if provider_metadata is not None
            else ProviderMode.fake_provider
        )
        policy_rejections = [
            issue
            for issue in validation.errors
            if issue.code
            in {
                "forbidden_action",
                "forbidden_risk_tier",
                "risk_tier_not_allowed",
                "missing_confirmation_requirement",
                "execution_not_allowed",
            }
        ]
        entry = LedgerEntry(
            run_id=run_id,
            user_goal_summary=safe_goal[:240],
            provider_mode=provider_mode,
            provider_metadata=provider_metadata,
            redaction_summary=redaction_summary,
            context_preview=context_preview,
            plan_snapshot=None,
            validation_results=validation,
            policy_rejections=policy_rejections,
        )
        self._ledger.append(entry)

    def _summarize_validation_error(self, exc: ValidationError) -> str:
        first_error = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first_error.get("loc", []))
        error_type = first_error.get("type", "validation_error")
        if location:
            return f"Provider output failed schema validation at {location}: {error_type}."
        return f"Provider output failed schema validation: {error_type}."
