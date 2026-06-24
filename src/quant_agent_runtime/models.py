from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RiskTier(str, Enum):
    read_only = "read_only"
    draft_only = "draft_only"
    reversible_write = "reversible_write"
    workflow_preflight = "workflow_preflight"
    expensive_compute = "expensive_compute"
    artifact_export = "artifact_export"
    governance_decision = "governance_decision"
    forbidden = "forbidden"


class ProviderMode(str, Enum):
    fake_provider = "fake_provider"
    disabled_or_local_fallback = "disabled_or_local_fallback"
    openai = "openai"
    ollama = "ollama"


class StepOperation(str, Enum):
    plan = "plan"
    execute = "execute"


def default_allowed_risk_tiers() -> list[RiskTier]:
    return [
        RiskTier.read_only,
        RiskTier.draft_only,
        RiskTier.reversible_write,
        RiskTier.workflow_preflight,
        RiskTier.expensive_compute,
        RiskTier.artifact_export,
    ]


def default_confirmation_required_tiers() -> list[RiskTier]:
    return [
        RiskTier.draft_only,
        RiskTier.reversible_write,
        RiskTier.expensive_compute,
        RiskTier.artifact_export,
        RiskTier.governance_decision,
    ]


class CapabilityDefinition(StrictModel):
    capability_id: str = Field(pattern=r"^[a-z0-9_]+\.[a-z0-9_]+$")
    app_id: str = Field(pattern=r"^[a-z0-9_]+$")
    display_name: str = Field(min_length=1)
    risk_tier: RiskTier
    enabled: bool = True
    required_fields: list[str] = Field(default_factory=list)
    preflight_required: bool = False
    confirmation_required: bool = False
    execution_supported: bool | None = None


class PolicySettings(StrictModel):
    provider_mode: ProviderMode = ProviderMode.fake_provider
    plan_only: bool = True
    allowed_risk_tiers: list[RiskTier] = Field(default_factory=default_allowed_risk_tiers)
    forbidden_action_ids: list[str] = Field(default_factory=list)
    confirmation_required_tiers: list[RiskTier] = Field(
        default_factory=default_confirmation_required_tiers
    )


class PlanRequest(StrictModel):
    user_goal: str = Field(min_length=1)
    context_summary: dict[str, Any] = Field(default_factory=dict)
    capabilities: list[CapabilityDefinition] | None = None
    policy: PolicySettings = Field(default_factory=PolicySettings)


class RedactionSummary(StrictModel):
    redacted: bool = False
    omitted_fields: list[str] = Field(default_factory=list)
    redacted_fields: list[str] = Field(default_factory=list)


class ContextPreview(StrictModel):
    context: dict[str, Any] = Field(default_factory=dict)
    context_sources: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    omitted_sensitive_fields: list[str] = Field(default_factory=list)
    omitted_row_level_fields: list[str] = Field(default_factory=list)
    context_char_count: int = 0
    data_policy: Literal["summaries_only"] = "summaries_only"
    row_level_data_included: Literal[False] = False


class ContextBuildResult(StrictModel):
    context_summary: dict[str, Any]
    context_preview: ContextPreview


class ProviderMetadata(StrictModel):
    provider: str
    model: str
    provider_mode: ProviderMode
    config_source: str = "internal_default"
    configured_provider_mode: str | None = None
    fallback_reason: str | None = None
    configuration_errors: list[str] = Field(default_factory=list)
    request_purpose: str = "plan_generation"
    supports_execution: bool = False


class ProviderRuntimeStatus(StrictModel):
    config_source: str
    configured_provider_mode: str
    effective_provider_mode: ProviderMode
    provider_identifier: str
    model_profile: str
    allowed_model_roles: list[str] = Field(default_factory=list)
    configured: bool
    supports_execution: Literal[False] = False
    hosted_provider_enabled: bool = False
    secret_reference_present: bool = False
    secrets_exposed: Literal[False] = False
    fallback_reason: str | None = None
    health_check_enabled: bool = False
    provider_reachable: bool = False
    configuration_errors: list[str] = Field(default_factory=list)
    timeout_seconds: int = 0
    max_context_chars: int = 18000
    retention_policy_label: str = "not_applicable"


class PlanStep(StrictModel):
    step_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    capability_id: str = Field(min_length=1)
    app_id: str = Field(min_length=1)
    risk_tier: RiskTier
    operation: StepOperation = StepOperation.plan
    preflight_required: bool = False
    requires_confirmation: bool = False
    action_input: dict[str, Any] = Field(default_factory=dict)
    expected_artifacts: list[str] = Field(default_factory=list)
    validation_checks: list[str] = Field(default_factory=list)


class ProviderPlanOutput(StrictModel):
    user_goal_summary: str = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    steps: list[PlanStep] = Field(default_factory=list)


class ConfirmationRequirement(StrictModel):
    step_id: str
    capability_id: str
    risk_tier: RiskTier
    reason: str


class StructuredPlan(StrictModel):
    plan_id: str
    user_goal_summary: str
    assumptions: list[str]
    missing_inputs: list[str]
    proposed_steps: list[PlanStep]
    risk_tiers: list[RiskTier]
    required_confirmations: list[ConfirmationRequirement]
    status: Literal["valid", "blocked"]
    execution_permitted: bool = False


class ValidationIssue(StrictModel):
    code: str
    message: str
    step_id: str | None = None
    capability_id: str | None = None


class PlanValidationResult(StrictModel):
    status: Literal["valid", "rejected"]
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)


RunState = Literal[
    "planned",
    "waiting_for_input",
    "waiting_for_confirmation",
    "preflight_blocked",
    "ready_for_execution_preview",
    "running",
    "paused",
    "completed",
    "completed_with_warnings",
    "failed_recoverable",
    "failed_terminal",
    "cancelled",
    "sample_reset",
]


OrchestrationStepStatus = Literal[
    "not_ready",
    "needs_preflight",
    "preflight_blocked",
    "needs_confirmation",
    "ready_for_action_request",
    "ready_for_execution",
    "running",
    "completed",
    "completed_with_warnings",
    "failed_recoverable",
    "failed_terminal",
    "cancelled",
    "informational",
    "unsupported",
]


class PlanResult(StrictModel):
    run_id: str
    run_state: RunState
    provider_metadata: ProviderMetadata
    redaction_summary: RedactionSummary
    context_preview: ContextPreview
    plan: StructuredPlan
    validation: PlanValidationResult
    ledger_recorded: bool


WorkflowScope = Literal["full_lifecycle", "app_workflow", "stage_range", "capability_set"]


class WorkflowRunRequest(StrictModel):
    goal: str = Field(min_length=1)
    workflow_scope: WorkflowScope
    source_app: str | None = None
    start_stage: str | None = None
    end_stage: str | None = None
    requested_capability_ids: list[str] = Field(default_factory=list)
    context_summary: dict[str, Any] = Field(default_factory=dict)
    policy: PolicySettings = Field(default_factory=PolicySettings)


class WorkflowRunScopeSummary(StrictModel):
    workflow_scope: WorkflowScope
    source_app: str | None = None
    start_stage: str | None = None
    end_stage: str | None = None
    requested_capability_ids: list[str] = Field(default_factory=list)
    selected_template_ids: list[str] = Field(default_factory=list)
    selected_capability_ids: list[str] = Field(default_factory=list)
    omitted_capability_ids: list[str] = Field(default_factory=list)
    workflow_gaps: list[dict[str, Any]] = Field(default_factory=list)


class WorkflowRunResult(StrictModel):
    run_id: str
    workflow_scope: WorkflowRunScopeSummary
    plan: StructuredPlan
    run_state: RunState
    orchestration: RunOrchestrationResult
    provider_metadata: ProviderMetadata
    redaction_summary: RedactionSummary
    context_preview: ContextPreview
    validation: PlanValidationResult
    ledger_recorded: bool


class WorkflowRunStatusResult(StrictModel):
    run_id: str
    workflow_scope: WorkflowRunScopeSummary | None = None
    run_status: RunStatusResult
    orchestration: RunOrchestrationResult
    validation: PlanValidationResult
    ledger_recorded: Literal[False] = False


class WorkflowScopeResolutionRequest(StrictModel):
    goal: str = Field(min_length=1)
    source_app: str | None = None
    source_stage: str | None = None
    context_summary: dict[str, Any] = Field(default_factory=dict)


class WorkflowScopeResolutionResult(StrictModel):
    goal: str
    resolution_status: Literal["resolved"]
    resolved_request: WorkflowRunRequest
    workflow_scope: WorkflowRunScopeSummary
    resolution_summary: dict[str, Any] = Field(default_factory=dict)
    validation: PlanValidationResult
    ledger_recorded: Literal[False] = False


class WorkflowAdvanceRequest(StrictModel):
    advance_intent: Literal["advance_workflow_one_step"] = "advance_workflow_one_step"


class WorkflowAdvanceUntilBlockedRequest(StrictModel):
    advance_intent: Literal["advance_workflow_until_blocked"] = "advance_workflow_until_blocked"
    max_steps: int = Field(default=10, ge=1, le=25)


class WorkflowAdvanceResult(StrictModel):
    run_id: str
    workflow_scope: WorkflowRunScopeSummary | None = None
    step_id: str | None = None
    capability_id: str | None = None
    selected_action: str | None = None
    advance_status: str
    delegated_result: dict[str, Any] | None = None
    run_state: RunState
    orchestration: RunOrchestrationResult
    validation: PlanValidationResult
    ledger_recorded: bool


class WorkflowAdvanceUntilBlockedResult(StrictModel):
    run_id: str
    workflow_scope: WorkflowRunScopeSummary | None = None
    advance_status: str
    completed_action_count: int = 0
    last_result: WorkflowAdvanceResult | None = None
    run_state: RunState
    orchestration: RunOrchestrationResult
    validation: PlanValidationResult
    ledger_recorded: bool


class PreflightRequest(StrictModel):
    run_id: str = Field(min_length=1)
    step_id: str = Field(min_length=1)


class PreflightResult(StrictModel):
    run_id: str
    step_id: str
    capability_id: str
    preflight: dict[str, Any]
    run_state: RunState
    validation: PlanValidationResult
    ledger_recorded: bool


class ConfirmationRequest(StrictModel):
    run_id: str = Field(min_length=1)
    step_id: str = Field(min_length=1)
    confirmation_intent: str = Field(min_length=1)


class ConfirmationResult(StrictModel):
    run_id: str
    step_id: str
    capability_id: str
    confirmation: dict[str, Any]
    run_state: RunState
    validation: PlanValidationResult
    ledger_recorded: bool


class ActionRequestPreviewRequest(StrictModel):
    run_id: str = Field(min_length=1)
    step_id: str = Field(min_length=1)


class ActionRequestPreviewResult(StrictModel):
    run_id: str
    step_id: str
    capability_id: str
    action_request: dict[str, Any]
    run_state: RunState
    validation: PlanValidationResult
    ledger_recorded: bool


class ExecutionRequest(StrictModel):
    run_id: str = Field(min_length=1)
    step_id: str = Field(min_length=1)


class ExecutionResult(StrictModel):
    run_id: str
    step_id: str
    capability_id: str
    action_request: dict[str, Any]
    action_result: dict[str, Any]
    run_state: RunState
    validation: PlanValidationResult
    ledger_recorded: bool


class RetryRequest(StrictModel):
    run_id: str = Field(min_length=1)
    step_id: str = Field(min_length=1)
    retry_intent: Literal["retry_failed_step"]


class RetryResult(StrictModel):
    run_id: str
    step_id: str
    capability_id: str
    retry_event: dict[str, Any]
    action_request: dict[str, Any]
    action_result: dict[str, Any]
    run_state: RunState
    orchestration: RunOrchestrationResult
    validation: PlanValidationResult
    ledger_recorded: bool


class RunProgressSummary(StrictModel):
    run_id: str
    parent_run_id: str | None = None
    parent_plan_id: str | None = None
    activated_revision_id: str | None = None
    child_run_ids: list[str] = Field(default_factory=list)
    plan_id: str | None = None
    run_state: RunState
    final_status: str
    total_steps: int = 0
    completed_steps: int = 0
    completed_with_warnings_steps: int = 0
    informational_steps: int = 0
    unsupported_steps: int = 0
    blocked_steps: int = 0
    failed_recoverable_steps: int = 0
    failed_terminal_steps: int = 0
    not_ready_steps: int = 0
    current_step_id: str | None = None
    current_step_title: str | None = None
    current_step_status: str | None = None
    current_blocker: str | None = None
    latest_record_counts: dict[str, int] = Field(default_factory=dict)
    allowed_next_actions: list[str] = Field(default_factory=list)


class StaleAssumptionSummary(StrictModel):
    status: Literal["not_evaluated", "fresh", "stale", "insufficient_context"] = "not_evaluated"
    current_context_provided: bool = False
    state_changed_since_planning: bool = False
    changed_sections: list[str] = Field(default_factory=list)
    added_sections: list[str] = Field(default_factory=list)
    missing_current_sections: list[str] = Field(default_factory=list)
    original_sections: list[str] = Field(default_factory=list)
    current_sections: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    revalidation_required: bool = False
    checked_at_utc: str | None = None


class GovernanceSummary(StrictModel):
    support_level: str = "not_available"
    environment_policy_pack_support_level: str = "not_available"
    release_evidence_support_level: str = "not_available"
    policy_pack_id: str
    environment: str
    actor_id: str | None = None
    actor_role: str
    effective_actor_role: str
    source: str
    fallback_active: bool = False
    fallback_reason: str | None = None
    allowed_routes: list[str] = Field(default_factory=list)
    denied_routes: list[str] = Field(default_factory=list)
    allowed_capability_ids: list[str] = Field(default_factory=list)
    denied_capability_ids: list[str] = Field(default_factory=list)
    diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    run_id: str | None = None


class SeparationOfDutiesSummary(StrictModel):
    support_level: str = "not_available"
    run_id: str | None = None
    actor_id: str
    actor_role: str
    effective_actor_role: str
    actor_exempt: bool = False
    active_rule_ids: list[str] = Field(default_factory=list)
    exempt_roles: list[str] = Field(default_factory=list)
    protected_routes: list[str] = Field(default_factory=list)
    blocked_routes: list[str] = Field(default_factory=list)
    blocked: bool = False
    blocker_reason: str | None = None
    latest_denial: dict[str, Any] | None = None
    diagnostics: list[dict[str, Any]] = Field(default_factory=list)


class LedgerIntegrity(StrictModel):
    status: str = "not_available"
    algorithm: str | None = None
    sequence_number: int = 0
    previous_hash: str | None = None
    payload_hash: str | None = None
    recorded_at_utc: str | None = None
    contract_schema: str | None = None
    journal_consistent: bool = False
    diagnostics: list[dict[str, Any]] = Field(default_factory=list)


class LedgerIntegritySummary(StrictModel):
    status: str = "not_available"
    algorithm: str | None = None
    sequence_number: int = 0
    previous_hash: str | None = None
    payload_hash: str | None = None
    recorded_at_utc: str | None = None
    contract_schema: str | None = None
    journal_consistent: bool = False
    diagnostics: list[dict[str, Any]] = Field(default_factory=list)


class RunStatusResult(StrictModel):
    run_id: str
    parent_run_id: str | None = None
    parent_plan_id: str | None = None
    activated_revision_id: str | None = None
    child_run_ids: list[str] = Field(default_factory=list)
    run_state: RunState
    final_status: str
    user_goal_summary: str
    plan: dict[str, Any] | None = None
    latest_preflight: dict[str, Any] | None = None
    latest_confirmation: dict[str, Any] | None = None
    latest_action_request: dict[str, Any] | None = None
    latest_action_result: dict[str, Any] | None = None
    latest_recovery: dict[str, Any] | None = None
    latest_cancellation: dict[str, Any] | None = None
    ledger_summary: dict[str, Any]
    external_approval_summary: dict[str, Any] | None = None
    external_approval_enforcement_summary: dict[str, Any] | None = None
    run_progress_summary: RunProgressSummary
    stale_assumption_summary: StaleAssumptionSummary
    ownership_summary: UserWorkflowOwnershipSummary | None = None
    plan_review_summary: UserPlanReviewSummary | None = None
    plan_approval_summary: UserPlanApprovalSummary | None = None
    readiness_summary: UserWorkflowReadinessSummary | None = None
    consent_summary: UserWorkflowConsentSummary | None = None
    allowed_user_owned_actions: list[str] = Field(default_factory=list)
    allowed_next_actions: list[str] = Field(default_factory=list)
    governance_summary: GovernanceSummary | None = None
    separation_of_duties_summary: SeparationOfDutiesSummary | None = None
    ledger_integrity_summary: LedgerIntegritySummary | None = None
    validation: PlanValidationResult


class RunSummary(StrictModel):
    run_id: str
    parent_run_id: str | None = None
    parent_plan_id: str | None = None
    activated_revision_id: str | None = None
    child_run_ids: list[str] = Field(default_factory=list)
    run_state: RunState
    final_status: str
    user_goal_summary: str
    lifecycle_id: str | None = None
    app_ids: list[str] = Field(default_factory=list)
    capability_ids: list[str] = Field(default_factory=list)
    latest_action_result: dict[str, Any] | None = None
    latest_recovery: dict[str, Any] | None = None
    latest_cancellation: dict[str, Any] | None = None
    latest_event_at_utc: str | None = None
    ledger_summary: dict[str, Any]
    external_approval_summary: dict[str, Any] | None = None
    external_approval_enforcement_summary: dict[str, Any] | None = None
    ownership_summary: UserWorkflowOwnershipSummary | None = None
    plan_review_summary: UserPlanReviewSummary | None = None
    plan_approval_summary: UserPlanApprovalSummary | None = None
    readiness_summary: UserWorkflowReadinessSummary | None = None
    consent_summary: UserWorkflowConsentSummary | None = None
    allowed_user_owned_actions: list[str] = Field(default_factory=list)
    governance_summary: GovernanceSummary | None = None
    separation_of_duties_summary: SeparationOfDutiesSummary | None = None
    ledger_integrity_summary: LedgerIntegritySummary | None = None


class RunListResult(StrictModel):
    runs: list[RunSummary]
    count: int
    validation: PlanValidationResult


class OrchestrationStepSummary(StrictModel):
    step_id: str
    capability_id: str
    app_id: str
    title: str
    risk_tier: str
    status: OrchestrationStepStatus
    is_current: bool = False
    preflight_required: bool = False
    confirmation_required: bool = False
    execution_supported: bool = False
    required_gate: str | None = None
    blocker_reason: str | None = None
    latest_preflight_reference: dict[str, Any] | None = None
    latest_confirmation_reference: dict[str, Any] | None = None
    latest_action_request_reference: dict[str, Any] | None = None
    latest_action_result_reference: dict[str, Any] | None = None
    allowed_actions: list[str] = Field(default_factory=list)


class RunOrchestrationResult(StrictModel):
    run_id: str
    parent_run_id: str | None = None
    parent_plan_id: str | None = None
    activated_revision_id: str | None = None
    child_run_ids: list[str] = Field(default_factory=list)
    run_state: RunState
    final_status: str
    plan_id: str | None = None
    current_step_id: str | None = None
    steps: list[OrchestrationStepSummary]
    allowed_next_actions: list[str] = Field(default_factory=list)
    ledger_summary: dict[str, Any]
    external_approval_summary: dict[str, Any] | None = None
    external_approval_enforcement_summary: dict[str, Any] | None = None
    run_progress_summary: RunProgressSummary
    stale_assumption_summary: StaleAssumptionSummary
    ownership_summary: UserWorkflowOwnershipSummary | None = None
    plan_review_summary: UserPlanReviewSummary | None = None
    plan_approval_summary: UserPlanApprovalSummary | None = None
    readiness_summary: UserWorkflowReadinessSummary | None = None
    consent_summary: UserWorkflowConsentSummary | None = None
    allowed_user_owned_actions: list[str] = Field(default_factory=list)
    governance_summary: GovernanceSummary | None = None
    separation_of_duties_summary: SeparationOfDutiesSummary | None = None
    ledger_integrity_summary: LedgerIntegritySummary | None = None
    validation: PlanValidationResult


class CancellationRequest(StrictModel):
    run_id: str = Field(min_length=1)
    cancellation_intent: Literal["cancel_run"]
    reason: str = Field(min_length=1)


class CancellationResult(StrictModel):
    run_id: str
    run_state: RunState
    cancellation: dict[str, Any]
    final_status: str
    allowed_next_actions: list[str] = Field(default_factory=list)
    validation: PlanValidationResult
    ledger_recorded: bool


class PauseRequest(StrictModel):
    run_id: str = Field(min_length=1)
    pause_intent: Literal["pause_run"]
    reason: str = Field(min_length=1)


class PauseResult(StrictModel):
    run_id: str
    run_state: RunState
    pause_event: dict[str, Any]
    final_status: str
    allowed_next_actions: list[str] = Field(default_factory=list)
    validation: PlanValidationResult
    ledger_recorded: bool


class ResumptionRequest(StrictModel):
    run_id: str = Field(min_length=1)
    resume_intent: Literal["resume_run"]


class ResumptionResult(StrictModel):
    run_id: str
    run_state: RunState
    resumption_event: dict[str, Any]
    final_status: str
    orchestration: RunOrchestrationResult
    allowed_next_actions: list[str] = Field(default_factory=list)
    validation: PlanValidationResult
    ledger_recorded: bool


PlanRevisionReason = Literal[
    "missing_inputs",
    "preflight_blocked",
    "stale_state",
    "failed_recoverable",
    "user_requested",
]


class PlanRevisionRequest(StrictModel):
    run_id: str = Field(min_length=1)
    revision_intent: Literal["revise_plan"]
    reason: PlanRevisionReason
    current_context_summary: dict[str, Any] = Field(default_factory=dict)


class PlanRevisionResult(StrictModel):
    run_id: str
    parent_plan_id: str
    revision_id: str
    revised_plan: dict[str, Any]
    revision_event: dict[str, Any]
    run_state: RunState
    orchestration: RunOrchestrationResult
    context_preview: ContextPreview
    stale_state_summary: dict[str, Any]
    validation: PlanValidationResult
    ledger_recorded: bool


class PlanRevisionActivationRequest(StrictModel):
    run_id: str = Field(min_length=1)
    revision_id: str = Field(min_length=1)
    activation_intent: Literal["activate_plan_revision"]


class PlanRevisionActivationResult(StrictModel):
    parent_run_id: str
    child_run_id: str
    revision_id: str
    activated_plan: dict[str, Any]
    activation_event: dict[str, Any]
    child_run_state: RunState
    child_orchestration: RunOrchestrationResult
    parent_run_state: RunState
    validation: PlanValidationResult
    ledger_recorded: bool


class RunRevalidationRequest(StrictModel):
    run_id: str = Field(min_length=1)
    revalidation_intent: Literal["check_current_context"]
    current_context_summary: dict[str, Any] = Field(default_factory=dict)


class RunRevalidationResult(StrictModel):
    run_id: str
    run_progress_summary: RunProgressSummary
    stale_assumption_summary: StaleAssumptionSummary
    orchestration: RunOrchestrationResult
    validation: PlanValidationResult
    ledger_recorded: bool


OwnershipClassification = Literal["sample_owned", "user_owned", "unknown"]


class UserWorkflowOwnershipSummary(StrictModel):
    ownership: OwnershipClassification
    lifecycle_id: str | None = None
    sample_workspace_id: str | None = None
    sample_owned: bool = False
    sample_workspace: bool = False
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    safe_labels: dict[str, Any] = Field(default_factory=dict)


class UserWorkflowReadinessSummary(StrictModel):
    status: Literal["ready", "blocked", "sample_owned", "not_checked"] = "not_checked"
    readiness_intent: str | None = None
    consent_required: bool = False
    allowed_preflight_capabilities: list[str] = Field(default_factory=list)
    allowed_execution_capabilities: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    data_policy: Literal["summaries_and_references_only"] = "summaries_and_references_only"
    checked_at_utc: str | None = None


class UserWorkflowConsentSummary(StrictModel):
    status: Literal["not_required", "not_recorded", "consented"] = "not_recorded"
    consent_intent: str | None = None
    consent_scope: str | None = None
    consented_by: str | None = None
    consented_at_utc: str | None = None
    execution_permitted: Literal[False] = False
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class UserPlanAssumptionReview(StrictModel):
    assumption_index: int = Field(ge=0)
    decision: Literal["accept", "revise"]
    safe_note: str | None = Field(default=None, max_length=500)


class UserPlanReviewSummary(StrictModel):
    status: Literal["not_required", "not_reviewed", "reviewed", "revision_requested", "blocked"] = "not_reviewed"
    plan_review_id: str | None = None
    plan_id: str | None = None
    review_intent: str | None = None
    total_assumption_count: int = 0
    accepted_assumption_count: int = 0
    revise_assumption_count: int = 0
    revision_notes: list[dict[str, Any]] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    data_policy: Literal["summaries_and_references_only"] = "summaries_and_references_only"
    reviewed_by: str | None = None
    reviewed_at_utc: str | None = None


class UserPlanApprovalSummary(StrictModel):
    status: Literal["not_required", "not_approved", "approved", "blocked"] = "not_approved"
    plan_approval_id: str | None = None
    plan_review_id: str | None = None
    plan_id: str | None = None
    approval_intent: str | None = None
    approved_by: str | None = None
    approved_at_utc: str | None = None
    execution_permitted: Literal[False] = False
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class UserPlanReviewRequest(StrictModel):
    run_id: str = Field(min_length=1)
    review_intent: Literal["review_plan_assumptions"]
    assumption_reviews: list[UserPlanAssumptionReview]
    current_context_summary: dict[str, Any] = Field(default_factory=dict)


class UserPlanReviewResult(StrictModel):
    run_id: str
    ownership_summary: UserWorkflowOwnershipSummary
    plan_review_summary: UserPlanReviewSummary
    plan_approval_summary: UserPlanApprovalSummary
    readiness_summary: UserWorkflowReadinessSummary
    consent_summary: UserWorkflowConsentSummary
    run_state: RunState
    orchestration: RunOrchestrationResult
    validation: PlanValidationResult
    ledger_recorded: bool


class UserPlanApprovalRequest(StrictModel):
    run_id: str = Field(min_length=1)
    approval_intent: Literal["approve_user_plan"]
    plan_review_id: str = Field(min_length=1)


class UserPlanApprovalResult(StrictModel):
    run_id: str
    ownership_summary: UserWorkflowOwnershipSummary
    plan_review_summary: UserPlanReviewSummary
    plan_approval_summary: UserPlanApprovalSummary
    readiness_summary: UserWorkflowReadinessSummary
    consent_summary: UserWorkflowConsentSummary
    run_state: RunState
    orchestration: RunOrchestrationResult
    validation: PlanValidationResult
    ledger_recorded: bool


class UserWorkflowReadinessRequest(StrictModel):
    run_id: str = Field(min_length=1)
    readiness_intent: Literal["check_user_owned_readiness"]
    current_context_summary: dict[str, Any] = Field(default_factory=dict)


class UserWorkflowReadinessResult(StrictModel):
    run_id: str
    ownership_summary: UserWorkflowOwnershipSummary
    plan_review_summary: UserPlanReviewSummary
    plan_approval_summary: UserPlanApprovalSummary
    readiness_summary: UserWorkflowReadinessSummary
    consent_summary: UserWorkflowConsentSummary
    run_state: RunState
    orchestration: RunOrchestrationResult
    validation: PlanValidationResult
    ledger_recorded: bool


class UserWorkflowConsentRequest(StrictModel):
    run_id: str = Field(min_length=1)
    consent_intent: Literal["approve_user_owned_guided_execution"]
    consent_scope: Literal["single_run_review_draft_actions"]


class UserWorkflowConsentResult(StrictModel):
    run_id: str
    ownership_summary: UserWorkflowOwnershipSummary
    plan_review_summary: UserPlanReviewSummary
    plan_approval_summary: UserPlanApprovalSummary
    readiness_summary: UserWorkflowReadinessSummary
    consent_summary: UserWorkflowConsentSummary
    run_state: RunState
    orchestration: RunOrchestrationResult
    validation: PlanValidationResult
    ledger_recorded: bool


class SampleAutopilotPreviewRequest(StrictModel):
    run_id: str = Field(min_length=1)
    autopilot_intent: Literal["preview_sample_autopilot"]
    current_context_summary: dict[str, Any] = Field(default_factory=dict)


class SampleAutopilotEligibility(StrictModel):
    eligible: bool
    status: Literal["eligible", "blocked"]
    sample_workspace_id: str | None = None
    sample_label: str | None = None
    lifecycle_id: str | None = None
    sample_owned: bool = False
    allowlisted: bool = False
    reset_boundary_available: bool = False
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    safe_labels: dict[str, Any] = Field(default_factory=dict)


class SampleAutopilotPreviewStep(StrictModel):
    step_id: str
    capability_id: str
    app_id: str
    title: str
    status: str
    dry_run_action: str
    allowed_manual_actions: list[str] = Field(default_factory=list)
    blocker_reason: str | None = None
    preflight_required: bool = False
    confirmation_required: bool = False
    execution_supported: bool = False


class SampleAutopilotPreview(StrictModel):
    dry_run_only: Literal[True] = True
    autonomous_execution_permitted: Literal[False] = False
    sample_workspace_id: str | None = None
    current_step_id: str | None = None
    step_count: int = 0
    blocked_step_count: int = 0
    next_manual_actions: list[str] = Field(default_factory=list)
    steps: list[SampleAutopilotPreviewStep] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SampleAutopilotPreviewResult(StrictModel):
    run_id: str
    sample_eligibility: SampleAutopilotEligibility
    autopilot_preview: SampleAutopilotPreview
    run_progress_summary: RunProgressSummary
    orchestration: RunOrchestrationResult
    validation: PlanValidationResult
    ledger_recorded: bool


class SampleAutopilotStepRequest(StrictModel):
    run_id: str = Field(min_length=1)
    autopilot_intent: Literal["advance_sample_autopilot_one_step"]
    current_context_summary: dict[str, Any] = Field(default_factory=dict)


class SampleAutopilotStepResult(StrictModel):
    run_id: str
    step_id: str | None = None
    capability_id: str | None = None
    selected_action: str | None = None
    advance_status: str
    autopilot_event: dict[str, Any]
    delegated_result: dict[str, Any] | None = None
    sample_eligibility: SampleAutopilotEligibility
    run_progress_summary: RunProgressSummary
    orchestration: RunOrchestrationResult
    validation: PlanValidationResult
    ledger_recorded: bool


class SampleResetPreviewRequest(StrictModel):
    run_id: str = Field(min_length=1)
    reset_intent: Literal["preview_sample_reset"]
    current_context_summary: dict[str, Any] = Field(default_factory=dict)


class SampleResetPreviewResult(StrictModel):
    run_id: str
    reset_preview_id: str | None = None
    sample_eligibility: SampleAutopilotEligibility
    reset_boundary_summary: dict[str, Any]
    run_progress_summary: RunProgressSummary
    orchestration: RunOrchestrationResult
    validation: PlanValidationResult
    ledger_recorded: bool


class SampleResetRequest(StrictModel):
    run_id: str = Field(min_length=1)
    reset_intent: Literal["reset_sample_demo"]
    reset_preview_id: str = Field(min_length=1)
    current_context_summary: dict[str, Any] = Field(default_factory=dict)


class SampleResetResult(StrictModel):
    run_id: str
    reset_preview_id: str
    reset_status: str
    reset_event: dict[str, Any]
    reset_result: dict[str, Any] | None = None
    sample_eligibility: SampleAutopilotEligibility
    reset_boundary_summary: dict[str, Any]
    run_progress_summary: RunProgressSummary
    orchestration: RunOrchestrationResult
    validation: PlanValidationResult
    ledger_recorded: bool


class DemoNarrativeSection(StrictModel):
    section_id: str
    title: str
    status: str
    summary: str
    step_id: str | None = None
    capability_id: str | None = None
    app_id: str | None = None
    evidence_references: list[dict[str, Any]] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DemoNarrativeResult(StrictModel):
    run_id: str
    demo_status: Literal["not_sample_demo", "in_progress", "completed", "sample_reset", "blocked"]
    sample_eligibility: SampleAutopilotEligibility
    narrative_sections: list[DemoNarrativeSection] = Field(default_factory=list)
    safety_summary: dict[str, Any]
    run_progress_summary: RunProgressSummary
    orchestration: RunOrchestrationResult
    ledger_summary: dict[str, Any]
    validation: PlanValidationResult


class AgentSupportBundleResult(StrictModel):
    schema_version: str = "1.0"
    data_policy: Literal["summaries_and_references_only"] = "summaries_and_references_only"
    bundle_id: str
    run_id: str
    generated_at_utc: str
    runtime_summary: dict[str, Any]
    provider_summary: dict[str, Any] | None = None
    governance_summary: GovernanceSummary | None = None
    separation_of_duties_summary: SeparationOfDutiesSummary | None = None
    run_status: RunStatusResult
    orchestration: RunOrchestrationResult
    ledger: dict[str, Any]
    ledger_integrity_summary: LedgerIntegritySummary
    contract_summary: dict[str, Any]
    redaction_report: dict[str, Any]
    validation: PlanValidationResult


class ExternalApprovalPreviewRequest(StrictModel):
    run_id: str = Field(min_length=1)
    approval_intent: Literal["preview_external_approval_request"]
    approval_scope: Literal["run", "step"]
    step_id: str | None = None


class ExternalApprovalPreviewResult(StrictModel):
    run_id: str
    step_id: str | None = None
    approval_request: dict[str, Any]
    run_status: RunStatusResult
    orchestration: RunOrchestrationResult
    validation: PlanValidationResult
    ledger_recorded: bool


class ExternalApprovalDecisionImportRequest(StrictModel):
    run_id: str = Field(min_length=1)
    decision_intent: Literal["import_external_approval_decision"]
    approval_decision: dict[str, Any]


class ExternalApprovalDecisionImportResult(StrictModel):
    run_id: str
    step_id: str | None = None
    approval_request_id: str
    approval_decision: dict[str, Any]
    external_approval_summary: dict[str, Any]
    run_status: RunStatusResult
    orchestration: RunOrchestrationResult
    validation: PlanValidationResult
    ledger_recorded: bool


class ExternalApprovalDecisionRefreshRequest(StrictModel):
    run_id: str = Field(min_length=1)
    decision_refresh_intent: Literal["refresh_external_approval_decision"]
    approval_request_id: str = Field(min_length=1)


class ExternalApprovalDecisionRefreshResult(StrictModel):
    run_id: str
    step_id: str | None = None
    approval_request_id: str
    decision_refresh: dict[str, Any]
    approval_decision: dict[str, Any] | None = None
    external_approval_summary: dict[str, Any]
    run_status: RunStatusResult
    orchestration: RunOrchestrationResult
    validation: PlanValidationResult
    ledger_recorded: bool


class ExternalApprovalSubmissionRequest(StrictModel):
    run_id: str = Field(min_length=1)
    submission_intent: Literal["submit_external_approval_request"]
    approval_request_id: str = Field(min_length=1)


class ExternalApprovalSubmissionResult(StrictModel):
    run_id: str
    step_id: str | None = None
    approval_request_id: str
    external_approval_submission: dict[str, Any]
    run_status: RunStatusResult
    orchestration: RunOrchestrationResult
    validation: PlanValidationResult
    ledger_recorded: bool


class ExternalApprovalSubmissionSummary(StrictModel):
    external_approval_submission_id: str
    approval_request_id: str
    approval_scope: str
    step_id: str | None = None
    capability_id: str | None = None
    adapter_mode: str
    submission_status: str
    adapter_delivery_status: str | None = None
    adapter_delivery_summary: dict[str, Any] = Field(default_factory=dict)
    outbox_status: Literal["present", "missing", "disabled", "not_checked"]
    submitted_at_utc: str | None = None
    submission_reference: dict[str, Any] = Field(default_factory=dict)
    latest_matching_decision: dict[str, Any] | None = None
    latest_decision_refresh: dict[str, Any] | None = None
    validation: PlanValidationResult
    ledger_integrity_summary: LedgerIntegritySummary | None = None


class ExternalApprovalSubmissionListResult(StrictModel):
    run_id: str
    submissions: list[ExternalApprovalSubmissionSummary]
    external_approval_summary: dict[str, Any]
    validation: PlanValidationResult
    ledger_recorded: Literal[False] = False


class LedgerEntry(StrictModel):
    schema_version: str = "1.0"
    data_policy: Literal["summaries_and_references_only"] = "summaries_and_references_only"
    run_id: str
    parent_run_id: str | None = None
    parent_plan_id: str | None = None
    activated_revision_id: str | None = None
    child_run_ids: list[str] = Field(default_factory=list)
    user_goal_summary: str
    provider_mode: ProviderMode
    provider_metadata: ProviderMetadata | None = None
    redaction_summary: RedactionSummary
    context_preview: ContextPreview | None = None
    plan_snapshot: dict[str, Any] | None = None
    capability_snapshot: list[dict[str, Any]] = Field(default_factory=list)
    preflight_records: list[dict[str, Any]] = Field(default_factory=list)
    confirmation_records: list[dict[str, Any]] = Field(default_factory=list)
    action_requests: list[dict[str, Any]] = Field(default_factory=list)
    action_results: list[dict[str, Any]] = Field(default_factory=list)
    validation_results: PlanValidationResult
    policy_rejections: list[ValidationIssue] = Field(default_factory=list)
    recovery_events: list[dict[str, Any]] = Field(default_factory=list)
    cancellation_events: list[dict[str, Any]] = Field(default_factory=list)
    final_status: str = "planned"
    safe_artifact_map: list[dict[str, Any]] = Field(default_factory=list)
    ledger_integrity: LedgerIntegrity | None = None


class RuntimeManifest(StrictModel):
    schema_version: str = "1.0"
    service_name: str
    runtime_version: str
    supported_quant_suite_contract_versions: list[str]
    plan_only_mode: bool
    execution_supported: bool
    supported_routes: list[str]
    supported_provider_modes: list[ProviderMode]
    supported_model_roles: list[str]
    supported_risk_tiers: list[RiskTier]
    policy_version: str
    runtime_health_endpoint: str
    capability_discovery_endpoints: list[str]
    capability_discovery: dict[str, Any] = Field(default_factory=dict)
    supported_preflight_capabilities: list[str] = Field(default_factory=list)
    supported_execution_capabilities: list[str] = Field(default_factory=list)
    ledger_support_level: str
    ledger_storage: dict[str, Any] = Field(default_factory=dict)
    ledger_integrity_support_level: str = "not_available"
    support_bundle_support_level: str = "not_available"
    recovery_support_level: str = "not_available"
    orchestration_support_level: str = "not_available"
    retry_support_level: str = "not_available"
    plan_revision_support_level: str = "not_available"
    plan_revision_activation_support_level: str = "not_available"
    revalidation_support_level: str = "not_available"
    user_workflow_support_level: str = "not_available"
    user_plan_approval_support_level: str = "not_available"
    workflow_run_support_level: str = "not_available"
    workflow_scope_resolution_support_level: str = "not_available"
    workflow_template_support_level: str = "not_available"
    long_running_action_support_level: str = "not_available"
    supported_workflow_scopes: list[str] = Field(default_factory=list)
    autopilot_support_level: str = "not_available"
    sample_reset_support_level: str = "not_available"
    demo_narrative_support_level: str = "not_available"
    external_approval_support_level: str = "not_available"
    external_approval_decision_support_level: str = "not_available"
    external_approval_enforcement_support_level: str = "not_available"
    external_approval_submission_support_level: str = "not_available"
    external_approval_submission_status_support_level: str = "not_available"
    external_approval_decision_refresh_support_level: str = "not_available"
    external_approval_adapter_support_level: str = "not_available"
    external_approval_submission_adapter: dict[str, Any] | None = None
    governance_support_level: str = "not_available"
    separation_of_duties_support_level: str = "not_available"
    environment_policy_pack_support_level: str = "not_available"
    release_evidence_support_level: str = "not_available"
    plan_only_support_level: str
    execution_support_level: str
    redaction_support_level: str
    validation_gates: list[str]
    contract_source: str
    canonical_agent_contracts_loaded: bool
    loaded_agent_contracts: list[str]
    temporary_internal_contract_fixtures: bool
    provider_status: ProviderRuntimeStatus
    governance_summary: GovernanceSummary | None = None
    separation_of_duties_summary: SeparationOfDutiesSummary | None = None
    external_approval_enforcement_summary: dict[str, Any] | None = None
    safety_boundaries: list[str]
