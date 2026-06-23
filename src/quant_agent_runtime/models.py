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
    run_progress_summary: RunProgressSummary
    stale_assumption_summary: StaleAssumptionSummary
    allowed_next_actions: list[str] = Field(default_factory=list)
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
    latest_event_at_utc: str | None = None
    ledger_summary: dict[str, Any]


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
    run_progress_summary: RunProgressSummary
    stale_assumption_summary: StaleAssumptionSummary
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
    recovery_support_level: str = "not_available"
    orchestration_support_level: str = "not_available"
    retry_support_level: str = "not_available"
    plan_revision_support_level: str = "not_available"
    plan_revision_activation_support_level: str = "not_available"
    revalidation_support_level: str = "not_available"
    autopilot_support_level: str = "not_available"
    sample_reset_support_level: str = "not_available"
    plan_only_support_level: str
    execution_support_level: str
    redaction_support_level: str
    validation_gates: list[str]
    contract_source: str
    canonical_agent_contracts_loaded: bool
    loaded_agent_contracts: list[str]
    temporary_internal_contract_fixtures: bool
    provider_status: ProviderRuntimeStatus
    safety_boundaries: list[str]
