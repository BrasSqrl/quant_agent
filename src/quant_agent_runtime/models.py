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
    hosted_provider_enabled: Literal[False] = False
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


class PlanResult(StrictModel):
    run_id: str
    provider_metadata: ProviderMetadata
    redaction_summary: RedactionSummary
    context_preview: ContextPreview
    plan: StructuredPlan
    validation: PlanValidationResult
    ledger_recorded: bool


class LedgerEntry(StrictModel):
    run_id: str
    user_goal_summary: str
    provider_mode: ProviderMode
    provider_metadata: ProviderMetadata | None = None
    redaction_summary: RedactionSummary
    context_preview: ContextPreview | None = None
    plan_snapshot: dict[str, Any] | None = None
    validation_results: PlanValidationResult
    policy_rejections: list[ValidationIssue] = Field(default_factory=list)


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
    ledger_support_level: str
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
