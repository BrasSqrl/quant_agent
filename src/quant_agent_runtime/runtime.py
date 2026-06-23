from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from quant_agent_runtime import __version__
from quant_agent_runtime.action_request import ActionRequestPreviewService
from quant_agent_runtime.capabilities import default_capabilities
from quant_agent_runtime.confirmation import ConfirmationService
from quant_agent_runtime.contracts import QuantSuiteContractLoader
from quant_agent_runtime.contracts.internal_test_fixtures import TEMPORARY_AGENT_CONTRACT_FIXTURES
from quant_agent_runtime.demo_narrative import DemoNarrativeService
from quant_agent_runtime.execution import ExecutionService
from quant_agent_runtime.external_approval import (
    EXTERNAL_APPROVAL_ADAPTER_SUPPORT_LEVEL,
    EXTERNAL_APPROVAL_DECISION_REFRESH_SUPPORT_LEVEL,
    EXTERNAL_APPROVAL_DECISION_SUPPORT_LEVEL,
    EXTERNAL_APPROVAL_SUPPORT_LEVEL,
    EXTERNAL_APPROVAL_SUBMISSION_SUPPORT_LEVEL,
    EXTERNAL_APPROVAL_SUBMISSION_STATUS_SUPPORT_LEVEL,
    ExternalApprovalService,
    ExternalApprovalSubmissionService,
    external_approval_submission_adapter_status,
)
from quant_agent_runtime.governance import (
    ALL_ROUTES,
    ENVIRONMENT_POLICY_PACK_SUPPORT_LEVEL,
    EXTERNAL_APPROVAL_ENFORCEMENT_SUPPORT_LEVEL,
    RELEASE_EVIDENCE_SUPPORT_LEVEL,
    GovernanceService,
)
from quant_agent_runtime.ledger import FileBackedLedger
from quant_agent_runtime.model_gateway import FakePlanProvider, ModelProvider, SharedLlmPlanProvider
from quant_agent_runtime.models import (
    AgentSupportBundleResult,
    ExternalApprovalDecisionRefreshRequest,
    ExternalApprovalDecisionRefreshResult,
    ExternalApprovalDecisionImportRequest,
    ExternalApprovalDecisionImportResult,
    ExternalApprovalPreviewRequest,
    ExternalApprovalPreviewResult,
    ExternalApprovalSubmissionRequest,
    ExternalApprovalSubmissionListResult,
    ExternalApprovalSubmissionResult,
    PlanValidationResult,
    ProviderMode,
    ProviderRuntimeStatus,
    RiskTier,
    RuntimeManifest,
    ValidationIssue,
)
from quant_agent_runtime.orchestration import OrchestrationService
from quant_agent_runtime.app_clients import LocalAgentAppClient
from quant_agent_runtime.capability_discovery import CapabilityDiscoveryService
from quant_agent_runtime.plan_revision import PlanRevisionService
from quant_agent_runtime.plan_revision_activation import PlanRevisionActivationService
from quant_agent_runtime.planner import PlannerService
from quant_agent_runtime.preflight import PreflightService
from quant_agent_runtime.provider_config import runtime_provider_status
from quant_agent_runtime.redaction import find_unsafe_payload_issues
from quant_agent_runtime.revalidation import RunRevalidationService
from quant_agent_runtime.retry import RetryService
from quant_agent_runtime.run_status import RunStatusService
from quant_agent_runtime.sample_autopilot import SampleAutopilotPreviewService, SampleAutopilotStepService
from quant_agent_runtime.sample_reset import SampleResetService
from quant_agent_runtime.user_workflow import UserWorkflowService
from quant_agent_runtime.validation.errors import RuntimeValidationError
from quant_agent_runtime.workflow_runner import (
    WORKFLOW_RUN_SUPPORT_LEVEL,
    WORKFLOW_TEMPLATE_SUPPORT_LEVEL,
    WorkflowRunService,
)


SUPPORT_BUNDLE_CONTRACT_SCHEMA = "agent_support_bundle.v1.schema.json"
SUPPORT_BUNDLE_SUPPORT_LEVEL = "redacted_run_bundle_json_v1"


@dataclass
class RuntimeContainer:
    planner: PlannerService
    preflight: PreflightService
    confirmation: ConfirmationService
    action_request: ActionRequestPreviewService
    execution: ExecutionService
    retry: RetryService
    run_status: RunStatusService
    orchestration: OrchestrationService
    plan_revision: PlanRevisionService
    plan_revision_activation: PlanRevisionActivationService
    revalidation: RunRevalidationService
    sample_autopilot: SampleAutopilotPreviewService
    sample_autopilot_step: SampleAutopilotStepService
    sample_reset: SampleResetService
    demo_narrative: DemoNarrativeService
    user_workflow: UserWorkflowService
    contract_loader: QuantSuiteContractLoader
    capability_discovery: CapabilityDiscoveryService
    provider_status: ProviderRuntimeStatus | None = None
    governance: GovernanceService | None = None
    external_approval: ExternalApprovalService | None = None
    external_approval_submission: ExternalApprovalSubmissionService | None = None
    workflow_runner: WorkflowRunService | None = None

    def support_bundle(self, run_id: str) -> AgentSupportBundleResult:
        manifest = self.manifest()
        run_status = self.run_status.get_run_status(run_id)
        orchestration = self.orchestration.get_run_orchestration(run_id)
        ledger_entry = self.run_status.get_ledger_entry(run_id)
        ledger_payload = ledger_entry.model_dump(mode="json")
        integrity_summary = (
            run_status.ledger_integrity_summary
            or self.planner.ledger.integrity_summary(run_id)
        )
        contract_result = self.contract_loader.load_agent_contracts()
        bundle = AgentSupportBundleResult(
            bundle_id=f"support_bundle_{uuid4().hex[:12]}",
            run_id=run_id,
            generated_at_utc=_utc_now_label(),
            runtime_summary={
                "service_name": manifest.service_name,
                "runtime_version": manifest.runtime_version,
                "contract_source": manifest.contract_source,
                "canonical_agent_contracts_loaded": manifest.canonical_agent_contracts_loaded,
                "execution_supported": manifest.execution_supported,
                "execution_support_level": manifest.execution_support_level,
                "ledger_support_level": manifest.ledger_support_level,
                "ledger_integrity_support_level": manifest.ledger_integrity_support_level,
                "support_bundle_support_level": manifest.support_bundle_support_level,
                "plan_only_mode": manifest.plan_only_mode,
                "safety_boundaries": manifest.safety_boundaries,
            },
            provider_summary=manifest.provider_status.model_dump(mode="json"),
            governance_summary=manifest.governance_summary,
            separation_of_duties_summary=manifest.separation_of_duties_summary,
            run_status=run_status,
            orchestration=orchestration,
            ledger=ledger_payload,
            ledger_integrity_summary=integrity_summary,
            contract_summary={
                "support_bundle_contract": SUPPORT_BUNDLE_CONTRACT_SCHEMA,
                "ledger_contract": "agent_execution_ledger.v1.schema.json",
                "loaded_agent_contract_count": len(contract_result.loaded_agent_contracts),
                "canonical_agent_contracts_loaded": contract_result.canonical_agent_contracts_loaded,
            },
            redaction_report={
                "data_policy": "summaries_and_references_only",
                "raw_payloads_included": False,
                "unsafe_issue_count": 0,
                "excluded_categories": [
                    "raw_rows",
                    "raw_paths",
                    "urls",
                    "bucket_names",
                    "secrets",
                    "credentials",
                    "raw_prompts",
                    "raw_provider_responses",
                    "raw_app_payloads",
                ],
            },
            validation=PlanValidationResult(status="valid"),
        )
        payload = bundle.model_dump(mode="json")
        unsafe_issues = find_unsafe_payload_issues(payload, root="support_bundle")
        if unsafe_issues:
            raise RuntimeValidationError(
                PlanValidationResult(
                    status="rejected",
                    errors=[
                        issue.model_copy(update={"code": "unsafe_support_bundle_payload"})
                        for issue in unsafe_issues
                    ],
                )
            )
        try:
            self.contract_loader.validate_agent_contract_payload(
                payload,
                SUPPORT_BUNDLE_CONTRACT_SCHEMA,
            )
        except Exception as exc:
            raise RuntimeValidationError(
                PlanValidationResult(
                    status="rejected",
                    errors=[
                        ValidationIssue(
                            code="support_bundle_contract_validation_failed",
                            message="The generated support bundle did not validate against the canonical contract.",
                        )
                    ],
                )
            ) from exc
        return bundle

    def preview_external_approval_request(
        self,
        request: ExternalApprovalPreviewRequest,
    ) -> ExternalApprovalPreviewResult:
        run_status = self.run_status.get_run_status(request.run_id)
        orchestration = self.orchestration.get_run_orchestration(request.run_id)
        support_bundle = self.support_bundle(request.run_id)
        service = self.external_approval or ExternalApprovalService(
            ledger=self.planner.ledger,
            contract_loader=self.contract_loader,
            governance=self.governance,
        )
        return service.preview_request(
            request,
            run_status=run_status,
            orchestration=orchestration,
            support_bundle=support_bundle,
        )

    def import_external_approval_decision(
        self,
        request: ExternalApprovalDecisionImportRequest,
    ) -> ExternalApprovalDecisionImportResult:
        run_status = self.run_status.get_run_status(request.run_id)
        orchestration = self.orchestration.get_run_orchestration(request.run_id)
        service = self.external_approval or ExternalApprovalService(
            ledger=self.planner.ledger,
            contract_loader=self.contract_loader,
            governance=self.governance,
        )
        return service.import_decision(
            request,
            run_status=run_status,
            orchestration=orchestration,
        )

    def refresh_external_approval_decision(
        self,
        request: ExternalApprovalDecisionRefreshRequest,
    ) -> ExternalApprovalDecisionRefreshResult:
        run_status = self.run_status.get_run_status(request.run_id)
        orchestration = self.orchestration.get_run_orchestration(request.run_id)
        service = self.external_approval or ExternalApprovalService(
            ledger=self.planner.ledger,
            contract_loader=self.contract_loader,
            governance=self.governance,
        )
        return service.refresh_decision(
            request,
            run_status=run_status,
            orchestration=orchestration,
        )

    def submit_external_approval_request(
        self,
        request: ExternalApprovalSubmissionRequest,
    ) -> ExternalApprovalSubmissionResult:
        run_status = self.run_status.get_run_status(request.run_id)
        orchestration = self.orchestration.get_run_orchestration(request.run_id)
        service = self.external_approval_submission or ExternalApprovalSubmissionService(
            ledger=self.planner.ledger,
            contract_loader=self.contract_loader,
            governance=self.governance,
        )
        return service.submit_request(
            request,
            run_status=run_status,
            orchestration=orchestration,
        )

    def list_external_approval_submissions(self, run_id: str) -> ExternalApprovalSubmissionListResult:
        service = self.external_approval_submission or ExternalApprovalSubmissionService(
            ledger=self.planner.ledger,
            contract_loader=self.contract_loader,
            governance=self.governance,
        )
        return service.list_submissions(run_id)

    def workflow_service(self) -> WorkflowRunService:
        if self.workflow_runner is None:
            self.workflow_runner = WorkflowRunService(
                planner=self.planner,
                ledger=self.planner.ledger,
                contract_loader=self.contract_loader,
                run_status=self.run_status,
                orchestration=self.orchestration,
                preflight=self.preflight,
                confirmation=self.confirmation,
                action_request=self.action_request,
                execution=self.execution,
                retry=self.retry,
                governance=self.governance,
            )
        return self.workflow_runner

    def manifest(self) -> RuntimeManifest:
        contract_result = self.contract_loader.load_agent_contracts()
        provider_status = self.provider_status or self.contract_loader.load_agent_provider_status()
        governance = self.governance or GovernanceService.local_fallback(ledger=self.planner.ledger)
        governance_summary = governance.manifest_summary()
        separation_of_duties_summary = governance.separation_of_duties_manifest_summary()
        external_approval_enforcement_summary = governance.external_approval_enforcement_manifest_summary()
        canonical_capabilities = self.contract_loader.load_agent_capabilities()
        capabilities = canonical_capabilities or default_capabilities()
        discovery_result = self.capability_discovery.discover(canonical_capabilities)
        loaded_contracts = (
            contract_result.loaded_agent_contracts
            if contract_result.canonical_agent_contracts_loaded
            else TEMPORARY_AGENT_CONTRACT_FIXTURES
        )
        contract_versions = [_contract_version_from_name(name) for name in loaded_contracts]
        return RuntimeManifest(
            service_name="quant-agent-runtime",
            runtime_version=__version__,
            supported_quant_suite_contract_versions=contract_versions,
            plan_only_mode=False,
            execution_supported=True,
            supported_routes=[
                *ALL_ROUTES,
            ],
            supported_provider_modes=[
                ProviderMode.fake_provider,
                ProviderMode.disabled_or_local_fallback,
                ProviderMode.openai,
                ProviderMode.ollama,
            ],
            supported_model_roles=["planner"],
            supported_risk_tiers=[
                RiskTier.read_only,
                RiskTier.draft_only,
                RiskTier.reversible_write,
                RiskTier.workflow_preflight,
                RiskTier.expensive_compute,
                RiskTier.artifact_export,
            ],
            policy_version="internal-policy.v0",
            runtime_health_endpoint="/health",
            capability_discovery_endpoints=[
                "quant_data:/api/agent/capabilities",
                "quant_studio:/api/agent/capabilities",
                "quant_documentation:/api/agent/capabilities",
                "quant_monitoring:/api/agent/capabilities",
            ],
            capability_discovery=discovery_result.diagnostics,
            supported_preflight_capabilities=discovery_result.supported_preflight_capabilities,
            supported_execution_capabilities=discovery_result.supported_execution_capabilities,
            ledger_support_level="local_json_file_backed",
            ledger_storage=self.planner.ledger.diagnostics(),
            ledger_integrity_support_level=self.planner.ledger.diagnostics().get(
                "integrity_support_level",
                "not_available",
            ),
            support_bundle_support_level=SUPPORT_BUNDLE_SUPPORT_LEVEL,
            external_approval_support_level=EXTERNAL_APPROVAL_SUPPORT_LEVEL,
            external_approval_decision_support_level=EXTERNAL_APPROVAL_DECISION_SUPPORT_LEVEL,
            external_approval_enforcement_support_level=EXTERNAL_APPROVAL_ENFORCEMENT_SUPPORT_LEVEL,
            external_approval_submission_support_level=EXTERNAL_APPROVAL_SUBMISSION_SUPPORT_LEVEL,
            external_approval_submission_status_support_level=EXTERNAL_APPROVAL_SUBMISSION_STATUS_SUPPORT_LEVEL,
            external_approval_decision_refresh_support_level=EXTERNAL_APPROVAL_DECISION_REFRESH_SUPPORT_LEVEL,
            external_approval_adapter_support_level=EXTERNAL_APPROVAL_ADAPTER_SUPPORT_LEVEL,
            external_approval_submission_adapter=external_approval_submission_adapter_status(),
            recovery_support_level="manual_pause_resume_only",
            orchestration_support_level="manual_guided_existing_steps_only",
            retry_support_level="manual_current_step_only",
            plan_revision_support_level="manual_preview_only",
            plan_revision_activation_support_level="manual_child_run_only",
            revalidation_support_level="manual_context_check_only",
            autopilot_support_level="sample_owned_one_step_manual_advance",
            sample_reset_support_level="sample_owned_studio_orchestrated_only",
            demo_narrative_support_level="sample_owned_ledger_narrative_only",
            governance_support_level=governance.support_level,
            separation_of_duties_support_level=governance.separation_of_duties_support_level,
            environment_policy_pack_support_level=ENVIRONMENT_POLICY_PACK_SUPPORT_LEVEL,
            release_evidence_support_level=RELEASE_EVIDENCE_SUPPORT_LEVEL,
            user_workflow_support_level="manual_user_owned_consent_gate_only",
            user_plan_approval_support_level="manual_active_plan_approval_only",
            workflow_run_support_level=WORKFLOW_RUN_SUPPORT_LEVEL,
            workflow_template_support_level=WORKFLOW_TEMPLATE_SUPPORT_LEVEL,
            supported_workflow_scopes=[
                "full_lifecycle",
                "app_workflow",
                "stage_range",
                "capability_set",
            ],
            plan_only_support_level="supported",
            execution_support_level="single_step_review_draft_actions_only",
            redaction_support_level="deterministic_context_redaction",
            validation_gates=[
                "provider_output_schema_validation",
                "capability_registry_validation",
                "policy_validation",
                "unsafe_context_rejection",
                "safe_ledger_scan",
                "action_request_contract_validation",
                "action_result_contract_validation",
            ],
            contract_source=contract_result.source_label,
            canonical_agent_contracts_loaded=contract_result.canonical_agent_contracts_loaded,
            loaded_agent_contracts=loaded_contracts,
            temporary_internal_contract_fixtures=not contract_result.canonical_agent_contracts_loaded,
            provider_status=provider_status,
            governance_summary=governance_summary,
            separation_of_duties_summary=separation_of_duties_summary,
            external_approval_enforcement_summary=external_approval_enforcement_summary,
            safety_boundaries=[
                "single_step_review_draft_execution_only",
                "no_generic_execution",
                "model_provider_planning_only",
                "server_side_provider_boundary",
                "app_owned_preflight_only",
                "confirmation_required_before_execution",
                "sanitized_context_only",
                "safe_ledger_only",
            ],
        )


def _contract_version_from_name(name: str) -> str:
    if name.endswith(".schema.json"):
        return name.removesuffix(".schema.json")
    return name


def _utc_now_label() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_runtime() -> RuntimeContainer:
    contract_loader = QuantSuiteContractLoader()
    ledger = FileBackedLedger(validate_contract=contract_loader.validate_agent_contract_payload)
    canonical_capabilities = contract_loader.load_agent_capabilities()
    provider_status = runtime_provider_status(
        base_status=contract_loader.load_agent_provider_status(),
    )
    model_provider = _planning_provider(provider_status)
    planner = PlannerService(
        provider=model_provider,
        ledger=ledger,
        default_capabilities=canonical_capabilities or None,
    )
    app_client = LocalAgentAppClient.from_environment()
    capability_discovery = CapabilityDiscoveryService(
        contract_loader=contract_loader,
        app_client=app_client,
    )
    governance = GovernanceService.from_contracts(
        ledger=ledger,
        contract_loader=contract_loader,
    )
    preflight = PreflightService(
        ledger=ledger,
        contract_loader=contract_loader,
        app_client=app_client,
        capability_discovery=capability_discovery,
    )
    confirmation = ConfirmationService(ledger=ledger)
    action_request = ActionRequestPreviewService(ledger=ledger, contract_loader=contract_loader)
    execution = ExecutionService(
        ledger=ledger,
        contract_loader=contract_loader,
        app_client=app_client,
        capability_discovery=capability_discovery,
    )
    retry = RetryService(
        ledger=ledger,
        execution=execution,
        app_client=app_client,
    )
    run_status = RunStatusService(
        ledger=ledger,
        capability_discovery=capability_discovery,
        governance=governance,
    )
    orchestration = OrchestrationService(
        ledger=ledger,
        governance=governance,
        capability_discovery=capability_discovery,
    )
    plan_revision = PlanRevisionService(
        provider=model_provider,
        ledger=ledger,
        contract_loader=contract_loader,
        default_capabilities=canonical_capabilities or None,
    )
    plan_revision_activation = PlanRevisionActivationService(
        ledger=ledger,
        contract_loader=contract_loader,
    )
    revalidation = RunRevalidationService(ledger=ledger)
    sample_autopilot = SampleAutopilotPreviewService(ledger=ledger)
    sample_autopilot_step = SampleAutopilotStepService(
        ledger=ledger,
        preflight=preflight,
        action_request=action_request,
        execution=execution,
    )
    sample_reset = SampleResetService(
        ledger=ledger,
        app_client=app_client,
    )
    demo_narrative = DemoNarrativeService(ledger=ledger)
    user_workflow = UserWorkflowService(ledger=ledger)
    external_approval = ExternalApprovalService(
        ledger=ledger,
        contract_loader=contract_loader,
        governance=governance,
    )
    external_approval_submission = ExternalApprovalSubmissionService(
        ledger=ledger,
        contract_loader=contract_loader,
        governance=governance,
    )
    workflow_runner = WorkflowRunService(
        planner=planner,
        ledger=ledger,
        contract_loader=contract_loader,
        run_status=run_status,
        orchestration=orchestration,
        preflight=preflight,
        confirmation=confirmation,
        action_request=action_request,
        execution=execution,
        retry=retry,
        governance=governance,
    )
    return RuntimeContainer(
        planner=planner,
        preflight=preflight,
        confirmation=confirmation,
        action_request=action_request,
        execution=execution,
        retry=retry,
        run_status=run_status,
        orchestration=orchestration,
        plan_revision=plan_revision,
        plan_revision_activation=plan_revision_activation,
        revalidation=revalidation,
        sample_autopilot=sample_autopilot,
        sample_autopilot_step=sample_autopilot_step,
        sample_reset=sample_reset,
        demo_narrative=demo_narrative,
        user_workflow=user_workflow,
        contract_loader=contract_loader,
        capability_discovery=capability_discovery,
        provider_status=provider_status,
        governance=governance,
        external_approval=external_approval,
        external_approval_submission=external_approval_submission,
        workflow_runner=workflow_runner,
    )


def _planning_provider(provider_status: ProviderRuntimeStatus) -> ModelProvider:
    if provider_status.effective_provider_mode in {ProviderMode.openai, ProviderMode.ollama}:
        return SharedLlmPlanProvider(provider_status=provider_status)
    return FakePlanProvider(provider_status=provider_status)
