from __future__ import annotations

from dataclasses import dataclass

from quant_agent_runtime import __version__
from quant_agent_runtime.action_request import ActionRequestPreviewService
from quant_agent_runtime.capabilities import default_capabilities
from quant_agent_runtime.confirmation import ConfirmationService
from quant_agent_runtime.contracts import QuantSuiteContractLoader
from quant_agent_runtime.contracts.internal_test_fixtures import TEMPORARY_AGENT_CONTRACT_FIXTURES
from quant_agent_runtime.execution import ExecutionService
from quant_agent_runtime.ledger import FileBackedLedger
from quant_agent_runtime.model_gateway import FakePlanProvider, ModelProvider, SharedLlmPlanProvider
from quant_agent_runtime.models import ProviderMode, ProviderRuntimeStatus, RiskTier, RuntimeManifest
from quant_agent_runtime.orchestration import OrchestrationService
from quant_agent_runtime.app_clients import LocalAgentAppClient
from quant_agent_runtime.capability_discovery import CapabilityDiscoveryService
from quant_agent_runtime.plan_revision import PlanRevisionService
from quant_agent_runtime.plan_revision_activation import PlanRevisionActivationService
from quant_agent_runtime.planner import PlannerService
from quant_agent_runtime.preflight import PreflightService
from quant_agent_runtime.provider_config import runtime_provider_status
from quant_agent_runtime.revalidation import RunRevalidationService
from quant_agent_runtime.retry import RetryService
from quant_agent_runtime.run_status import RunStatusService
from quant_agent_runtime.sample_autopilot import SampleAutopilotPreviewService, SampleAutopilotStepService
from quant_agent_runtime.sample_reset import SampleResetService


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
    contract_loader: QuantSuiteContractLoader
    capability_discovery: CapabilityDiscoveryService
    provider_status: ProviderRuntimeStatus | None = None

    def manifest(self) -> RuntimeManifest:
        contract_result = self.contract_loader.load_agent_contracts()
        provider_status = self.provider_status or self.contract_loader.load_agent_provider_status()
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
                "GET /health",
                "GET /runtime/manifest",
                "POST /plans",
                "POST /preflights",
                "POST /confirmations",
                "POST /action-requests",
                "POST /executions",
                "POST /retries",
                "GET /runs",
                "GET /runs/{run_id}",
                "GET /runs/{run_id}/orchestration",
                "GET /runs/{run_id}/ledger",
                "POST /cancellations",
                "POST /pauses",
                "POST /resumptions",
                "POST /plan-revisions",
                "POST /plan-revision-activations",
                "POST /run-revalidations",
                "POST /autopilot-previews",
                "POST /autopilot-steps",
                "POST /sample-reset-previews",
                "POST /sample-resets",
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
            recovery_support_level="manual_pause_resume_only",
            orchestration_support_level="manual_guided_existing_steps_only",
            retry_support_level="manual_current_step_only",
            plan_revision_support_level="manual_preview_only",
            plan_revision_activation_support_level="manual_child_run_only",
            revalidation_support_level="manual_context_check_only",
            autopilot_support_level="sample_owned_one_step_manual_advance",
            sample_reset_support_level="sample_owned_studio_orchestrated_only",
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
    run_status = RunStatusService(ledger=ledger, capability_discovery=capability_discovery)
    orchestration = OrchestrationService(ledger=ledger)
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
        contract_loader=contract_loader,
        capability_discovery=capability_discovery,
        provider_status=provider_status,
    )


def _planning_provider(provider_status: ProviderRuntimeStatus) -> ModelProvider:
    if provider_status.effective_provider_mode in {ProviderMode.openai, ProviderMode.ollama}:
        return SharedLlmPlanProvider(provider_status=provider_status)
    return FakePlanProvider(provider_status=provider_status)
