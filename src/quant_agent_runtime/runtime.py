from __future__ import annotations

from dataclasses import dataclass

from quant_agent_runtime import __version__
from quant_agent_runtime.action_request import ActionRequestPreviewService
from quant_agent_runtime.capabilities import default_capabilities
from quant_agent_runtime.confirmation import ConfirmationService
from quant_agent_runtime.contracts import QuantSuiteContractLoader
from quant_agent_runtime.contracts.internal_test_fixtures import TEMPORARY_AGENT_CONTRACT_FIXTURES
from quant_agent_runtime.execution import ExecutionService
from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.model_gateway import FakePlanProvider
from quant_agent_runtime.models import ProviderMode, ProviderRuntimeStatus, RiskTier, RuntimeManifest
from quant_agent_runtime.app_clients import LocalAgentAppClient
from quant_agent_runtime.capability_discovery import CapabilityDiscoveryService
from quant_agent_runtime.planner import PlannerService
from quant_agent_runtime.preflight import PreflightService


@dataclass
class RuntimeContainer:
    planner: PlannerService
    preflight: PreflightService
    confirmation: ConfirmationService
    action_request: ActionRequestPreviewService
    execution: ExecutionService
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
            ],
            supported_provider_modes=[
                ProviderMode.fake_provider,
                ProviderMode.disabled_or_local_fallback,
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
                "quant_monitoring:/api/agent/capabilities",
            ],
            capability_discovery=discovery_result.diagnostics,
            supported_preflight_capabilities=discovery_result.supported_preflight_capabilities,
            supported_execution_capabilities=discovery_result.supported_execution_capabilities,
            ledger_support_level="plan_preflight_confirmation_execution_in_memory",
            plan_only_support_level="supported",
            execution_support_level="single_step_studio_draft_only",
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
                "single_step_studio_draft_execution_only",
                "no_generic_execution",
                "no_real_provider",
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
    ledger = InMemoryLedger()
    contract_loader = QuantSuiteContractLoader()
    canonical_capabilities = contract_loader.load_agent_capabilities()
    provider_status = contract_loader.load_agent_provider_status()
    planner = PlannerService(
        provider=FakePlanProvider(provider_status=provider_status),
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
    return RuntimeContainer(
        planner=planner,
        preflight=preflight,
        confirmation=confirmation,
        action_request=action_request,
        execution=execution,
        contract_loader=contract_loader,
        capability_discovery=capability_discovery,
        provider_status=provider_status,
    )
