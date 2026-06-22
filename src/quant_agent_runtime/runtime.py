from __future__ import annotations

from dataclasses import dataclass

from quant_agent_runtime import __version__
from quant_agent_runtime.contracts import QuantSuiteContractLoader
from quant_agent_runtime.contracts.internal_test_fixtures import TEMPORARY_AGENT_CONTRACT_FIXTURES
from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.model_gateway import FakePlanProvider
from quant_agent_runtime.models import ProviderMode, RiskTier, RuntimeManifest
from quant_agent_runtime.planner import PlannerService


@dataclass
class RuntimeContainer:
    planner: PlannerService
    contract_loader: QuantSuiteContractLoader

    def manifest(self) -> RuntimeManifest:
        contract_result = self.contract_loader.load_agent_contracts()
        loaded_contracts = (
            contract_result.loaded_agent_contracts
            if contract_result.canonical_agent_contracts_loaded
            else TEMPORARY_AGENT_CONTRACT_FIXTURES
        )
        return RuntimeManifest(
            service_name="quant-agent-runtime",
            runtime_version=__version__,
            plan_only_mode=True,
            execution_supported=False,
            supported_routes=[
                "GET /health",
                "GET /runtime/manifest",
                "POST /plans",
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
            contract_source=contract_result.source_label,
            canonical_agent_contracts_loaded=contract_result.canonical_agent_contracts_loaded,
            loaded_agent_contracts=loaded_contracts,
            temporary_internal_contract_fixtures=not contract_result.canonical_agent_contracts_loaded,
            safety_boundaries=[
                "plan_only",
                "no_app_execution",
                "no_real_provider",
                "server_side_provider_boundary",
                "sanitized_context_only",
                "safe_ledger_only",
            ],
        )


def build_runtime() -> RuntimeContainer:
    ledger = InMemoryLedger()
    planner = PlannerService(provider=FakePlanProvider(), ledger=ledger)
    return RuntimeContainer(
        planner=planner,
        contract_loader=QuantSuiteContractLoader(),
    )
