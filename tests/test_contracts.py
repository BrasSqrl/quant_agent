from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from quant_agent_runtime.api import create_app
from quant_agent_runtime.capabilities import default_capabilities
from quant_agent_runtime.contracts import QuantSuiteContractLoader
from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.model_gateway import FakePlanProvider, ModelProvider, ProviderPlanRequest, ProviderResult
from quant_agent_runtime.models import PlanRequest, ProviderMetadata
from quant_agent_runtime.planner import PlannerService
from quant_agent_runtime.runtime import RuntimeContainer
from quant_agent_runtime.validation.errors import RuntimeValidationError


AGENT_ROOT = Path(__file__).resolve().parents[1]
QUANT_SUITE_ROOT = AGENT_ROOT.parent / "quant_suite"
CONTRACTS_DIR = QUANT_SUITE_ROOT / "contracts"
EXPECTED_AGENT_CONTRACTS = {
    "agent_capability.v1.schema.json",
    "agent_execution_ledger.v1.schema.json",
    "agent_plan.v1.schema.json",
    "agent_policy.v1.schema.json",
    "agent_provider_config.v1.schema.json",
    "agent_runtime_manifest.v1.schema.json",
}


class StaticProvider(ModelProvider):
    def __init__(self, raw_output: dict[str, object]) -> None:
        self.raw_output = raw_output

    def generate_plan(self, request: ProviderPlanRequest) -> ProviderResult:
        return ProviderResult(
            raw_output=self.raw_output,
            metadata=ProviderMetadata(
                provider="test",
                model="static",
                provider_mode=request.policy.provider_mode,
                supports_execution=False,
            ),
        )


def valid_provider_output() -> dict[str, object]:
    return {
        "user_goal_summary": "safe goal",
        "assumptions": [],
        "missing_inputs": [],
        "steps": [
            {
                "step_id": "step_1",
                "title": "Inspect lifecycle",
                "capability_id": "quant_suite.inspect_lifecycle_context",
                "app_id": "quant_suite",
                "risk_tier": "read_only",
                "operation": "plan",
                "requires_confirmation": False,
                "action_input": {"lifecycle_summary": "safe summary"},
                "expected_artifacts": [],
                "validation_checks": ["capability_known"],
            }
        ],
    }


def load_suite_validator():
    validator_path = QUANT_SUITE_ROOT / "scripts" / "validate_contracts.py"
    if not validator_path.is_file():
        pytest.fail(f"Quant Suite validator was not found at {validator_path}")
    spec = importlib.util.spec_from_file_location("quant_suite_validate_contracts", validator_path)
    if spec is None or spec.loader is None:
        pytest.fail("Quant Suite validator could not be loaded.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_schema(schema_name: str) -> dict[str, object]:
    schema_path = CONTRACTS_DIR / schema_name
    if not schema_path.is_file():
        pytest.fail(f"Expected canonical contract was not found: {schema_path}")
    return load_suite_validator().load_json(schema_path)


def validate_against_contract(payload: object, schema_name: str) -> None:
    validator = load_suite_validator()
    schema = load_schema(schema_name)
    validator.validate_schema(payload, schema)


def runtime_with_loader(loader: QuantSuiteContractLoader) -> RuntimeContainer:
    capabilities = loader.load_agent_capabilities()
    return RuntimeContainer(
        planner=PlannerService(
            provider=FakePlanProvider(),
            ledger=InMemoryLedger(),
            default_capabilities=capabilities or None,
        ),
        contract_loader=loader,
    )


def test_canonical_agent_contracts_are_discovered_from_quant_suite() -> None:
    result = QuantSuiteContractLoader(QUANT_SUITE_ROOT).load_agent_contracts()

    assert result.canonical_agent_contracts_loaded is True
    assert EXPECTED_AGENT_CONTRACTS.issubset(set(result.loaded_agent_contracts))


def test_canonical_capability_example_is_loaded_and_mapped() -> None:
    capabilities = QuantSuiteContractLoader(QUANT_SUITE_ROOT).load_agent_capabilities()

    assert [item.capability_id for item in capabilities] == [
        "quant_data.run_source_preflight",
        "quant_studio.prepare_model_config_draft",
        "quant_documentation.inspect_package",
        "quant_monitoring.validate_bundle",
    ]
    assert capabilities[1].confirmation_required is True
    assert capabilities[1].required_fields == ["target_summary"]


def test_internal_default_capabilities_remain_available_when_canonical_contracts_are_absent() -> None:
    missing_suite_root = AGENT_ROOT / "__missing_quant_suite_for_contract_test__"
    capabilities = QuantSuiteContractLoader(missing_suite_root).load_agent_capabilities()

    assert capabilities == []
    assert default_capabilities()[0].capability_id == "quant_suite.inspect_lifecycle_context"


def test_internal_fixtures_are_used_only_when_canonical_contracts_are_absent() -> None:
    missing_suite_root = AGENT_ROOT / "__missing_quant_suite_for_contract_test__"
    runtime = runtime_with_loader(QuantSuiteContractLoader(missing_suite_root))

    manifest = runtime.manifest()

    assert manifest.canonical_agent_contracts_loaded is False
    assert manifest.temporary_internal_contract_fixtures is True
    assert manifest.loaded_agent_contracts
    assert all(name.startswith("internal.") for name in manifest.loaded_agent_contracts)


def test_runtime_manifest_validates_against_canonical_contract() -> None:
    runtime = runtime_with_loader(QuantSuiteContractLoader(QUANT_SUITE_ROOT))
    client = TestClient(create_app(runtime))

    response = client.get("/runtime/manifest")

    assert response.status_code == 200
    payload = response.json()
    assert payload["canonical_agent_contracts_loaded"] is True
    assert payload["temporary_internal_contract_fixtures"] is False
    validate_against_contract(payload, "agent_runtime_manifest.v1.schema.json")


def test_fake_provider_plan_validates_against_agent_plan_contract() -> None:
    runtime = runtime_with_loader(QuantSuiteContractLoader(QUANT_SUITE_ROOT))
    client = TestClient(create_app(runtime))

    response = client.post(
        "/plans",
        json={
            "user_goal": "Build a conservative PD scorecard plan.",
            "context_summary": {
                "lifecycle_summary": "Lifecycle exists.",
                "source_summary": "Development sample is registered.",
                "target_summary": "Default flag is the candidate target.",
                "package_summary": "No documentation package exists yet.",
                "bundle_summary": "Monitoring bundle is not available.",
            },
        },
    )

    assert response.status_code == 200
    plan = response.json()["plan"]
    assert [step["app_id"] for step in plan["proposed_steps"]] == [
        "quant_data",
        "quant_studio",
        "quant_documentation",
        "quant_monitoring",
    ]
    validate_against_contract(plan, "agent_plan.v1.schema.json")


def test_missing_context_fields_still_produce_schema_valid_blocked_plan() -> None:
    runtime = runtime_with_loader(QuantSuiteContractLoader(QUANT_SUITE_ROOT))
    client = TestClient(create_app(runtime))

    response = client.post(
        "/plans",
        json={
            "user_goal": "Build a conservative PD scorecard plan.",
            "context_summary": {},
        },
    )

    assert response.status_code == 200
    plan = response.json()["plan"]
    assert plan["status"] == "blocked"
    assert "quant_data.run_source_preflight requires source_summary." in plan["missing_inputs"]
    assert plan["proposed_steps"][0]["action_input"]["source_summary"] == "[missing]"
    validate_against_contract(plan, "agent_plan.v1.schema.json")


def test_ledger_entry_validates_against_agent_execution_ledger_contract() -> None:
    planner = PlannerService(provider=FakePlanProvider(), ledger=InMemoryLedger())

    planner.create_plan(
        PlanRequest(
            user_goal="Plan from safe summaries.",
            context_summary={
                "lifecycle_summary": "Lifecycle exists.",
                "source_summary": "Source summary only.",
                "target_summary": "Target summary only.",
                "package_summary": "Package summary only.",
                "bundle_summary": "Bundle summary only.",
            },
        )
    )

    entry = planner.ledger.list_entries()[0].model_dump(mode="json")
    validate_against_contract(entry, "agent_execution_ledger.v1.schema.json")


def test_policy_rejection_ledger_validates_against_agent_execution_ledger_contract() -> None:
    planner = PlannerService(provider=StaticProvider(valid_provider_output()), ledger=InMemoryLedger())

    with pytest.raises(RuntimeValidationError):
        planner.create_plan(
            PlanRequest(
                user_goal="Plan from safe summaries.",
                context_summary={"lifecycle_summary": "Lifecycle exists."},
                policy={"forbidden_action_ids": ["quant_suite.inspect_lifecycle_context"]},
            )
        )

    entry = planner.ledger.list_entries()[0].model_dump(mode="json")
    assert entry["validation_results"]["status"] == "rejected"
    assert entry["policy_rejections"]
    validate_against_contract(entry, "agent_execution_ledger.v1.schema.json")
