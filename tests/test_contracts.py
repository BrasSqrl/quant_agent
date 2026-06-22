from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from quant_agent_runtime.action_request import ActionRequestPreviewService
from quant_agent_runtime.api import create_app
from quant_agent_runtime.capabilities import default_capabilities
from quant_agent_runtime.capability_discovery import CapabilityDiscoveryService
from quant_agent_runtime.confirmation import ConfirmationService
from quant_agent_runtime.contracts import QuantSuiteContractLoader
from quant_agent_runtime.execution import ExecutionService
from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.model_gateway import FakePlanProvider, ModelProvider, ProviderPlanRequest, ProviderResult
from quant_agent_runtime.models import (
    LedgerEntry,
    PlanRequest,
    PlanValidationResult,
    ProviderMetadata,
    ProviderMode,
    RedactionSummary,
)
from quant_agent_runtime.preflight import PreflightService
from quant_agent_runtime.planner import PlannerService
from quant_agent_runtime.runtime import RuntimeContainer
from quant_agent_runtime.validation.errors import RuntimeValidationError


AGENT_ROOT = Path(__file__).resolve().parents[1]
QUANT_SUITE_ROOT = AGENT_ROOT.parent / "quant_suite"
CONTRACTS_DIR = QUANT_SUITE_ROOT / "contracts"
EXPECTED_AGENT_CONTRACTS = {
    "agent_action_preflight.v1.schema.json",
    "agent_action_request.v1.schema.json",
    "agent_action_result.v1.schema.json",
    "agent_capability.v1.schema.json",
    "agent_execution_ledger.v1.schema.json",
    "agent_plan.v1.schema.json",
    "agent_policy.v1.schema.json",
    "agent_provider_config.v1.schema.json",
    "agent_runtime_manifest.v1.schema.json",
}
PHASE3_AGENT_EXAMPLES = [
    ("agent_action_preflight.v1.example.json", "agent_action_preflight.v1.schema.json"),
    ("agent_action_request.v1.example.json", "agent_action_request.v1.schema.json"),
    ("agent_action_result.v1.example.json", "agent_action_result.v1.schema.json"),
    ("agent_execution_ledger.v1.example.json", "agent_execution_ledger.v1.schema.json"),
]


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


class FakePreflightAppClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response

    def discover_capabilities(self, *, app_id: str) -> dict[str, object]:
        if app_id == "quant_data":
            capabilities = [
                {
                    "capability_id": "quant_data.run_source_preflight",
                    "app_id": "quant_data",
                    "risk_tier": "workflow_preflight",
                    "enabled": True,
                    "preflight_required": True,
                    "confirmation_required": False,
                }
            ]
        elif app_id == "quant_monitoring":
            capabilities = [
                {
                    "capability_id": "quant_monitoring.validate_bundle",
                    "app_id": "quant_monitoring",
                    "risk_tier": "workflow_preflight",
                    "enabled": True,
                    "preflight_required": True,
                    "confirmation_required": False,
                }
            ]
        elif app_id == "quant_studio":
            capabilities = [
                {
                    "capability_id": "quant_studio.prepare_model_config_draft",
                    "app_id": "quant_studio",
                    "risk_tier": "draft_only",
                    "enabled": True,
                    "preflight_required": False,
                    "confirmation_required": True,
                    "execution_supported": True,
                }
            ]
        else:
            capabilities = []
        return {
            "schema_version": "1.0",
            "data_policy": "summaries_and_references_only",
            "app_id": app_id,
            "capabilities": capabilities,
        }

    def create_preflight(
        self,
        *,
        app_id: str,
        capability_id: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        return dict(self.response)

    def execute_action(
        self,
        *,
        app_id: str,
        capability_id: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        return {}


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


def load_contract_example(example_name: str) -> object:
    example_path = CONTRACTS_DIR / example_name
    if not example_path.is_file():
        pytest.fail(f"Expected canonical example was not found: {example_path}")
    return load_suite_validator().load_json(example_path)


def runtime_with_loader(loader: QuantSuiteContractLoader) -> RuntimeContainer:
    capabilities = loader.load_agent_capabilities()
    provider_status = loader.load_agent_provider_status()
    ledger = InMemoryLedger()
    app_client = FakePreflightAppClient({})
    discovery = CapabilityDiscoveryService(contract_loader=loader, app_client=app_client)
    return RuntimeContainer(
        planner=PlannerService(
            provider=FakePlanProvider(provider_status=provider_status),
            ledger=ledger,
            default_capabilities=capabilities or None,
        ),
        preflight=PreflightService(
            ledger=ledger,
            contract_loader=loader,
            app_client=app_client,
            capability_discovery=discovery,
        ),
        confirmation=ConfirmationService(ledger=ledger),
        action_request=ActionRequestPreviewService(ledger=ledger, contract_loader=loader),
        execution=ExecutionService(
            ledger=ledger,
            contract_loader=loader,
            app_client=app_client,
            capability_discovery=discovery,
        ),
        contract_loader=loader,
        capability_discovery=discovery,
        provider_status=provider_status,
    )


def test_canonical_agent_contracts_are_discovered_from_quant_suite() -> None:
    result = QuantSuiteContractLoader(QUANT_SUITE_ROOT).load_agent_contracts()

    assert result.canonical_agent_contracts_loaded is True
    assert EXPECTED_AGENT_CONTRACTS.issubset(set(result.loaded_agent_contracts))


@pytest.mark.parametrize(("example_name", "schema_name"), PHASE3_AGENT_EXAMPLES)
def test_phase3_agent_examples_validate_against_canonical_contracts(
    example_name: str,
    schema_name: str,
) -> None:
    validate_against_contract(load_contract_example(example_name), schema_name)


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


def test_canonical_provider_config_example_is_loaded_and_mapped() -> None:
    status = QuantSuiteContractLoader(QUANT_SUITE_ROOT).load_agent_provider_status()

    assert status.config_source == "configured_path"
    assert status.configured_provider_mode == "disabled_or_local_fallback"
    assert status.effective_provider_mode == ProviderMode.disabled_or_local_fallback
    assert status.provider_identifier == "disabled"
    assert status.model_profile == "deterministic_plan_fixture"
    assert status.allowed_model_roles == ["planner"]
    assert status.configured is True
    assert status.supports_execution is False
    assert status.hosted_provider_enabled is False
    assert status.secret_reference_present is False
    assert status.secrets_exposed is False
    assert status.fallback_reason
    assert "cost_accounting_bucket" not in status.model_dump(mode="json")


def test_internal_default_capabilities_remain_available_when_canonical_contracts_are_absent() -> None:
    missing_suite_root = AGENT_ROOT / "__missing_quant_suite_for_contract_test__"
    capabilities = QuantSuiteContractLoader(missing_suite_root).load_agent_capabilities()

    assert capabilities == []
    assert default_capabilities()[0].capability_id == "quant_suite.inspect_lifecycle_context"


def test_internal_provider_status_is_used_when_canonical_config_is_absent() -> None:
    missing_suite_root = AGENT_ROOT / "__missing_quant_suite_for_contract_test__"
    status = QuantSuiteContractLoader(missing_suite_root).load_agent_provider_status()

    assert status.config_source == "internal_default"
    assert status.configured_provider_mode == "fake_provider"
    assert status.effective_provider_mode == ProviderMode.fake_provider
    assert status.provider_identifier == "fake"
    assert status.model_profile == "deterministic-plan-fixture"
    assert status.fallback_reason is None
    assert status.secret_reference_present is False
    assert status.secrets_exposed is False


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
    assert payload["provider_status"]["effective_provider_mode"] == "disabled_or_local_fallback"
    assert payload["provider_status"]["supports_execution"] is False
    assert payload["provider_status"]["hosted_provider_enabled"] is False
    assert payload["provider_status"]["secrets_exposed"] is False
    assert "cost_accounting_bucket" not in payload["provider_status"]
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
    validate_against_contract(
        response.json()["context_preview"],
        "assistant_context_preview.v1.schema.json",
    )


def test_disabled_provider_fallback_metadata_and_ledger_validate() -> None:
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
    payload = response.json()
    metadata = payload["provider_metadata"]
    assert metadata["provider"] == "disabled"
    assert metadata["model"] == "deterministic_plan_fixture"
    assert metadata["provider_mode"] == "disabled_or_local_fallback"
    assert metadata["config_source"] == "configured_path"
    assert metadata["configured_provider_mode"] == "disabled_or_local_fallback"
    assert metadata["supports_execution"] is False
    assert metadata["fallback_reason"]
    assert metadata["configuration_errors"] == []
    assert "cost_accounting_bucket" not in metadata
    validate_against_contract(payload["plan"], "agent_plan.v1.schema.json")

    entry = runtime.planner.ledger.list_entries()[0].model_dump(mode="json")
    assert entry["provider_metadata"]["provider_mode"] == "disabled_or_local_fallback"
    assert entry["provider_metadata"]["supports_execution"] is False
    validate_against_contract(entry, "agent_execution_ledger.v1.schema.json")


def test_plan_context_preview_omits_unsafe_request_context() -> None:
    runtime = runtime_with_loader(QuantSuiteContractLoader(QUANT_SUITE_ROOT))
    client = TestClient(create_app(runtime))

    response = client.post(
        "/plans",
        json={
            "user_goal": "Build a conservative PD scorecard plan.",
            "context_summary": {
                "source_summary": "Development sample is registered.",
                "target_summary": "Default flag is the candidate target.",
                "package_summary": "Documentation package is ready.",
                "bundle_summary": "Monitoring bundle is ready.",
                "links": [{"frontend_url": "http://127.0.0.1:5810/?query=raw"}],
                "query": {"run_id": "raw-route-query"},
                "records": [{"borrower_id": "A12345"}],
                "raw_path": "C:\\Users\\matth\\Desktop\\private\\raw.csv",
                "bucket_name": "private-bucket",
                "hidden_commands": ["rm -rf ."],
                "safe_note": "Open http://127.0.0.1:5810/?secret=value after review.",
            },
        },
    )

    assert response.status_code == 200
    preview = response.json()["context_preview"]
    dumped_preview = str(preview)
    assert "links" not in preview["context"]
    assert "query" not in preview["context"]
    assert "records" not in preview["context"]
    assert "raw_path" not in preview["context"]
    assert "bucket_name" not in preview["context"]
    assert "hidden_commands" not in preview["context"]
    assert preview["context"]["safe_note"] == "Open [redacted] after review."
    assert "records" in preview["omitted_row_level_fields"]
    assert "links" in preview["omitted_sensitive_fields"]
    assert "query" in preview["omitted_sensitive_fields"]
    assert "raw_path" in preview["omitted_sensitive_fields"]
    assert "bucket_name" in preview["omitted_sensitive_fields"]
    assert "hidden_commands" in preview["omitted_sensitive_fields"]
    assert "safe_note" in preview["omitted_sensitive_fields"]
    assert "A12345" not in dumped_preview
    assert "private-bucket" not in dumped_preview
    assert "rm -rf" not in dumped_preview
    assert "C:\\Users\\matth\\Desktop\\private\\raw.csv" not in dumped_preview
    assert "http://127.0.0.1:5810" not in dumped_preview
    validate_against_contract(preview, "assistant_context_preview.v1.schema.json")


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
    assert entry["context_preview"]["context"]["source_summary"] == "Source summary only."
    validate_against_contract(
        entry["context_preview"],
        "assistant_context_preview.v1.schema.json",
    )
    validate_against_contract(entry, "agent_execution_ledger.v1.schema.json")


def test_ledger_with_preflight_action_and_result_sections_validates() -> None:
    action_request = load_contract_example("agent_action_request.v1.example.json")
    preflight_record = load_contract_example("agent_action_preflight.v1.example.json")
    action_result = load_contract_example("agent_action_result.v1.example.json")
    entry = LedgerEntry(
        run_id="run_plan_only_contract_sections",
        user_goal_summary="Validate safe plan-only ledger sections.",
        provider_mode=ProviderMode.disabled_or_local_fallback,
        redaction_summary=RedactionSummary(),
        plan_snapshot={
            "plan_id": "plan_contract_sections",
            "status": "blocked",
            "execution_permitted": False,
        },
        preflight_records=[preflight_record],
        confirmation_records=[
            {
                "confirmation_id": "confirmation_model_config_draft_001",
                "step_id": "step_2",
                "capability_id": "quant_studio.prepare_model_config_draft",
                "status": "required",
                "reason": "Draft-only Studio changes require explicit review before execution.",
            }
        ],
        action_requests=[action_request],
        action_results=[action_result],
        validation_results=PlanValidationResult(status="valid"),
        policy_rejections=[],
    )

    snapshot = entry.model_dump(mode="json")
    dumped = str(snapshot)
    assert snapshot["action_requests"][0]["execution_permitted"] is False
    assert snapshot["action_results"][0]["execution_status"] == "not_executed"
    assert "raw.csv" not in dumped
    assert "private-bucket" not in dumped
    assert "rm -rf" not in dumped
    assert "provider_prompt" not in dumped
    assert "provider_response" not in dumped
    validate_against_contract(snapshot["action_requests"][0], "agent_action_request.v1.schema.json")
    validate_against_contract(
        snapshot["preflight_records"][0],
        "agent_action_preflight.v1.schema.json",
    )
    validate_against_contract(snapshot["action_results"][0], "agent_action_result.v1.schema.json")
    validate_against_contract(snapshot, "agent_execution_ledger.v1.schema.json")


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
    validate_against_contract(
        entry["context_preview"],
        "assistant_context_preview.v1.schema.json",
    )
    validate_against_contract(entry, "agent_execution_ledger.v1.schema.json")
