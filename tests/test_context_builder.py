from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from quant_agent_runtime.action_request import ActionRequestPreviewService
from quant_agent_runtime.api import create_app
from quant_agent_runtime.capability_discovery import CapabilityDiscoveryService
from quant_agent_runtime.confirmation import ConfirmationService
from quant_agent_runtime.context_builder import LifecycleContextBuilder
from quant_agent_runtime.contracts import QuantSuiteContractLoader
from quant_agent_runtime.demo_narrative import DemoNarrativeService
from quant_agent_runtime.execution import ExecutionService
from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.model_gateway import FakePlanProvider
from quant_agent_runtime.orchestration import OrchestrationService
from quant_agent_runtime.plan_revision import PlanRevisionService
from quant_agent_runtime.plan_revision_activation import PlanRevisionActivationService
from quant_agent_runtime.planner import PlannerService
from quant_agent_runtime.preflight import PreflightService
from quant_agent_runtime.revalidation import RunRevalidationService
from quant_agent_runtime.retry import RetryService
from quant_agent_runtime.runtime import RuntimeContainer
from quant_agent_runtime.run_status import RunStatusService
from quant_agent_runtime.sample_autopilot import SampleAutopilotPreviewService, SampleAutopilotStepService
from quant_agent_runtime.sample_reset import SampleResetService
from quant_agent_runtime.user_workflow import UserWorkflowService


AGENT_ROOT = Path(__file__).resolve().parents[1]
QUANT_SUITE_ROOT = AGENT_ROOT.parent / "quant_suite"
CONTRACTS_DIR = QUANT_SUITE_ROOT / "contracts"
LIFECYCLE_FIXTURE_PATH = (
    QUANT_SUITE_ROOT
    / "fixtures"
    / "sample_workspaces"
    / "credit_pd_scorecard_panel"
    / "quant_lifecycle_manifest.v1.json"
)


def load_suite_validator() -> Any:
    validator_path = QUANT_SUITE_ROOT / "scripts" / "validate_contracts.py"
    if not validator_path.is_file():
        pytest.fail(f"Quant Suite validator was not found at {validator_path}")
    spec = importlib.util.spec_from_file_location(
        "quant_suite_validate_contracts_context_builder",
        validator_path,
    )
    if spec is None or spec.loader is None:
        pytest.fail("Quant Suite validator could not be loaded.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_against_contract(payload: object, schema_name: str) -> None:
    schema_path = CONTRACTS_DIR / schema_name
    if not schema_path.is_file():
        pytest.fail(f"Expected canonical contract was not found: {schema_path}")
    validator = load_suite_validator()
    schema = validator.load_json(schema_path)
    validator.validate_schema(payload, schema)


def runtime_with_canonical_capabilities() -> RuntimeContainer:
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    capabilities = loader.load_agent_capabilities()
    ledger = InMemoryLedger()
    app_client = FakePreflightAppClient()
    discovery = CapabilityDiscoveryService(contract_loader=loader, app_client=app_client)
    execution = ExecutionService(
        ledger=ledger,
        contract_loader=loader,
        app_client=app_client,
        capability_discovery=discovery,
    )
    preflight = PreflightService(
        ledger=ledger,
        contract_loader=loader,
        app_client=app_client,
        capability_discovery=discovery,
    )
    action_request = ActionRequestPreviewService(ledger=ledger, contract_loader=loader)
    return RuntimeContainer(
        planner=PlannerService(
            provider=FakePlanProvider(),
            ledger=ledger,
            default_capabilities=capabilities or None,
        ),
        preflight=preflight,
        confirmation=ConfirmationService(ledger=ledger),
        action_request=action_request,
        execution=execution,
        retry=RetryService(ledger=ledger, execution=execution, app_client=app_client),
        run_status=RunStatusService(ledger=ledger),
        orchestration=OrchestrationService(ledger=ledger),
        plan_revision=PlanRevisionService(
            provider=FakePlanProvider(),
            ledger=ledger,
            contract_loader=loader,
            default_capabilities=capabilities or None,
        ),
        plan_revision_activation=PlanRevisionActivationService(
            ledger=ledger,
            contract_loader=loader,
        ),
        revalidation=RunRevalidationService(ledger=ledger),
        sample_autopilot=SampleAutopilotPreviewService(
            ledger=ledger,
            sample_workspace_root=QUANT_SUITE_ROOT / "fixtures" / "sample_workspaces",
        ),
        sample_autopilot_step=SampleAutopilotStepService(
            ledger=ledger,
            preflight=preflight,
            action_request=action_request,
            execution=execution,
        ),
        sample_reset=SampleResetService(
            ledger=ledger,
            app_client=app_client,
            sample_workspace_root=QUANT_SUITE_ROOT / "fixtures" / "sample_workspaces",
        ),
        demo_narrative=DemoNarrativeService(
            ledger=ledger,
            sample_workspace_root=QUANT_SUITE_ROOT / "fixtures" / "sample_workspaces",
        ),
        user_workflow=UserWorkflowService(ledger=ledger),
        contract_loader=loader,
        capability_discovery=discovery,
    )


class FakePreflightAppClient:
    def discover_capabilities(self, *, app_id: str) -> dict[str, Any]:
        capabilities: list[dict[str, Any]]
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
        elif app_id == "quant_documentation":
            capabilities = [
                {
                    "capability_id": "quant_documentation.inspect_package",
                    "app_id": "quant_documentation",
                    "risk_tier": "read_only",
                    "enabled": True,
                    "preflight_required": False,
                    "confirmation_required": False,
                    "execution_supported": True,
                },
                {
                    "capability_id": "quant_documentation.create_draft_workspace",
                    "app_id": "quant_documentation",
                    "risk_tier": "draft_only",
                    "enabled": True,
                    "preflight_required": False,
                    "confirmation_required": True,
                    "execution_supported": True,
                },
                {
                    "capability_id": "quant_documentation.draft_section",
                    "app_id": "quant_documentation",
                    "risk_tier": "draft_only",
                    "enabled": True,
                    "preflight_required": False,
                    "confirmation_required": True,
                    "execution_supported": True,
                },
                {
                    "capability_id": "quant_documentation.find_unsupported_claims",
                    "app_id": "quant_documentation",
                    "risk_tier": "read_only",
                    "enabled": True,
                    "preflight_required": False,
                    "confirmation_required": False,
                    "execution_supported": True,
                },
                {
                    "capability_id": "quant_documentation.export_markdown_review_package",
                    "app_id": "quant_documentation",
                    "risk_tier": "artifact_export",
                    "enabled": True,
                    "preflight_required": False,
                    "confirmation_required": True,
                    "execution_supported": True,
                },
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
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return {}

    def execute_action(
        self,
        *,
        app_id: str,
        capability_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return {}

    def reset_sample_workspaces(self) -> dict[str, Any]:
        return {
            "status": "reset",
            "deleted_lifecycle_ids": ["sample_credit_pd_scorecard_panel"],
            "warnings": [],
            "lifecycle_response": {"manifests": []},
        }


def load_lifecycle_fixture() -> dict[str, object]:
    with LIFECYCLE_FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        pytest.fail("Lifecycle fixture must be a JSON object.")
    return payload


def test_context_builder_builds_required_summaries_from_credit_fixture() -> None:
    context = LifecycleContextBuilder().build_from_path(LIFECYCLE_FIXTURE_PATH)

    assert set(context) == {
        "lifecycle_summary",
        "source_summary",
        "target_summary",
        "package_summary",
        "bundle_summary",
        "app_availability",
    }
    assert "Credit Risk PD Scorecard" in str(context["lifecycle_summary"])
    assert "Sample workspace credit_pd_scorecard_panel is sample-owned." in str(
        context["lifecycle_summary"]
    )
    assert "Credit PD panel sample source" in str(context["source_summary"])
    assert "Credit PD panel EDA handoff" in str(context["source_summary"])
    assert "Logistic scorecard candidate completed" in str(context["target_summary"])
    assert "Sample credit PD scorecard champion" in str(context["target_summary"])
    assert "Credit PD methodology documentation package" in str(context["package_summary"])
    assert "Credit PD methodology draft" in str(context["package_summary"])
    assert "Credit PD scorecard monitoring bundle" in str(context["bundle_summary"])
    assert "Credit PD monitoring run completed" in str(context["bundle_summary"])
    assert "retrain" in str(context["bundle_summary"]).lower()

    availability = context["app_availability"]
    assert isinstance(availability, dict)
    assert availability["quant_data"]["summary_count"] == 2
    assert availability["quant_studio"]["summary_count"] == 3
    assert availability["quant_documentation"]["summary_count"] == 2
    assert availability["quant_monitoring"]["summary_count"] == 3


def test_context_builder_preview_validates_against_contract() -> None:
    builder = LifecycleContextBuilder()
    result = builder.build_result_from_path(LIFECYCLE_FIXTURE_PATH)
    preview = result.context_preview.model_dump(mode="json")

    assert result.context_summary == builder.build_from_path(LIFECYCLE_FIXTURE_PATH)
    assert preview["data_policy"] == "summaries_only"
    assert preview["row_level_data_included"] is False
    assert preview["context_char_count"] > 0
    assert "Lifecycle: Credit Risk PD Scorecard" in preview["context_sources"]
    assert "quant_data source_reference: Credit PD panel sample source" in preview[
        "context_sources"
    ]
    assert "links" in preview["omitted_sensitive_fields"]
    assert any(field.endswith("query") for field in preview["omitted_sensitive_fields"])
    validate_against_contract(preview, "assistant_context_preview.v1.schema.json")


def test_context_builder_omits_queries_links_raw_records_and_hidden_commands() -> None:
    manifest = copy.deepcopy(load_lifecycle_fixture())
    source_reference = manifest["source_references"][0]
    assert isinstance(source_reference, dict)
    source_reference["query"]["secret"] = "do-not-leak"
    source_reference["records"] = [{"borrower_id": "A12345"}]
    source_reference["raw_path"] = "C:\\Users\\matth\\Desktop\\private\\raw.csv"
    source_reference["bucket_name"] = "private-bucket"
    source_reference["hidden_commands"] = ["rm -rf ."]
    source_reference["summary"] = (
        f"{source_reference['summary']} Raw copy was staged at "
        "C:\\Users\\matth\\Desktop\\private\\raw.csv and "
        "s3://private-bucket/raw.csv."
    )

    result = LifecycleContextBuilder().build_result_from_manifest(manifest)
    context = result.context_summary
    dumped = json.dumps(context, sort_keys=True)
    preview = result.context_preview.model_dump(mode="json")
    dumped_preview = json.dumps(preview, sort_keys=True)

    assert '"links"' not in dumped
    assert '"query"' not in dumped
    assert "frontend_url" not in dumped
    assert "do-not-leak" not in dumped
    assert "A12345" not in dumped
    assert "raw_path" not in dumped
    assert "private-bucket" not in dumped
    assert "hidden_commands" not in dumped
    assert "rm -rf" not in dumped
    assert "C:\\Users\\matth\\Desktop\\private\\raw.csv" not in dumped
    assert "s3://private-bucket/raw.csv" not in dumped
    assert "[redacted]" in dumped
    assert "source_references.records" in preview["omitted_row_level_fields"]
    assert "links" in preview["omitted_sensitive_fields"]
    assert "source_references.query" in preview["omitted_sensitive_fields"]
    assert "source_references.raw_path" in preview["omitted_sensitive_fields"]
    assert "source_references.bucket_name" in preview["omitted_sensitive_fields"]
    assert "source_references.hidden_commands" in preview["omitted_sensitive_fields"]
    assert "do-not-leak" not in dumped_preview
    assert "A12345" not in dumped_preview
    assert "private-bucket" not in dumped_preview
    assert "rm -rf" not in dumped_preview
    assert "C:\\Users\\matth\\Desktop\\private\\raw.csv" not in dumped_preview
    assert "s3://private-bucket/raw.csv" not in dumped_preview
    validate_against_contract(preview, "assistant_context_preview.v1.schema.json")


def test_lifecycle_context_produces_valid_plan_and_safe_ledger() -> None:
    runtime = runtime_with_canonical_capabilities()
    client = TestClient(create_app(runtime))
    context_result = LifecycleContextBuilder().build_result_from_path(LIFECYCLE_FIXTURE_PATH)
    context = context_result.context_summary

    response = client.post(
        "/plans",
        json={
            "user_goal": "Plan the sample credit PD lifecycle from Data through Monitoring.",
            "context_summary": context,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["validation"]["status"] == "valid"
    assert payload["plan"]["status"] == "valid"
    assert payload["plan"]["missing_inputs"] == []
    assert [step["app_id"] for step in payload["plan"]["proposed_steps"]] == [
        "quant_data",
        "quant_studio",
        "quant_documentation",
        "quant_documentation",
        "quant_monitoring",
    ]
    assert {
        item["capability_id"] for item in payload["plan"]["required_confirmations"]
    } == {
        "quant_studio.prepare_model_config_draft",
        "quant_documentation.create_draft_workspace",
    }

    validate_against_contract(payload["plan"], "agent_plan.v1.schema.json")
    validate_against_contract(
        context_result.context_preview.model_dump(mode="json"),
        "assistant_context_preview.v1.schema.json",
    )
    assert payload["context_preview"]["context"] == context
    validate_against_contract(payload["context_preview"], "assistant_context_preview.v1.schema.json")

    ledger_entries = runtime.planner.ledger.list_entries()
    assert len(ledger_entries) == 1
    ledger_entry = ledger_entries[0].model_dump(mode="json")
    validate_against_contract(ledger_entry, "agent_execution_ledger.v1.schema.json")
    validate_against_contract(
        ledger_entry["context_preview"],
        "assistant_context_preview.v1.schema.json",
    )
    dumped_ledger = json.dumps(ledger_entry, sort_keys=True)
    assert '"query"' not in dumped_ledger
    assert '"links"' not in dumped_ledger
    assert "frontend_url" not in dumped_ledger
