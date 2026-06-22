from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from quant_agent_runtime.api import create_app
from quant_agent_runtime.context_builder import LifecycleContextBuilder
from quant_agent_runtime.contracts import QuantSuiteContractLoader
from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.model_gateway import FakePlanProvider
from quant_agent_runtime.planner import PlannerService
from quant_agent_runtime.runtime import RuntimeContainer


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
    return RuntimeContainer(
        planner=PlannerService(
            provider=FakePlanProvider(),
            ledger=InMemoryLedger(),
            default_capabilities=capabilities or None,
        ),
        contract_loader=loader,
    )


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

    context = LifecycleContextBuilder().build_from_manifest(manifest)
    dumped = json.dumps(context, sort_keys=True)

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


def test_lifecycle_context_produces_valid_plan_and_safe_ledger() -> None:
    runtime = runtime_with_canonical_capabilities()
    client = TestClient(create_app(runtime))
    context = LifecycleContextBuilder().build_from_path(LIFECYCLE_FIXTURE_PATH)

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
        "quant_monitoring",
    ]
    assert {
        item["capability_id"] for item in payload["plan"]["required_confirmations"]
    } == {"quant_studio.prepare_model_config_draft"}

    validate_against_contract(payload["plan"], "agent_plan.v1.schema.json")

    ledger_entries = runtime.planner.ledger.list_entries()
    assert len(ledger_entries) == 1
    ledger_entry = ledger_entries[0].model_dump(mode="json")
    validate_against_contract(ledger_entry, "agent_execution_ledger.v1.schema.json")
    dumped_ledger = json.dumps(ledger_entry, sort_keys=True)
    assert '"query"' not in dumped_ledger
    assert '"links"' not in dumped_ledger
    assert "frontend_url" not in dumped_ledger
