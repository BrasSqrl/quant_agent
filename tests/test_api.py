import copy
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from quant_agent_runtime.action_request import ActionRequestPreviewService
from quant_agent_runtime.app_clients import AppClientError
from quant_agent_runtime.api import create_app
from quant_agent_runtime.capability_discovery import CapabilityDiscoveryService
from quant_agent_runtime.confirmation import ConfirmationService
from quant_agent_runtime.contracts import QuantSuiteContractLoader
from quant_agent_runtime.execution import ExecutionService
from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.model_gateway import FakePlanProvider
from quant_agent_runtime.models import (
    ContextPreview,
    LedgerEntry,
    PlanValidationResult,
    ProviderMode,
    RedactionSummary,
)
from quant_agent_runtime.planner import PlannerService
from quant_agent_runtime.preflight import PreflightService
from quant_agent_runtime.runtime import RuntimeContainer


AGENT_ROOT = Path(__file__).resolve().parents[1]
QUANT_SUITE_ROOT = AGENT_ROOT.parent / "quant_suite"


class FakePreflightAppClient:
    def __init__(
        self,
        response: dict[str, Any] | None = None,
        responses_by_capability: dict[str, dict[str, Any]] | None = None,
        execution_response: dict[str, Any] | None = None,
        execution_error: AppClientError | None = None,
        discovery_payloads_by_app: dict[str, dict[str, Any]] | None = None,
        discovery_errors_by_app: dict[str, AppClientError] | None = None,
        error: AppClientError | None = None,
    ) -> None:
        self.response = response or _valid_preflight_response()
        self.responses_by_capability = responses_by_capability or {}
        self.execution_response = execution_response or _valid_action_result()
        self.execution_error = execution_error
        default_discovery_payloads = {
            "quant_data": _capabilities_payload("quant_data"),
            "quant_studio": _capabilities_payload("quant_studio"),
            "quant_monitoring": _capabilities_payload("quant_monitoring"),
        }
        if discovery_payloads_by_app:
            default_discovery_payloads.update(discovery_payloads_by_app)
        self.discovery_payloads_by_app = default_discovery_payloads
        self.discovery_errors_by_app = discovery_errors_by_app or {}
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self.execution_calls: list[dict[str, Any]] = []
        self.discovery_calls: list[str] = []

    def discover_capabilities(self, *, app_id: str) -> dict[str, Any]:
        self.discovery_calls.append(app_id)
        if app_id in self.discovery_errors_by_app:
            raise self.discovery_errors_by_app[app_id]
        return copy.deepcopy(
            self.discovery_payloads_by_app.get(
                app_id,
                {
                    "schema_version": "1.0",
                    "data_policy": "summaries_and_references_only",
                    "app_id": app_id,
                    "capabilities": [],
                },
            )
        )

    def create_preflight(
        self,
        *,
        app_id: str,
        capability_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "app_id": app_id,
                "capability_id": capability_id,
                "payload": payload,
            }
        )
        if self.error is not None:
            raise self.error
        response = self.responses_by_capability.get(capability_id, self.response)
        return dict(response)

    def execute_action(
        self,
        *,
        app_id: str,
        capability_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self.execution_calls.append(
            {
                "app_id": app_id,
                "capability_id": capability_id,
                "payload": payload,
            }
        )
        if self.execution_error is not None:
            raise self.execution_error
        return copy.deepcopy(self.execution_response)


def runtime_with_preflight_client(app_client: FakePreflightAppClient) -> RuntimeContainer:
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    capabilities = loader.load_agent_capabilities()
    provider_status = loader.load_agent_provider_status()
    ledger = InMemoryLedger()
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


def _valid_preflight_response(
    status: str = "ready",
    *,
    capability_id: str = "quant_data.run_source_preflight",
    app_id: str = "quant_data",
) -> dict[str, Any]:
    reference_type = "source_reference" if app_id == "quant_data" else "monitoring_bundle"
    reference_id = "source_ref_test" if app_id == "quant_data" else "bundle_ref_test"
    evidence_check_id = "source_summary_present" if app_id == "quant_data" else "bundle_summary_present"
    summary_key = "source_summary" if app_id == "quant_data" else "bundle_summary"
    return {
        "schema_version": "1.0",
        "data_policy": "summaries_and_references_only",
        "preflight_id": f"preflight_test_{app_id}",
        "capability_id": capability_id,
        "app_id": app_id,
        "status": status,
        "input_summary": {
            summary_key: "Safe preflight summary is available.",
            "lifecycle_id": "lifecycle_test",
        },
        "blockers": [],
        "warnings": [],
        "required_user_inputs": [],
        "required_confirmations": [],
        "stale_state_signals": [],
        "estimated_cost": {"cost_label": "none", "billable": False},
        "estimated_duration_seconds": 5,
        "safe_artifact_references": [
            {
                "reference_type": reference_type,
                "reference_id": reference_id,
                "label": "Safe preflight reference",
            }
        ],
        "app_validation_evidence": [
            {
                "check_id": evidence_check_id,
                "status": "passed",
                "summary": "Preflight summary is available.",
            }
        ],
        "expires_at_utc": None,
        "revalidation_required": status != "ready",
    }


def _valid_action_result(
    execution_status: str = "succeeded",
    *,
    capability_id: str = "quant_studio.prepare_model_config_draft",
    app_id: str = "quant_studio",
    step_id: str = "step_2",
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "data_policy": "summaries_and_references_only",
        "action_run_id": "action_studio_draft_test",
        "step_id": step_id,
        "capability_id": capability_id,
        "app_id": app_id,
        "execution_status": execution_status,
        "accepted_input_summary": {
            "target_summary": "Default flag is the candidate target.",
            "lifecycle_id": "lifecycle_test",
        },
        "output_references": [
            {
                "reference_type": "model_config_draft",
                "reference_id": "model_config_draft_test",
                "label": "Model configuration draft",
            }
        ],
        "warnings": [],
        "recoverable_errors": [],
        "terminal_errors": [],
        "artifact_references": [],
        "app_run_reference": None,
        "validation_results": {"status": "valid", "errors": [], "warnings": []},
        "recommended_next_step": {
            "label": "Review the draft model configuration.",
            "target_app": "quant_studio",
            "review_only": True,
        },
        "retry_allowed": False,
        "state_changed_since_planning": False,
    }


def _capabilities_payload(app_id: str, capabilities: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if capabilities is None:
        if app_id == "quant_data":
            capabilities = [
                {
                    "capability_id": "quant_data.run_source_preflight",
                    "app_id": "quant_data",
                    "version": "1.0-draft",
                    "display_name": "Run source preflight",
                    "summary": "Checks safe source readiness evidence before downstream planning.",
                    "risk_tier": "workflow_preflight",
                    "enabled": True,
                    "preflight_required": True,
                    "confirmation_required": False,
                    "idempotent": True,
                    "reversible": True,
                    "side_effects": ["none"],
                    "input_schema": {"required_fields": ["source_summary"]},
                    "output_schema": {"safe_reference_types": ["preflight_summary", "source_reference"]},
                    "data_policy": "summaries_and_references_only",
                }
            ]
        elif app_id == "quant_studio":
            capabilities = [
                {
                    "capability_id": "quant_studio.prepare_model_config_draft",
                    "app_id": "quant_studio",
                    "version": "1.0-draft",
                    "display_name": "Prepare model configuration draft",
                    "summary": "Creates a review-only model configuration draft reference from safe target summaries.",
                    "risk_tier": "draft_only",
                    "enabled": True,
                    "preflight_required": False,
                    "confirmation_required": True,
                    "execution_supported": True,
                    "idempotent": False,
                    "reversible": True,
                    "side_effects": ["draft_only_after_confirmation"],
                    "input_schema": {"required_fields": ["target_summary"]},
                    "output_schema": {"safe_reference_types": ["model_config_draft"]},
                    "data_policy": "summaries_and_references_only",
                }
            ]
        elif app_id == "quant_monitoring":
            capabilities = [
                {
                    "capability_id": "quant_monitoring.validate_bundle",
                    "app_id": "quant_monitoring",
                    "version": "1.0-draft",
                    "display_name": "Validate monitoring bundle",
                    "summary": "Checks safe monitoring bundle readiness evidence before downstream planning.",
                    "risk_tier": "workflow_preflight",
                    "enabled": True,
                    "preflight_required": True,
                    "confirmation_required": False,
                    "idempotent": True,
                    "reversible": True,
                    "side_effects": ["none"],
                    "input_schema": {"required_fields": ["bundle_summary"]},
                    "output_schema": {
                        "safe_reference_types": ["bundle_validation_summary", "monitoring_bundle"]
                    },
                    "data_policy": "summaries_and_references_only",
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


def _create_plan(client: TestClient) -> dict[str, Any]:
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
    return response.json()


def _create_plan_with_lifecycle_reference(client: TestClient) -> dict[str, Any]:
    response = client.post(
        "/plans",
        json={
            "user_goal": "Build a conservative PD scorecard plan.",
            "context_summary": {
                "lifecycle_summary": {
                    "lifecycle_id": "lifecycle_test",
                    "state": "ready_for_modeling",
                    "summary": "Lifecycle has safe source and target summaries.",
                },
                "source_summary": "Development sample is registered.",
                "target_summary": "Default flag is the candidate target.",
                "package_summary": "No documentation package exists yet.",
                "bundle_summary": "Monitoring bundle is available.",
            },
        },
    )
    assert response.status_code == 200
    return response.json()


def _create_confirmed_studio_preview(client: TestClient) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan_payload = _create_plan_with_lifecycle_reference(client)
    studio_step = next(
        step
        for step in plan_payload["plan"]["proposed_steps"]
        if step["capability_id"] == "quant_studio.prepare_model_config_draft"
    )
    confirmation_response = client.post(
        "/confirmations",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": studio_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    )
    assert confirmation_response.status_code == 200
    preview_response = client.post(
        "/action-requests",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )
    assert preview_response.status_code == 200
    return plan_payload, studio_step, preview_response.json()


def test_health_endpoint_works() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["plan_only_mode"] is False
    assert response.json()["execution_supported"] is True
    assert response.json()["execution_support_level"] == "single_step_studio_draft_only"


def test_runtime_manifest_returns_supported_modes() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    response = client.get("/runtime/manifest")

    assert response.status_code == 200
    manifest = response.json()
    assert manifest["plan_only_mode"] is False
    assert manifest["execution_supported"] is True
    assert "fake_provider" in manifest["supported_provider_modes"]
    assert "disabled_or_local_fallback" in manifest["supported_provider_modes"]
    assert "POST /plans" in manifest["supported_routes"]
    assert "POST /preflights" in manifest["supported_routes"]
    assert "POST /confirmations" in manifest["supported_routes"]
    assert "POST /action-requests" in manifest["supported_routes"]
    assert "POST /executions" in manifest["supported_routes"]
    assert manifest["runtime_health_endpoint"] == "/health"
    assert manifest["execution_support_level"] == "single_step_studio_draft_only"
    assert manifest["provider_status"]["supports_execution"] is False
    assert manifest["provider_status"]["hosted_provider_enabled"] is False
    assert "quant_data:/api/agent/capabilities" in manifest["capability_discovery_endpoints"]
    assert "quant_studio:/api/agent/capabilities" in manifest["capability_discovery_endpoints"]
    assert "quant_monitoring:/api/agent/capabilities" in manifest["capability_discovery_endpoints"]
    assert manifest["supported_preflight_capabilities"] == [
        "quant_data.run_source_preflight",
        "quant_monitoring.validate_bundle",
    ]
    assert manifest["capability_discovery"]["discovered_apps"] == [
        "quant_data",
        "quant_studio",
        "quant_monitoring",
    ]
    assert manifest["capability_discovery"]["unavailable_apps"] == []
    assert manifest["capability_discovery"]["unsupported_capability_ids"] == []
    assert manifest["capability_discovery"]["supported_preflight_capabilities"] == [
        "quant_data.run_source_preflight",
        "quant_monitoring.validate_bundle",
    ]
    assert manifest["supported_execution_capabilities"] == [
        "quant_studio.prepare_model_config_draft"
    ]
    assert manifest["capability_discovery"]["supported_execution_capabilities"] == [
        "quant_studio.prepare_model_config_draft"
    ]


def test_runtime_manifest_reports_unavailable_app_capability_discovery() -> None:
    app_client = FakePreflightAppClient(
        discovery_errors_by_app={
            "quant_monitoring": AppClientError(
                "Quant Monitoring capability discovery app is unavailable.",
                status_code=503,
            )
        }
    )
    client = TestClient(create_app(runtime_with_preflight_client(app_client)))

    response = client.get("/runtime/manifest")

    assert response.status_code == 200
    discovery = response.json()["capability_discovery"]
    assert discovery["discovered_apps"] == ["quant_data", "quant_studio"]
    assert discovery["unavailable_apps"] == ["quant_monitoring"]
    assert discovery["supported_preflight_capabilities"] == ["quant_data.run_source_preflight"]
    assert discovery["supported_execution_capabilities"] == [
        "quant_studio.prepare_model_config_draft"
    ]
    assert "quant_monitoring.validate_bundle" not in response.json()["supported_preflight_capabilities"]
    assert discovery["reconciliation_warnings"][0]["code"] == "app_capability_discovery_unavailable"


def test_runtime_manifest_rejects_unsafe_capability_discovery_payload() -> None:
    data_payload = _capabilities_payload("quant_data")
    data_payload["raw_path"] = "C:\\Users\\matth\\Desktop\\private\\raw.csv"
    app_client = FakePreflightAppClient(
        discovery_payloads_by_app={
            "quant_data": data_payload,
            "quant_monitoring": _capabilities_payload("quant_monitoring"),
        }
    )
    client = TestClient(create_app(runtime_with_preflight_client(app_client)))

    response = client.get("/runtime/manifest")

    assert response.status_code == 200
    payload = response.json()
    discovery = payload["capability_discovery"]
    assert discovery["unavailable_apps"] == ["quant_data"]
    assert payload["supported_preflight_capabilities"] == ["quant_monitoring.validate_bundle"]
    assert discovery["reconciliation_warnings"][0]["code"] == "unsafe_capability_discovery_payload"
    assert "private\\raw.csv" not in response.text


def test_runtime_manifest_warns_on_unknown_app_capability_without_supporting_it() -> None:
    unknown_capability = {
        "capability_id": "quant_data.unknown_preflight",
        "app_id": "quant_data",
        "risk_tier": "workflow_preflight",
        "enabled": True,
        "preflight_required": True,
        "confirmation_required": False,
    }
    app_client = FakePreflightAppClient(
        discovery_payloads_by_app={
            "quant_data": _capabilities_payload("quant_data", [unknown_capability]),
            "quant_monitoring": _capabilities_payload("quant_monitoring"),
        }
    )
    client = TestClient(create_app(runtime_with_preflight_client(app_client)))

    response = client.get("/runtime/manifest")

    assert response.status_code == 200
    discovery = response.json()["capability_discovery"]
    assert "quant_data.unknown_preflight" in discovery["unsupported_capability_ids"]
    assert "quant_data.run_source_preflight" in discovery["unsupported_capability_ids"]
    assert discovery["supported_preflight_capabilities"] == ["quant_monitoring.validate_bundle"]
    warning_codes = {warning["code"] for warning in discovery["reconciliation_warnings"]}
    assert "missing_canonical_capability" in warning_codes
    assert "canonical_capability_not_advertised" in warning_codes


def test_runtime_manifest_warns_on_app_and_preflight_policy_mismatch() -> None:
    mismatched_app = {
        "capability_id": "quant_data.run_source_preflight",
        "app_id": "quant_monitoring",
        "risk_tier": "workflow_preflight",
        "enabled": True,
        "preflight_required": True,
        "confirmation_required": False,
    }
    mismatched_policy = {
        "capability_id": "quant_monitoring.validate_bundle",
        "app_id": "quant_monitoring",
        "risk_tier": "workflow_preflight",
        "enabled": True,
        "preflight_required": False,
        "confirmation_required": False,
    }
    app_client = FakePreflightAppClient(
        discovery_payloads_by_app={
            "quant_data": _capabilities_payload("quant_data", [mismatched_app]),
            "quant_monitoring": _capabilities_payload("quant_monitoring", [mismatched_policy]),
        }
    )
    client = TestClient(create_app(runtime_with_preflight_client(app_client)))

    response = client.get("/runtime/manifest")

    assert response.status_code == 200
    discovery = response.json()["capability_discovery"]
    assert discovery["supported_preflight_capabilities"] == []
    warning_codes = {warning["code"] for warning in discovery["reconciliation_warnings"]}
    assert "capability_app_mismatch" in warning_codes
    assert "capability_preflight_policy_mismatch" in warning_codes


def test_runtime_manifest_reports_invalid_json_discovery_as_recoverable_warning() -> None:
    app_client = FakePreflightAppClient(
        discovery_errors_by_app={
            "quant_data": AppClientError(
                "Quant Data capability discovery returned invalid JSON.",
                status_code=502,
            )
        }
    )
    client = TestClient(create_app(runtime_with_preflight_client(app_client)))

    response = client.get("/runtime/manifest")

    assert response.status_code == 200
    discovery = response.json()["capability_discovery"]
    assert discovery["unavailable_apps"] == ["quant_data"]
    assert discovery["reconciliation_warnings"][0]["code"] == "app_capability_discovery_unavailable"
    assert "invalid JSON" in discovery["reconciliation_warnings"][0]["message"]


def test_runtime_manifest_reports_non_object_discovery_as_recoverable_warning() -> None:
    app_client = FakePreflightAppClient(
        discovery_errors_by_app={
            "quant_data": AppClientError(
                "Quant Data capability discovery returned a non-object response.",
                status_code=502,
            )
        }
    )
    client = TestClient(create_app(runtime_with_preflight_client(app_client)))

    response = client.get("/runtime/manifest")

    assert response.status_code == 200
    discovery = response.json()["capability_discovery"]
    assert discovery["unavailable_apps"] == ["quant_data"]
    assert discovery["reconciliation_warnings"][0]["code"] == "app_capability_discovery_unavailable"
    assert "non-object response" in discovery["reconciliation_warnings"][0]["message"]


def test_runtime_manifest_warns_on_malformed_capability_entries() -> None:
    app_client = FakePreflightAppClient(
        discovery_payloads_by_app={
            "quant_data": _capabilities_payload("quant_data", ["not an object"]),  # type: ignore[list-item]
            "quant_monitoring": _capabilities_payload("quant_monitoring"),
        }
    )
    client = TestClient(create_app(runtime_with_preflight_client(app_client)))

    response = client.get("/runtime/manifest")

    assert response.status_code == 200
    discovery = response.json()["capability_discovery"]
    assert discovery["supported_preflight_capabilities"] == ["quant_monitoring.validate_bundle"]
    warning_codes = {warning["code"] for warning in discovery["reconciliation_warnings"]}
    assert "malformed_capability_entry" in warning_codes
    assert "canonical_capability_not_advertised" in warning_codes


def test_local_browser_cors_preflight_allows_plan_requests() -> None:
    client = TestClient(create_app())

    response = client.options(
        "/plans",
        headers={
            "Origin": "http://127.0.0.1:5810",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5810"
    assert "POST" in response.headers["access-control-allow-methods"]


def test_fake_provider_can_produce_valid_plan() -> None:
    client = TestClient(create_app())

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
    assert payload["validation"]["status"] == "valid"
    assert payload["run_state"] == "waiting_for_confirmation"
    assert payload["provider_metadata"]["supports_execution"] is False
    assert payload["provider_metadata"]["provider_mode"] in {
        "fake_provider",
        "disabled_or_local_fallback",
    }
    assert payload["plan"]["execution_permitted"] is False
    assert payload["ledger_recorded"] is True
    assert payload["plan"]["proposed_steps"]
    assert [step["app_id"] for step in payload["plan"]["proposed_steps"]] == [
        "quant_data",
        "quant_studio",
        "quant_documentation",
        "quant_monitoring",
    ]
    assert payload["plan"]["required_confirmations"][0]["capability_id"] == (
        "quant_studio.prepare_model_config_draft"
    )


def test_missing_required_context_fields_become_missing_inputs() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/plans",
        json={
            "user_goal": "Build the lifecycle plan from whatever summaries are available.",
            "context_summary": {},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["validation"]["status"] == "valid"
    assert payload["run_state"] == "waiting_for_input"
    assert payload["plan"]["status"] == "blocked"
    assert payload["plan"]["missing_inputs"]
    assert payload["plan"]["proposed_steps"][0]["action_input"]["source_summary"] == "[missing]"


def test_no_execution_endpoint_exists() -> None:
    client = TestClient(create_app())

    assert client.post("/runs", json={}).status_code == 404
    assert client.post("/execute", json={}).status_code == 404
    assert client.post("/preflight", json={}).status_code == 404
    assert client.post("/runtime/preflight", json={}).status_code == 404
    assert client.post("/runtime/confirm", json={}).status_code == 404


def test_preflight_resolves_recorded_plan_step_and_ledgers_response() -> None:
    app_client = FakePreflightAppClient()
    runtime = runtime_with_preflight_client(app_client)
    client = TestClient(create_app(runtime))
    plan_payload = _create_plan(client)
    source_step = plan_payload["plan"]["proposed_steps"][0]

    response = client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": source_step["step_id"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == plan_payload["run_id"]
    assert payload["step_id"] == source_step["step_id"]
    assert payload["capability_id"] == "quant_data.run_source_preflight"
    assert payload["validation"]["status"] == "valid"
    assert payload["run_state"] == "waiting_for_confirmation"
    assert payload["ledger_recorded"] is True
    assert payload["preflight"]["status"] == "ready"
    assert source_step["preflight_required"] is True
    assert len(app_client.calls) == 1
    app_call = app_client.calls[0]
    assert app_call["app_id"] == "quant_data"
    assert app_call["capability_id"] == "quant_data.run_source_preflight"
    assert app_call["payload"]["action_input"] == source_step["action_input"]
    assert app_call["payload"]["context_summary"]["source_summary"] == "Development sample is registered."
    assert "raw_paths" not in app_call["payload"]["context_summary"]
    entry = runtime.planner.ledger.list_entries()[0]
    assert entry.preflight_records[0]["preflight_id"] == "preflight_test_quant_data"


def test_preflight_resolves_monitoring_plan_step_and_ledgers_response() -> None:
    monitoring_response = _valid_preflight_response(
        capability_id="quant_monitoring.validate_bundle",
        app_id="quant_monitoring",
    )
    app_client = FakePreflightAppClient(
        responses_by_capability={
            "quant_data.run_source_preflight": _valid_preflight_response(),
            "quant_monitoring.validate_bundle": monitoring_response,
        }
    )
    runtime = runtime_with_preflight_client(app_client)
    client = TestClient(create_app(runtime))
    plan_payload = _create_plan(client)
    monitoring_step = next(
        step
        for step in plan_payload["plan"]["proposed_steps"]
        if step["capability_id"] == "quant_monitoring.validate_bundle"
    )

    response = client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": monitoring_step["step_id"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["capability_id"] == "quant_monitoring.validate_bundle"
    assert payload["run_state"] == "waiting_for_confirmation"
    assert payload["preflight"]["app_id"] == "quant_monitoring"
    assert payload["preflight"]["safe_artifact_references"][0]["reference_type"] == "monitoring_bundle"
    assert len(app_client.calls) == 1
    app_call = app_client.calls[0]
    assert app_call["app_id"] == "quant_monitoring"
    assert app_call["capability_id"] == "quant_monitoring.validate_bundle"
    assert app_call["payload"]["action_input"] == monitoring_step["action_input"]
    assert app_call["payload"]["context_summary"]["bundle_summary"] == "Monitoring bundle is not available."
    entry = runtime.planner.ledger.list_entries()[0]
    assert entry.preflight_records[0]["capability_id"] == "quant_monitoring.validate_bundle"


def test_preflight_can_record_multiple_app_owned_preflights_and_validate_ledger() -> None:
    app_client = FakePreflightAppClient(
        responses_by_capability={
            "quant_data.run_source_preflight": _valid_preflight_response(),
            "quant_monitoring.validate_bundle": _valid_preflight_response(
                capability_id="quant_monitoring.validate_bundle",
                app_id="quant_monitoring",
            ),
        }
    )
    runtime = runtime_with_preflight_client(app_client)
    client = TestClient(create_app(runtime))
    plan_payload = _create_plan(client)
    source_step = next(
        step
        for step in plan_payload["plan"]["proposed_steps"]
        if step["capability_id"] == "quant_data.run_source_preflight"
    )
    monitoring_step = next(
        step
        for step in plan_payload["plan"]["proposed_steps"]
        if step["capability_id"] == "quant_monitoring.validate_bundle"
    )

    source_response = client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": source_step["step_id"]},
    )
    monitoring_response = client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": monitoring_step["step_id"]},
    )

    assert source_response.status_code == 200
    assert monitoring_response.status_code == 200
    entry = runtime.planner.ledger.list_entries()[0]
    assert [record["capability_id"] for record in entry.preflight_records] == [
        "quant_data.run_source_preflight",
        "quant_monitoring.validate_bundle",
    ]
    runtime.contract_loader.validate_agent_contract_payload(
        entry.model_dump(mode="json"),
        "agent_execution_ledger.v1.schema.json",
    )


def test_preflight_blocked_response_sets_preflight_blocked_run_state() -> None:
    blocked_response = _valid_preflight_response(status="blocked")
    blocked_response["blockers"] = [
        {
            "code": "missing_safe_source_reference",
            "message": "A safe source reference is required.",
        }
    ]
    app_client = FakePreflightAppClient(response=blocked_response)
    client = TestClient(create_app(runtime_with_preflight_client(app_client)))
    plan_payload = _create_plan(client)
    source_step = plan_payload["plan"]["proposed_steps"][0]

    response = client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": source_step["step_id"]},
    )

    assert response.status_code == 200
    assert response.json()["run_state"] == "preflight_blocked"


def test_confirmation_records_required_step_and_validates_ledger() -> None:
    app_client = FakePreflightAppClient()
    runtime = runtime_with_preflight_client(app_client)
    client = TestClient(create_app(runtime))
    plan_payload = _create_plan(client)
    studio_step = next(
        step
        for step in plan_payload["plan"]["proposed_steps"]
        if step["capability_id"] == "quant_studio.prepare_model_config_draft"
    )

    response = client.post(
        "/confirmations",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": studio_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == plan_payload["run_id"]
    assert payload["step_id"] == studio_step["step_id"]
    assert payload["capability_id"] == "quant_studio.prepare_model_config_draft"
    assert payload["run_state"] == "ready_for_execution_preview"
    assert payload["validation"]["status"] == "valid"
    assert payload["ledger_recorded"] is True
    assert payload["confirmation"]["status"] == "confirmed"
    assert payload["confirmation"]["confirmation_intent"] == "approve_plan_step"
    assert payload["confirmation"]["confirmed_by"] == "local_user"
    assert payload["confirmation"]["execution_permitted"] is False
    entry = runtime.planner.ledger.list_entries()[0]
    assert entry.confirmation_records[0]["capability_id"] == (
        "quant_studio.prepare_model_config_draft"
    )
    runtime.contract_loader.validate_agent_contract_payload(
        entry.model_dump(mode="json"),
        "agent_execution_ledger.v1.schema.json",
    )


def test_action_request_preview_records_confirmed_studio_step_and_validates_ledger() -> None:
    app_client = FakePreflightAppClient()
    runtime = runtime_with_preflight_client(app_client)
    client = TestClient(create_app(runtime))
    plan_payload = _create_plan_with_lifecycle_reference(client)
    studio_step = next(
        step
        for step in plan_payload["plan"]["proposed_steps"]
        if step["capability_id"] == "quant_studio.prepare_model_config_draft"
    )
    confirmation_response = client.post(
        "/confirmations",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": studio_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    )
    assert confirmation_response.status_code == 200

    response = client.post(
        "/action-requests",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == plan_payload["run_id"]
    assert payload["step_id"] == studio_step["step_id"]
    assert payload["capability_id"] == "quant_studio.prepare_model_config_draft"
    assert payload["run_state"] == "ready_for_execution_preview"
    assert payload["validation"]["status"] == "valid"
    assert payload["ledger_recorded"] is True
    action_request = payload["action_request"]
    assert action_request["schema_version"] == "1.0"
    assert action_request["data_policy"] == "summaries_and_references_only"
    assert action_request["execution_permitted"] is False
    assert action_request["agent_run_id"] == plan_payload["run_id"]
    assert action_request["plan_id"] == plan_payload["plan"]["plan_id"]
    assert action_request["action_input"] == studio_step["action_input"]
    assert action_request["input_schema_version"] == "1.0-draft"
    assert action_request["confirmation_reference"]["status"] == "confirmed"
    assert action_request["confirmation_reference"]["execution_permitted"] is False
    assert action_request["preflight_reference"] is None
    assert action_request["lifecycle_state_reference"] == {
        "lifecycle_id": "lifecycle_test",
        "state": "ready_for_modeling",
        "summary": "Lifecycle has safe source and target summaries.",
    }
    assert action_request["idempotency_key"] == (
        f"idem_{plan_payload['run_id']}_{studio_step['step_id']}_"
        "quant_studio.prepare_model_config_draft"
    )
    assert "raw_paths" not in str(action_request)
    entry = runtime.planner.ledger.list_entries()[0]
    assert len(entry.action_requests) == 1
    runtime.contract_loader.validate_agent_contract_payload(
        entry.action_requests[0],
        "agent_action_request.v1.schema.json",
    )
    runtime.contract_loader.validate_agent_contract_payload(
        entry.model_dump(mode="json"),
        "agent_execution_ledger.v1.schema.json",
    )


def test_action_request_preview_records_data_and_monitoring_preflight_steps() -> None:
    app_client = FakePreflightAppClient(
        responses_by_capability={
            "quant_data.run_source_preflight": _valid_preflight_response(status="warning"),
            "quant_monitoring.validate_bundle": _valid_preflight_response(
                status="ready",
                capability_id="quant_monitoring.validate_bundle",
                app_id="quant_monitoring",
            ),
        }
    )
    runtime = runtime_with_preflight_client(app_client)
    client = TestClient(create_app(runtime))
    plan_payload = _create_plan_with_lifecycle_reference(client)
    source_step = next(
        step
        for step in plan_payload["plan"]["proposed_steps"]
        if step["capability_id"] == "quant_data.run_source_preflight"
    )
    monitoring_step = next(
        step
        for step in plan_payload["plan"]["proposed_steps"]
        if step["capability_id"] == "quant_monitoring.validate_bundle"
    )
    source_preflight = client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": source_step["step_id"]},
    )
    monitoring_preflight = client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": monitoring_step["step_id"]},
    )
    assert source_preflight.status_code == 200
    assert monitoring_preflight.status_code == 200

    source_preview = client.post(
        "/action-requests",
        json={"run_id": plan_payload["run_id"], "step_id": source_step["step_id"]},
    )
    monitoring_preview = client.post(
        "/action-requests",
        json={"run_id": plan_payload["run_id"], "step_id": monitoring_step["step_id"]},
    )

    assert source_preview.status_code == 200
    assert monitoring_preview.status_code == 200
    assert source_preview.json()["action_request"]["preflight_reference"]["status"] == "warning"
    assert source_preview.json()["action_request"]["confirmation_reference"] is None
    assert monitoring_preview.json()["action_request"]["preflight_reference"]["app_id"] == "quant_monitoring"
    assert monitoring_preview.json()["action_request"]["execution_permitted"] is False
    entry = runtime.planner.ledger.list_entries()[0]
    assert [record["capability_id"] for record in entry.action_requests] == [
        "quant_data.run_source_preflight",
        "quant_monitoring.validate_bundle",
    ]


def test_action_request_preview_is_idempotent_for_existing_request() -> None:
    runtime = runtime_with_preflight_client(FakePreflightAppClient())
    client = TestClient(create_app(runtime))
    plan_payload = _create_plan_with_lifecycle_reference(client)
    studio_step = next(
        step
        for step in plan_payload["plan"]["proposed_steps"]
        if step["capability_id"] == "quant_studio.prepare_model_config_draft"
    )
    confirmation_response = client.post(
        "/confirmations",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": studio_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    )
    assert confirmation_response.status_code == 200

    first = client.post(
        "/action-requests",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )
    second = client.post(
        "/action-requests",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["action_request"] == first.json()["action_request"]
    entry = runtime.planner.ledger.list_entries()[0]
    assert len(entry.action_requests) == 1


def test_execution_runs_confirmed_studio_draft_step_and_validates_ledger() -> None:
    app_client = FakePreflightAppClient()
    runtime = runtime_with_preflight_client(app_client)
    client = TestClient(create_app(runtime))
    plan_payload, studio_step, preview_payload = _create_confirmed_studio_preview(client)

    response = client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == plan_payload["run_id"]
    assert payload["step_id"] == studio_step["step_id"]
    assert payload["capability_id"] == "quant_studio.prepare_model_config_draft"
    assert payload["run_state"] == "completed"
    assert payload["validation"]["status"] == "valid"
    assert payload["ledger_recorded"] is True
    action_request = payload["action_request"]
    assert action_request["execution_permitted"] is True
    assert action_request["execution_request"] is True
    assert action_request["preview_idempotency_key"] == preview_payload["action_request"]["idempotency_key"]
    assert action_request["idempotency_key"] == (
        f"exec_{preview_payload['action_request']['idempotency_key']}"
    )
    assert action_request["action_input"] == studio_step["action_input"]
    assert payload["action_result"]["execution_status"] == "succeeded"
    assert payload["action_result"]["output_references"][0]["reference_type"] == "model_config_draft"
    assert len(app_client.execution_calls) == 1
    app_call = app_client.execution_calls[0]
    assert app_call["app_id"] == "quant_studio"
    assert app_call["capability_id"] == "quant_studio.prepare_model_config_draft"
    assert app_call["payload"] == {"action_request": action_request}
    assert "raw_path" not in str(app_call)
    entry = runtime.planner.ledger.list_entries()[0]
    assert [record["execution_permitted"] for record in entry.action_requests] == [False, True]
    assert len(entry.action_results) == 1
    runtime.contract_loader.validate_agent_contract_payload(
        entry.action_requests[1],
        "agent_action_request.v1.schema.json",
    )
    runtime.contract_loader.validate_agent_contract_payload(
        entry.action_results[0],
        "agent_action_result.v1.schema.json",
    )
    runtime.contract_loader.validate_agent_contract_payload(
        entry.model_dump(mode="json"),
        "agent_execution_ledger.v1.schema.json",
    )


def test_execution_duplicate_call_returns_existing_result_without_calling_app_again() -> None:
    app_client = FakePreflightAppClient()
    runtime = runtime_with_preflight_client(app_client)
    client = TestClient(create_app(runtime))
    plan_payload, studio_step, _preview_payload = _create_confirmed_studio_preview(client)
    request = {"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]}

    first = client.post("/executions", json=request)
    second = client.post("/executions", json=request)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["action_result"] == first.json()["action_result"]
    assert second.json()["action_request"]["execution_permitted"] is True
    assert len(app_client.execution_calls) == 1
    entry = runtime.planner.ledger.list_entries()[0]
    assert len(entry.action_requests) == 2
    assert len(entry.action_results) == 1


def test_execution_rejects_missing_preview_missing_confirmation_and_unsupported_step() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))
    plan_payload = _create_plan_with_lifecycle_reference(client)
    source_step = next(
        step
        for step in plan_payload["plan"]["proposed_steps"]
        if step["capability_id"] == "quant_data.run_source_preflight"
    )
    studio_step = next(
        step
        for step in plan_payload["plan"]["proposed_steps"]
        if step["capability_id"] == "quant_studio.prepare_model_config_draft"
    )

    unsupported = client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": source_step["step_id"]},
    )
    missing_confirmation = client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )
    assert client.post(
        "/confirmations",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": studio_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    ).status_code == 200
    missing_preview = client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )

    assert unsupported.status_code == 422
    assert unsupported.json()["detail"]["errors"][0]["code"] == "unsupported_execution_capability"
    assert missing_confirmation.status_code == 422
    assert missing_confirmation.json()["detail"]["errors"][0]["code"] == (
        "missing_confirmation_for_execution"
    )
    assert missing_preview.status_code == 422
    assert missing_preview.json()["detail"]["errors"][0]["code"] == "missing_action_request_preview"


def test_execution_rejects_when_studio_app_unavailable_or_capability_not_advertised() -> None:
    unavailable_client = FakePreflightAppClient(
        discovery_errors_by_app={
            "quant_studio": AppClientError(
                "Quant Studio capability discovery app is unavailable.",
                status_code=503,
            )
        }
    )
    unavailable_runtime = runtime_with_preflight_client(unavailable_client)
    unavailable_api = TestClient(create_app(unavailable_runtime))
    plan_payload, studio_step, _preview_payload = _create_confirmed_studio_preview(unavailable_api)
    unavailable_response = unavailable_api.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )

    disabled_client = FakePreflightAppClient(
        discovery_payloads_by_app={"quant_studio": _capabilities_payload("quant_studio", [])}
    )
    disabled_runtime = runtime_with_preflight_client(disabled_client)
    disabled_api = TestClient(create_app(disabled_runtime))
    disabled_plan, disabled_step, _disabled_preview = _create_confirmed_studio_preview(disabled_api)
    disabled_response = disabled_api.post(
        "/executions",
        json={"run_id": disabled_plan["run_id"], "step_id": disabled_step["step_id"]},
    )

    assert unavailable_response.status_code == 422
    assert unavailable_response.json()["detail"]["errors"][0]["code"] == "execution_app_unavailable"
    assert unavailable_client.execution_calls == []
    assert disabled_response.status_code == 422
    assert disabled_response.json()["detail"]["errors"][0]["code"] == (
        "execution_capability_unavailable"
    )
    assert disabled_client.execution_calls == []


def test_execution_rejects_app_unavailable_malformed_and_unsafe_results() -> None:
    app_unavailable = FakePreflightAppClient(
        execution_error=AppClientError("Quant Studio execution app is unavailable.", status_code=503)
    )
    app_unavailable_api = TestClient(create_app(runtime_with_preflight_client(app_unavailable)))
    plan_payload, studio_step, _preview_payload = _create_confirmed_studio_preview(app_unavailable_api)
    unavailable_response = app_unavailable_api.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )

    malformed_result = _valid_action_result()
    malformed_result.pop("action_run_id")
    malformed = FakePreflightAppClient(execution_response=malformed_result)
    malformed_api = TestClient(create_app(runtime_with_preflight_client(malformed)))
    malformed_plan, malformed_step, _malformed_preview = _create_confirmed_studio_preview(
        malformed_api
    )
    malformed_response = malformed_api.post(
        "/executions",
        json={"run_id": malformed_plan["run_id"], "step_id": malformed_step["step_id"]},
    )

    unsafe_result = _valid_action_result()
    unsafe_result["raw_path"] = "C:\\Users\\matth\\Desktop\\private\\raw.csv"
    unsafe = FakePreflightAppClient(execution_response=unsafe_result)
    unsafe_api = TestClient(create_app(runtime_with_preflight_client(unsafe)))
    unsafe_plan, unsafe_step, _unsafe_preview = _create_confirmed_studio_preview(unsafe_api)
    unsafe_response = unsafe_api.post(
        "/executions",
        json={"run_id": unsafe_plan["run_id"], "step_id": unsafe_step["step_id"]},
    )

    assert unavailable_response.status_code == 503
    assert unavailable_response.json()["detail"]["code"] == "app_unavailable"
    assert malformed_response.status_code == 422
    assert malformed_response.json()["detail"]["errors"][0]["code"] == "malformed_app_action_result"
    assert unsafe_response.status_code == 422
    assert unsafe_response.json()["detail"]["errors"][0]["code"] == "unsafe_app_action_result"
    assert "private\\raw.csv" not in unsafe_response.text


def test_action_request_rejects_unknown_run_unknown_step_and_browser_action_payload() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    unknown_run = client.post(
        "/action-requests",
        json={"run_id": "run_missing", "step_id": "step_1"},
    )
    plan_payload = _create_plan_with_lifecycle_reference(client)
    unknown_step = client.post(
        "/action-requests",
        json={"run_id": plan_payload["run_id"], "step_id": "step_missing"},
    )
    extra_payload = client.post(
        "/action-requests",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": plan_payload["plan"]["proposed_steps"][0]["step_id"],
            "action_input": {"source_summary": "Browser must not provide action input."},
        },
    )

    assert unknown_run.status_code == 422
    assert unknown_run.json()["detail"]["errors"][0]["code"] == "unknown_run"
    assert unknown_step.status_code == 422
    assert unknown_step.json()["detail"]["errors"][0]["code"] == "unknown_step"
    assert extra_payload.status_code == 422


def test_action_request_rejects_missing_lifecycle_preflight_confirmation_and_blocked_plan() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))
    plan_payload = _create_plan(client)
    studio_step = next(
        step
        for step in plan_payload["plan"]["proposed_steps"]
        if step["capability_id"] == "quant_studio.prepare_model_config_draft"
    )
    confirmation_response = client.post(
        "/confirmations",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": studio_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    )
    assert confirmation_response.status_code == 200
    missing_lifecycle = client.post(
        "/action-requests",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )

    fresh_client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))
    gated_payload = _create_plan_with_lifecycle_reference(fresh_client)
    source_step = gated_payload["plan"]["proposed_steps"][0]
    studio_step = next(
        step
        for step in gated_payload["plan"]["proposed_steps"]
        if step["capability_id"] == "quant_studio.prepare_model_config_draft"
    )
    missing_preflight = fresh_client.post(
        "/action-requests",
        json={"run_id": gated_payload["run_id"], "step_id": source_step["step_id"]},
    )
    missing_confirmation = fresh_client.post(
        "/action-requests",
        json={"run_id": gated_payload["run_id"], "step_id": studio_step["step_id"]},
    )
    blocked_plan_response = fresh_client.post(
        "/plans",
        json={"user_goal": "Plan with missing summaries.", "context_summary": {}},
    )
    assert blocked_plan_response.status_code == 200
    blocked_step = blocked_plan_response.json()["plan"]["proposed_steps"][0]
    blocked_preview = fresh_client.post(
        "/action-requests",
        json={"run_id": blocked_plan_response.json()["run_id"], "step_id": blocked_step["step_id"]},
    )

    assert missing_lifecycle.status_code == 422
    assert missing_lifecycle.json()["detail"]["errors"][0]["code"] == "missing_lifecycle_state_reference"
    assert missing_preflight.status_code == 422
    assert missing_preflight.json()["detail"]["errors"][0]["code"] == "missing_preflight_for_action_request"
    assert missing_confirmation.status_code == 422
    assert missing_confirmation.json()["detail"]["errors"][0]["code"] == (
        "missing_confirmation_for_action_request"
    )
    assert blocked_preview.status_code == 422
    assert blocked_preview.json()["detail"]["errors"][0]["code"] == "blocked_plan_action_request"


def test_action_request_rejects_blocked_preflight_stale_capability_unsafe_input_and_malformed_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked_response = _valid_preflight_response(status="blocked")
    blocked_response["blockers"] = [
        {
            "code": "missing_safe_source_reference",
            "message": "A safe source reference is required.",
        }
    ]
    blocked_client = TestClient(
        create_app(runtime_with_preflight_client(FakePreflightAppClient(response=blocked_response)))
    )
    blocked_payload = _create_plan_with_lifecycle_reference(blocked_client)
    source_step = blocked_payload["plan"]["proposed_steps"][0]
    assert blocked_client.post(
        "/preflights",
        json={"run_id": blocked_payload["run_id"], "step_id": source_step["step_id"]},
    ).status_code == 200
    blocked_preview = blocked_client.post(
        "/action-requests",
        json={"run_id": blocked_payload["run_id"], "step_id": source_step["step_id"]},
    )

    stale_runtime = runtime_with_preflight_client(FakePreflightAppClient())
    stale_client = TestClient(create_app(stale_runtime))
    stale_payload = _create_plan_with_lifecycle_reference(stale_client)
    stale_studio_step = next(
        step
        for step in stale_payload["plan"]["proposed_steps"]
        if step["capability_id"] == "quant_studio.prepare_model_config_draft"
    )
    assert stale_client.post(
        "/confirmations",
        json={
            "run_id": stale_payload["run_id"],
            "step_id": stale_studio_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    ).status_code == 200
    stale_entry = stale_runtime.planner.ledger.list_entries()[0]
    stale_entry.capability_snapshot[1]["risk_tier"] = "read_only"
    stale_preview = stale_client.post(
        "/action-requests",
        json={"run_id": stale_payload["run_id"], "step_id": stale_studio_step["step_id"]},
    )

    unsafe_runtime = runtime_with_preflight_client(FakePreflightAppClient())
    unsafe_client = TestClient(create_app(unsafe_runtime))
    unsafe_payload = _create_plan_with_lifecycle_reference(unsafe_client)
    unsafe_studio_step = next(
        step
        for step in unsafe_payload["plan"]["proposed_steps"]
        if step["capability_id"] == "quant_studio.prepare_model_config_draft"
    )
    assert unsafe_client.post(
        "/confirmations",
        json={
            "run_id": unsafe_payload["run_id"],
            "step_id": unsafe_studio_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    ).status_code == 200
    unsafe_entry = unsafe_runtime.planner.ledger.list_entries()[0]
    unsafe_step = unsafe_entry.plan_snapshot["proposed_steps"][1]  # noqa: SLF001
    unsafe_step["action_input"]["raw_path"] = "C:\\Users\\matth\\Desktop\\private\\raw.csv"
    unsafe_preview = unsafe_client.post(
        "/action-requests",
        json={"run_id": unsafe_payload["run_id"], "step_id": unsafe_studio_step["step_id"]},
    )

    malformed_runtime = runtime_with_preflight_client(FakePreflightAppClient())
    malformed_client = TestClient(create_app(malformed_runtime))
    malformed_payload = _create_plan_with_lifecycle_reference(malformed_client)
    malformed_studio_step = next(
        step
        for step in malformed_payload["plan"]["proposed_steps"]
        if step["capability_id"] == "quant_studio.prepare_model_config_draft"
    )
    assert malformed_client.post(
        "/confirmations",
        json={
            "run_id": malformed_payload["run_id"],
            "step_id": malformed_studio_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    ).status_code == 200

    original_validate = malformed_runtime.contract_loader.validate_agent_contract_payload

    def rejecting_validate(payload: dict[str, Any], schema_name: str) -> None:
        if schema_name == "agent_action_request.v1.schema.json":
            raise ValueError("Generated request is malformed.")
        original_validate(payload, schema_name)

    monkeypatch.setattr(
        malformed_runtime.contract_loader,
        "validate_agent_contract_payload",
        rejecting_validate,
    )
    malformed_preview = malformed_client.post(
        "/action-requests",
        json={"run_id": malformed_payload["run_id"], "step_id": malformed_studio_step["step_id"]},
    )

    assert blocked_preview.status_code == 422
    assert blocked_preview.json()["detail"]["errors"][0]["code"] == "preflight_blocked_action_request"
    assert stale_preview.status_code == 422
    assert stale_preview.json()["detail"]["errors"][0]["code"] == "stale_action_request_capability_snapshot"
    assert unsafe_preview.status_code == 422
    assert unsafe_preview.json()["detail"]["errors"][0]["code"] == "unsafe_action_input"
    assert "private\\raw.csv" not in unsafe_preview.text
    assert malformed_preview.status_code == 422
    assert malformed_preview.json()["detail"]["errors"][0]["code"] == "malformed_generated_action_request"


def test_confirmation_rejects_duplicate_confirmation() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))
    plan_payload = _create_plan(client)
    studio_step = next(
        step
        for step in plan_payload["plan"]["proposed_steps"]
        if step["capability_id"] == "quant_studio.prepare_model_config_draft"
    )
    request = {
        "run_id": plan_payload["run_id"],
        "step_id": studio_step["step_id"],
        "confirmation_intent": "approve_plan_step",
    }

    first_response = client.post("/confirmations", json=request)
    duplicate_response = client.post("/confirmations", json=request)

    assert first_response.status_code == 200
    assert duplicate_response.status_code == 422
    assert duplicate_response.json()["detail"]["errors"][0]["code"] == "duplicate_confirmation"


def test_confirmation_rejects_unknown_run_and_unknown_step() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    unknown_run = client.post(
        "/confirmations",
        json={
            "run_id": "run_missing",
            "step_id": "step_1",
            "confirmation_intent": "approve_plan_step",
        },
    )
    plan_payload = _create_plan(client)
    unknown_step = client.post(
        "/confirmations",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": "step_missing",
            "confirmation_intent": "approve_plan_step",
        },
    )

    assert unknown_run.status_code == 422
    assert unknown_run.json()["detail"]["errors"][0]["code"] == "unknown_run"
    assert unknown_step.status_code == 422
    assert unknown_step.json()["detail"]["errors"][0]["code"] == "unknown_step"


def test_confirmation_rejects_step_without_confirmation_requirement() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))
    plan_payload = _create_plan(client)
    source_step = plan_payload["plan"]["proposed_steps"][0]

    response = client.post(
        "/confirmations",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": source_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["code"] == "confirmation_not_required"


def test_confirmation_rejects_blocked_plan() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))
    response = client.post(
        "/plans",
        json={
            "user_goal": "Build the lifecycle plan from whatever summaries are available.",
            "context_summary": {},
        },
    )
    assert response.status_code == 200
    plan_payload = response.json()
    studio_step = next(
        step
        for step in plan_payload["plan"]["proposed_steps"]
        if step["capability_id"] == "quant_studio.prepare_model_config_draft"
    )

    confirmation_response = client.post(
        "/confirmations",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": studio_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    )

    assert confirmation_response.status_code == 422
    assert confirmation_response.json()["detail"]["errors"][0]["code"] == "blocked_plan_confirmation"


def test_confirmation_rejects_when_preflight_is_blocked() -> None:
    blocked_response = _valid_preflight_response(status="blocked")
    blocked_response["blockers"] = [
        {
            "code": "missing_safe_source_reference",
            "message": "A safe source reference is required.",
        }
    ]
    client = TestClient(
        create_app(runtime_with_preflight_client(FakePreflightAppClient(response=blocked_response)))
    )
    plan_payload = _create_plan(client)
    source_step = plan_payload["plan"]["proposed_steps"][0]
    studio_step = next(
        step
        for step in plan_payload["plan"]["proposed_steps"]
        if step["capability_id"] == "quant_studio.prepare_model_config_draft"
    )
    preflight_response = client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": source_step["step_id"]},
    )
    assert preflight_response.status_code == 200

    confirmation_response = client.post(
        "/confirmations",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": studio_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    )

    assert confirmation_response.status_code == 422
    assert confirmation_response.json()["detail"]["errors"][0]["code"] == (
        "preflight_blocked_confirmation"
    )


def test_confirmation_rejects_attempted_execution_intent() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))
    plan_payload = _create_plan(client)
    studio_step = next(
        step
        for step in plan_payload["plan"]["proposed_steps"]
        if step["capability_id"] == "quant_studio.prepare_model_config_draft"
    )

    response = client.post(
        "/confirmations",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": studio_step["step_id"],
            "confirmation_intent": "execute_step",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["code"] == "unsupported_confirmation_intent"


def test_confirmation_rejects_stale_capability_snapshot() -> None:
    runtime = runtime_with_preflight_client(FakePreflightAppClient())
    runtime.planner.ledger.append(
        LedgerEntry(
            run_id="run_stale_capability",
            user_goal_summary="Test stale capability.",
            provider_mode=ProviderMode.fake_provider,
            redaction_summary=RedactionSummary(),
            context_preview=ContextPreview(context={"target_summary": "Safe summary"}),
            plan_snapshot={
                "plan_id": "plan_stale",
                "status": "valid",
                "execution_permitted": False,
                "proposed_steps": [
                    {
                        "step_id": "step_stale",
                        "title": "Prepare model configuration draft",
                        "capability_id": "quant_studio.prepare_model_config_draft",
                        "app_id": "quant_studio",
                        "risk_tier": "draft_only",
                        "operation": "plan",
                        "requires_confirmation": True,
                        "action_input": {"target_summary": "Safe summary"},
                    }
                ],
                "required_confirmations": [
                    {
                        "step_id": "step_stale",
                        "capability_id": "quant_studio.prepare_model_config_draft",
                        "risk_tier": "draft_only",
                        "reason": "Human review required.",
                    }
                ],
            },
            capability_snapshot=[
                {
                    "capability_id": "quant_studio.prepare_model_config_draft",
                    "app_id": "quant_studio",
                    "risk_tier": "read_only",
                    "enabled": True,
                }
            ],
            validation_results=PlanValidationResult(status="valid"),
        )
    )
    client = TestClient(create_app(runtime))

    response = client.post(
        "/confirmations",
        json={
            "run_id": "run_stale_capability",
            "step_id": "step_stale",
            "confirmation_intent": "approve_plan_step",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["code"] == (
        "stale_confirmation_capability_snapshot"
    )


def test_confirmation_rejects_unsafe_confirmation_record_without_leaking_value() -> None:
    runtime = runtime_with_preflight_client(FakePreflightAppClient())
    runtime.planner.ledger.append(
        LedgerEntry(
            run_id="run_unsafe_confirmation",
            user_goal_summary="Test unsafe confirmation.",
            provider_mode=ProviderMode.fake_provider,
            redaction_summary=RedactionSummary(),
            context_preview=ContextPreview(context={"target_summary": "Safe summary"}),
            plan_snapshot={
                "plan_id": "plan_unsafe",
                "status": "valid",
                "execution_permitted": False,
                "proposed_steps": [
                    {
                        "step_id": "step_unsafe",
                        "title": "Prepare model configuration draft",
                        "capability_id": "quant_studio.prepare_model_config_draft",
                        "app_id": "quant_studio",
                        "risk_tier": "draft_only",
                        "operation": "plan",
                        "requires_confirmation": True,
                        "action_input": {"target_summary": "Safe summary"},
                    }
                ],
                "required_confirmations": [
                    {
                        "step_id": "step_unsafe",
                        "capability_id": "quant_studio.prepare_model_config_draft",
                        "risk_tier": "draft_only",
                        "reason": "Review safe evidence before confirmation.",
                    }
                ],
            },
            capability_snapshot=[
                {
                    "capability_id": "quant_studio.prepare_model_config_draft",
                    "app_id": "quant_studio",
                    "risk_tier": "draft_only",
                    "enabled": True,
                }
            ],
            validation_results=PlanValidationResult(status="valid"),
        )
    )
    entry = runtime.planner.ledger.list_entries()[0]
    unsafe_snapshot = copy.deepcopy(entry.plan_snapshot)
    assert isinstance(unsafe_snapshot, dict)
    required = unsafe_snapshot["required_confirmations"]
    assert isinstance(required, list)
    required[0]["reason"] = "Review http://127.0.0.1/private before confirmation."
    runtime.planner.ledger._entries[0] = entry.model_copy(  # noqa: SLF001 - corrupts test ledger intentionally.
        update={"plan_snapshot": unsafe_snapshot},
        deep=True,
    )
    client = TestClient(create_app(runtime))

    response = client.post(
        "/confirmations",
        json={
            "run_id": "run_unsafe_confirmation",
            "step_id": "step_unsafe",
            "confirmation_intent": "approve_plan_step",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["code"] == "unsafe_confirmation_record"
    assert "127.0.0.1/private" not in response.text


def test_preflight_rejects_unknown_run_without_app_call() -> None:
    app_client = FakePreflightAppClient()
    client = TestClient(create_app(runtime_with_preflight_client(app_client)))

    response = client.post("/preflights", json={"run_id": "run_missing", "step_id": "step_1"})

    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["code"] == "unknown_run"
    assert app_client.calls == []


def test_preflight_rejects_unsupported_step_without_app_call() -> None:
    app_client = FakePreflightAppClient()
    client = TestClient(create_app(runtime_with_preflight_client(app_client)))
    plan_payload = _create_plan(client)
    studio_step = plan_payload["plan"]["proposed_steps"][1]

    response = client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["code"] == "unsupported_preflight_step"
    assert app_client.calls == []


def test_preflight_rejects_unknown_capability_snapshot_without_app_call() -> None:
    app_client = FakePreflightAppClient()
    runtime = runtime_with_preflight_client(app_client)
    runtime.planner.ledger.append(
        LedgerEntry(
            run_id="run_unknown_capability",
            user_goal_summary="Test unknown capability.",
            provider_mode=ProviderMode.fake_provider,
            redaction_summary=RedactionSummary(),
            context_preview=ContextPreview(context={"source_summary": "Safe summary"}),
            plan_snapshot={
                "proposed_steps": [
                    {
                        "step_id": "step_unknown",
                        "capability_id": "quant_data.run_source_preflight",
                        "app_id": "quant_data",
                        "action_input": {"source_summary": "Safe summary"},
                    }
                ]
            },
            capability_snapshot=[],
            validation_results=PlanValidationResult(status="valid"),
        )
    )
    client = TestClient(create_app(runtime))

    response = client.post(
        "/preflights",
        json={"run_id": "run_unknown_capability", "step_id": "step_unknown"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["code"] == "unknown_preflight_capability"
    assert app_client.calls == []


def test_preflight_reports_app_unavailable() -> None:
    app_client = FakePreflightAppClient(error=AppClientError("Quant Data unavailable.", status_code=503))
    client = TestClient(create_app(runtime_with_preflight_client(app_client)))
    plan_payload = _create_plan(client)
    source_step = plan_payload["plan"]["proposed_steps"][0]

    response = client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": source_step["step_id"]},
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "app_unavailable"


def test_preflight_rejects_when_owning_app_capability_discovery_is_unavailable() -> None:
    app_client = FakePreflightAppClient(
        discovery_errors_by_app={
            "quant_data": AppClientError(
                "Quant Data capability discovery app is unavailable.",
                status_code=503,
            )
        }
    )
    client = TestClient(create_app(runtime_with_preflight_client(app_client)))
    plan_payload = _create_plan(client)
    source_step = plan_payload["plan"]["proposed_steps"][0]

    response = client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": source_step["step_id"]},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["code"] == "preflight_app_unavailable"
    assert app_client.calls == []


def test_preflight_rejects_when_capability_is_no_longer_advertised() -> None:
    app_client = FakePreflightAppClient(
        discovery_payloads_by_app={
            "quant_data": _capabilities_payload("quant_data", []),
            "quant_monitoring": _capabilities_payload("quant_monitoring"),
        }
    )
    client = TestClient(create_app(runtime_with_preflight_client(app_client)))
    plan_payload = _create_plan(client)
    source_step = plan_payload["plan"]["proposed_steps"][0]

    response = client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": source_step["step_id"]},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["code"] == "preflight_capability_unavailable"
    assert app_client.calls == []


def test_preflight_rejects_malformed_app_response() -> None:
    malformed = _valid_preflight_response()
    malformed.pop("preflight_id")
    app_client = FakePreflightAppClient(response=malformed)
    client = TestClient(create_app(runtime_with_preflight_client(app_client)))
    plan_payload = _create_plan(client)
    source_step = plan_payload["plan"]["proposed_steps"][0]

    response = client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": source_step["step_id"]},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["code"] == "malformed_app_preflight_response"


def test_preflight_rejects_app_capability_mismatch() -> None:
    app_client = FakePreflightAppClient(response=_valid_preflight_response())
    client = TestClient(create_app(runtime_with_preflight_client(app_client)))
    plan_payload = _create_plan(client)
    monitoring_step = next(
        step
        for step in plan_payload["plan"]["proposed_steps"]
        if step["capability_id"] == "quant_monitoring.validate_bundle"
    )

    response = client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": monitoring_step["step_id"]},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["code"] == "preflight_capability_mismatch"


def test_preflight_rejects_unsafe_app_response() -> None:
    unsafe = _valid_preflight_response()
    unsafe["raw_paths"] = ["C:\\Users\\matth\\Desktop\\private\\raw.csv"]
    app_client = FakePreflightAppClient(response=unsafe)
    client = TestClient(create_app(runtime_with_preflight_client(app_client)))
    plan_payload = _create_plan(client)
    source_step = plan_payload["plan"]["proposed_steps"][0]

    response = client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": source_step["step_id"]},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["code"] == "unsafe_app_preflight_response"
    assert "private\\raw.csv" not in response.text


def test_unsupported_provider_mode_is_rejected_before_planning() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/plans",
        json={
            "user_goal": "Plan safely.",
            "context_summary": {},
            "policy": {"provider_mode": "vendor_managed_saas"},
        },
    )

    assert response.status_code == 422
