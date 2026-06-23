import copy
import json
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
from quant_agent_runtime.ledger import FileBackedLedger, InMemoryLedger
from quant_agent_runtime.model_gateway import FakePlanProvider
from quant_agent_runtime.model_gateway.provider import ProviderPlanRequest, ProviderResult
from quant_agent_runtime.models import (
    ContextPreview,
    LedgerEntry,
    PlanValidationResult,
    ProviderMode,
    RedactionSummary,
)
from quant_agent_runtime.orchestration import OrchestrationService
from quant_agent_runtime.plan_revision import PlanRevisionService
from quant_agent_runtime.plan_revision_activation import PlanRevisionActivationService
from quant_agent_runtime.planner import PlannerService
from quant_agent_runtime.preflight import PreflightService
from quant_agent_runtime.revalidation import RunRevalidationService
from quant_agent_runtime.retry import RetryService
from quant_agent_runtime.runtime import RuntimeContainer
from quant_agent_runtime.run_status import RunStatusService
from quant_agent_runtime.sample_autopilot import (
    SampleAutopilotPreviewService,
    SampleAutopilotStepService,
)
from quant_agent_runtime.sample_reset import SampleResetService


AGENT_ROOT = Path(__file__).resolve().parents[1]
QUANT_SUITE_ROOT = AGENT_ROOT.parent / "quant_suite"


class FakePreflightAppClient:
    def __init__(
        self,
        response: dict[str, Any] | None = None,
        responses_by_capability: dict[str, dict[str, Any]] | None = None,
        execution_response: dict[str, Any] | None = None,
        execution_responses_by_capability: dict[str, dict[str, Any]] | None = None,
        execution_error: AppClientError | None = None,
        reset_response: dict[str, Any] | None = None,
        reset_error: AppClientError | None = None,
        discovery_payloads_by_app: dict[str, dict[str, Any]] | None = None,
        discovery_errors_by_app: dict[str, AppClientError] | None = None,
        error: AppClientError | None = None,
    ) -> None:
        self.response = response or _valid_preflight_response()
        self.responses_by_capability = responses_by_capability or {}
        self.execution_response = execution_response or _valid_action_result()
        self.execution_responses_by_capability = execution_responses_by_capability or {}
        self.execution_error = execution_error
        self.reset_response = reset_response or _valid_sample_reset_response()
        self.reset_error = reset_error
        default_discovery_payloads = {
            "quant_data": _capabilities_payload("quant_data"),
            "quant_studio": _capabilities_payload("quant_studio"),
            "quant_documentation": _capabilities_payload("quant_documentation"),
            "quant_monitoring": _capabilities_payload("quant_monitoring"),
        }
        if discovery_payloads_by_app:
            default_discovery_payloads.update(discovery_payloads_by_app)
        self.discovery_payloads_by_app = default_discovery_payloads
        self.discovery_errors_by_app = discovery_errors_by_app or {}
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self.execution_calls: list[dict[str, Any]] = []
        self.reset_calls: list[dict[str, Any]] = []
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
        response = self.execution_responses_by_capability.get(capability_id)
        if response is None and self.execution_response.get("capability_id") == capability_id:
            response = self.execution_response
        if response is None and capability_id == "quant_studio.prepare_model_config_draft":
            response = _valid_action_result()
        if response is None and capability_id == "quant_documentation.create_draft_workspace":
            response = _valid_documentation_action_result(step_id="step_4")
        return copy.deepcopy(response or self.execution_response)

    def reset_sample_workspaces(self) -> dict[str, Any]:
        self.reset_calls.append({"app_id": "quant_studio", "route": "/api/sample-workspaces/reset"})
        if self.reset_error is not None:
            raise self.reset_error
        return copy.deepcopy(self.reset_response)


class RevisionProvider:
    def __init__(self, raw_output: dict[str, Any]) -> None:
        self.raw_output = raw_output

    def generate_plan(self, request: ProviderPlanRequest) -> ProviderResult:
        return ProviderResult(
            raw_output=copy.deepcopy(self.raw_output),
            metadata=FakePlanProvider().generate_plan(request).metadata,
        )


def runtime_with_preflight_client(
    app_client: FakePreflightAppClient,
    *,
    ledger: InMemoryLedger | None = None,
    sample_workspace_root: Path | None = None,
) -> RuntimeContainer:
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    capabilities = loader.load_agent_capabilities()
    provider_status = loader.load_agent_provider_status()
    ledger = ledger or InMemoryLedger()
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
            provider=FakePlanProvider(provider_status=provider_status),
            ledger=ledger,
            default_capabilities=capabilities or None,
        ),
        preflight=preflight,
        confirmation=ConfirmationService(ledger=ledger),
        action_request=action_request,
        execution=execution,
        retry=RetryService(ledger=ledger, execution=execution, app_client=app_client),
        run_status=RunStatusService(ledger=ledger, capability_discovery=discovery),
        orchestration=OrchestrationService(ledger=ledger),
        plan_revision=PlanRevisionService(
            provider=FakePlanProvider(provider_status=provider_status),
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
            sample_workspace_root=sample_workspace_root or QUANT_SUITE_ROOT / "fixtures" / "sample_workspaces",
        ),
        sample_autopilot_step=SampleAutopilotStepService(
            ledger=ledger,
            preflight=preflight,
            action_request=action_request,
            execution=execution,
            sample_workspace_root=sample_workspace_root or QUANT_SUITE_ROOT / "fixtures" / "sample_workspaces",
        ),
        sample_reset=SampleResetService(
            ledger=ledger,
            app_client=app_client,
            sample_workspace_root=sample_workspace_root or QUANT_SUITE_ROOT / "fixtures" / "sample_workspaces",
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


def _valid_sample_reset_response() -> dict[str, Any]:
    return {
        "status": "reset",
        "deleted_lifecycle_ids": ["sample_credit_pd_scorecard_panel"],
        "warnings": ["quant_data sample reset skipped in test harness"],
        "lifecycle_response": {
            "manifests": [
                {
                    "lifecycle_id": "lifecycle_user_keep",
                    "label": "User lifecycle preserved",
                }
            ]
        },
    }


def _valid_documentation_action_result(
    execution_status: str = "succeeded",
    *,
    step_id: str = "step_4",
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "data_policy": "summaries_and_references_only",
        "action_run_id": "action_documentation_draft_test",
        "step_id": step_id,
        "capability_id": "quant_documentation.create_draft_workspace",
        "app_id": "quant_documentation",
        "execution_status": execution_status,
        "accepted_input_summary": {
            "package_summary": "Documentation package summary is ready for draft workspace setup.",
            "lifecycle_id": "lifecycle_test",
        },
        "output_references": [
            {
                "reference_type": "documentation_draft",
                "reference_id": "documentation_draft_workspace_test",
                "label": "Documentation draft workspace",
                "review_status": "manual_review_required",
            }
        ],
        "warnings": [
            {
                "code": "human_review_required",
                "message": "No prose was generated; the workspace requires human drafting and review.",
            }
        ],
        "recoverable_errors": [],
        "terminal_errors": [],
        "artifact_references": [],
        "app_run_reference": None,
        "validation_results": {"status": "valid", "errors": [], "warnings": []},
        "recommended_next_step": {
            "label": "Review the draft workspace in Quant Documentation.",
            "target_app": "quant_documentation",
            "review_only": True,
        },
        "retry_allowed": False,
        "state_changed_since_planning": False,
    }


def _safe_documentation_package_summary() -> dict[str, Any]:
    return {
        "package_metadata": {
            "documentation_package_id": "documentation_package_test",
            "source_run_id": "studio_run_test",
            "checksum_sha256": "abc123safe",
            "status": "ready_for_draft_workspace",
        },
        "documentation_packages": [
            {
                "reference_id": "documentation_package_test",
                "reference_type": "documentation_package",
                "label": "Safe documentation package",
                "status": "ready",
                "summary": "Package summaries and citation evidence are ready for review.",
            }
        ],
        "section_evidence_map": [
            {
                "section_id": "model_design",
                "document_section": "Model design",
                "parent_heading": "Concept and methodology",
                "toc_level": 2,
                "display_order": 1,
                "required_evidence": ["approved_claim_model_design"],
            }
        ],
        "evidence_summary": {
            "citable_evidence_count": 4,
            "citation_map_rows": 6,
        },
        "known_gaps": [],
        "safety_warnings": [],
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
        elif app_id == "quant_documentation":
            capabilities = [
                {
                    "capability_id": "quant_documentation.create_draft_workspace",
                    "app_id": "quant_documentation",
                    "version": "1.0-draft",
                    "display_name": "Create documentation draft workspace",
                    "summary": "Creates a review-only documentation draft workspace reference from safe package summaries.",
                    "risk_tier": "draft_only",
                    "enabled": True,
                    "preflight_required": False,
                    "confirmation_required": True,
                    "execution_supported": True,
                    "idempotent": True,
                    "reversible": True,
                    "side_effects": ["draft_only_after_confirmation"],
                    "input_schema": {"required_fields": ["package_summary"]},
                    "output_schema": {"safe_reference_types": ["documentation_draft"]},
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
            "context_summary": _safe_lifecycle_context(),
        },
    )
    assert response.status_code == 200
    return response.json()


def _safe_lifecycle_context(
    *,
    lifecycle_state: str = "ready_for_modeling",
    lifecycle_summary: str = "Lifecycle has safe source and target summaries.",
) -> dict[str, Any]:
    return {
        "lifecycle_summary": {
            "lifecycle_id": "lifecycle_test",
            "state": lifecycle_state,
            "summary": lifecycle_summary,
        },
        "source_summary": "Development sample is registered.",
        "target_summary": "Default flag is the candidate target.",
        "package_summary": "No documentation package exists yet.",
        "bundle_summary": "Monitoring bundle is available.",
    }


def _safe_sample_lifecycle_context(
    *,
    sample_workspace_id: str = "credit_pd_scorecard_panel",
    sample_owned: bool = True,
    lifecycle_id: str = "sample_credit_pd_scorecard_panel",
) -> dict[str, Any]:
    context = _safe_lifecycle_context(
        lifecycle_state="retrain_recommended",
        lifecycle_summary="Credit PD sample lifecycle has safe seeded evidence.",
    )
    context["lifecycle_summary"] = {
        **context["lifecycle_summary"],
        "lifecycle_id": lifecycle_id,
        "sample_workspace": {
            "sample_workspace": True,
            "sample_workspace_id": sample_workspace_id,
            "sample_owned": sample_owned,
        },
    }
    context["package_summary"] = {
        "documentation_packages": [
            {
                "reference_id": "sample_credit_pd_documentation_package",
                "label": "Sample methodology package",
                "status": "available",
                "summary": "Safe documentation package summary is available.",
            }
        ],
        "documentation_drafts": [
            {
                "reference_id": "sample_credit_pd_documentation_draft",
                "label": "Sample methodology draft",
                "status": "available",
                "summary": "Safe draft summary is available.",
            }
        ],
    }
    context["bundle_summary"] = {
        "monitoring_bundles": [
            {
                "reference_id": "sample_credit_pd_monitoring_bundle",
                "label": "Sample monitoring bundle",
                "status": "available",
                "summary": "Safe monitoring bundle summary is available.",
            }
        ],
    }
    return context


def _create_plan_with_documentation_package_reference(client: TestClient) -> dict[str, Any]:
    response = client.post(
        "/plans",
        json={
            "user_goal": "Create a review-only documentation draft workspace from safe package summaries.",
            "context_summary": {
                "lifecycle_summary": {
                    "lifecycle_id": "lifecycle_test",
                    "state": "ready_for_documentation",
                    "summary": "Lifecycle has safe source, target, package, and monitoring summaries.",
                },
                "source_summary": "Development sample is registered.",
                "target_summary": "Default flag is the candidate target.",
                "package_summary": _safe_documentation_package_summary(),
                "bundle_summary": "Monitoring bundle is available.",
            },
        },
    )
    assert response.status_code == 200
    return response.json()


def _step_for_capability(plan_payload: dict[str, Any], capability_id: str) -> dict[str, Any]:
    return next(
        step
        for step in plan_payload["plan"]["proposed_steps"]
        if step["capability_id"] == capability_id
    )


def _run_source_preflight(client: TestClient, plan_payload: dict[str, Any]) -> dict[str, Any]:
    source_step = _step_for_capability(plan_payload, "quant_data.run_source_preflight")
    response = client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": source_step["step_id"]},
    )
    assert response.status_code == 200
    return response.json()


def _create_studio_preview(
    client: TestClient,
    plan_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    studio_step = _step_for_capability(plan_payload, "quant_studio.prepare_model_config_draft")
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
    return studio_step, preview_response.json()


def _complete_studio_step(
    client: TestClient,
    plan_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    studio_step, preview_payload = _create_studio_preview(client, plan_payload)
    execution_response = client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )
    assert execution_response.status_code == 200
    return studio_step, preview_payload


def _create_documentation_preview(
    client: TestClient,
    plan_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    documentation_step = _step_for_capability(
        plan_payload,
        "quant_documentation.create_draft_workspace",
    )
    confirmation_response = client.post(
        "/confirmations",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": documentation_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    )
    assert confirmation_response.status_code == 200
    preview_response = client.post(
        "/action-requests",
        json={"run_id": plan_payload["run_id"], "step_id": documentation_step["step_id"]},
    )
    assert preview_response.status_code == 200
    return documentation_step, preview_response.json()


def _complete_documentation_step(
    client: TestClient,
    plan_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    documentation_step, preview_payload = _create_documentation_preview(client, plan_payload)
    execution_response = client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": documentation_step["step_id"]},
    )
    assert execution_response.status_code == 200
    return documentation_step, preview_payload


def _advance_to_studio_step(client: TestClient, plan_payload: dict[str, Any]) -> None:
    _run_source_preflight(client, plan_payload)


def _advance_to_documentation_step(client: TestClient, plan_payload: dict[str, Any]) -> None:
    _advance_to_studio_step(client, plan_payload)
    _complete_studio_step(client, plan_payload)


def _advance_to_monitoring_step(client: TestClient, plan_payload: dict[str, Any]) -> None:
    _advance_to_documentation_step(client, plan_payload)
    _complete_documentation_step(client, plan_payload)


def _create_confirmed_studio_preview(client: TestClient) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan_payload = _create_plan_with_lifecycle_reference(client)
    _advance_to_studio_step(client, plan_payload)
    studio_step, preview_payload = _create_studio_preview(client, plan_payload)
    return plan_payload, studio_step, preview_payload


def _create_confirmed_documentation_preview(
    client: TestClient,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan_payload = _create_plan_with_documentation_package_reference(client)
    _advance_to_documentation_step(client, plan_payload)
    documentation_step, preview_payload = _create_documentation_preview(client, plan_payload)
    return plan_payload, documentation_step, preview_payload


def _create_missing_input_revision(client: TestClient) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_payload = client.post(
        "/plans",
        json={"user_goal": "Plan with missing summaries.", "context_summary": {}},
    ).json()
    revision_payload = client.post(
        "/plan-revisions",
        json={
            "run_id": plan_payload["run_id"],
            "revision_intent": "revise_plan",
            "reason": "missing_inputs",
            "current_context_summary": {
                "lifecycle_summary": {"lifecycle_id": "lifecycle_test", "state": "ready_for_modeling"},
                "source_summary": "Development sample is registered.",
                "target_summary": "Default flag is the candidate target.",
                "package_summary": "Documentation package is available.",
                "bundle_summary": "Monitoring bundle is available.",
            },
        },
    ).json()
    return plan_payload, revision_payload


def test_health_endpoint_works() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["plan_only_mode"] is False
    assert response.json()["execution_supported"] is True
    assert response.json()["execution_support_level"] == "single_step_review_draft_actions_only"


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
    assert "POST /retries" in manifest["supported_routes"]
    assert "GET /runs" in manifest["supported_routes"]
    assert "GET /runs/{run_id}" in manifest["supported_routes"]
    assert "GET /runs/{run_id}/orchestration" in manifest["supported_routes"]
    assert "GET /runs/{run_id}/ledger" in manifest["supported_routes"]
    assert "POST /cancellations" in manifest["supported_routes"]
    assert "POST /pauses" in manifest["supported_routes"]
    assert "POST /resumptions" in manifest["supported_routes"]
    assert "POST /plan-revisions" in manifest["supported_routes"]
    assert "POST /plan-revision-activations" in manifest["supported_routes"]
    assert "POST /run-revalidations" in manifest["supported_routes"]
    assert "POST /autopilot-previews" in manifest["supported_routes"]
    assert "POST /autopilot-steps" in manifest["supported_routes"]
    assert "POST /sample-reset-previews" in manifest["supported_routes"]
    assert "POST /sample-resets" in manifest["supported_routes"]
    assert manifest["runtime_health_endpoint"] == "/health"
    assert manifest["execution_support_level"] == "single_step_review_draft_actions_only"
    assert manifest["ledger_support_level"] == "local_json_file_backed"
    assert manifest["recovery_support_level"] == "manual_pause_resume_only"
    assert manifest["orchestration_support_level"] == "manual_guided_existing_steps_only"
    assert manifest["retry_support_level"] == "manual_current_step_only"
    assert manifest["plan_revision_support_level"] == "manual_preview_only"
    assert manifest["plan_revision_activation_support_level"] == "manual_child_run_only"
    assert manifest["revalidation_support_level"] == "manual_context_check_only"
    assert manifest["autopilot_support_level"] == "sample_owned_one_step_manual_advance"
    assert manifest["sample_reset_support_level"] == "sample_owned_studio_orchestrated_only"
    assert manifest["ledger_storage"]["storage_mode"] == "memory"
    assert manifest["provider_status"]["supports_execution"] is False
    assert manifest["provider_status"]["hosted_provider_enabled"] is False
    assert "quant_data:/api/agent/capabilities" in manifest["capability_discovery_endpoints"]
    assert "quant_studio:/api/agent/capabilities" in manifest["capability_discovery_endpoints"]
    assert "quant_documentation:/api/agent/capabilities" in manifest["capability_discovery_endpoints"]
    assert "quant_monitoring:/api/agent/capabilities" in manifest["capability_discovery_endpoints"]
    assert manifest["supported_preflight_capabilities"] == [
        "quant_data.run_source_preflight",
        "quant_monitoring.validate_bundle",
    ]
    assert manifest["capability_discovery"]["discovered_apps"] == [
        "quant_data",
        "quant_studio",
        "quant_documentation",
        "quant_monitoring",
    ]
    assert manifest["capability_discovery"]["unavailable_apps"] == []
    assert manifest["capability_discovery"]["unsupported_capability_ids"] == []
    assert manifest["capability_discovery"]["supported_preflight_capabilities"] == [
        "quant_data.run_source_preflight",
        "quant_monitoring.validate_bundle",
    ]
    assert manifest["supported_execution_capabilities"] == [
        "quant_studio.prepare_model_config_draft",
        "quant_documentation.create_draft_workspace",
    ]
    assert manifest["capability_discovery"]["supported_execution_capabilities"] == [
        "quant_studio.prepare_model_config_draft",
        "quant_documentation.create_draft_workspace",
    ]


def test_file_backed_ledger_persists_plan_and_exposes_safe_ledger(tmp_path: Path) -> None:
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path,
        validate_contract=loader.validate_agent_contract_payload,
    )
    runtime = runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)
    client = TestClient(create_app(runtime))

    plan_payload = _create_plan_with_lifecycle_reference(client)
    ledger_files = list(tmp_path.glob("*.json"))
    assert len(ledger_files) == 1
    stored_payload = json.loads(ledger_files[0].read_text(encoding="utf-8"))
    assert stored_payload["run_id"] == plan_payload["run_id"]
    loader.validate_agent_contract_payload(stored_payload, "agent_execution_ledger.v1.schema.json")

    ledger_response = client.get(f"/runs/{plan_payload['run_id']}/ledger")
    assert ledger_response.status_code == 200
    exported = ledger_response.json()
    assert exported["run_id"] == plan_payload["run_id"]
    assert exported["data_policy"] == "summaries_and_references_only"
    assert str(tmp_path) not in ledger_response.text
    assert "raw_path" not in ledger_response.text
    loader.validate_agent_contract_payload(exported, "agent_execution_ledger.v1.schema.json")


def test_plan_revision_preview_for_missing_inputs_validates_and_preserves_active_plan() -> None:
    runtime = runtime_with_preflight_client(FakePreflightAppClient())
    client = TestClient(create_app(runtime))
    blocked_response = client.post(
        "/plans",
        json={"user_goal": "Plan with missing summaries.", "context_summary": {}},
    )
    assert blocked_response.status_code == 200
    blocked_payload = blocked_response.json()
    parent_plan_id = blocked_payload["plan"]["plan_id"]
    original_snapshot = copy.deepcopy(runtime.planner.ledger.get(blocked_payload["run_id"]).plan_snapshot)

    request = {
        "run_id": blocked_payload["run_id"],
        "revision_intent": "revise_plan",
        "reason": "missing_inputs",
        "current_context_summary": {
            "lifecycle_summary": {"lifecycle_id": "lifecycle_test", "state": "ready_for_modeling"},
            "source_summary": "Development sample is registered.",
            "target_summary": "Default flag is the candidate target.",
            "package_summary": "Documentation package is available.",
            "bundle_summary": "Monitoring bundle is available.",
        },
    }
    first = client.post("/plan-revisions", json=request)
    second = client.post("/plan-revisions", json=request)

    assert first.status_code == 200
    assert second.status_code == 200
    payload = first.json()
    assert payload["parent_plan_id"] == parent_plan_id
    assert payload["revised_plan"]["status"] == "valid"
    assert payload["revised_plan"]["parent_plan_id"] == parent_plan_id
    assert payload["revised_plan"]["revision_source_run_id"] == blocked_payload["run_id"]
    assert payload["revision_event"]["event_type"] == "plan_revision_preview"
    assert payload["revision_event"]["execution_permitted"] is False
    assert payload["orchestration"]["plan_id"] == parent_plan_id
    assert second.json()["revision_id"] == payload["revision_id"]
    entry = runtime.planner.ledger.get(blocked_payload["run_id"])
    assert entry.plan_snapshot == original_snapshot
    assert [
        event.get("event_type")
        for event in entry.recovery_events
        if event.get("event_type") == "plan_revision_preview"
    ] == ["plan_revision_preview"]
    runtime.contract_loader.validate_agent_contract_payload(
        payload["revised_plan"],
        "agent_plan.v1.schema.json",
    )
    runtime.contract_loader.validate_agent_contract_payload(
        entry.model_dump(mode="json"),
        "agent_execution_ledger.v1.schema.json",
    )


def test_plan_revision_preview_supports_preflight_blocked_stale_and_recoverable_failure() -> None:
    blocked_preflight = _valid_preflight_response(status="blocked")
    blocked_preflight["blockers"] = [
        {"code": "missing_safe_source_reference", "message": "Safe source evidence is missing."}
    ]
    preflight_runtime = runtime_with_preflight_client(
        FakePreflightAppClient(response=blocked_preflight)
    )
    preflight_client = TestClient(create_app(preflight_runtime))
    preflight_plan = _create_plan_with_lifecycle_reference(preflight_client)
    source_step = _step_for_capability(preflight_plan, "quant_data.run_source_preflight")
    assert preflight_client.post(
        "/preflights",
        json={"run_id": preflight_plan["run_id"], "step_id": source_step["step_id"]},
    ).status_code == 200
    preflight_revision = preflight_client.post(
        "/plan-revisions",
        json={
            "run_id": preflight_plan["run_id"],
            "revision_intent": "revise_plan",
            "reason": "preflight_blocked",
            "current_context_summary": {
                "lifecycle_summary": {"lifecycle_id": "lifecycle_test"},
                "source_summary": "Safe source reference has been refreshed.",
                "target_summary": "Default flag is the candidate target.",
                "package_summary": "Documentation package is available.",
                "bundle_summary": "Monitoring bundle is available.",
            },
        },
    )

    stale_client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))
    stale_plan = _create_plan_with_lifecycle_reference(stale_client)
    stale_revision = stale_client.post(
        "/plan-revisions",
        json={
            "run_id": stale_plan["run_id"],
            "revision_intent": "revise_plan",
            "reason": "stale_state",
            "current_context_summary": {
                "lifecycle_summary": {
                    "lifecycle_id": "lifecycle_test",
                    "state": "ready_for_documentation",
                    "summary": "Lifecycle state changed after planning.",
                },
                "source_summary": "Development sample is registered.",
                "target_summary": "Default flag target was reviewed.",
                "package_summary": "Documentation package is now available.",
                "bundle_summary": "Monitoring bundle is available.",
            },
        },
    )

    failed_runtime = runtime_with_preflight_client(
        FakePreflightAppClient(
            execution_error=AppClientError("Quant Studio execution app is unavailable.", status_code=503)
        )
    )
    failed_client = TestClient(create_app(failed_runtime))
    failed_plan, failed_step, _preview = _create_confirmed_studio_preview(failed_client)
    failed_execution = failed_client.post(
        "/executions",
        json={"run_id": failed_plan["run_id"], "step_id": failed_step["step_id"]},
    )
    assert failed_execution.status_code == 200
    assert failed_execution.json()["run_state"] == "failed_recoverable"
    failed_revision = failed_client.post(
        "/plan-revisions",
        json={
            "run_id": failed_plan["run_id"],
            "revision_intent": "revise_plan",
            "reason": "failed_recoverable",
            "current_context_summary": {
                "lifecycle_summary": {"lifecycle_id": "lifecycle_test"},
                "source_summary": "Development sample is registered.",
                "target_summary": "Retry after Studio availability recovers.",
                "package_summary": "Documentation package is available.",
                "bundle_summary": "Monitoring bundle is available.",
            },
        },
    )

    assert preflight_revision.status_code == 200
    assert preflight_revision.json()["stale_state_summary"]["current_context_provided"] is True
    assert preflight_revision.json()["revision_event"]["blocker_summary"]["latest_preflight_status"] == "blocked"
    assert stale_revision.status_code == 200
    assert stale_revision.json()["stale_state_summary"]["state_changed_since_planning"] is True
    assert failed_revision.status_code == 200
    assert failed_revision.json()["revision_event"]["blocker_summary"]["latest_action_result_status"] == (
        "failed_recoverable"
    )


def test_plan_revision_preview_rejects_invalid_states_context_and_provider_output() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))
    unknown = client.post(
        "/plan-revisions",
        json={"run_id": "run_missing", "revision_intent": "revise_plan", "reason": "user_requested"},
    )
    valid_plan = _create_plan_with_lifecycle_reference(client)
    no_need = client.post(
        "/plan-revisions",
        json={
            "run_id": valid_plan["run_id"],
            "revision_intent": "revise_plan",
            "reason": "user_requested",
        },
    )
    unsafe = client.post(
        "/plan-revisions",
        json={
            "run_id": valid_plan["run_id"],
            "revision_intent": "revise_plan",
            "reason": "stale_state",
            "current_context_summary": {
                "lifecycle_summary": {"lifecycle_id": "lifecycle_test"},
                "raw_path": "C:\\Users\\matth\\Desktop\\private\\raw.csv",
            },
        },
    )

    paused_runtime = runtime_with_preflight_client(FakePreflightAppClient())
    paused_client = TestClient(create_app(paused_runtime))
    paused_plan = _create_plan_with_lifecycle_reference(paused_client)
    assert paused_client.post(
        "/pauses",
        json={
            "run_id": paused_plan["run_id"],
            "pause_intent": "pause_run",
            "reason": "user_paused",
        },
    ).status_code == 200
    paused = paused_client.post(
        "/plan-revisions",
        json={"run_id": paused_plan["run_id"], "revision_intent": "revise_plan", "reason": "user_requested"},
    )

    cancelled_runtime = runtime_with_preflight_client(FakePreflightAppClient())
    cancelled_client = TestClient(create_app(cancelled_runtime))
    cancelled_plan = _create_plan_with_lifecycle_reference(cancelled_client)
    assert cancelled_client.post(
        "/cancellations",
        json={
            "run_id": cancelled_plan["run_id"],
            "cancellation_intent": "cancel_run",
            "reason": "user_cancelled",
        },
    ).status_code == 200
    cancelled = cancelled_client.post(
        "/plan-revisions",
        json={"run_id": cancelled_plan["run_id"], "revision_intent": "revise_plan", "reason": "user_requested"},
    )

    missing_plan_runtime = runtime_with_preflight_client(FakePreflightAppClient())
    missing_plan_runtime.planner.ledger.append(
        LedgerEntry(
            run_id="run_missing_plan",
            user_goal_summary="Missing plan.",
            provider_mode=ProviderMode.fake_provider,
            redaction_summary=RedactionSummary(),
            context_preview=ContextPreview(context={}),
            plan_snapshot=None,
            validation_results=PlanValidationResult(status="valid"),
        )
    )
    missing_plan_client = TestClient(create_app(missing_plan_runtime))
    missing_plan = missing_plan_client.post(
        "/plan-revisions",
        json={"run_id": "run_missing_plan", "revision_intent": "revise_plan", "reason": "user_requested"},
    )

    malformed_runtime = runtime_with_preflight_client(FakePreflightAppClient())
    malformed_runtime.plan_revision._provider = RevisionProvider({"missing": "shape"})  # noqa: SLF001
    malformed_client = TestClient(create_app(malformed_runtime))
    malformed_plan = malformed_client.post(
        "/plans",
        json={"user_goal": "Plan with missing summaries.", "context_summary": {}},
    ).json()
    malformed = malformed_client.post(
        "/plan-revisions",
        json={
            "run_id": malformed_plan["run_id"],
            "revision_intent": "revise_plan",
            "reason": "missing_inputs",
        },
    )

    unsupported_runtime = runtime_with_preflight_client(FakePreflightAppClient())
    unsupported_runtime.plan_revision._provider = RevisionProvider(  # noqa: SLF001
        {
            "user_goal_summary": "Unsafe revision.",
            "assumptions": [],
            "missing_inputs": [],
            "steps": [
                {
                    "step_id": "step_unknown",
                    "title": "Unknown capability",
                    "capability_id": "quant_unknown.perform_action",
                    "app_id": "quant_unknown",
                    "risk_tier": "read_only",
                    "operation": "plan",
                    "requires_confirmation": False,
                    "action_input": {},
                    "expected_artifacts": [],
                    "validation_checks": [],
                }
            ],
        }
    )
    unsupported_client = TestClient(create_app(unsupported_runtime))
    unsupported_plan = unsupported_client.post(
        "/plans",
        json={"user_goal": "Plan with missing summaries.", "context_summary": {}},
    ).json()
    unsupported = unsupported_client.post(
        "/plan-revisions",
        json={
            "run_id": unsupported_plan["run_id"],
            "revision_intent": "revise_plan",
            "reason": "missing_inputs",
        },
    )

    execution_runtime = runtime_with_preflight_client(FakePreflightAppClient())
    execution_runtime.plan_revision._provider = RevisionProvider(  # noqa: SLF001
        {
            "user_goal_summary": "Attempt execution.",
            "assumptions": [],
            "missing_inputs": [],
            "steps": [
                {
                    "step_id": "step_execute",
                    "title": "Execute",
                    "capability_id": "quant_data.run_source_preflight",
                    "app_id": "quant_data",
                    "risk_tier": "workflow_preflight",
                    "operation": "execute",
                    "preflight_required": True,
                    "requires_confirmation": False,
                    "action_input": {"source_summary": "Safe summary"},
                    "expected_artifacts": [],
                    "validation_checks": [],
                }
            ],
        }
    )
    execution_client = TestClient(create_app(execution_runtime))
    execution_plan = execution_client.post(
        "/plans",
        json={"user_goal": "Plan with missing summaries.", "context_summary": {}},
    ).json()
    attempted_execution = execution_client.post(
        "/plan-revisions",
        json={
            "run_id": execution_plan["run_id"],
            "revision_intent": "revise_plan",
            "reason": "missing_inputs",
        },
    )

    assert unknown.status_code == 422
    assert unknown.json()["detail"]["errors"][0]["code"] == "unknown_run"
    assert no_need.status_code == 422
    assert no_need.json()["detail"]["errors"][0]["code"] == "no_plan_revision_needed"
    assert unsafe.status_code == 422
    assert unsafe.json()["detail"]["errors"][0]["code"] == "unsafe_revision_context"
    assert "private\\raw.csv" not in unsafe.text
    assert paused.status_code == 422
    assert paused.json()["detail"]["errors"][0]["code"] == "paused_run_plan_revision"
    assert cancelled.status_code == 422
    assert cancelled.json()["detail"]["errors"][0]["code"] == "terminal_run_plan_revision"
    assert missing_plan.status_code == 422
    assert missing_plan.json()["detail"]["errors"][0]["code"] == "missing_plan_revision_source"
    assert malformed.status_code == 422
    assert malformed.json()["detail"]["errors"][0]["code"] == "malformed_revision_provider_output"
    assert unsupported.status_code == 422
    assert unsupported.json()["detail"]["errors"][0]["code"] == "unknown_capability"
    assert attempted_execution.status_code == 422
    assert attempted_execution.json()["detail"]["errors"][0]["code"] == "execution_not_allowed"


def test_plan_revision_activation_creates_child_run_and_preserves_parent_plan() -> None:
    runtime = runtime_with_preflight_client(FakePreflightAppClient())
    client = TestClient(create_app(runtime))
    blocked_response = client.post(
        "/plans",
        json={"user_goal": "Plan with missing summaries.", "context_summary": {}},
    )
    assert blocked_response.status_code == 200
    parent_payload = blocked_response.json()
    parent_entry = runtime.planner.ledger.get(parent_payload["run_id"])
    assert parent_entry is not None
    original_parent_plan = copy.deepcopy(parent_entry.plan_snapshot)
    revision = client.post(
        "/plan-revisions",
        json={
            "run_id": parent_payload["run_id"],
            "revision_intent": "revise_plan",
            "reason": "missing_inputs",
            "current_context_summary": {
                "lifecycle_summary": {"lifecycle_id": "lifecycle_test", "state": "ready_for_modeling"},
                "source_summary": "Development sample is registered.",
                "target_summary": "Default flag is the candidate target.",
                "package_summary": "Documentation package is available.",
                "bundle_summary": "Monitoring bundle is available.",
            },
        },
    )
    assert revision.status_code == 200
    revision_payload = revision.json()
    first_activation = client.post(
        "/plan-revision-activations",
        json={
            "run_id": parent_payload["run_id"],
            "revision_id": revision_payload["revision_id"],
            "activation_intent": "activate_plan_revision",
        },
    )
    second_activation = client.post(
        "/plan-revision-activations",
        json={
            "run_id": parent_payload["run_id"],
            "revision_id": revision_payload["revision_id"],
            "activation_intent": "activate_plan_revision",
        },
    )

    assert first_activation.status_code == 200
    assert second_activation.status_code == 200
    payload = first_activation.json()
    child_run_id = payload["child_run_id"]
    assert child_run_id != parent_payload["run_id"]
    assert second_activation.json()["child_run_id"] == child_run_id
    assert payload["parent_run_id"] == parent_payload["run_id"]
    assert payload["revision_id"] == revision_payload["revision_id"]
    assert payload["activation_event"]["event_type"] == "plan_revision_activation"
    assert payload["activation_event"]["active_plan_replaced"] is False
    assert payload["activation_event"]["execution_permitted"] is False
    assert payload["activated_plan"]["plan_id"] == revision_payload["revised_plan"]["plan_id"]
    assert payload["child_orchestration"]["parent_run_id"] == parent_payload["run_id"]
    assert payload["child_orchestration"]["activated_revision_id"] == revision_payload["revision_id"]
    assert payload["child_run_state"] == payload["child_orchestration"]["run_state"]

    recorded_parent = runtime.planner.ledger.get(parent_payload["run_id"])
    recorded_child = runtime.planner.ledger.get(child_run_id)
    assert recorded_parent is not None
    assert recorded_child is not None
    assert recorded_parent.plan_snapshot == original_parent_plan
    assert recorded_parent.child_run_ids == [child_run_id]
    assert [
        event.get("event_type")
        for event in recorded_parent.recovery_events
        if event.get("event_type") == "plan_revision_activation"
    ] == ["plan_revision_activation"]
    assert recorded_child.parent_run_id == parent_payload["run_id"]
    assert recorded_child.parent_plan_id == parent_payload["plan"]["plan_id"]
    assert recorded_child.activated_revision_id == revision_payload["revision_id"]
    assert recorded_child.plan_snapshot == revision_payload["revised_plan"]

    parent_status = client.get(f"/runs/{parent_payload['run_id']}")
    child_status = client.get(f"/runs/{child_run_id}")
    run_list = client.get("/runs")
    parent_ledger = client.get(f"/runs/{parent_payload['run_id']}/ledger")
    child_ledger = client.get(f"/runs/{child_run_id}/ledger")
    assert parent_status.status_code == 200
    assert parent_status.json()["child_run_ids"] == [child_run_id]
    assert parent_status.json()["plan"] == original_parent_plan
    assert child_status.status_code == 200
    assert child_status.json()["parent_run_id"] == parent_payload["run_id"]
    assert child_status.json()["activated_revision_id"] == revision_payload["revision_id"]
    assert run_list.status_code == 200
    summaries = {item["run_id"]: item for item in run_list.json()["runs"]}
    assert summaries[parent_payload["run_id"]]["child_run_ids"] == [child_run_id]
    assert summaries[child_run_id]["parent_run_id"] == parent_payload["run_id"]
    runtime.contract_loader.validate_agent_contract_payload(
        parent_ledger.json(),
        "agent_execution_ledger.v1.schema.json",
    )
    runtime.contract_loader.validate_agent_contract_payload(
        child_ledger.json(),
        "agent_execution_ledger.v1.schema.json",
    )


def test_plan_revision_activation_rejects_invalid_states_and_payloads() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))
    unknown_run = client.post(
        "/plan-revision-activations",
        json={
            "run_id": "run_missing",
            "revision_id": "revision_missing",
            "activation_intent": "activate_plan_revision",
        },
    )
    parent = client.post(
        "/plans",
        json={"user_goal": "Plan with missing summaries.", "context_summary": {}},
    ).json()
    unknown_revision = client.post(
        "/plan-revision-activations",
        json={
            "run_id": parent["run_id"],
            "revision_id": "revision_missing",
            "activation_intent": "activate_plan_revision",
        },
    )
    extra_payload = client.post(
        "/plan-revision-activations",
        json={
            "run_id": parent["run_id"],
            "revision_id": "revision_missing",
            "activation_intent": "activate_plan_revision",
            "steps": [],
        },
    )

    paused_runtime = runtime_with_preflight_client(FakePreflightAppClient())
    paused_client = TestClient(create_app(paused_runtime))
    paused_plan, paused_revision = _create_missing_input_revision(paused_client)
    assert paused_client.post(
        "/pauses",
        json={"run_id": paused_plan["run_id"], "pause_intent": "pause_run", "reason": "user_paused"},
    ).status_code == 200
    paused = paused_client.post(
        "/plan-revision-activations",
        json={
            "run_id": paused_plan["run_id"],
            "revision_id": paused_revision["revision_id"],
            "activation_intent": "activate_plan_revision",
        },
    )

    cancelled_runtime = runtime_with_preflight_client(FakePreflightAppClient())
    cancelled_client = TestClient(create_app(cancelled_runtime))
    cancelled_plan, cancelled_revision = _create_missing_input_revision(cancelled_client)
    assert cancelled_client.post(
        "/cancellations",
        json={
            "run_id": cancelled_plan["run_id"],
            "cancellation_intent": "cancel_run",
            "reason": "user_cancelled",
        },
    ).status_code == 200
    cancelled = cancelled_client.post(
        "/plan-revision-activations",
        json={
            "run_id": cancelled_plan["run_id"],
            "revision_id": cancelled_revision["revision_id"],
            "activation_intent": "activate_plan_revision",
        },
    )

    malformed_runtime = runtime_with_preflight_client(FakePreflightAppClient())
    malformed_client = TestClient(create_app(malformed_runtime))
    malformed_plan = _create_plan_with_lifecycle_reference(malformed_client)
    malformed_entry = malformed_runtime.planner.ledger.get(malformed_plan["run_id"])
    assert malformed_entry is not None
    malformed_runtime.planner.ledger._entries[0] = malformed_entry.model_copy(  # noqa: SLF001
        update={
            "recovery_events": [
                {
                    "recovery_event_id": "revision_malformed",
                    "revision_id": "revision_malformed",
                    "event_type": "plan_revision_preview",
                    "status": "previewed",
                    "parent_plan_id": malformed_plan["plan"]["plan_id"],
                    "execution_permitted": False,
                }
            ]
        },
        deep=True,
    )
    malformed = malformed_client.post(
        "/plan-revision-activations",
        json={
            "run_id": malformed_plan["run_id"],
            "revision_id": "revision_malformed",
            "activation_intent": "activate_plan_revision",
        },
    )

    stale_runtime = runtime_with_preflight_client(FakePreflightAppClient())
    stale_client = TestClient(create_app(stale_runtime))
    stale_plan, stale_revision = _create_missing_input_revision(stale_client)
    stale_entry = stale_runtime.planner.ledger.get(stale_plan["run_id"])
    assert stale_entry is not None
    stale_snapshot = copy.deepcopy(stale_entry.capability_snapshot)
    stale_snapshot[0]["enabled"] = False
    stale_runtime.planner.ledger._entries[0] = stale_entry.model_copy(  # noqa: SLF001
        update={"capability_snapshot": stale_snapshot},
        deep=True,
    )
    stale = stale_client.post(
        "/plan-revision-activations",
        json={
            "run_id": stale_plan["run_id"],
            "revision_id": stale_revision["revision_id"],
            "activation_intent": "activate_plan_revision",
        },
    )

    unsupported_runtime = runtime_with_preflight_client(FakePreflightAppClient())
    unsupported_client = TestClient(create_app(unsupported_runtime))
    unsupported_plan, unsupported_revision = _create_missing_input_revision(unsupported_client)
    unsupported_entry = unsupported_runtime.planner.ledger.get(unsupported_plan["run_id"])
    assert unsupported_entry is not None
    first_revised_capability = unsupported_revision["revised_plan"]["proposed_steps"][0]["capability_id"]
    unsupported_snapshot = [
        capability
        for capability in unsupported_entry.capability_snapshot
        if capability.get("capability_id") != first_revised_capability
    ]
    unsupported_runtime.planner.ledger._entries[0] = unsupported_entry.model_copy(  # noqa: SLF001
        update={"capability_snapshot": unsupported_snapshot},
        deep=True,
    )
    unsupported = unsupported_client.post(
        "/plan-revision-activations",
        json={
            "run_id": unsupported_plan["run_id"],
            "revision_id": unsupported_revision["revision_id"],
            "activation_intent": "activate_plan_revision",
        },
    )

    assert unknown_run.status_code == 422
    assert unknown_run.json()["detail"]["errors"][0]["code"] == "unknown_run"
    assert unknown_revision.status_code == 422
    assert unknown_revision.json()["detail"]["errors"][0]["code"] == "unknown_plan_revision"
    assert extra_payload.status_code == 422
    assert paused.status_code == 422
    assert paused.json()["detail"]["errors"][0]["code"] == "paused_run_revision_activation"
    assert cancelled.status_code == 422
    assert cancelled.json()["detail"]["errors"][0]["code"] == "terminal_run_revision_activation"
    assert malformed.status_code == 422
    assert malformed.json()["detail"]["errors"][0]["code"] == "malformed_plan_revision_event"
    assert stale.status_code == 422
    assert stale.json()["detail"]["errors"][0]["code"] == "stale_capability_snapshot"
    assert unsupported.status_code == 422
    assert unsupported.json()["detail"]["errors"][0]["code"] == "unsupported_revision_capability"


def test_file_backed_ledger_reload_restores_run_status_and_list_filters(tmp_path: Path) -> None:
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    first_ledger = FileBackedLedger(
        tmp_path,
        validate_contract=loader.validate_agent_contract_payload,
    )
    first_runtime = runtime_with_preflight_client(FakePreflightAppClient(), ledger=first_ledger)
    first_client = TestClient(create_app(first_runtime))
    plan_payload, documentation_step, _preview_payload = _create_confirmed_documentation_preview(first_client)

    second_ledger = FileBackedLedger(
        tmp_path,
        validate_contract=loader.validate_agent_contract_payload,
    )
    second_runtime = runtime_with_preflight_client(FakePreflightAppClient(), ledger=second_ledger)
    second_client = TestClient(create_app(second_runtime))

    status = second_client.get(f"/runs/{plan_payload['run_id']}")
    assert status.status_code == 200
    assert status.json()["run_id"] == plan_payload["run_id"]
    assert status.json()["latest_action_request"]["step_id"] == documentation_step["step_id"]

    all_runs = second_client.get("/runs")
    by_lifecycle = second_client.get("/runs", params={"lifecycle_id": "lifecycle_test"})
    by_app = second_client.get("/runs", params={"app_id": "quant_documentation"})
    by_capability = second_client.get(
        "/runs",
        params={"capability_id": "quant_documentation.create_draft_workspace"},
    )
    by_missing = second_client.get("/runs", params={"lifecycle_id": "lifecycle_missing"})

    assert all_runs.status_code == 200
    assert all_runs.json()["count"] == 1
    assert all_runs.json()["runs"][0]["run_id"] == plan_payload["run_id"]
    assert all_runs.json()["runs"][0]["lifecycle_id"] == "lifecycle_test"
    assert "quant_documentation" in all_runs.json()["runs"][0]["app_ids"]
    assert "quant_documentation.create_draft_workspace" in all_runs.json()["runs"][0]["capability_ids"]
    assert by_lifecycle.json()["count"] == 1
    assert by_app.json()["count"] == 1
    assert by_capability.json()["count"] == 1
    assert by_missing.json()["count"] == 0


def test_file_backed_ledger_ignores_malformed_files_without_leaking_paths(tmp_path: Path) -> None:
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    (tmp_path / "bad.json").write_text('{"not": "a ledger"}', encoding="utf-8")

    ledger = FileBackedLedger(
        tmp_path,
        validate_contract=loader.validate_agent_contract_payload,
    )
    runtime = runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)
    client = TestClient(create_app(runtime))

    manifest = client.get("/runtime/manifest")
    assert manifest.status_code == 200
    storage = manifest.json()["ledger_storage"]
    assert storage["storage_mode"] == "local_json_file_backed"
    assert storage["loaded_entry_count"] == 0
    assert storage["invalid_entry_count"] == 1
    assert str(tmp_path) not in manifest.text


def test_file_backed_ledger_persists_failure_and_cancellation_records(tmp_path: Path) -> None:
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    failure_ledger = FileBackedLedger(
        tmp_path / "failure",
        validate_contract=loader.validate_agent_contract_payload,
    )
    failing_client = FakePreflightAppClient(
        execution_error=AppClientError("Quant Studio execution app is unavailable.", status_code=503)
    )
    failure_runtime = runtime_with_preflight_client(failing_client, ledger=failure_ledger)
    failure_api = TestClient(create_app(failure_runtime))
    failure_plan, failure_step, _preview_payload = _create_confirmed_studio_preview(failure_api)

    execution = failure_api.post(
        "/executions",
        json={"run_id": failure_plan["run_id"], "step_id": failure_step["step_id"]},
    )
    assert execution.status_code == 200
    assert execution.json()["run_state"] == "failed_recoverable"
    failure_file = next((tmp_path / "failure").glob("*.json"))
    failure_payload = json.loads(failure_file.read_text(encoding="utf-8"))
    assert failure_payload["action_results"][0]["execution_status"] == "failed_recoverable"
    loader.validate_agent_contract_payload(failure_payload, "agent_execution_ledger.v1.schema.json")

    cancellation_ledger = FileBackedLedger(
        tmp_path / "cancellation",
        validate_contract=loader.validate_agent_contract_payload,
    )
    cancellation_runtime = runtime_with_preflight_client(
        FakePreflightAppClient(),
        ledger=cancellation_ledger,
    )
    cancellation_api = TestClient(create_app(cancellation_runtime))
    cancellation_plan = _create_plan_with_lifecycle_reference(cancellation_api)
    cancellation = cancellation_api.post(
        "/cancellations",
        json={
            "run_id": cancellation_plan["run_id"],
            "cancellation_intent": "cancel_run",
            "reason": "user_cancelled",
        },
    )
    assert cancellation.status_code == 200
    cancellation_file = next((tmp_path / "cancellation").glob("*.json"))
    cancellation_payload = json.loads(cancellation_file.read_text(encoding="utf-8"))
    assert cancellation_payload["final_status"] == "cancelled"
    assert cancellation_payload["cancellation_events"][0]["status"] == "cancelled"
    loader.validate_agent_contract_payload(cancellation_payload, "agent_execution_ledger.v1.schema.json")


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
    assert discovery["discovered_apps"] == ["quant_data", "quant_studio", "quant_documentation"]
    assert discovery["unavailable_apps"] == ["quant_monitoring"]
    assert discovery["supported_preflight_capabilities"] == ["quant_data.run_source_preflight"]
    assert discovery["supported_execution_capabilities"] == [
        "quant_studio.prepare_model_config_draft",
        "quant_documentation.create_draft_workspace",
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
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

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
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

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
    assert payload["run_state"] == "planned"
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
        "quant_documentation",
        "quant_monitoring",
    ]
    assert {
        item["capability_id"] for item in payload["plan"]["required_confirmations"]
    } == {
        "quant_studio.prepare_model_config_draft",
        "quant_documentation.create_draft_workspace",
    }


def test_missing_required_context_fields_become_missing_inputs() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

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
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    assert client.post("/runs", json={}).status_code in {404, 405}
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
    plan_payload = _create_plan_with_lifecycle_reference(client)
    _advance_to_monitoring_step(client, plan_payload)
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
    assert payload["run_state"] == "completed"
    assert payload["preflight"]["app_id"] == "quant_monitoring"
    assert payload["preflight"]["safe_artifact_references"][0]["reference_type"] == "monitoring_bundle"
    assert len(app_client.calls) == 2
    app_call = app_client.calls[-1]
    assert app_call["app_id"] == "quant_monitoring"
    assert app_call["capability_id"] == "quant_monitoring.validate_bundle"
    assert app_call["payload"]["action_input"] == monitoring_step["action_input"]
    assert app_call["payload"]["context_summary"]["bundle_summary"] == "Monitoring bundle is available."
    entry = runtime.planner.ledger.list_entries()[0]
    assert entry.preflight_records[-1]["capability_id"] == "quant_monitoring.validate_bundle"


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

    source_response = client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": source_step["step_id"]},
    )
    _complete_studio_step(client, plan_payload)
    _complete_documentation_step(client, plan_payload)
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
    _advance_to_studio_step(client, plan_payload)
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
    _advance_to_studio_step(client, plan_payload)
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


def test_action_request_preview_rejects_preflight_only_orchestration_steps() -> None:
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
    assert monitoring_preflight.status_code == 422
    assert monitoring_preflight.json()["detail"]["errors"][0]["code"] == (
        "orchestration_step_not_ready"
    )

    source_preview = client.post(
        "/action-requests",
        json={"run_id": plan_payload["run_id"], "step_id": source_step["step_id"]},
    )
    _complete_studio_step(client, plan_payload)
    _complete_documentation_step(client, plan_payload)
    monitoring_preflight = client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": monitoring_step["step_id"]},
    )
    monitoring_preview = client.post(
        "/action-requests",
        json={"run_id": plan_payload["run_id"], "step_id": monitoring_step["step_id"]},
    )

    assert source_preview.status_code == 422
    assert source_preview.json()["detail"]["errors"][0]["code"] == "orchestration_step_not_current"
    assert monitoring_preflight.status_code == 200
    assert monitoring_preview.status_code == 422
    assert monitoring_preview.json()["detail"]["errors"][0]["code"] == "orchestration_run_terminal"
    entry = runtime.planner.ledger.list_entries()[0]
    assert "quant_monitoring.validate_bundle" in [
        record["capability_id"] for record in entry.preflight_records
    ]


def test_action_request_preview_is_idempotent_for_existing_request() -> None:
    runtime = runtime_with_preflight_client(FakePreflightAppClient())
    client = TestClient(create_app(runtime))
    plan_payload = _create_plan_with_lifecycle_reference(client)
    _advance_to_studio_step(client, plan_payload)
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
    assert payload["run_state"] == "waiting_for_confirmation"
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
    assert [record["execution_permitted"] for record in entry.action_requests[-2:]] == [False, True]
    assert len(entry.action_results) == 1
    runtime.contract_loader.validate_agent_contract_payload(
        entry.action_requests[-1],
        "agent_action_request.v1.schema.json",
    )
    runtime.contract_loader.validate_agent_contract_payload(
        entry.action_results[-1],
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


def test_execution_runs_confirmed_documentation_draft_workspace_and_validates_ledger() -> None:
    app_client = FakePreflightAppClient(
        execution_response=_valid_documentation_action_result(step_id="step_4")
    )
    runtime = runtime_with_preflight_client(app_client)
    client = TestClient(create_app(runtime))
    plan_payload, documentation_step, preview_payload = _create_confirmed_documentation_preview(client)

    response = client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": documentation_step["step_id"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == plan_payload["run_id"]
    assert payload["step_id"] == documentation_step["step_id"]
    assert payload["capability_id"] == "quant_documentation.create_draft_workspace"
    assert payload["run_state"] == "planned"
    assert payload["validation"]["status"] == "valid"
    assert payload["ledger_recorded"] is True
    action_request = payload["action_request"]
    assert action_request["execution_permitted"] is True
    assert action_request["execution_request"] is True
    assert action_request["preview_idempotency_key"] == preview_payload["action_request"]["idempotency_key"]
    assert action_request["action_input"] == documentation_step["action_input"]
    assert isinstance(action_request["action_input"]["package_summary"], dict)
    assert payload["action_result"]["execution_status"] == "succeeded"
    assert payload["action_result"]["output_references"][0]["reference_type"] == "documentation_draft"
    documentation_calls = [
        call
        for call in app_client.execution_calls
        if call["capability_id"] == "quant_documentation.create_draft_workspace"
    ]
    assert len(documentation_calls) == 1
    app_call = documentation_calls[0]
    assert app_call["app_id"] == "quant_documentation"
    assert app_call["capability_id"] == "quant_documentation.create_draft_workspace"
    assert app_call["payload"] == {"action_request": action_request}
    assert "raw_path" not in str(app_call)
    run_status = client.get(f"/runs/{plan_payload['run_id']}")
    assert run_status.status_code == 200
    assert run_status.json()["latest_action_result"]["capability_id"] == (
        "quant_documentation.create_draft_workspace"
    )
    entry = runtime.planner.ledger.list_entries()[0]
    assert [record["execution_permitted"] for record in entry.action_requests[-2:]] == [False, True]
    assert len(entry.action_results) == 2
    runtime.contract_loader.validate_agent_contract_payload(
        entry.action_requests[-1],
        "agent_action_request.v1.schema.json",
    )
    runtime.contract_loader.validate_agent_contract_payload(
        entry.action_results[-1],
        "agent_action_result.v1.schema.json",
    )
    runtime.contract_loader.validate_agent_contract_payload(
        entry.model_dump(mode="json"),
        "agent_execution_ledger.v1.schema.json",
    )


def test_documentation_execution_duplicate_call_returns_existing_result_without_calling_app_again() -> None:
    app_client = FakePreflightAppClient(
        execution_response=_valid_documentation_action_result(step_id="step_4")
    )
    runtime = runtime_with_preflight_client(app_client)
    client = TestClient(create_app(runtime))
    plan_payload, documentation_step, _preview_payload = _create_confirmed_documentation_preview(client)
    request = {"run_id": plan_payload["run_id"], "step_id": documentation_step["step_id"]}

    first = client.post("/executions", json=request)
    second = client.post("/executions", json=request)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["action_result"] == first.json()["action_result"]
    assert second.json()["action_request"]["capability_id"] == (
        "quant_documentation.create_draft_workspace"
    )
    documentation_calls = [
        call
        for call in app_client.execution_calls
        if call["capability_id"] == "quant_documentation.create_draft_workspace"
    ]
    assert len(documentation_calls) == 1
    entry = runtime.planner.ledger.list_entries()[0]
    assert len(entry.action_requests) == 4
    assert len(entry.action_results) == 2


def test_run_status_tracks_execution_lifecycle_states() -> None:
    app_client = FakePreflightAppClient(
        responses_by_capability={
            "quant_monitoring.validate_bundle": _valid_preflight_response(
                capability_id="quant_monitoring.validate_bundle",
                app_id="quant_monitoring",
            )
        }
    )
    runtime = runtime_with_preflight_client(app_client)
    client = TestClient(create_app(runtime))
    plan_payload = _create_plan_with_lifecycle_reference(client)
    studio_step = next(
        step
        for step in plan_payload["plan"]["proposed_steps"]
        if step["capability_id"] == "quant_studio.prepare_model_config_draft"
    )
    documentation_step = next(
        step
        for step in plan_payload["plan"]["proposed_steps"]
        if step["capability_id"] == "quant_documentation.create_draft_workspace"
    )

    planned_status = client.get(f"/runs/{plan_payload['run_id']}")
    assert planned_status.status_code == 200
    assert planned_status.json()["run_state"] == "planned"
    assert "run_preflight" in planned_status.json()["allowed_next_actions"]
    assert planned_status.json()["ledger_summary"]["action_result_count"] == 0

    source_step = next(
        step
        for step in plan_payload["plan"]["proposed_steps"]
        if step["capability_id"] == "quant_data.run_source_preflight"
    )
    assert client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": source_step["step_id"]},
    ).status_code == 200
    preflighted_status = client.get(f"/runs/{plan_payload['run_id']}")
    assert preflighted_status.status_code == 200
    assert preflighted_status.json()["run_state"] == "waiting_for_confirmation"
    assert "confirm_step" in preflighted_status.json()["allowed_next_actions"]

    assert client.post(
        "/confirmations",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": studio_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    ).status_code == 200
    confirmed_status = client.get(f"/runs/{plan_payload['run_id']}")
    assert confirmed_status.status_code == 200
    assert confirmed_status.json()["run_state"] == "ready_for_execution_preview"
    assert "preview_action_request" in confirmed_status.json()["allowed_next_actions"]
    assert confirmed_status.json()["latest_confirmation"]["status"] == "confirmed"

    assert client.post(
        "/action-requests",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    ).status_code == 200
    preview_status = client.get(f"/runs/{plan_payload['run_id']}")
    assert preview_status.status_code == 200
    assert preview_status.json()["run_state"] == "ready_for_execution_preview"
    assert "execute_step" in preview_status.json()["allowed_next_actions"]
    assert preview_status.json()["latest_action_request"]["execution_permitted"] is False

    assert client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    ).status_code == 200
    studio_completed_status = client.get(f"/runs/{plan_payload['run_id']}")
    assert studio_completed_status.status_code == 200
    assert studio_completed_status.json()["run_state"] == "waiting_for_confirmation"
    assert "confirm_step" in studio_completed_status.json()["allowed_next_actions"]
    assert studio_completed_status.json()["latest_action_result"]["execution_status"] == "succeeded"

    assert client.post(
        "/confirmations",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": documentation_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    ).status_code == 200
    assert client.post(
        "/action-requests",
        json={"run_id": plan_payload["run_id"], "step_id": documentation_step["step_id"]},
    ).status_code == 200
    assert client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": documentation_step["step_id"]},
    ).status_code == 200
    doc_completed_status = client.get(f"/runs/{plan_payload['run_id']}")
    assert doc_completed_status.status_code == 200
    assert doc_completed_status.json()["run_state"] == "planned"
    assert "run_preflight" in doc_completed_status.json()["allowed_next_actions"]

    monitoring_step = next(
        step
        for step in plan_payload["plan"]["proposed_steps"]
        if step["capability_id"] == "quant_monitoring.validate_bundle"
    )
    assert client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": monitoring_step["step_id"]},
    ).status_code == 200
    completed_status = client.get(f"/runs/{plan_payload['run_id']}")
    assert completed_status.status_code == 200
    assert completed_status.json()["run_state"] == "completed"
    assert completed_status.json()["allowed_next_actions"] == []


def test_run_status_unknown_run_is_deterministic() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    response = client.get("/runs/run_missing")

    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["code"] == "unknown_run"


def test_orchestration_tracks_current_step_and_blocks_out_of_order_actions() -> None:
    runtime = runtime_with_preflight_client(FakePreflightAppClient())
    client = TestClient(create_app(runtime))
    plan_payload = _create_plan_with_lifecycle_reference(client)
    source_step = _step_for_capability(plan_payload, "quant_data.run_source_preflight")
    studio_step = _step_for_capability(plan_payload, "quant_studio.prepare_model_config_draft")
    monitoring_step = _step_for_capability(plan_payload, "quant_monitoring.validate_bundle")

    initial = client.get(f"/runs/{plan_payload['run_id']}/orchestration")
    assert initial.status_code == 200
    initial_payload = initial.json()
    assert initial_payload["run_state"] == "planned"
    assert initial_payload["current_step_id"] == source_step["step_id"]
    assert initial_payload["steps"][0]["status"] == "needs_preflight"
    assert initial_payload["steps"][1]["status"] == "not_ready"
    assert initial_payload["steps"][0]["allowed_actions"] == ["run_preflight"]

    future_confirmation = client.post(
        "/confirmations",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": studio_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    )
    assert future_confirmation.status_code == 422
    assert future_confirmation.json()["detail"]["errors"][0]["code"] == (
        "orchestration_step_not_ready"
    )

    assert client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": source_step["step_id"]},
    ).status_code == 200
    preflighted = client.get(f"/runs/{plan_payload['run_id']}/orchestration").json()
    assert preflighted["current_step_id"] == studio_step["step_id"]
    assert preflighted["steps"][0]["status"] == "completed"
    assert preflighted["steps"][1]["status"] == "needs_confirmation"
    assert preflighted["allowed_next_actions"] == ["cancel_run", "confirm_step"]

    future_preflight = client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": monitoring_step["step_id"]},
    )
    assert future_preflight.status_code == 422
    assert future_preflight.json()["detail"]["errors"][0]["code"] == (
        "orchestration_step_not_ready"
    )

    assert client.post(
        "/confirmations",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": studio_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    ).status_code == 200
    confirmed = client.get(f"/runs/{plan_payload['run_id']}/orchestration").json()
    assert confirmed["current_step_id"] == studio_step["step_id"]
    assert confirmed["steps"][1]["status"] == "ready_for_action_request"
    assert confirmed["steps"][1]["allowed_actions"] == ["preview_action_request"]


def test_orchestration_restores_current_step_from_durable_ledger(tmp_path: Path) -> None:
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    first_ledger = FileBackedLedger(
        tmp_path,
        validate_contract=loader.validate_agent_contract_payload,
    )
    first_runtime = runtime_with_preflight_client(FakePreflightAppClient(), ledger=first_ledger)
    first_client = TestClient(create_app(first_runtime))
    plan_payload = _create_plan_with_lifecycle_reference(first_client)
    _run_source_preflight(first_client, plan_payload)

    reloaded_ledger = FileBackedLedger(
        tmp_path,
        validate_contract=loader.validate_agent_contract_payload,
    )
    reloaded_runtime = runtime_with_preflight_client(
        FakePreflightAppClient(),
        ledger=reloaded_ledger,
    )
    reloaded_client = TestClient(create_app(reloaded_runtime))
    response = reloaded_client.get(f"/runs/{plan_payload['run_id']}/orchestration")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_state"] == "waiting_for_confirmation"
    current = next(step for step in payload["steps"] if step["is_current"])
    assert current["capability_id"] == "quant_studio.prepare_model_config_draft"
    assert current["status"] == "needs_confirmation"


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
    _advance_to_studio_step(client, plan_payload)
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


def test_execution_ledgers_safe_failure_results_for_app_unavailable_malformed_and_unsafe_results() -> None:
    app_unavailable = FakePreflightAppClient(
        execution_error=AppClientError("Quant Studio execution app is unavailable.", status_code=503)
    )
    app_unavailable_runtime = runtime_with_preflight_client(app_unavailable)
    app_unavailable_api = TestClient(create_app(app_unavailable_runtime))
    plan_payload, studio_step, _preview_payload = _create_confirmed_studio_preview(app_unavailable_api)
    unavailable_response = app_unavailable_api.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )

    malformed_result = _valid_action_result()
    malformed_result.pop("action_run_id")
    malformed = FakePreflightAppClient(execution_response=malformed_result)
    malformed_runtime = runtime_with_preflight_client(malformed)
    malformed_api = TestClient(create_app(malformed_runtime))
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
    unsafe_runtime = runtime_with_preflight_client(unsafe)
    unsafe_api = TestClient(create_app(unsafe_runtime))
    unsafe_plan, unsafe_step, _unsafe_preview = _create_confirmed_studio_preview(unsafe_api)
    unsafe_response = unsafe_api.post(
        "/executions",
        json={"run_id": unsafe_plan["run_id"], "step_id": unsafe_step["step_id"]},
    )

    assert unavailable_response.status_code == 200
    assert unavailable_response.json()["run_state"] == "failed_recoverable"
    assert unavailable_response.json()["action_result"]["execution_status"] == "failed_recoverable"
    assert unavailable_response.json()["action_result"]["recoverable_errors"][0]["code"] == "app_unavailable"
    assert unavailable_response.json()["action_result"]["retry_allowed"] is True
    assert malformed_response.status_code == 200
    assert malformed_response.json()["run_state"] == "failed_terminal"
    assert malformed_response.json()["action_result"]["execution_status"] == "failed_terminal"
    assert malformed_response.json()["action_result"]["terminal_errors"][0]["code"] == "malformed_app_action_result"
    assert unsafe_response.status_code == 200
    assert unsafe_response.json()["run_state"] == "failed_terminal"
    assert unsafe_response.json()["action_result"]["execution_status"] == "failed_terminal"
    assert unsafe_response.json()["action_result"]["terminal_errors"][0]["code"] == "unsafe_app_action_result"
    assert "private\\raw.csv" not in unsafe_response.text
    for runtime in [app_unavailable_runtime, malformed_runtime, unsafe_runtime]:
        entry = runtime.planner.ledger.list_entries()[0]
        runtime.contract_loader.validate_agent_contract_payload(
            entry.action_results[0],
            "agent_action_result.v1.schema.json",
        )
        runtime.contract_loader.validate_agent_contract_payload(
            entry.model_dump(mode="json"),
            "agent_execution_ledger.v1.schema.json",
        )


def test_execution_duplicate_after_failure_returns_existing_failure_without_calling_app_again() -> None:
    app_client = FakePreflightAppClient(
        execution_error=AppClientError("Quant Studio execution app is unavailable.", status_code=503)
    )
    runtime = runtime_with_preflight_client(app_client)
    client = TestClient(create_app(runtime))
    plan_payload, studio_step, _preview_payload = _create_confirmed_studio_preview(client)
    request = {"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]}

    first = client.post("/executions", json=request)
    second = client.post("/executions", json=request)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["action_result"] == first.json()["action_result"]
    assert len(app_client.execution_calls) == 1
    entry = runtime.planner.ledger.list_entries()[0]
    assert len(entry.action_results) == 1


def test_retry_retries_recoverable_studio_failure_and_validates_ledger() -> None:
    app_client = FakePreflightAppClient(
        execution_error=AppClientError("Quant Studio execution app is unavailable.", status_code=503)
    )
    runtime = runtime_with_preflight_client(app_client)
    client = TestClient(create_app(runtime))
    plan_payload, studio_step, _preview_payload = _create_confirmed_studio_preview(client)
    execution = client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )
    assert execution.status_code == 200
    assert execution.json()["run_state"] == "failed_recoverable"
    orchestration = client.get(f"/runs/{plan_payload['run_id']}/orchestration")
    assert orchestration.status_code == 200
    assert orchestration.json()["run_state"] == "failed_recoverable"
    assert orchestration.json()["allowed_next_actions"] == ["cancel_run", "retry_failed_step"]

    app_client.execution_error = None
    retry = client.post(
        "/retries",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": studio_step["step_id"],
            "retry_intent": "retry_failed_step",
        },
    )
    duplicate = client.post(
        "/retries",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": studio_step["step_id"],
            "retry_intent": "retry_failed_step",
        },
    )

    assert retry.status_code == 200
    assert duplicate.status_code == 200
    payload = retry.json()
    assert duplicate.json()["action_result"] == payload["action_result"]
    assert payload["retry_event"]["event_type"] == "retry"
    assert payload["retry_event"]["status"] == "retried"
    assert payload["retry_event"]["execution_permitted"] is False
    assert payload["retry_event"]["failed_action_run_id"] == execution.json()["action_result"]["action_run_id"]
    assert payload["action_request"]["retry_request"] is True
    assert payload["action_request"]["retry_intent"] == "retry_failed_step"
    assert payload["action_request"]["retry_source_action_run_id"] == (
        execution.json()["action_result"]["action_run_id"]
    )
    assert payload["action_request"]["execution_permitted"] is True
    assert payload["action_result"]["execution_status"] == "succeeded"
    assert payload["run_state"] == "waiting_for_confirmation"
    assert payload["orchestration"]["run_state"] == "waiting_for_confirmation"
    assert len(app_client.execution_calls) == 2
    assert app_client.execution_calls[-1]["payload"] == {"action_request": payload["action_request"]}
    entry = runtime.planner.ledger.list_entries()[0]
    assert [event["event_type"] for event in entry.recovery_events] == ["retry"]
    assert len(entry.action_results) == 2
    assert len(entry.action_requests) == 3
    runtime.contract_loader.validate_agent_contract_payload(
        entry.action_requests[-1],
        "agent_action_request.v1.schema.json",
    )
    runtime.contract_loader.validate_agent_contract_payload(
        entry.action_results[-1],
        "agent_action_result.v1.schema.json",
    )
    runtime.contract_loader.validate_agent_contract_payload(
        entry.model_dump(mode="json"),
        "agent_execution_ledger.v1.schema.json",
    )


def test_retry_retries_recoverable_documentation_failure() -> None:
    app_client = FakePreflightAppClient()
    runtime = runtime_with_preflight_client(app_client)
    client = TestClient(create_app(runtime))
    plan_payload, documentation_step, _preview_payload = _create_confirmed_documentation_preview(client)

    app_client.execution_error = AppClientError(
        "Quant Documentation execution app is unavailable.",
        status_code=503,
    )
    failed = client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": documentation_step["step_id"]},
    )
    assert failed.status_code == 200
    assert failed.json()["run_state"] == "failed_recoverable"

    app_client.execution_error = None
    retry = client.post(
        "/retries",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": documentation_step["step_id"],
            "retry_intent": "retry_failed_step",
        },
    )

    assert retry.status_code == 200
    payload = retry.json()
    assert payload["capability_id"] == "quant_documentation.create_draft_workspace"
    assert payload["action_request"]["capability_id"] == "quant_documentation.create_draft_workspace"
    assert payload["action_result"]["execution_status"] == "succeeded"
    assert payload["action_result"]["output_references"][0]["reference_type"] == "documentation_draft"


def test_retry_rejects_invalid_states_and_non_retryable_failures() -> None:
    runtime = runtime_with_preflight_client(FakePreflightAppClient())
    client = TestClient(create_app(runtime))
    plan_payload, studio_step, _preview_payload = _create_confirmed_studio_preview(client)

    unknown_run = client.post(
        "/retries",
        json={"run_id": "missing_run", "step_id": studio_step["step_id"], "retry_intent": "retry_failed_step"},
    )
    extra_payload = client.post(
        "/retries",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": studio_step["step_id"],
            "retry_intent": "retry_failed_step",
            "action_input": {"target_summary": "browser supplied"},
        },
    )
    no_failure = client.post(
        "/retries",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": studio_step["step_id"],
            "retry_intent": "retry_failed_step",
        },
    )

    assert client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    ).status_code == 200
    completed_retry = client.post(
        "/retries",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": studio_step["step_id"],
            "retry_intent": "retry_failed_step",
        },
    )

    terminal_app = FakePreflightAppClient(execution_response=_valid_action_result("failed_terminal"))
    terminal_runtime = runtime_with_preflight_client(terminal_app)
    terminal_client = TestClient(create_app(terminal_runtime))
    terminal_plan, terminal_step, _terminal_preview = _create_confirmed_studio_preview(terminal_client)
    assert terminal_client.post(
        "/executions",
        json={"run_id": terminal_plan["run_id"], "step_id": terminal_step["step_id"]},
    ).status_code == 200
    terminal_retry = terminal_client.post(
        "/retries",
        json={
            "run_id": terminal_plan["run_id"],
            "step_id": terminal_step["step_id"],
            "retry_intent": "retry_failed_step",
        },
    )

    no_retry_result = _valid_action_result("failed_recoverable")
    no_retry_result["retry_allowed"] = False
    no_retry_app = FakePreflightAppClient(execution_response=no_retry_result)
    no_retry_runtime = runtime_with_preflight_client(no_retry_app)
    no_retry_client = TestClient(create_app(no_retry_runtime))
    no_retry_plan, no_retry_step, _no_retry_preview = _create_confirmed_studio_preview(no_retry_client)
    assert no_retry_client.post(
        "/executions",
        json={"run_id": no_retry_plan["run_id"], "step_id": no_retry_step["step_id"]},
    ).status_code == 200
    no_retry_response = no_retry_client.post(
        "/retries",
        json={
            "run_id": no_retry_plan["run_id"],
            "step_id": no_retry_step["step_id"],
            "retry_intent": "retry_failed_step",
        },
    )

    assert unknown_run.status_code == 422
    assert unknown_run.json()["detail"]["errors"][0]["code"] == "unknown_run"
    assert extra_payload.status_code == 422
    assert no_failure.status_code == 422
    assert no_failure.json()["detail"]["errors"][0]["code"] == "orchestration_action_not_allowed"
    assert completed_retry.status_code == 422
    assert completed_retry.json()["detail"]["errors"][0]["code"] == "orchestration_step_not_current"
    assert terminal_retry.status_code == 422
    assert terminal_retry.json()["detail"]["errors"][0]["code"] == "terminal_run_retry"
    assert no_retry_response.status_code == 422
    assert no_retry_response.json()["detail"]["errors"][0]["code"] == (
        "orchestration_action_not_allowed"
    )


def test_retry_revalidates_capabilities_and_ledgers_safe_retry_failures() -> None:
    unavailable_app = FakePreflightAppClient(
        execution_error=AppClientError("Quant Studio execution app is unavailable.", status_code=503)
    )
    unavailable_runtime = runtime_with_preflight_client(unavailable_app)
    unavailable_client = TestClient(create_app(unavailable_runtime))
    plan_payload, studio_step, _preview_payload = _create_confirmed_studio_preview(unavailable_client)
    assert unavailable_client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    ).status_code == 200
    unavailable_app.execution_error = None
    unavailable_app.discovery_errors_by_app["quant_studio"] = AppClientError(
        "Quant Studio capability discovery app is unavailable.",
        status_code=503,
    )
    unavailable_retry = unavailable_client.post(
        "/retries",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": studio_step["step_id"],
            "retry_intent": "retry_failed_step",
        },
    )

    malformed_app = FakePreflightAppClient(
        execution_error=AppClientError("Quant Studio execution app is unavailable.", status_code=503)
    )
    malformed_runtime = runtime_with_preflight_client(malformed_app)
    malformed_client = TestClient(create_app(malformed_runtime))
    malformed_plan, malformed_step, _malformed_preview = _create_confirmed_studio_preview(malformed_client)
    assert malformed_client.post(
        "/executions",
        json={"run_id": malformed_plan["run_id"], "step_id": malformed_step["step_id"]},
    ).status_code == 200
    malformed_app.execution_error = None
    malformed_response = _valid_action_result()
    malformed_response.pop("action_run_id")
    malformed_app.execution_response = malformed_response
    malformed_retry = malformed_client.post(
        "/retries",
        json={
            "run_id": malformed_plan["run_id"],
            "step_id": malformed_step["step_id"],
            "retry_intent": "retry_failed_step",
        },
    )

    unsafe_app = FakePreflightAppClient(
        execution_error=AppClientError("Quant Studio execution app is unavailable.", status_code=503)
    )
    unsafe_runtime = runtime_with_preflight_client(unsafe_app)
    unsafe_client = TestClient(create_app(unsafe_runtime))
    unsafe_plan, unsafe_step, _unsafe_preview = _create_confirmed_studio_preview(unsafe_client)
    assert unsafe_client.post(
        "/executions",
        json={"run_id": unsafe_plan["run_id"], "step_id": unsafe_step["step_id"]},
    ).status_code == 200
    unsafe_app.execution_error = None
    unsafe_response = _valid_action_result()
    unsafe_response["raw_path"] = "C:\\Users\\matth\\Desktop\\private\\raw.csv"
    unsafe_app.execution_response = unsafe_response
    unsafe_retry = unsafe_client.post(
        "/retries",
        json={
            "run_id": unsafe_plan["run_id"],
            "step_id": unsafe_step["step_id"],
            "retry_intent": "retry_failed_step",
        },
    )

    app_failure = FakePreflightAppClient(
        execution_error=AppClientError("Quant Studio execution app is unavailable.", status_code=503)
    )
    app_failure_runtime = runtime_with_preflight_client(app_failure)
    app_failure_client = TestClient(create_app(app_failure_runtime))
    app_failure_plan, app_failure_step, _app_failure_preview = _create_confirmed_studio_preview(
        app_failure_client
    )
    assert app_failure_client.post(
        "/executions",
        json={"run_id": app_failure_plan["run_id"], "step_id": app_failure_step["step_id"]},
    ).status_code == 200
    app_failure.execution_error = AppClientError(
        "Quant Studio retry app is unavailable.",
        status_code=503,
    )
    app_failure_retry = app_failure_client.post(
        "/retries",
        json={
            "run_id": app_failure_plan["run_id"],
            "step_id": app_failure_step["step_id"],
            "retry_intent": "retry_failed_step",
        },
    )

    assert unavailable_retry.status_code == 422
    assert unavailable_retry.json()["detail"]["errors"][0]["code"] == "execution_app_unavailable"
    assert malformed_retry.status_code == 200
    assert malformed_retry.json()["action_result"]["execution_status"] == "failed_terminal"
    assert malformed_retry.json()["action_result"]["terminal_errors"][0]["code"] == (
        "malformed_app_action_result"
    )
    assert unsafe_retry.status_code == 200
    assert unsafe_retry.json()["action_result"]["execution_status"] == "failed_terminal"
    assert unsafe_retry.json()["action_result"]["terminal_errors"][0]["code"] == (
        "unsafe_app_action_result"
    )
    assert "private\\raw.csv" not in unsafe_retry.text
    assert app_failure_retry.status_code == 200
    assert app_failure_retry.json()["action_result"]["execution_status"] == "failed_recoverable"
    assert app_failure_retry.json()["action_result"]["retry_allowed"] is False
    for runtime in [malformed_runtime, unsafe_runtime, app_failure_runtime]:
        entry = runtime.planner.ledger.list_entries()[0]
        assert len(entry.recovery_events) == 1
        runtime.contract_loader.validate_agent_contract_payload(
            entry.action_results[-1],
            "agent_action_result.v1.schema.json",
        )
        runtime.contract_loader.validate_agent_contract_payload(
            entry.model_dump(mode="json"),
            "agent_execution_ledger.v1.schema.json",
        )


def test_cancellation_is_ledgered_idempotent_and_blocks_further_actions() -> None:
    runtime = runtime_with_preflight_client(FakePreflightAppClient())
    client = TestClient(create_app(runtime))
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

    first = client.post(
        "/cancellations",
        json={
            "run_id": plan_payload["run_id"],
            "cancellation_intent": "cancel_run",
            "reason": "user_cancelled",
        },
    )
    second = client.post(
        "/cancellations",
        json={
            "run_id": plan_payload["run_id"],
            "cancellation_intent": "cancel_run",
            "reason": "user_cancelled",
        },
    )
    status = client.get(f"/runs/{plan_payload['run_id']}")
    preflight = client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": source_step["step_id"]},
    )
    confirmation = client.post(
        "/confirmations",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": studio_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    )
    action_request = client.post(
        "/action-requests",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )
    execution = client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )

    assert first.status_code == 200
    assert first.json()["run_state"] == "cancelled"
    assert first.json()["final_status"] == "cancelled"
    assert first.json()["cancellation"]["status"] == "cancelled"
    assert first.json()["cancellation"]["execution_permitted"] is False
    assert second.status_code == 200
    assert second.json()["cancellation"] == first.json()["cancellation"]
    assert status.status_code == 200
    assert status.json()["run_state"] == "cancelled"
    assert status.json()["latest_cancellation"] == first.json()["cancellation"]
    assert status.json()["allowed_next_actions"] == []
    assert preflight.status_code == 422
    assert preflight.json()["detail"]["errors"][0]["code"] == "cancelled_run_preflight"
    assert confirmation.status_code == 422
    assert confirmation.json()["detail"]["errors"][0]["code"] == "cancelled_run_confirmation"
    assert action_request.status_code == 422
    assert action_request.json()["detail"]["errors"][0]["code"] == "cancelled_run_action_request"
    assert execution.status_code == 422
    assert execution.json()["detail"]["errors"][0]["code"] == "cancelled_run_execution"
    entry = runtime.planner.ledger.list_entries()[0]
    assert len(entry.cancellation_events) == 1
    runtime.contract_loader.validate_agent_contract_payload(
        entry.model_dump(mode="json"),
        "agent_execution_ledger.v1.schema.json",
    )


def test_cancellation_rejects_unknown_terminal_and_unsafe_runs() -> None:
    completed_runtime = runtime_with_preflight_client(
        FakePreflightAppClient(
            responses_by_capability={
                "quant_monitoring.validate_bundle": _valid_preflight_response(
                    capability_id="quant_monitoring.validate_bundle",
                    app_id="quant_monitoring",
                )
            }
        )
    )
    completed_api = TestClient(create_app(completed_runtime))
    completed_plan, completed_step, _preview = _create_confirmed_studio_preview(completed_api)
    assert completed_api.post(
        "/executions",
        json={"run_id": completed_plan["run_id"], "step_id": completed_step["step_id"]},
    ).status_code == 200
    _complete_documentation_step(completed_api, completed_plan)
    completed_monitoring_step = _step_for_capability(
        completed_plan,
        "quant_monitoring.validate_bundle",
    )
    assert completed_api.post(
        "/preflights",
        json={
            "run_id": completed_plan["run_id"],
            "step_id": completed_monitoring_step["step_id"],
        },
    ).status_code == 200

    terminal_client = FakePreflightAppClient(execution_response=_valid_action_result("failed_terminal"))
    terminal_runtime = runtime_with_preflight_client(terminal_client)
    terminal_api = TestClient(create_app(terminal_runtime))
    terminal_plan, terminal_step, _terminal_preview = _create_confirmed_studio_preview(terminal_api)
    assert terminal_api.post(
        "/executions",
        json={"run_id": terminal_plan["run_id"], "step_id": terminal_step["step_id"]},
    ).status_code == 200

    recoverable_client = FakePreflightAppClient(
        execution_error=AppClientError("Quant Studio execution app is unavailable.", status_code=503)
    )
    recoverable_runtime = runtime_with_preflight_client(recoverable_client)
    recoverable_api = TestClient(create_app(recoverable_runtime))
    recoverable_plan, recoverable_step, _recoverable_preview = _create_confirmed_studio_preview(
        recoverable_api
    )
    assert recoverable_api.post(
        "/executions",
        json={"run_id": recoverable_plan["run_id"], "step_id": recoverable_step["step_id"]},
    ).status_code == 200

    unknown = completed_api.post(
        "/cancellations",
        json={
            "run_id": "run_missing",
            "cancellation_intent": "cancel_run",
            "reason": "user_cancelled",
        },
    )
    completed = completed_api.post(
        "/cancellations",
        json={
            "run_id": completed_plan["run_id"],
            "cancellation_intent": "cancel_run",
            "reason": "user_cancelled",
        },
    )
    terminal = terminal_api.post(
        "/cancellations",
        json={
            "run_id": terminal_plan["run_id"],
            "cancellation_intent": "cancel_run",
            "reason": "user_cancelled",
        },
    )
    unsafe = recoverable_api.post(
        "/cancellations",
        json={
            "run_id": recoverable_plan["run_id"],
            "cancellation_intent": "cancel_run",
            "reason": "C:\\Users\\matth\\Desktop\\private\\raw.csv",
        },
    )
    recoverable = recoverable_api.post(
        "/cancellations",
        json={
            "run_id": recoverable_plan["run_id"],
            "cancellation_intent": "cancel_run",
            "reason": "user_cancelled",
        },
    )

    assert unknown.status_code == 422
    assert unknown.json()["detail"]["errors"][0]["code"] == "unknown_run"
    assert completed.status_code == 422
    assert completed.json()["detail"]["errors"][0]["code"] == "terminal_run_cancellation"
    assert terminal.status_code == 422
    assert terminal.json()["detail"]["errors"][0]["code"] == "terminal_run_cancellation"
    assert unsafe.status_code == 422
    assert unsafe.json()["detail"]["errors"][0]["code"] == "unsafe_cancellation_record"
    assert "private\\raw.csv" not in unsafe.text
    assert recoverable.status_code == 200
    assert recoverable.json()["run_state"] == "cancelled"


def test_pause_resume_is_ledgered_idempotent_and_revalidates_current_step() -> None:
    runtime = runtime_with_preflight_client(FakePreflightAppClient())
    client = TestClient(create_app(runtime))
    plan_payload = _create_plan_with_lifecycle_reference(client)
    source_step = _step_for_capability(plan_payload, "quant_data.run_source_preflight")

    first_pause = client.post(
        "/pauses",
        json={
            "run_id": plan_payload["run_id"],
            "pause_intent": "pause_run",
            "reason": "user_paused",
        },
    )
    duplicate_pause = client.post(
        "/pauses",
        json={
            "run_id": plan_payload["run_id"],
            "pause_intent": "pause_run",
            "reason": "user_paused",
        },
    )
    status = client.get(f"/runs/{plan_payload['run_id']}")
    orchestration = client.get(f"/runs/{plan_payload['run_id']}/orchestration")
    blocked_preflight = client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": source_step["step_id"]},
    )
    resume = client.post(
        "/resumptions",
        json={"run_id": plan_payload["run_id"], "resume_intent": "resume_run"},
    )
    resumed_orchestration = client.get(f"/runs/{plan_payload['run_id']}/orchestration")

    assert first_pause.status_code == 200
    assert first_pause.json()["run_state"] == "paused"
    assert first_pause.json()["pause_event"]["event_type"] == "pause"
    assert first_pause.json()["pause_event"]["execution_permitted"] is False
    assert first_pause.json()["allowed_next_actions"] == ["resume_run", "cancel_run"]
    assert duplicate_pause.status_code == 200
    assert duplicate_pause.json()["pause_event"] == first_pause.json()["pause_event"]
    assert status.status_code == 200
    assert status.json()["run_state"] == "paused"
    assert status.json()["latest_recovery"] == first_pause.json()["pause_event"]
    assert status.json()["allowed_next_actions"] == ["resume_run", "cancel_run"]
    assert orchestration.status_code == 200
    assert orchestration.json()["run_state"] == "paused"
    assert orchestration.json()["allowed_next_actions"] == ["resume_run", "cancel_run"]
    assert all(step["allowed_actions"] == [] for step in orchestration.json()["steps"])
    assert blocked_preflight.status_code == 422
    assert blocked_preflight.json()["detail"]["errors"][0]["code"] == "paused_run_preflight"
    assert resume.status_code == 200
    assert resume.json()["run_state"] == "planned"
    assert resume.json()["resumption_event"]["event_type"] == "resume"
    assert resume.json()["resumption_event"]["execution_permitted"] is False
    assert resume.json()["resumption_event"]["revalidation_summary"]["current_step_id"] == (
        source_step["step_id"]
    )
    assert "run_preflight" in resume.json()["allowed_next_actions"]
    assert resumed_orchestration.status_code == 200
    assert resumed_orchestration.json()["current_step_id"] == source_step["step_id"]
    assert resumed_orchestration.json()["steps"][0]["allowed_actions"] == ["run_preflight"]
    entry = runtime.planner.ledger.list_entries()[0]
    assert [event["event_type"] for event in entry.recovery_events] == ["pause", "resume"]
    runtime.contract_loader.validate_agent_contract_payload(
        entry.model_dump(mode="json"),
        "agent_execution_ledger.v1.schema.json",
    )


def test_pause_resume_from_multiple_recoverable_states() -> None:
    runtime = runtime_with_preflight_client(FakePreflightAppClient())
    client = TestClient(create_app(runtime))

    waiting_input = client.post(
        "/plans",
        json={
            "user_goal": "Build the lifecycle plan from whatever summaries are available.",
            "context_summary": {},
        },
    ).json()
    waiting_input_pause = client.post(
        "/pauses",
        json={
            "run_id": waiting_input["run_id"],
            "pause_intent": "pause_run",
            "reason": "user_paused",
        },
    )

    waiting_confirmation = _create_plan_with_lifecycle_reference(client)
    _advance_to_studio_step(client, waiting_confirmation)
    waiting_confirmation_pause = client.post(
        "/pauses",
        json={
            "run_id": waiting_confirmation["run_id"],
            "pause_intent": "pause_run",
            "reason": "user_paused",
        },
    )
    waiting_confirmation_resume = client.post(
        "/resumptions",
        json={"run_id": waiting_confirmation["run_id"], "resume_intent": "resume_run"},
    )

    blocked_preflight_response = _valid_preflight_response(status="blocked")
    blocked_preflight_response["blockers"] = [
        {
            "code": "missing_safe_source_reference",
            "message": "A safe source reference is required.",
        }
    ]
    blocked_runtime = runtime_with_preflight_client(
        FakePreflightAppClient(response=blocked_preflight_response)
    )
    blocked_client = TestClient(create_app(blocked_runtime))
    blocked_plan = _create_plan_with_lifecycle_reference(blocked_client)
    _run_source_preflight(blocked_client, blocked_plan)
    blocked_pause = blocked_client.post(
        "/pauses",
        json={
            "run_id": blocked_plan["run_id"],
            "pause_intent": "pause_run",
            "reason": "user_paused",
        },
    )

    recoverable_runtime = runtime_with_preflight_client(
        FakePreflightAppClient(
            execution_error=AppClientError("Quant Studio execution app is unavailable.", status_code=503)
        )
    )
    recoverable_client = TestClient(create_app(recoverable_runtime))
    recoverable_plan, recoverable_step, _preview = _create_confirmed_studio_preview(
        recoverable_client
    )
    assert recoverable_client.post(
        "/executions",
        json={"run_id": recoverable_plan["run_id"], "step_id": recoverable_step["step_id"]},
    ).status_code == 200
    recoverable_pause = recoverable_client.post(
        "/pauses",
        json={
            "run_id": recoverable_plan["run_id"],
            "pause_intent": "pause_run",
            "reason": "user_paused",
        },
    )

    assert waiting_input_pause.status_code == 200
    assert waiting_input_pause.json()["run_state"] == "paused"
    assert waiting_confirmation_pause.status_code == 200
    assert waiting_confirmation_pause.json()["run_state"] == "paused"
    assert waiting_confirmation_resume.status_code == 200
    assert waiting_confirmation_resume.json()["run_state"] == "waiting_for_confirmation"
    assert "confirm_step" in waiting_confirmation_resume.json()["allowed_next_actions"]
    assert blocked_pause.status_code == 200
    assert blocked_pause.json()["run_state"] == "paused"
    assert recoverable_pause.status_code == 200
    assert recoverable_pause.json()["run_state"] == "paused"


def test_paused_runs_reject_all_gated_actions_before_idempotent_results() -> None:
    runtime = runtime_with_preflight_client(FakePreflightAppClient())
    client = TestClient(create_app(runtime))
    plan_payload, studio_step, _preview_payload = _create_confirmed_studio_preview(client)
    source_step = _step_for_capability(plan_payload, "quant_data.run_source_preflight")

    pause = client.post(
        "/pauses",
        json={
            "run_id": plan_payload["run_id"],
            "pause_intent": "pause_run",
            "reason": "user_paused",
        },
    )
    preflight = client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": source_step["step_id"]},
    )
    confirmation = client.post(
        "/confirmations",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": studio_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    )
    action_request = client.post(
        "/action-requests",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )
    execution = client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )

    assert pause.status_code == 200
    assert preflight.status_code == 422
    assert preflight.json()["detail"]["errors"][0]["code"] == "paused_run_preflight"
    assert confirmation.status_code == 422
    assert confirmation.json()["detail"]["errors"][0]["code"] == "paused_run_confirmation"
    assert action_request.status_code == 422
    assert action_request.json()["detail"]["errors"][0]["code"] == "paused_run_action_request"
    assert execution.status_code == 422
    assert execution.json()["detail"]["errors"][0]["code"] == "paused_run_execution"


def test_resume_rejects_invalid_or_unrevalidatable_runs() -> None:
    unpaused_api = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))
    unpaused_plan = _create_plan_with_lifecycle_reference(unpaused_api)

    paused_unavailable_client = FakePreflightAppClient(
        discovery_errors_by_app={
            "quant_data": AppClientError(
                "Quant Data capability discovery app is unavailable.",
                status_code=503,
            )
        }
    )
    paused_unavailable_runtime = runtime_with_preflight_client(paused_unavailable_client)
    paused_unavailable_api = TestClient(create_app(paused_unavailable_runtime))
    paused_unavailable_plan = _create_plan_with_lifecycle_reference(paused_unavailable_api)
    assert paused_unavailable_api.post(
        "/pauses",
        json={
            "run_id": paused_unavailable_plan["run_id"],
            "pause_intent": "pause_run",
            "reason": "user_paused",
        },
    ).status_code == 200

    stale_runtime = runtime_with_preflight_client(FakePreflightAppClient())
    stale_api = TestClient(create_app(stale_runtime))
    stale_plan = _create_plan_with_lifecycle_reference(stale_api)
    assert stale_api.post(
        "/pauses",
        json={
            "run_id": stale_plan["run_id"],
            "pause_intent": "pause_run",
            "reason": "user_paused",
        },
    ).status_code == 200
    stale_entry = stale_runtime.planner.ledger.list_entries()[0]
    stale_snapshot = copy.deepcopy(stale_entry.capability_snapshot)
    stale_snapshot[0]["enabled"] = False
    stale_runtime.planner.ledger._entries[0] = stale_entry.model_copy(  # noqa: SLF001 - test corruption.
        update={"capability_snapshot": stale_snapshot},
        deep=True,
    )

    completed_runtime = runtime_with_preflight_client(
        FakePreflightAppClient(
            responses_by_capability={
                "quant_monitoring.validate_bundle": _valid_preflight_response(
                    capability_id="quant_monitoring.validate_bundle",
                    app_id="quant_monitoring",
                )
            }
        )
    )
    completed_api = TestClient(create_app(completed_runtime))
    completed_plan, completed_step, _preview = _create_confirmed_studio_preview(completed_api)
    assert completed_api.post(
        "/executions",
        json={"run_id": completed_plan["run_id"], "step_id": completed_step["step_id"]},
    ).status_code == 200
    _complete_documentation_step(completed_api, completed_plan)
    completed_monitoring_step = _step_for_capability(
        completed_plan,
        "quant_monitoring.validate_bundle",
    )
    assert completed_api.post(
        "/preflights",
        json={
            "run_id": completed_plan["run_id"],
            "step_id": completed_monitoring_step["step_id"],
        },
    ).status_code == 200

    unknown = paused_unavailable_api.post(
        "/resumptions",
        json={"run_id": "run_missing", "resume_intent": "resume_run"},
    )
    unpaused = unpaused_api.post(
        "/resumptions",
        json={"run_id": unpaused_plan["run_id"], "resume_intent": "resume_run"},
    )
    completed = completed_api.post(
        "/resumptions",
        json={"run_id": completed_plan["run_id"], "resume_intent": "resume_run"},
    )
    app_unavailable = paused_unavailable_api.post(
        "/resumptions",
        json={"run_id": paused_unavailable_plan["run_id"], "resume_intent": "resume_run"},
    )
    stale = stale_api.post(
        "/resumptions",
        json={"run_id": stale_plan["run_id"], "resume_intent": "resume_run"},
    )
    cancelled = paused_unavailable_api.post(
        "/cancellations",
        json={
            "run_id": paused_unavailable_plan["run_id"],
            "cancellation_intent": "cancel_run",
            "reason": "user_cancelled",
        },
    )
    cancelled_resume = paused_unavailable_api.post(
        "/resumptions",
        json={"run_id": paused_unavailable_plan["run_id"], "resume_intent": "resume_run"},
    )

    assert unknown.status_code == 422
    assert unknown.json()["detail"]["errors"][0]["code"] == "unknown_run"
    assert unpaused.status_code == 422
    assert unpaused.json()["detail"]["errors"][0]["code"] == "run_not_paused"
    assert completed.status_code == 422
    assert completed.json()["detail"]["errors"][0]["code"] == "terminal_run_resumption"
    assert app_unavailable.status_code == 422
    assert app_unavailable.json()["detail"]["errors"][0]["code"] == "resume_app_unavailable"
    assert stale.status_code == 422
    assert stale.json()["detail"]["errors"][0]["code"] == "stale_resume_capability_snapshot"
    assert cancelled.status_code == 200
    assert cancelled_resume.status_code == 422
    assert cancelled_resume.json()["detail"]["errors"][0]["code"] == "terminal_run_resumption"


def test_pause_rejects_unsafe_reason_without_leaking_value() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))
    plan_payload = _create_plan_with_lifecycle_reference(client)

    response = client.post(
        "/pauses",
        json={
            "run_id": plan_payload["run_id"],
            "pause_intent": "pause_run",
            "reason": "C:\\Users\\matth\\Desktop\\private\\raw.csv",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["code"] == "unsafe_recovery_record"
    assert "private\\raw.csv" not in response.text


def test_run_progress_summary_is_returned_on_status_and_orchestration() -> None:
    runtime = runtime_with_preflight_client(FakePreflightAppClient())
    client = TestClient(create_app(runtime))
    plan_payload = _create_plan_with_lifecycle_reference(client)
    source_step = _step_for_capability(plan_payload, "quant_data.run_source_preflight")

    planned_status = client.get(f"/runs/{plan_payload['run_id']}")
    planned_orchestration = client.get(f"/runs/{plan_payload['run_id']}/orchestration")
    preflight = client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": source_step["step_id"]},
    )
    post_preflight_status = client.get(f"/runs/{plan_payload['run_id']}")

    assert planned_status.status_code == 200
    assert planned_orchestration.status_code == 200
    assert planned_status.json()["run_progress_summary"]["total_steps"] >= 1
    assert planned_status.json()["run_progress_summary"]["current_step_id"] == source_step["step_id"]
    assert planned_status.json()["run_progress_summary"]["current_step_status"] == "needs_preflight"
    assert planned_status.json()["stale_assumption_summary"]["status"] == "not_evaluated"
    assert planned_orchestration.json()["run_progress_summary"] == planned_status.json()["run_progress_summary"]
    assert preflight.status_code == 200
    assert post_preflight_status.json()["run_progress_summary"]["latest_record_counts"]["preflight_records"] == 1
    assert post_preflight_status.json()["run_progress_summary"]["current_step_status"] == "needs_confirmation"


def test_run_revalidation_records_fresh_stale_insufficient_and_paused_summaries() -> None:
    runtime = runtime_with_preflight_client(FakePreflightAppClient())
    client = TestClient(create_app(runtime))
    plan_payload = _create_plan_with_lifecycle_reference(client)
    run_id = plan_payload["run_id"]

    fresh = client.post(
        "/run-revalidations",
        json={
            "run_id": run_id,
            "revalidation_intent": "check_current_context",
            "current_context_summary": _safe_lifecycle_context(),
        },
    )
    stale = client.post(
        "/run-revalidations",
        json={
            "run_id": run_id,
            "revalidation_intent": "check_current_context",
            "current_context_summary": _safe_lifecycle_context(
                lifecycle_state="ready_for_documentation",
                lifecycle_summary="Lifecycle state changed after planning.",
            ),
        },
    )
    insufficient = client.post(
        "/run-revalidations",
        json={
            "run_id": run_id,
            "revalidation_intent": "check_current_context",
            "current_context_summary": {
                "lifecycle_summary": {
                    "lifecycle_id": "lifecycle_test",
                    "state": "ready_for_modeling",
                }
            },
        },
    )
    assert client.post(
        "/pauses",
        json={"run_id": run_id, "pause_intent": "pause_run", "reason": "user_paused"},
    ).status_code == 200
    paused = client.post(
        "/run-revalidations",
        json={
            "run_id": run_id,
            "revalidation_intent": "check_current_context",
            "current_context_summary": _safe_lifecycle_context(),
        },
    )
    status = client.get(f"/runs/{run_id}")
    orchestration = client.get(f"/runs/{run_id}/orchestration")

    assert fresh.status_code == 200
    assert fresh.json()["stale_assumption_summary"]["status"] == "fresh"
    assert fresh.json()["stale_assumption_summary"]["revalidation_required"] is False
    assert stale.status_code == 200
    assert stale.json()["stale_assumption_summary"]["status"] == "stale"
    assert stale.json()["stale_assumption_summary"]["state_changed_since_planning"] is True
    assert stale.json()["stale_assumption_summary"]["changed_sections"] == ["lifecycle_summary"]
    assert insufficient.status_code == 200
    assert insufficient.json()["stale_assumption_summary"]["status"] == "insufficient_context"
    assert "source_summary" in insufficient.json()["stale_assumption_summary"]["missing_current_sections"]
    assert paused.status_code == 200
    assert paused.json()["orchestration"]["run_state"] == "paused"
    assert paused.json()["run_progress_summary"]["allowed_next_actions"] == ["resume_run", "cancel_run"]
    assert status.json()["stale_assumption_summary"]["status"] == "fresh"
    assert orchestration.json()["stale_assumption_summary"]["status"] == "fresh"

    entry = runtime.planner.ledger.get(run_id)
    assert entry is not None
    assert [event["event_type"] for event in entry.recovery_events].count("run_revalidation") == 4
    assert entry.recovery_events[-1]["execution_permitted"] is False
    assert "context_fingerprint" in entry.recovery_events[-1]
    runtime.contract_loader.validate_agent_contract_payload(
        entry.model_dump(mode="json"),
        "agent_execution_ledger.v1.schema.json",
    )


def test_run_revalidation_rejects_unknown_unsafe_cancelled_and_completed_runs() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))
    plan_payload = _create_plan_with_lifecycle_reference(client)
    run_id = plan_payload["run_id"]

    unknown = client.post(
        "/run-revalidations",
        json={
            "run_id": "run_missing",
            "revalidation_intent": "check_current_context",
            "current_context_summary": _safe_lifecycle_context(),
        },
    )
    unsafe = client.post(
        "/run-revalidations",
        json={
            "run_id": run_id,
            "revalidation_intent": "check_current_context",
            "current_context_summary": {
                "lifecycle_summary": {"lifecycle_id": "lifecycle_test"},
                "raw_path": "C:\\Users\\matth\\Desktop\\private\\raw.csv",
            },
        },
    )
    assert client.post(
        "/cancellations",
        json={"run_id": run_id, "cancellation_intent": "cancel_run", "reason": "user_cancelled"},
    ).status_code == 200
    cancelled = client.post(
        "/run-revalidations",
        json={
            "run_id": run_id,
            "revalidation_intent": "check_current_context",
            "current_context_summary": _safe_lifecycle_context(),
        },
    )

    completed_runtime = runtime_with_preflight_client(
        FakePreflightAppClient(
            responses_by_capability={
                "quant_monitoring.validate_bundle": _valid_preflight_response(
                    capability_id="quant_monitoring.validate_bundle",
                    app_id="quant_monitoring",
                )
            }
        )
    )
    completed_client = TestClient(create_app(completed_runtime))
    completed_plan = _create_plan_with_lifecycle_reference(completed_client)
    _advance_to_monitoring_step(completed_client, completed_plan)
    monitoring_step = _step_for_capability(completed_plan, "quant_monitoring.validate_bundle")
    assert completed_client.post(
        "/preflights",
        json={"run_id": completed_plan["run_id"], "step_id": monitoring_step["step_id"]},
    ).status_code == 200
    completed = completed_client.post(
        "/run-revalidations",
        json={
            "run_id": completed_plan["run_id"],
            "revalidation_intent": "check_current_context",
            "current_context_summary": _safe_lifecycle_context(),
        },
    )

    assert unknown.status_code == 422
    assert unknown.json()["detail"]["errors"][0]["code"] == "unknown_run"
    assert unsafe.status_code == 422
    assert unsafe.json()["detail"]["errors"][0]["code"] == "unsafe_revalidation_context"
    assert "private\\raw.csv" not in unsafe.text
    assert cancelled.status_code == 422
    assert cancelled.json()["detail"]["errors"][0]["code"] == "terminal_run_revalidation"
    assert completed.status_code == 422
    assert completed.json()["detail"]["errors"][0]["code"] == "terminal_run_revalidation"


def test_sample_autopilot_preview_allows_credit_pd_sample_and_validates_ledger() -> None:
    runtime = runtime_with_preflight_client(FakePreflightAppClient())
    client = TestClient(create_app(runtime))
    response = client.post(
        "/plans",
        json={
            "user_goal": "Preview the sample autopilot path.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    )
    assert response.status_code == 200
    run_id = response.json()["run_id"]

    preview = client.post(
        "/autopilot-previews",
        json={
            "run_id": run_id,
            "autopilot_intent": "preview_sample_autopilot",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )

    assert preview.status_code == 200
    payload = preview.json()
    assert payload["sample_eligibility"]["eligible"] is True
    assert payload["sample_eligibility"]["sample_workspace_id"] == "credit_pd_scorecard_panel"
    assert payload["sample_eligibility"]["reset_boundary_available"] is True
    assert payload["autopilot_preview"]["dry_run_only"] is True
    assert payload["autopilot_preview"]["autonomous_execution_permitted"] is False
    assert payload["autopilot_preview"]["steps"][0]["dry_run_action"] == "request_manual_preflight"
    assert payload["autopilot_preview"]["next_manual_actions"] == ["cancel_run", "run_preflight"]
    assert payload["ledger_recorded"] is True
    entry = runtime.planner.ledger.get(run_id)
    assert entry is not None
    event = entry.recovery_events[-1]
    assert event["event_type"] == "sample_autopilot_preview"
    assert event["status"] == "eligible_previewed"
    assert event["dry_run_only"] is True
    assert event["execution_permitted"] is False
    runtime.contract_loader.validate_agent_contract_payload(
        entry.model_dump(mode="json"),
        "agent_execution_ledger.v1.schema.json",
    )


def test_sample_autopilot_preview_blocks_non_demo_shaped_active_plan() -> None:
    runtime = runtime_with_preflight_client(FakePreflightAppClient())
    client = TestClient(create_app(runtime))
    plan = client.post(
        "/plans",
        json={
            "user_goal": "Preview a sample autopilot path with stale active plan shape.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    entry = runtime.planner.ledger.get(plan["run_id"])
    assert entry is not None
    stale_snapshot = copy.deepcopy(entry.plan_snapshot)
    stale_snapshot["proposed_steps"] = [
        stale_snapshot["proposed_steps"][1],
        stale_snapshot["proposed_steps"][0],
        *stale_snapshot["proposed_steps"][2:],
    ]
    runtime.planner.ledger._entries[0] = entry.model_copy(  # noqa: SLF001 - corrupts test ledger intentionally.
        update={"plan_snapshot": stale_snapshot},
        deep=True,
    )

    preview = client.post(
        "/autopilot-previews",
        json={
            "run_id": plan["run_id"],
            "autopilot_intent": "preview_sample_autopilot",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )

    assert preview.status_code == 200
    assert preview.json()["sample_eligibility"]["status"] == "blocked"
    assert any(
        "Phase 7 sample demo capability sequence" in blocker
        for blocker in preview.json()["sample_eligibility"]["blockers"]
    )


def test_sample_autopilot_preview_blocks_stale_demo_gate_metadata() -> None:
    runtime = runtime_with_preflight_client(FakePreflightAppClient())
    client = TestClient(create_app(runtime))
    plan = client.post(
        "/plans",
        json={
            "user_goal": "Preview a sample autopilot path with stale gate metadata.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    entry = runtime.planner.ledger.get(plan["run_id"])
    assert entry is not None
    stale_snapshot = copy.deepcopy(entry.plan_snapshot)
    stale_snapshot["proposed_steps"][0]["preflight_required"] = False
    runtime.planner.ledger._entries[0] = entry.model_copy(  # noqa: SLF001 - corrupts test ledger intentionally.
        update={"plan_snapshot": stale_snapshot},
        deep=True,
    )

    preview = client.post(
        "/autopilot-previews",
        json={
            "run_id": plan["run_id"],
            "autopilot_intent": "preview_sample_autopilot",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )

    assert preview.status_code == 200
    assert preview.json()["sample_eligibility"]["status"] == "blocked"
    assert any(
        "gate metadata is stale for quant_data.run_source_preflight" in blocker
        for blocker in preview.json()["sample_eligibility"]["blockers"]
    )


def test_sample_autopilot_preview_reflects_current_gated_orchestration_state() -> None:
    runtime = runtime_with_preflight_client(FakePreflightAppClient())
    client = TestClient(create_app(runtime))
    plan = client.post(
        "/plans",
        json={
            "user_goal": "Preview the sample autopilot path after source preflight.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    source_step = plan["plan"]["proposed_steps"][0]
    assert client.post(
        "/preflights",
        json={"run_id": plan["run_id"], "step_id": source_step["step_id"]},
    ).status_code == 200

    preview = client.post(
        "/autopilot-previews",
        json={
            "run_id": plan["run_id"],
            "autopilot_intent": "preview_sample_autopilot",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )

    assert preview.status_code == 200
    payload = preview.json()
    studio_step = next(
        step
        for step in payload["autopilot_preview"]["steps"]
        if step["capability_id"] == "quant_studio.prepare_model_config_draft"
    )
    assert studio_step["status"] == "needs_confirmation"
    assert studio_step["dry_run_action"] == "request_manual_confirmation"
    assert payload["autopilot_preview"]["next_manual_actions"] == ["cancel_run", "confirm_step"]


def test_sample_autopilot_preview_returns_blocked_for_non_sample_and_invalid_sample_contexts(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))
    user_plan = _create_plan_with_lifecycle_reference(client)

    user_owned = client.post(
        "/autopilot-previews",
        json={
            "run_id": user_plan["run_id"],
            "autopilot_intent": "preview_sample_autopilot",
            "current_context_summary": _safe_lifecycle_context(),
        },
    )
    assert user_owned.status_code == 200
    assert user_owned.json()["sample_eligibility"]["status"] == "blocked"
    assert any(
        "sample_workspace_id" in blocker
        for blocker in user_owned.json()["sample_eligibility"]["blockers"]
    )

    non_allowlisted_plan = client.post(
        "/plans",
        json={
            "user_goal": "Preview a non-allowlisted sample.",
            "context_summary": _safe_sample_lifecycle_context(sample_workspace_id="monitoring_drift_review"),
        },
    ).json()
    non_allowlisted = client.post(
        "/autopilot-previews",
        json={
            "run_id": non_allowlisted_plan["run_id"],
            "autopilot_intent": "preview_sample_autopilot",
            "current_context_summary": _safe_sample_lifecycle_context(sample_workspace_id="monitoring_drift_review"),
        },
    )
    assert non_allowlisted.status_code == 200
    assert non_allowlisted.json()["sample_eligibility"]["status"] == "blocked"
    assert any(
        "not allowlisted" in blocker
        for blocker in non_allowlisted.json()["sample_eligibility"]["blockers"]
    )

    stale_plan = client.post(
        "/plans",
        json={
            "user_goal": "Preview a mismatched sample.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    mismatch = client.post(
        "/autopilot-previews",
        json={
            "run_id": stale_plan["run_id"],
            "autopilot_intent": "preview_sample_autopilot",
            "current_context_summary": _safe_sample_lifecycle_context(sample_workspace_id="monitoring_drift_review"),
        },
    )
    assert mismatch.status_code == 200
    assert mismatch.json()["sample_eligibility"]["status"] == "blocked"
    assert any(
        "does not match" in blocker
        for blocker in mismatch.json()["sample_eligibility"]["blockers"]
    )

    sample_root = tmp_path / "samples"
    sample_dir = sample_root / "credit_pd_scorecard_panel"
    sample_dir.mkdir(parents=True)
    (sample_dir / "sample_workspace.v1.json").write_text(
        json.dumps(
            {
                "sample_workspace_id": "credit_pd_scorecard_panel",
                "label": "Credit Risk PD Scorecard",
                "owned_marker": {
                    "sample_workspace": True,
                    "sample_workspace_id": "credit_pd_scorecard_panel",
                    "sample_owned": True,
                },
                "lifecycle_id": "sample_credit_pd_scorecard_panel",
                "reset_scope": {"sample_owned_only": False},
            }
        ),
        encoding="utf-8",
    )
    missing_reset_runtime = runtime_with_preflight_client(
        FakePreflightAppClient(),
        sample_workspace_root=sample_root,
    )
    missing_reset_client = TestClient(create_app(missing_reset_runtime))
    missing_reset_plan = missing_reset_client.post(
        "/plans",
        json={
            "user_goal": "Preview a sample without reset boundary.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    missing_reset = missing_reset_client.post(
        "/autopilot-previews",
        json={
            "run_id": missing_reset_plan["run_id"],
            "autopilot_intent": "preview_sample_autopilot",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )
    assert missing_reset.status_code == 200
    assert missing_reset.json()["sample_eligibility"]["status"] == "blocked"
    assert any(
        "safe reset boundary" in blocker
        for blocker in missing_reset.json()["sample_eligibility"]["blockers"]
    )


def test_sample_autopilot_preview_blocks_paused_terminal_unsafe_and_stale_capability_runs() -> None:
    runtime = runtime_with_preflight_client(FakePreflightAppClient())
    client = TestClient(create_app(runtime))
    plan = client.post(
        "/plans",
        json={
            "user_goal": "Preview blocked sample autopilot states.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    run_id = plan["run_id"]

    unsafe = client.post(
        "/autopilot-previews",
        json={
            "run_id": run_id,
            "autopilot_intent": "preview_sample_autopilot",
            "current_context_summary": {
                **_safe_sample_lifecycle_context(),
                "raw_local_path": "C:\\private\\rows.csv",
            },
        },
    )
    assert unsafe.status_code == 200
    assert unsafe.json()["sample_eligibility"]["status"] == "blocked"
    assert "private\\rows.csv" not in unsafe.text
    assert any(
        "unsafe fields" in blocker
        for blocker in unsafe.json()["sample_eligibility"]["blockers"]
    )

    paused = client.post(
        "/pauses",
        json={"run_id": run_id, "pause_intent": "pause_run", "reason": "user_paused"},
    )
    assert paused.status_code == 200
    paused_preview = client.post(
        "/autopilot-previews",
        json={
            "run_id": run_id,
            "autopilot_intent": "preview_sample_autopilot",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )
    assert paused_preview.status_code == 200
    assert any(
        "Paused runs" in blocker
        for blocker in paused_preview.json()["sample_eligibility"]["blockers"]
    )

    stale_runtime = runtime_with_preflight_client(FakePreflightAppClient())
    stale_client = TestClient(create_app(stale_runtime))
    stale_plan = stale_client.post(
        "/plans",
        json={
            "user_goal": "Preview stale capability sample autopilot.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    entry = stale_runtime.planner.ledger.get(stale_plan["run_id"])
    assert entry is not None
    entry.capability_snapshot[0]["enabled"] = False
    stale_capability = stale_client.post(
        "/autopilot-previews",
        json={
            "run_id": stale_plan["run_id"],
            "autopilot_intent": "preview_sample_autopilot",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )
    assert stale_capability.status_code == 200
    assert any(
        "disabled" in blocker
        for blocker in stale_capability.json()["sample_eligibility"]["blockers"]
    )

    completed_runtime = runtime_with_preflight_client(
        FakePreflightAppClient(
            responses_by_capability={
                "quant_monitoring.validate_bundle": _valid_preflight_response(
                    capability_id="quant_monitoring.validate_bundle",
                    app_id="quant_monitoring",
                )
            }
        )
    )
    completed_client = TestClient(create_app(completed_runtime))
    completed_plan = completed_client.post(
        "/plans",
        json={
            "user_goal": "Preview completed sample autopilot.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    _advance_to_monitoring_step(completed_client, completed_plan)
    monitoring_step = _step_for_capability(completed_plan, "quant_monitoring.validate_bundle")
    assert completed_client.post(
        "/preflights",
        json={"run_id": completed_plan["run_id"], "step_id": monitoring_step["step_id"]},
    ).status_code == 200
    completed = completed_client.post(
        "/autopilot-previews",
        json={
            "run_id": completed_plan["run_id"],
            "autopilot_intent": "preview_sample_autopilot",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )
    assert completed.status_code == 200
    assert any(
        "Terminal or cancelled" in blocker
        for blocker in completed.json()["sample_eligibility"]["blockers"]
    )


def test_sample_autopilot_preview_rejects_unknown_run_and_extra_payload_fields() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    unknown = client.post(
        "/autopilot-previews",
        json={
            "run_id": "run_missing",
            "autopilot_intent": "preview_sample_autopilot",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )
    extra = client.post(
        "/autopilot-previews",
        json={
            "run_id": "run_missing",
            "autopilot_intent": "preview_sample_autopilot",
            "current_context_summary": _safe_sample_lifecycle_context(),
            "execution_flags": {"run": True},
        },
    )

    assert unknown.status_code == 422
    assert unknown.json()["detail"]["errors"][0]["code"] == "unknown_run"
    assert extra.status_code == 422


def test_sample_autopilot_step_advances_data_preflight_and_validates_ledger() -> None:
    app_client = FakePreflightAppClient()
    runtime = runtime_with_preflight_client(app_client)
    client = TestClient(create_app(runtime))
    plan = client.post(
        "/plans",
        json={
            "user_goal": "Advance the sample autopilot path one step.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()

    response = client.post(
        "/autopilot-steps",
        json={
            "run_id": plan["run_id"],
            "autopilot_intent": "advance_sample_autopilot_one_step",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_action"] == "run_preflight"
    assert payload["advance_status"] == "advanced"
    assert payload["delegated_result"]["preflight"]["status"] == "ready"
    assert payload["orchestration"]["current_step_id"] == _step_for_capability(
        plan,
        "quant_studio.prepare_model_config_draft",
    )["step_id"]
    assert app_client.calls == [
        {
            "app_id": "quant_data",
            "capability_id": "quant_data.run_source_preflight",
            "payload": app_client.calls[0]["payload"],
        }
    ]
    assert app_client.execution_calls == []
    entry = runtime.planner.ledger.get(plan["run_id"])
    assert entry is not None
    assert entry.preflight_records[-1]["capability_id"] == "quant_data.run_source_preflight"
    assert entry.recovery_events[-1]["event_type"] == "sample_autopilot_step"
    assert entry.recovery_events[-1]["selected_action"] == "run_preflight"
    assert entry.recovery_events[-1]["single_step_only"] is True
    assert entry.recovery_events[-1]["execution_permitted"] is False
    runtime.contract_loader.validate_agent_contract_payload(
        entry.model_dump(mode="json"),
        "agent_execution_ledger.v1.schema.json",
    )


def test_sample_autopilot_step_stops_for_manual_confirmation_without_confirming() -> None:
    runtime = runtime_with_preflight_client(FakePreflightAppClient())
    client = TestClient(create_app(runtime))
    plan = client.post(
        "/plans",
        json={
            "user_goal": "Advance until the manual Studio confirmation gate.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    _run_source_preflight(client, plan)

    response = client.post(
        "/autopilot-steps",
        json={
            "run_id": plan["run_id"],
            "autopilot_intent": "advance_sample_autopilot_one_step",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )

    assert response.status_code == 200
    assert response.json()["selected_action"] == "confirm_step"
    assert response.json()["advance_status"] == "manual_confirmation_required"
    entry = runtime.planner.ledger.get(plan["run_id"])
    assert entry is not None
    assert entry.confirmation_records == []
    assert entry.action_requests == []
    assert entry.action_results == []
    assert entry.recovery_events[-1]["status"] == "manual_confirmation_required"


def test_sample_autopilot_step_advances_studio_preview_then_execution_after_confirmation() -> None:
    app_client = FakePreflightAppClient()
    runtime = runtime_with_preflight_client(app_client)
    client = TestClient(create_app(runtime))
    plan = client.post(
        "/plans",
        json={
            "user_goal": "Advance the confirmed Studio sample step.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    _advance_to_studio_step(client, plan)
    studio_step = _step_for_capability(plan, "quant_studio.prepare_model_config_draft")
    assert client.post(
        "/confirmations",
        json={
            "run_id": plan["run_id"],
            "step_id": studio_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    ).status_code == 200

    preview = client.post(
        "/autopilot-steps",
        json={
            "run_id": plan["run_id"],
            "autopilot_intent": "advance_sample_autopilot_one_step",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )
    execution = client.post(
        "/autopilot-steps",
        json={
            "run_id": plan["run_id"],
            "autopilot_intent": "advance_sample_autopilot_one_step",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )

    assert preview.status_code == 200
    assert preview.json()["selected_action"] == "preview_action_request"
    assert preview.json()["advance_status"] == "advanced"
    assert preview.json()["delegated_result"]["action_request"]["execution_permitted"] is False
    assert execution.status_code == 200
    assert execution.json()["selected_action"] == "execute_step"
    assert execution.json()["advance_status"] == "advanced"
    assert execution.json()["delegated_result"]["action_result"]["execution_status"] == "succeeded"
    assert len(app_client.execution_calls) == 1
    assert app_client.execution_calls[0]["capability_id"] == "quant_studio.prepare_model_config_draft"
    entry = runtime.planner.ledger.get(plan["run_id"])
    assert entry is not None
    assert entry.action_requests[-1]["execution_permitted"] is True
    assert entry.recovery_events[-2]["selected_action"] == "preview_action_request"
    assert entry.recovery_events[-1]["selected_action"] == "execute_step"


def test_sample_autopilot_step_advances_documentation_and_monitoring_paths() -> None:
    responses_by_capability = {
        "quant_monitoring.validate_bundle": _valid_preflight_response(
            capability_id="quant_monitoring.validate_bundle",
            app_id="quant_monitoring",
        )
    }
    app_client = FakePreflightAppClient(responses_by_capability=responses_by_capability)
    runtime = runtime_with_preflight_client(app_client)
    client = TestClient(create_app(runtime))
    plan = client.post(
        "/plans",
        json={
            "user_goal": "Advance documentation and monitoring sample steps.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    _advance_to_documentation_step(client, plan)
    documentation_step = _step_for_capability(plan, "quant_documentation.create_draft_workspace")
    assert client.post(
        "/confirmations",
        json={
            "run_id": plan["run_id"],
            "step_id": documentation_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    ).status_code == 200

    documentation_preview = client.post(
        "/autopilot-steps",
        json={
            "run_id": plan["run_id"],
            "autopilot_intent": "advance_sample_autopilot_one_step",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )
    documentation_execution = client.post(
        "/autopilot-steps",
        json={
            "run_id": plan["run_id"],
            "autopilot_intent": "advance_sample_autopilot_one_step",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )
    monitoring_preflight = client.post(
        "/autopilot-steps",
        json={
            "run_id": plan["run_id"],
            "autopilot_intent": "advance_sample_autopilot_one_step",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )

    assert documentation_preview.status_code == 200
    assert documentation_preview.json()["selected_action"] == "preview_action_request"
    assert documentation_preview.json()["capability_id"] == "quant_documentation.create_draft_workspace"
    assert documentation_execution.status_code == 200
    assert documentation_execution.json()["selected_action"] == "execute_step"
    assert documentation_execution.json()["delegated_result"]["action_result"]["capability_id"] == (
        "quant_documentation.create_draft_workspace"
    )
    assert monitoring_preflight.status_code == 200
    assert monitoring_preflight.json()["selected_action"] == "run_preflight"
    assert monitoring_preflight.json()["capability_id"] == "quant_monitoring.validate_bundle"
    assert app_client.calls[-1]["capability_id"] == "quant_monitoring.validate_bundle"
    assert app_client.execution_calls[-1]["capability_id"] == (
        "quant_documentation.create_draft_workspace"
    )


def test_sample_autopilot_step_blocks_invalid_states_without_app_calls() -> None:
    app_client = FakePreflightAppClient()
    client = TestClient(create_app(runtime_with_preflight_client(app_client)))
    unknown = client.post(
        "/autopilot-steps",
        json={
            "run_id": "run_missing",
            "autopilot_intent": "advance_sample_autopilot_one_step",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )
    extra = client.post(
        "/autopilot-steps",
        json={
            "run_id": "run_missing",
            "autopilot_intent": "advance_sample_autopilot_one_step",
            "current_context_summary": _safe_sample_lifecycle_context(),
            "step_id": "step_browser_supplied",
        },
    )
    user_plan = _create_plan_with_lifecycle_reference(client)
    user_owned = client.post(
        "/autopilot-steps",
        json={
            "run_id": user_plan["run_id"],
            "autopilot_intent": "advance_sample_autopilot_one_step",
            "current_context_summary": _safe_lifecycle_context(),
        },
    )
    unsafe_plan = client.post(
        "/plans",
        json={
            "user_goal": "Plan a safe sample path.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    unsafe = client.post(
        "/autopilot-steps",
        json={
            "run_id": unsafe_plan["run_id"],
            "autopilot_intent": "advance_sample_autopilot_one_step",
            "current_context_summary": {
                **_safe_sample_lifecycle_context(),
                "raw_path": "C:\\private\\raw.csv",
            },
        },
    )

    assert unknown.status_code == 422
    assert unknown.json()["detail"]["errors"][0]["code"] == "unknown_run"
    assert extra.status_code == 422
    assert user_owned.status_code == 200
    assert user_owned.json()["advance_status"] == "blocked"
    assert unsafe.status_code == 200
    assert unsafe.json()["advance_status"] == "blocked"
    assert "private\\raw.csv" not in unsafe.text
    assert app_client.calls == []
    assert app_client.execution_calls == []


def test_sample_autopilot_step_blocks_non_demo_paused_cancelled_and_retry_states() -> None:
    non_demo_runtime = runtime_with_preflight_client(FakePreflightAppClient())
    non_demo_client = TestClient(create_app(non_demo_runtime))
    non_demo_plan = client_plan = non_demo_client.post(
        "/plans",
        json={
            "user_goal": "Advance a stale sample plan.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    entry = non_demo_runtime.planner.ledger.get(client_plan["run_id"])
    assert entry is not None
    stale_snapshot = copy.deepcopy(entry.plan_snapshot)
    stale_snapshot["proposed_steps"] = stale_snapshot["proposed_steps"][:2]
    non_demo_runtime.planner.ledger._entries[0] = entry.model_copy(  # noqa: SLF001 - corrupts test ledger intentionally.
        update={"plan_snapshot": stale_snapshot},
        deep=True,
    )
    non_demo = non_demo_client.post(
        "/autopilot-steps",
        json={
            "run_id": non_demo_plan["run_id"],
            "autopilot_intent": "advance_sample_autopilot_one_step",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )

    paused_runtime = runtime_with_preflight_client(FakePreflightAppClient())
    paused_client = TestClient(create_app(paused_runtime))
    paused_plan = paused_client.post(
        "/plans",
        json={
            "user_goal": "Advance paused sample plan.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    assert paused_client.post(
        "/pauses",
        json={"run_id": paused_plan["run_id"], "pause_intent": "pause_run", "reason": "user_paused"},
    ).status_code == 200
    paused = paused_client.post(
        "/autopilot-steps",
        json={
            "run_id": paused_plan["run_id"],
            "autopilot_intent": "advance_sample_autopilot_one_step",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )

    cancelled_runtime = runtime_with_preflight_client(FakePreflightAppClient())
    cancelled_client = TestClient(create_app(cancelled_runtime))
    cancelled_plan = cancelled_client.post(
        "/plans",
        json={
            "user_goal": "Advance cancelled sample plan.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    assert cancelled_client.post(
        "/cancellations",
        json={
            "run_id": cancelled_plan["run_id"],
            "cancellation_intent": "cancel_run",
            "reason": "user_cancelled",
        },
    ).status_code == 200
    cancelled = cancelled_client.post(
        "/autopilot-steps",
        json={
            "run_id": cancelled_plan["run_id"],
            "autopilot_intent": "advance_sample_autopilot_one_step",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )

    failed_result = _valid_action_result("failed_recoverable")
    failed_result["retry_allowed"] = True
    retry_runtime = runtime_with_preflight_client(FakePreflightAppClient(execution_response=failed_result))
    retry_client = TestClient(create_app(retry_runtime))
    retry_plan = retry_client.post(
        "/plans",
        json={
            "user_goal": "Advance retry-required sample plan.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    _advance_to_studio_step(retry_client, retry_plan)
    studio_step = _step_for_capability(retry_plan, "quant_studio.prepare_model_config_draft")
    assert retry_client.post(
        "/confirmations",
        json={
            "run_id": retry_plan["run_id"],
            "step_id": studio_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    ).status_code == 200
    assert retry_client.post(
        "/action-requests",
        json={"run_id": retry_plan["run_id"], "step_id": studio_step["step_id"]},
    ).status_code == 200
    assert retry_client.post(
        "/executions",
        json={"run_id": retry_plan["run_id"], "step_id": studio_step["step_id"]},
    ).status_code == 200
    retry_required = retry_client.post(
        "/autopilot-steps",
        json={
            "run_id": retry_plan["run_id"],
            "autopilot_intent": "advance_sample_autopilot_one_step",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )

    assert non_demo.status_code == 200
    assert non_demo.json()["advance_status"] == "blocked"
    assert paused.status_code == 200
    assert paused.json()["advance_status"] == "blocked"
    assert cancelled.status_code == 200
    assert cancelled.json()["advance_status"] == "blocked"
    assert retry_required.status_code == 200
    assert retry_required.json()["selected_action"] == "retry_failed_step"
    assert retry_required.json()["advance_status"] == "manual_retry_required"


def test_sample_autopilot_step_ledgers_delegated_failures_without_leaking_values() -> None:
    unavailable_app = FakePreflightAppClient(error=AppClientError("Quant Data unavailable.", status_code=503))
    unavailable_runtime = runtime_with_preflight_client(unavailable_app)
    unavailable_client = TestClient(create_app(unavailable_runtime))
    unavailable_plan = unavailable_client.post(
        "/plans",
        json={
            "user_goal": "Advance unavailable sample preflight.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    unavailable = unavailable_client.post(
        "/autopilot-steps",
        json={
            "run_id": unavailable_plan["run_id"],
            "autopilot_intent": "advance_sample_autopilot_one_step",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )

    unsafe_response = _valid_preflight_response()
    unsafe_response["raw_path"] = "C:\\private\\rows.csv"
    unsafe_app = FakePreflightAppClient(response=unsafe_response)
    unsafe_runtime = runtime_with_preflight_client(unsafe_app)
    unsafe_client = TestClient(create_app(unsafe_runtime))
    unsafe_plan = unsafe_client.post(
        "/plans",
        json={
            "user_goal": "Advance unsafe sample preflight.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    unsafe = unsafe_client.post(
        "/autopilot-steps",
        json={
            "run_id": unsafe_plan["run_id"],
            "autopilot_intent": "advance_sample_autopilot_one_step",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )

    assert unavailable.status_code == 200
    assert unavailable.json()["advance_status"] == "delegated_app_unavailable"
    assert unavailable.json()["validation"]["status"] == "rejected"
    assert "Quant Data unavailable" not in unavailable.text
    unavailable_entry = unavailable_runtime.planner.ledger.get(unavailable_plan["run_id"])
    assert unavailable_entry is not None
    assert unavailable_entry.recovery_events[-1]["status"] == "delegated_app_unavailable"
    assert unavailable_entry.preflight_records == []
    unavailable_runtime.contract_loader.validate_agent_contract_payload(
        unavailable_entry.model_dump(mode="json"),
        "agent_execution_ledger.v1.schema.json",
    )

    assert unsafe.status_code == 200
    assert unsafe.json()["advance_status"] == "delegated_rejected"
    assert unsafe.json()["validation"]["status"] == "rejected"
    assert "private\\rows.csv" not in unsafe.text
    unsafe_entry = unsafe_runtime.planner.ledger.get(unsafe_plan["run_id"])
    assert unsafe_entry is not None
    assert unsafe_entry.preflight_records == []
    assert unsafe_entry.recovery_events[-1]["status"] == "delegated_rejected"


def test_sample_reset_preview_and_reset_mark_run_terminal_and_block_gated_actions() -> None:
    app_client = FakePreflightAppClient()
    runtime = runtime_with_preflight_client(app_client)
    client = TestClient(create_app(runtime))
    plan = client.post(
        "/plans",
        json={
            "user_goal": "Reset the sample demo after review.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()

    preview = client.post(
        "/sample-reset-previews",
        json={
            "run_id": plan["run_id"],
            "reset_intent": "preview_sample_reset",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )
    assert preview.status_code == 200
    preview_payload = preview.json()
    reset_preview_id = preview_payload["reset_preview_id"]
    assert preview_payload["sample_eligibility"]["eligible"] is True
    assert preview_payload["reset_boundary_summary"]["sample_owned_only"] is True
    assert "sample_lifecycle_manifest" in preview_payload["reset_boundary_summary"]["allowed_delete_scopes"]

    reset = client.post(
        "/sample-resets",
        json={
            "run_id": plan["run_id"],
            "reset_intent": "reset_sample_demo",
            "reset_preview_id": reset_preview_id,
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )
    duplicate = client.post(
        "/sample-resets",
        json={
            "run_id": plan["run_id"],
            "reset_intent": "reset_sample_demo",
            "reset_preview_id": reset_preview_id,
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )

    assert reset.status_code == 200
    reset_payload = reset.json()
    assert reset_payload["reset_status"] == "reset"
    assert reset_payload["orchestration"]["run_state"] == "sample_reset"
    assert reset_payload["orchestration"]["final_status"] == "sample_reset"
    assert reset_payload["reset_result"] == {
        "result_type": "sample_workspace_reset",
        "status": "reset",
        "deleted_lifecycle_ids": ["sample_credit_pd_scorecard_panel"],
        "deleted_lifecycle_count": 1,
        "warning_count": 1,
        "warning_labels": ["warning_1"],
    }
    assert "lifecycle_response" not in json.dumps(reset_payload["reset_result"])
    assert duplicate.status_code == 200
    assert duplicate.json()["reset_status"] == "reset"
    assert app_client.reset_calls == [
        {"app_id": "quant_studio", "route": "/api/sample-workspaces/reset"}
    ]
    assert app_client.calls == []
    assert app_client.execution_calls == []

    entry = runtime.planner.ledger.get(plan["run_id"])
    assert entry is not None
    assert entry.final_status == "sample_reset"
    assert [event["event_type"] for event in entry.recovery_events[-2:]] == [
        "sample_reset_preview",
        "sample_reset",
    ]
    assert entry.recovery_events[-1]["sample_owned_only"] is True
    assert entry.recovery_events[-1]["execution_permitted"] is False
    runtime.contract_loader.validate_agent_contract_payload(
        entry.model_dump(mode="json"),
        "agent_execution_ledger.v1.schema.json",
    )

    source_step = _step_for_capability(plan, "quant_data.run_source_preflight")
    studio_step = _step_for_capability(plan, "quant_studio.prepare_model_config_draft")
    preflight = client.post(
        "/preflights",
        json={"run_id": plan["run_id"], "step_id": source_step["step_id"]},
    )
    confirmation = client.post(
        "/confirmations",
        json={
            "run_id": plan["run_id"],
            "step_id": studio_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    )
    action_request = client.post(
        "/action-requests",
        json={"run_id": plan["run_id"], "step_id": studio_step["step_id"]},
    )
    execution = client.post(
        "/executions",
        json={"run_id": plan["run_id"], "step_id": studio_step["step_id"]},
    )
    retry = client.post(
        "/retries",
        json={
            "run_id": plan["run_id"],
            "step_id": studio_step["step_id"],
            "retry_intent": "retry_failed_step",
        },
    )
    revision = client.post(
        "/plan-revisions",
        json={
            "run_id": plan["run_id"],
            "revision_intent": "revise_plan",
            "reason": "user_requested",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )
    autopilot_step = client.post(
        "/autopilot-steps",
        json={
            "run_id": plan["run_id"],
            "autopilot_intent": "advance_sample_autopilot_one_step",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )

    assert preflight.status_code == 422
    assert preflight.json()["detail"]["errors"][0]["code"] == "terminal_run_preflight"
    assert confirmation.status_code == 422
    assert confirmation.json()["detail"]["errors"][0]["code"] == "terminal_run_confirmation"
    assert action_request.status_code == 422
    assert action_request.json()["detail"]["errors"][0]["code"] == "terminal_run_action_request"
    assert execution.status_code == 422
    assert execution.json()["detail"]["errors"][0]["code"] == "terminal_run_execution"
    assert retry.status_code == 422
    assert retry.json()["detail"]["errors"][0]["code"] == "terminal_run_retry"
    assert revision.status_code == 422
    assert revision.json()["detail"]["errors"][0]["code"] == "terminal_run_plan_revision"
    assert autopilot_step.status_code == 200
    assert autopilot_step.json()["advance_status"] == "blocked"


def test_sample_reset_preview_blocks_ineligible_contexts_and_extra_payload_fields(tmp_path: Path) -> None:
    app_client = FakePreflightAppClient()
    runtime = runtime_with_preflight_client(app_client)
    client = TestClient(create_app(runtime))
    user_plan = _create_plan_with_lifecycle_reference(client)
    sample_plan = client.post(
        "/plans",
        json={
            "user_goal": "Preview sample reset blockers.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()

    user_owned = client.post(
        "/sample-reset-previews",
        json={
            "run_id": user_plan["run_id"],
            "reset_intent": "preview_sample_reset",
            "current_context_summary": _safe_lifecycle_context(),
        },
    )
    unsafe = client.post(
        "/sample-reset-previews",
        json={
            "run_id": sample_plan["run_id"],
            "reset_intent": "preview_sample_reset",
            "current_context_summary": {
                **_safe_sample_lifecycle_context(),
                "raw_path": "C:\\private\\raw.csv",
            },
        },
    )
    assert client.post(
        "/pauses",
        json={"run_id": sample_plan["run_id"], "pause_intent": "pause_run", "reason": "user_paused"},
    ).status_code == 200
    paused = client.post(
        "/sample-reset-previews",
        json={
            "run_id": sample_plan["run_id"],
            "reset_intent": "preview_sample_reset",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )
    extra = client.post(
        "/sample-reset-previews",
        json={
            "run_id": sample_plan["run_id"],
            "reset_intent": "preview_sample_reset",
            "current_context_summary": _safe_sample_lifecycle_context(),
            "reset_scope": "browser_supplied",
        },
    )

    assert user_owned.status_code == 200
    assert user_owned.json()["sample_eligibility"]["status"] == "blocked"
    assert unsafe.status_code == 200
    assert unsafe.json()["sample_eligibility"]["status"] == "blocked"
    assert "private\\raw.csv" not in unsafe.text
    assert paused.status_code == 200
    assert any("Paused runs" in blocker for blocker in paused.json()["sample_eligibility"]["blockers"])
    assert extra.status_code == 422
    assert app_client.reset_calls == []

    sample_root = tmp_path / "samples"
    sample_dir = sample_root / "credit_pd_scorecard_panel"
    sample_dir.mkdir(parents=True)
    (sample_dir / "sample_workspace.v1.json").write_text(
        json.dumps(
            {
                "sample_workspace_id": "credit_pd_scorecard_panel",
                "label": "Credit Risk PD Scorecard",
                "owned_marker": {
                    "sample_workspace": True,
                    "sample_workspace_id": "credit_pd_scorecard_panel",
                    "sample_owned": True,
                },
                "lifecycle_id": "sample_credit_pd_scorecard_panel",
                "reset_scope": {"sample_owned_only": False},
            }
        ),
        encoding="utf-8",
    )
    missing_reset_runtime = runtime_with_preflight_client(
        FakePreflightAppClient(),
        sample_workspace_root=sample_root,
    )
    missing_reset_client = TestClient(create_app(missing_reset_runtime))
    missing_reset_plan = missing_reset_client.post(
        "/plans",
        json={
            "user_goal": "Preview a sample reset without reset boundary.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    missing_reset = missing_reset_client.post(
        "/sample-reset-previews",
        json={
            "run_id": missing_reset_plan["run_id"],
            "reset_intent": "preview_sample_reset",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )
    assert missing_reset.status_code == 200
    assert missing_reset.json()["sample_eligibility"]["status"] == "blocked"
    assert any(
        "safe reset boundary" in blocker
        for blocker in missing_reset.json()["sample_eligibility"]["blockers"]
    )


def test_sample_reset_requires_matching_preview_and_handles_app_failures_safely() -> None:
    app_client = FakePreflightAppClient()
    runtime = runtime_with_preflight_client(app_client)
    client = TestClient(create_app(runtime))
    plan = client.post(
        "/plans",
        json={
            "user_goal": "Reset requires a ledgered preview.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()

    missing_preview = client.post(
        "/sample-resets",
        json={
            "run_id": plan["run_id"],
            "reset_intent": "reset_sample_demo",
            "reset_preview_id": "preview_missing",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )
    preview = client.post(
        "/sample-reset-previews",
        json={
            "run_id": plan["run_id"],
            "reset_intent": "preview_sample_reset",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    stale_marker = client.post(
        "/sample-resets",
        json={
            "run_id": plan["run_id"],
            "reset_intent": "reset_sample_demo",
            "reset_preview_id": preview["reset_preview_id"],
            "current_context_summary": _safe_sample_lifecycle_context(sample_workspace_id="monitoring_drift_review"),
        },
    )

    assert missing_preview.status_code == 200
    assert missing_preview.json()["reset_status"] == "blocked"
    assert missing_preview.json()["validation"]["status"] == "rejected"
    assert stale_marker.status_code == 200
    assert stale_marker.json()["reset_status"] == "blocked"
    assert app_client.reset_calls == []

    unavailable_app = FakePreflightAppClient(
        reset_error=AppClientError("Quant Studio reset unavailable.", status_code=503)
    )
    unavailable_runtime = runtime_with_preflight_client(unavailable_app)
    unavailable_client = TestClient(create_app(unavailable_runtime))
    unavailable_plan = unavailable_client.post(
        "/plans",
        json={
            "user_goal": "Reset with unavailable Studio.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    unavailable_preview = unavailable_client.post(
        "/sample-reset-previews",
        json={
            "run_id": unavailable_plan["run_id"],
            "reset_intent": "preview_sample_reset",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    unavailable = unavailable_client.post(
        "/sample-resets",
        json={
            "run_id": unavailable_plan["run_id"],
            "reset_intent": "reset_sample_demo",
            "reset_preview_id": unavailable_preview["reset_preview_id"],
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )
    assert unavailable.status_code == 200
    assert unavailable.json()["reset_status"] == "app_unavailable"
    assert "Quant Studio reset unavailable" not in unavailable.text
    unavailable_entry = unavailable_runtime.planner.ledger.get(unavailable_plan["run_id"])
    assert unavailable_entry is not None
    assert unavailable_entry.final_status == "planned"
    assert unavailable_entry.recovery_events[-1]["status"] == "app_unavailable"

    unsafe_app = FakePreflightAppClient(
        reset_response={
            **_valid_sample_reset_response(),
            "secret": "do-not-ledger",
            "raw_path": "C:\\private\\sample.csv",
        }
    )
    unsafe_runtime = runtime_with_preflight_client(unsafe_app)
    unsafe_client = TestClient(create_app(unsafe_runtime))
    unsafe_plan = unsafe_client.post(
        "/plans",
        json={
            "user_goal": "Reset with unsafe Studio response.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    unsafe_preview = unsafe_client.post(
        "/sample-reset-previews",
        json={
            "run_id": unsafe_plan["run_id"],
            "reset_intent": "preview_sample_reset",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    unsafe = unsafe_client.post(
        "/sample-resets",
        json={
            "run_id": unsafe_plan["run_id"],
            "reset_intent": "reset_sample_demo",
            "reset_preview_id": unsafe_preview["reset_preview_id"],
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )
    assert unsafe.status_code == 200
    assert unsafe.json()["reset_status"] == "app_rejected"
    assert "do-not-ledger" not in unsafe.text
    assert "private\\sample.csv" not in unsafe.text
    unsafe_entry = unsafe_runtime.planner.ledger.get(unsafe_plan["run_id"])
    assert unsafe_entry is not None
    assert unsafe_entry.final_status == "planned"
    assert unsafe_entry.recovery_events[-1]["status"] == "app_rejected"


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
    _advance_to_studio_step(client, plan_payload)
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
    assert missing_preflight.json()["detail"]["errors"][0]["code"] == "orchestration_action_not_allowed"
    assert missing_confirmation.status_code == 422
    assert missing_confirmation.json()["detail"]["errors"][0]["code"] == (
        "orchestration_step_not_ready"
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
    _advance_to_studio_step(stale_client, stale_payload)
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
    _advance_to_studio_step(unsafe_client, unsafe_payload)
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
    _advance_to_studio_step(malformed_client, malformed_payload)
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
    _advance_to_studio_step(client, plan_payload)
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
    plan_payload = _create_plan_with_lifecycle_reference(client)
    _advance_to_monitoring_step(client, plan_payload)
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
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    response = client.post(
        "/plans",
        json={
            "user_goal": "Plan safely.",
            "context_summary": {},
            "policy": {"provider_mode": "vendor_managed_saas"},
        },
    )

    assert response.status_code == 422
