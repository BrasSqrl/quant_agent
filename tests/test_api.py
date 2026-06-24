import copy
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import quant_agent_runtime.external_approval as external_approval_module
from quant_agent_runtime.action_request import ActionRequestPreviewService
from quant_agent_runtime.app_clients import AppClientError
from quant_agent_runtime.api import create_app
from quant_agent_runtime.capability_discovery import CapabilityDiscoveryService
from quant_agent_runtime.confirmation import ConfirmationService
from quant_agent_runtime.contracts import QuantSuiteContractLoader
from quant_agent_runtime.demo_narrative import DemoNarrativeService
from quant_agent_runtime.execution import ExecutionService
from quant_agent_runtime.governance import GovernanceService
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
from quant_agent_runtime.user_workflow import UserWorkflowService


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
        self.has_explicit_execution_response = execution_response is not None
        self.execution_response = execution_response or _valid_action_result()
        self._default_execution_response = self.execution_response
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
        response = self.responses_by_capability.get(capability_id)
        if response is None and self.response.get("capability_id") == capability_id:
            response = self.response
        if response is None:
            response = _valid_preflight_response(capability_id=capability_id, app_id=app_id)
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
        if (
            response is None
            and (self.has_explicit_execution_response or self.execution_response is not self._default_execution_response)
            and self.execution_response.get("capability_id") == capability_id
        ):
            response = self.execution_response
        if response is None and capability_id.startswith("quant_data."):
            action_request = payload.get("action_request") if isinstance(payload, dict) else {}
            response = _valid_action_result(
                capability_id=capability_id,
                app_id="quant_data",
                step_id=str(action_request.get("step_id") or "step_data"),
            )
        if response is None and capability_id.startswith("quant_studio."):
            action_request = payload.get("action_request") if isinstance(payload, dict) else {}
            response = _valid_action_result(
                capability_id=capability_id,
                app_id="quant_studio",
                step_id=str(action_request.get("step_id") or "step_studio"),
            )
        if response is None and capability_id.startswith("quant_documentation."):
            action_request = payload.get("action_request") if isinstance(payload, dict) else {}
            response = _valid_action_result(
                capability_id=capability_id,
                app_id="quant_documentation",
                step_id=str(action_request.get("step_id") or "step_documentation"),
            )
        if response is None and capability_id.startswith("quant_monitoring."):
            action_request = payload.get("action_request") if isinstance(payload, dict) else {}
            response = _valid_action_result(
                capability_id=capability_id,
                app_id="quant_monitoring",
                step_id=str(action_request.get("step_id") or "step_monitoring"),
            )
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
    governance = GovernanceService.from_contracts(ledger=ledger, contract_loader=loader)
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
        run_status=RunStatusService(
            ledger=ledger,
            capability_discovery=discovery,
            governance=governance,
        ),
        orchestration=OrchestrationService(
            ledger=ledger,
            governance=governance,
            capability_discovery=discovery,
        ),
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
        demo_narrative=DemoNarrativeService(
            ledger=ledger,
            sample_workspace_root=sample_workspace_root or QUANT_SUITE_ROOT / "fixtures" / "sample_workspaces",
        ),
        user_workflow=UserWorkflowService(ledger=ledger),
        contract_loader=loader,
        capability_discovery=discovery,
        provider_status=provider_status,
        governance=governance,
    )


def _valid_preflight_response(
    status: str = "ready",
    *,
    capability_id: str = "quant_data.run_source_preflight",
    app_id: str = "quant_data",
) -> dict[str, Any]:
    if app_id == "quant_data":
        if capability_id == "quant_data.run_eda_review":
            reference_type = "eda_plan"
            reference_id = "eda_plan_test"
        else:
            reference_type = "source_reference"
            reference_id = "source_ref_test"
        evidence_check_id = "source_summary_present"
        summary_key = "source_summary"
    elif app_id == "quant_studio":
        reference_type = (
            "model_readiness_summary"
            if capability_id == "quant_studio.run_model_readiness_check"
            else "model_config_draft"
        )
        reference_id = "model_readiness_summary_test" if reference_type == "model_readiness_summary" else "model_config_draft_test"
        evidence_check_id = "target_summary_present"
        summary_key = "target_summary"
    else:
        if capability_id == "quant_monitoring.run_monitoring_review":
            reference_type = "bundle_validation_summary"
            reference_id = "bundle_validation_summary_test"
        else:
            reference_type = "monitoring_bundle"
            reference_id = "bundle_ref_test"
        evidence_check_id = "bundle_summary_present"
        summary_key = "bundle_summary"
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
        "safe_artifact_references": _preflight_references_for(capability_id, reference_type, reference_id),
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


def _preflight_references_for(capability_id: str, reference_type: str, reference_id: str) -> list[dict[str, str]]:
    references = [
        {
            "reference_type": reference_type,
            "reference_id": reference_id,
            "label": "Safe preflight reference",
        }
    ]
    if capability_id == "quant_data.run_source_preflight":
        references.append(
            {
                "reference_type": "preflight_summary",
                "reference_id": "preflight_summary_test",
                "label": "Source preflight summary",
            }
        )
    if capability_id == "quant_monitoring.validate_bundle":
        references.append(
            {
                "reference_type": "bundle_validation_summary",
                "reference_id": "bundle_validation_summary_test",
                "label": "Bundle validation summary",
            }
        )
    return references


def _valid_action_result(
    execution_status: str = "succeeded",
    *,
    capability_id: str = "quant_studio.prepare_model_config_draft",
    app_id: str = "quant_studio",
    step_id: str = "step_2",
) -> dict[str, Any]:
    output_reference_by_capability = {
        "quant_data.register_source_reference": {
            "reference_type": "source_reference",
            "reference_id": "source_reference_test",
            "label": "Source reference",
        },
        "quant_data.create_eda_plan": {
            "reference_type": "eda_plan",
            "reference_id": "eda_plan_test",
            "label": "EDA plan",
        },
        "quant_data.run_eda_review": {
            "reference_type": "eda_package",
            "reference_id": "eda_package_test",
            "label": "EDA package",
        },
        "quant_data.export_eda_handoff": {
            "reference_type": "eda_handoff",
            "reference_id": "eda_handoff_test",
            "label": "EDA handoff",
        },
        "quant_studio.prepare_model_config_draft": {
            "reference_type": "model_config_draft",
            "reference_id": "model_config_draft_test",
            "label": "Model configuration draft",
        },
        "quant_studio.fit_candidate_model": {
            "reference_type": "studio_run",
            "reference_id": "studio_run_test",
            "label": "Candidate Studio run",
        },
        "quant_studio.compare_candidate_runs": {
            "reference_type": "champion_recommendation",
            "reference_id": "champion_recommendation_test",
            "label": "Champion recommendation",
        },
        "quant_studio.create_documentation_package": {
            "reference_type": "documentation_package",
            "reference_id": "documentation_package_test",
            "label": "Documentation package",
        },
        "quant_documentation.inspect_package": {
            "reference_type": "documentation_package_summary",
            "reference_id": "documentation_package_summary_test",
            "label": "Documentation package summary",
        },
        "quant_documentation.create_draft_workspace": {
            "reference_type": "documentation_draft",
            "reference_id": "documentation_draft_workspace_test",
            "label": "Documentation draft workspace",
        },
        "quant_documentation.draft_section": {
            "reference_type": "draft_section",
            "reference_id": "draft_section_test",
            "label": "Reviewable draft section",
        },
        "quant_documentation.find_unsupported_claims": {
            "reference_type": "claim_review_summary",
            "reference_id": "claim_review_summary_test",
            "label": "Claim review summary",
        },
        "quant_documentation.export_markdown_review_package": {
            "reference_type": "documentation_review_package",
            "reference_id": "documentation_review_package_test",
            "label": "Documentation review package",
        },
        "quant_monitoring.inspect_bundle": {
            "reference_type": "bundle_summary",
            "reference_id": "bundle_summary_test",
            "label": "Monitoring bundle summary",
        },
        "quant_monitoring.prepare_profile_draft": {
            "reference_type": "monitoring_profile_draft",
            "reference_id": "monitoring_profile_draft_test",
            "label": "Monitoring profile draft",
        },
        "quant_monitoring.run_monitoring_review": {
            "reference_type": "monitoring_run",
            "reference_id": "monitoring_run_test",
            "label": "Monitoring run",
        },
        "quant_monitoring.create_feedback_signal": {
            "reference_type": "feedback_signal",
            "reference_id": "feedback_signal_test",
            "label": "Feedback signal",
        },
    }
    output_reference = output_reference_by_capability.get(
        capability_id,
        output_reference_by_capability["quant_studio.prepare_model_config_draft"],
    )
    output_references = [output_reference]
    if capability_id == "quant_studio.create_documentation_package":
        output_references.append(
            {
                "reference_type": "monitoring_bundle",
                "reference_id": "monitoring_bundle_test",
                "label": "Monitoring bundle",
            }
        )
    return {
        "schema_version": "1.0",
        "data_policy": "summaries_and_references_only",
        "action_run_id": "action_studio_draft_test",
        "step_id": step_id,
        "capability_id": capability_id,
        "app_id": app_id,
        "execution_status": execution_status,
        "accepted_input_summary": {
            (
                "source_summary"
                if app_id == "quant_data"
                else "package_summary"
                if app_id == "quant_documentation"
                else "bundle_summary"
                if app_id == "quant_monitoring"
                else "target_summary"
            ): (
                "Reviewed source summary."
                if app_id == "quant_data"
                else "Documentation package summary is ready."
                if app_id == "quant_documentation"
                else "Monitoring bundle summary is ready."
                if app_id == "quant_monitoring"
                else "Default flag is the candidate target."
            ),
            "lifecycle_id": "lifecycle_test",
        },
        "output_references": output_references,
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
                    "capability_id": "quant_data.register_source_reference",
                    "app_id": "quant_data",
                    "version": "1.0",
                    "display_name": "Register source reference",
                    "summary": "Registers a summarized source reference for a governed Quant Data workflow.",
                    "risk_tier": "reversible_write",
                    "enabled": True,
                    "preflight_required": False,
                    "confirmation_required": True,
                    "execution_supported": True,
                    "idempotent": True,
                    "reversible": True,
                    "side_effects": ["app_owned_reference_write_after_confirmation"],
                    "input_schema": {"required_fields": ["source_summary"]},
                    "output_schema": {"safe_reference_types": ["source_reference"]},
                    "data_policy": "summaries_and_references_only",
                },
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
                },
                {
                    "capability_id": "quant_data.create_eda_plan",
                    "app_id": "quant_data",
                    "version": "1.0",
                    "display_name": "Create EDA plan",
                    "summary": "Creates a safe reviewable EDA plan from source summaries.",
                    "risk_tier": "draft_only",
                    "enabled": True,
                    "preflight_required": False,
                    "confirmation_required": True,
                    "execution_supported": True,
                    "idempotent": True,
                    "reversible": True,
                    "side_effects": ["draft_only_after_confirmation"],
                    "input_schema": {"required_fields": ["source_summary"]},
                    "output_schema": {"safe_reference_types": ["eda_plan"]},
                    "data_policy": "summaries_and_references_only",
                },
                {
                    "capability_id": "quant_data.run_eda_review",
                    "app_id": "quant_data",
                    "version": "1.0",
                    "display_name": "Run EDA review",
                    "summary": "Runs app-owned EDA review from safe source references and returns summaries only.",
                    "risk_tier": "expensive_compute",
                    "enabled": True,
                    "preflight_required": True,
                    "confirmation_required": True,
                    "execution_supported": True,
                    "idempotent": False,
                    "reversible": False,
                    "side_effects": ["app_owned_compute_after_confirmation"],
                    "input_schema": {"required_fields": ["source_summary"]},
                    "output_schema": {"safe_reference_types": ["eda_package"]},
                    "data_policy": "summaries_and_references_only",
                },
                {
                    "capability_id": "quant_data.export_eda_handoff",
                    "app_id": "quant_data",
                    "version": "1.0",
                    "display_name": "Export EDA handoff",
                    "summary": "Exports a safe EDA handoff reference for Quant Studio.",
                    "risk_tier": "artifact_export",
                    "enabled": True,
                    "preflight_required": False,
                    "confirmation_required": True,
                    "execution_supported": True,
                    "idempotent": True,
                    "reversible": True,
                    "side_effects": ["safe_artifact_reference_export_after_confirmation"],
                    "input_schema": {"required_fields": ["source_summary"]},
                    "output_schema": {"safe_reference_types": ["eda_handoff"]},
                    "data_policy": "summaries_and_references_only",
                },
            ]
        elif app_id == "quant_studio":
            capabilities = [
                {
                    "capability_id": "quant_studio.run_model_readiness_check",
                    "app_id": "quant_studio",
                    "version": "1.0",
                    "display_name": "Run model readiness check",
                    "summary": "Checks Studio modeling readiness from safe target and handoff summaries.",
                    "risk_tier": "workflow_preflight",
                    "enabled": True,
                    "preflight_required": True,
                    "confirmation_required": False,
                    "execution_supported": False,
                    "idempotent": True,
                    "reversible": True,
                    "side_effects": ["none_preflight_only"],
                    "input_schema": {"required_fields": ["target_summary"]},
                    "output_schema": {"safe_reference_types": ["model_readiness_summary"]},
                    "data_policy": "summaries_and_references_only",
                },
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
                },
                {
                    "capability_id": "quant_studio.fit_candidate_model",
                    "app_id": "quant_studio",
                    "version": "1.0",
                    "display_name": "Fit candidate model",
                    "summary": "Runs an app-owned candidate model fit and returns safe model-run references.",
                    "risk_tier": "expensive_compute",
                    "enabled": True,
                    "preflight_required": True,
                    "confirmation_required": True,
                    "execution_supported": True,
                    "idempotent": False,
                    "reversible": False,
                    "side_effects": ["app_owned_model_compute_after_confirmation"],
                    "input_schema": {"required_fields": ["target_summary"]},
                    "output_schema": {"safe_reference_types": ["studio_run"]},
                    "data_policy": "summaries_and_references_only",
                },
                {
                    "capability_id": "quant_studio.compare_candidate_runs",
                    "app_id": "quant_studio",
                    "version": "1.0",
                    "display_name": "Compare candidate runs",
                    "summary": "Compares safe candidate run summaries and creates a reviewable recommendation.",
                    "risk_tier": "draft_only",
                    "enabled": True,
                    "preflight_required": False,
                    "confirmation_required": True,
                    "execution_supported": True,
                    "idempotent": True,
                    "reversible": True,
                    "side_effects": ["draft_only_after_confirmation"],
                    "input_schema": {"required_fields": ["target_summary"]},
                    "output_schema": {"safe_reference_types": ["champion_recommendation"]},
                    "data_policy": "summaries_and_references_only",
                },
                {
                    "capability_id": "quant_studio.create_documentation_package",
                    "app_id": "quant_studio",
                    "version": "1.0",
                    "display_name": "Create documentation package",
                    "summary": "Creates safe documentation and monitoring package references.",
                    "risk_tier": "artifact_export",
                    "enabled": True,
                    "preflight_required": False,
                    "confirmation_required": True,
                    "execution_supported": True,
                    "idempotent": True,
                    "reversible": True,
                    "side_effects": ["safe_artifact_reference_export_after_confirmation"],
                    "input_schema": {"required_fields": ["target_summary"]},
                    "output_schema": {"safe_reference_types": ["documentation_package", "monitoring_bundle"]},
                    "data_policy": "summaries_and_references_only",
                },
            ]
        elif app_id == "quant_documentation":
            capabilities = [
                {
                    "capability_id": "quant_documentation.inspect_package",
                    "app_id": "quant_documentation",
                    "version": "1.0-draft",
                    "display_name": "Inspect documentation package",
                    "summary": "Inspects safe package references and documentation readiness summaries.",
                    "risk_tier": "read_only",
                    "enabled": True,
                    "preflight_required": False,
                    "confirmation_required": False,
                    "execution_supported": True,
                    "idempotent": True,
                    "reversible": True,
                    "side_effects": ["none_read_only"],
                    "input_schema": {"required_fields": ["package_summary"]},
                    "output_schema": {"safe_reference_types": ["documentation_package_summary"]},
                    "data_policy": "summaries_and_references_only",
                },
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
                },
                {
                    "capability_id": "quant_documentation.draft_section",
                    "app_id": "quant_documentation",
                    "version": "1.0",
                    "display_name": "Draft documentation section",
                    "summary": "Creates a reviewable draft-section reference from safe package evidence.",
                    "risk_tier": "draft_only",
                    "enabled": True,
                    "preflight_required": False,
                    "confirmation_required": True,
                    "execution_supported": True,
                    "idempotent": False,
                    "reversible": True,
                    "side_effects": ["draft_only_after_confirmation"],
                    "input_schema": {"required_fields": ["package_summary"]},
                    "output_schema": {"safe_reference_types": ["draft_section"]},
                    "data_policy": "summaries_and_references_only",
                },
                {
                    "capability_id": "quant_documentation.find_unsupported_claims",
                    "app_id": "quant_documentation",
                    "version": "1.0",
                    "display_name": "Find unsupported documentation claims",
                    "summary": "Reviews draft summaries for unsupported claims using safe citation references.",
                    "risk_tier": "read_only",
                    "enabled": True,
                    "preflight_required": False,
                    "confirmation_required": False,
                    "execution_supported": True,
                    "idempotent": True,
                    "reversible": True,
                    "side_effects": ["none_read_only"],
                    "input_schema": {"required_fields": ["package_summary"]},
                    "output_schema": {"safe_reference_types": ["claim_review_summary"]},
                    "data_policy": "summaries_and_references_only",
                },
                {
                    "capability_id": "quant_documentation.export_markdown_review_package",
                    "app_id": "quant_documentation",
                    "version": "1.0",
                    "display_name": "Export Markdown review package",
                    "summary": "Creates a safe Markdown review package reference after claim review.",
                    "risk_tier": "artifact_export",
                    "enabled": True,
                    "preflight_required": False,
                    "confirmation_required": True,
                    "execution_supported": True,
                    "idempotent": True,
                    "reversible": True,
                    "side_effects": ["safe_artifact_reference_export_after_confirmation"],
                    "input_schema": {"required_fields": ["package_summary"]},
                    "output_schema": {"safe_reference_types": ["documentation_review_package"]},
                    "data_policy": "summaries_and_references_only",
                },
            ]
        elif app_id == "quant_monitoring":
            capabilities = [
                {
                    "capability_id": "quant_monitoring.inspect_bundle",
                    "app_id": "quant_monitoring",
                    "version": "1.0",
                    "display_name": "Inspect monitoring bundle",
                    "summary": "Inspects safe monitoring bundle references before setup.",
                    "risk_tier": "read_only",
                    "enabled": True,
                    "preflight_required": False,
                    "confirmation_required": False,
                    "execution_supported": True,
                    "idempotent": True,
                    "reversible": True,
                    "side_effects": ["none_read_only"],
                    "input_schema": {"required_fields": ["bundle_summary"]},
                    "output_schema": {"safe_reference_types": ["bundle_summary"]},
                    "data_policy": "summaries_and_references_only",
                },
                {
                    "capability_id": "quant_monitoring.prepare_profile_draft",
                    "app_id": "quant_monitoring",
                    "version": "1.0",
                    "display_name": "Prepare monitoring profile draft",
                    "summary": "Creates a reviewable monitoring profile or threshold draft from safe bundle summaries.",
                    "risk_tier": "draft_only",
                    "enabled": True,
                    "preflight_required": False,
                    "confirmation_required": True,
                    "execution_supported": True,
                    "idempotent": True,
                    "reversible": True,
                    "side_effects": ["draft_only_after_confirmation"],
                    "input_schema": {"required_fields": ["bundle_summary"]},
                    "output_schema": {"safe_reference_types": ["monitoring_profile_draft"]},
                    "data_policy": "summaries_and_references_only",
                },
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
                    "execution_supported": False,
                    "idempotent": True,
                    "reversible": True,
                    "side_effects": ["none"],
                    "input_schema": {"required_fields": ["bundle_summary"]},
                    "output_schema": {
                        "safe_reference_types": ["bundle_validation_summary", "monitoring_bundle"]
                    },
                    "data_policy": "summaries_and_references_only",
                },
                {
                    "capability_id": "quant_monitoring.run_monitoring_review",
                    "app_id": "quant_monitoring",
                    "version": "1.0",
                    "display_name": "Run monitoring review",
                    "summary": "Runs an app-owned monitoring review from safe bundle and validation references.",
                    "risk_tier": "expensive_compute",
                    "enabled": True,
                    "preflight_required": True,
                    "confirmation_required": True,
                    "execution_supported": True,
                    "idempotent": False,
                    "reversible": False,
                    "side_effects": ["app_owned_monitoring_compute_after_confirmation"],
                    "input_schema": {"required_fields": ["bundle_summary"]},
                    "output_schema": {"safe_reference_types": ["monitoring_run"]},
                    "data_policy": "summaries_and_references_only",
                },
                {
                    "capability_id": "quant_monitoring.create_feedback_signal",
                    "app_id": "quant_monitoring",
                    "version": "1.0",
                    "display_name": "Create feedback signal",
                    "summary": "Creates a safe feedback or retrain signal reference from monitoring review summaries.",
                    "risk_tier": "reversible_write",
                    "enabled": True,
                    "preflight_required": False,
                    "confirmation_required": True,
                    "execution_supported": True,
                    "idempotent": True,
                    "reversible": True,
                    "side_effects": ["app_owned_feedback_reference_write_after_confirmation"],
                    "input_schema": {"required_fields": ["bundle_summary"]},
                    "output_schema": {"safe_reference_types": ["feedback_signal"]},
                    "data_policy": "summaries_and_references_only",
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


def _safe_user_owned_lifecycle_context() -> dict[str, Any]:
    context = _safe_lifecycle_context()
    context["lifecycle_summary"] = {
        **context["lifecycle_summary"],
        "ownership": "user_owned",
        "sample_workspace": None,
    }
    return context


def _create_user_owned_plan(client: TestClient) -> dict[str, Any]:
    return _create_user_owned_plan_with_policy(client)


def _create_user_owned_plan_with_policy(
    client: TestClient,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "user_goal": "Build a governed user-owned PD scorecard plan.",
        "context_summary": _safe_user_owned_lifecycle_context(),
    }
    if policy is not None:
        payload["policy"] = policy
    response = client.post(
        "/plans",
        json=payload,
    )
    assert response.status_code == 200
    return response.json()


def _check_user_owned_readiness(client: TestClient, plan_payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post(
        "/user-workflow-readiness",
        json={
            "run_id": plan_payload["run_id"],
            "readiness_intent": "check_user_owned_readiness",
            "current_context_summary": _safe_user_owned_lifecycle_context(),
        },
    )
    assert response.status_code == 200
    return response.json()


def _review_user_plan(
    client: TestClient,
    plan_payload: dict[str, Any],
    *,
    decision: str = "accept",
    safe_note: str | None = None,
) -> dict[str, Any]:
    assumptions = plan_payload["plan"].get("assumptions", [])
    response = client.post(
        "/user-plan-reviews",
        json={
            "run_id": plan_payload["run_id"],
            "review_intent": "review_plan_assumptions",
            "assumption_reviews": [
                {
                    "assumption_index": index,
                    "decision": decision,
                    **({"safe_note": safe_note or "Revise this assumption before approval."} if decision == "revise" else {}),
                }
                for index, _assumption in enumerate(assumptions)
            ],
            "current_context_summary": _safe_user_owned_lifecycle_context(),
        },
    )
    assert response.status_code == 200
    return response.json()


def _approve_user_plan(
    client: TestClient,
    plan_payload: dict[str, Any],
    review_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    review_payload = review_payload or _review_user_plan(client, plan_payload)
    response = client.post(
        "/user-plan-approvals",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "approve_user_plan",
            "plan_review_id": review_payload["plan_review_summary"]["plan_review_id"],
        },
    )
    assert response.status_code == 200
    return response.json()


def _approve_user_owned_consent(client: TestClient, plan_payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post(
        "/user-workflow-consents",
        json={
            "run_id": plan_payload["run_id"],
            "consent_intent": "approve_user_owned_guided_execution",
            "consent_scope": "single_run_review_draft_actions",
        },
    )
    assert response.status_code == 200
    return response.json()


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


def _expected_demo_narrative_fixture() -> dict[str, Any]:
    fixture_path = (
        QUANT_SUITE_ROOT
        / "fixtures"
        / "sample_workspaces"
        / "credit_pd_scorecard_panel"
        / "agent_demo_expected_narrative.v1.json.fixture"
    )
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def _expected_demo_certification_fixture() -> dict[str, Any]:
    fixture_path = (
        QUANT_SUITE_ROOT
        / "fixtures"
        / "sample_workspaces"
        / "credit_pd_scorecard_panel"
        / "agent_demo_certification_expected_ledger.v1.json.fixture"
    )
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def _expected_user_owned_phase8_certification_fixture() -> dict[str, Any]:
    fixture_path = (
        QUANT_SUITE_ROOT
        / "fixtures"
        / "agent_certification"
        / "user_owned_phase8_expected_ledger_shape.json.fixture"
    )
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def _expected_phase9_governance_certification_fixture() -> dict[str, Any]:
    fixture_path = (
        QUANT_SUITE_ROOT
        / "fixtures"
        / "agent_certification"
        / "phase9_governance_expected_evidence.json.fixture"
    )
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def _expected_phase10_external_approval_adapter_fixture() -> dict[str, Any]:
    fixture_path = (
        QUANT_SUITE_ROOT
        / "fixtures"
        / "agent_certification"
        / "phase10_external_approval_adapter_expected_evidence.json.fixture"
    )
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def _assert_safe_ledger_export_text(serialized: str) -> None:
    for unsafe_term in [
        "C:\\",
        "/Users/",
        "http://",
        "https://",
        "raw_path",
        "raw_paths",
        "raw_rows",
        "bucket_name",
        "OPENAI_API_KEY",
        "sk-test",
        "paste-your-openai-api-key",
        "do-not-ledger",
        "\"links\"",
        "\"query\"",
    ]:
        assert unsafe_term not in serialized


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


def _full_lifecycle_capability_ids() -> list[str]:
    return [
        "quant_data.register_source_reference",
        "quant_data.run_source_preflight",
        "quant_data.create_eda_plan",
        "quant_data.run_eda_review",
        "quant_data.export_eda_handoff",
        "quant_studio.run_model_readiness_check",
        "quant_studio.prepare_model_config_draft",
        "quant_studio.fit_candidate_model",
        "quant_studio.compare_candidate_runs",
        "quant_studio.create_documentation_package",
        "quant_documentation.inspect_package",
        "quant_documentation.create_draft_workspace",
        "quant_documentation.draft_section",
        "quant_documentation.find_unsupported_claims",
        "quant_documentation.export_markdown_review_package",
        "quant_monitoring.inspect_bundle",
        "quant_monitoring.prepare_profile_draft",
        "quant_monitoring.validate_bundle",
        "quant_monitoring.run_monitoring_review",
        "quant_monitoring.create_feedback_signal",
    ]


def _write_governance_policy_pack(
    path: Path,
    *,
    role_id: str = "local_developer_operator",
    allowed_routes: list[str] | None = None,
    denied_routes: list[str] | None = None,
    allowed_capability_ids: list[str] | None = None,
    denied_capability_ids: list[str] | None = None,
    separation_of_duties_rules: list[dict[str, Any]] | None = None,
    external_approval_rules: list[dict[str, Any]] | None = None,
) -> None:
    payload = {
        "schema_version": "1.0",
        "data_policy": "summaries_and_references_only",
        "policy_pack_id": "test_governance_policy_pack",
        "environment_label": "local_development",
        "default_actor_role": role_id,
        "roles": [
            {
                "role_id": role_id,
                "display_name": "Test role",
                "description": "Role used by governance tests.",
                "actor_kind": "local_user",
            }
        ],
        "route_permissions": [
            {
                "role_id": role_id,
                "allowed_routes": allowed_routes or ["*"],
                "denied_routes": denied_routes or [],
            }
        ],
        "capability_permissions": [
            {
                "role_id": role_id,
                "allowed_capability_ids": allowed_capability_ids or ["*"],
                "denied_capability_ids": denied_capability_ids or [],
            }
        ],
        "audit_requirements": {
            "ledger_denials": True,
            "redaction_policy": "summaries_and_references_only",
            "actor_label_policy": "local_user_with_configured_role",
        },
        "fallback_behavior": {
            "missing_pack": "allow_existing_local_behavior_with_diagnostics",
            "invalid_pack": "allow_existing_local_behavior_with_diagnostics",
            "unknown_role": "allow_existing_local_behavior_with_diagnostics",
        },
        "separation_of_duties_rules": separation_of_duties_rules or [],
        "external_approval_rules": external_approval_rules or [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _blocking_sod_rule(*, exempt_roles: list[str] | None = None) -> dict[str, Any]:
    return {
        "rule_id": "draft_action_gate_and_execution_separation",
        "display_name": "Draft action gate and execution separation",
        "description": "Non-exempt actors cannot execute or retry after recording prior governance gates.",
        "enforcement_mode": "blocking",
        "protected_routes": ["POST /executions", "POST /retries"],
        "exempt_roles": exempt_roles or [],
    }


def _blocking_external_approval_rule(
    *,
    enforcement_mode: str = "blocking",
    exempt_roles: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "rule_id": "draft_action_external_approval_gate",
        "display_name": "Draft action external approval gate",
        "description": "Governed draft execution and retry require external approval evidence.",
        "enforcement_mode": enforcement_mode,
        "protected_routes": ["POST /executions", "POST /retries"],
        "protected_capability_ids": [
            "quant_studio.prepare_model_config_draft",
            "quant_documentation.create_draft_workspace",
        ],
        "accepted_decision_statuses": ["approved"],
        "allowed_scopes": ["run", "step"],
        "exempt_roles": exempt_roles or [],
    }


def _prepare_user_owned_studio_execution(
    client: TestClient,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_payload = _create_user_owned_plan(client)
    _check_user_owned_readiness(client, plan_payload)
    _approve_user_plan(client, plan_payload)
    _approve_user_owned_consent(client, plan_payload)
    _run_source_preflight(client, plan_payload)
    studio_step, _preview = _create_studio_preview(client, plan_payload)
    return plan_payload, studio_step


def _client_for_governance_environment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ledger: InMemoryLedger,
    app_client: FakePreflightAppClient,
    environment: str,
    actor_role: str,
    actor_id: str,
) -> TestClient:
    monkeypatch.delenv("QUANT_AGENT_GOVERNANCE_POLICY_PACK_PATH", raising=False)
    monkeypatch.setenv("QUANT_AGENT_GOVERNANCE_ENVIRONMENT", environment)
    monkeypatch.setenv("QUANT_AGENT_ACTOR_ROLE", actor_role)
    monkeypatch.setenv("QUANT_AGENT_ACTOR_ID", actor_id)
    return TestClient(create_app(runtime_with_preflight_client(app_client, ledger=ledger)))


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
    _complete_documentation_inspection_step(client, plan_payload)
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


def _complete_documentation_inspection_step(
    client: TestClient,
    plan_payload: dict[str, Any],
) -> None:
    inspection_step = next(
        (
            step
            for step in plan_payload["plan"]["proposed_steps"]
            if step["capability_id"] == "quant_documentation.inspect_package"
        ),
        None,
    )
    if inspection_step is None:
        return
    orchestration_response = client.get(f"/runs/{plan_payload['run_id']}/orchestration")
    assert orchestration_response.status_code == 200
    inspection_summary = next(
        (
            step
            for step in orchestration_response.json()["steps"]
            if step["step_id"] == inspection_step["step_id"]
        ),
        None,
    )
    if inspection_summary is None or inspection_summary["status"] in {
        "completed",
        "completed_with_warnings",
        "informational",
    }:
        return
    if inspection_summary["status"] == "ready_for_action_request":
        preview_response = client.post(
            "/action-requests",
            json={
                "run_id": plan_payload["run_id"],
                "step_id": inspection_step["step_id"],
            },
        )
        assert preview_response.status_code == 200
    execution_response = client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": inspection_step["step_id"]},
    )
    assert execution_response.status_code == 200


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
    _complete_documentation_inspection_step(client, plan_payload)


def _advance_to_monitoring_step(client: TestClient, plan_payload: dict[str, Any]) -> None:
    _advance_to_documentation_step(client, plan_payload)
    _complete_documentation_step(client, plan_payload)


def _run_monitoring_preflight(client: TestClient, plan_payload: dict[str, Any]) -> dict[str, Any]:
    monitoring_step = _step_for_capability(plan_payload, "quant_monitoring.validate_bundle")
    response = client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": monitoring_step["step_id"]},
    )
    assert response.status_code == 200
    return response.json()


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


def test_runtime_manifest_returns_supported_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QUANT_AGENT_GOVERNANCE_POLICY_PACK_PATH", raising=False)
    monkeypatch.delenv("QUANT_AGENT_GOVERNANCE_ENVIRONMENT", raising=False)
    monkeypatch.delenv("QUANT_AGENT_ACTOR_ROLE", raising=False)
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
    assert "GET /runs/{run_id}/demo-narrative" in manifest["supported_routes"]
    assert "POST /user-plan-reviews" in manifest["supported_routes"]
    assert "POST /user-plan-approvals" in manifest["supported_routes"]
    assert "POST /user-workflow-readiness" in manifest["supported_routes"]
    assert "POST /user-workflow-consents" in manifest["supported_routes"]
    assert "POST /external-approval-requests" in manifest["supported_routes"]
    assert "POST /external-approval-submissions" in manifest["supported_routes"]
    assert "POST /external-approval-decisions" in manifest["supported_routes"]
    assert "POST /external-approval-decision-refreshes" in manifest["supported_routes"]
    assert "GET /runs/{run_id}/external-approval-submissions" in manifest["supported_routes"]
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
    assert manifest["demo_narrative_support_level"] == "sample_owned_ledger_narrative_only"
    assert manifest["external_approval_support_level"] == "manual_approval_package_preview_only"
    assert manifest["external_approval_decision_support_level"] == "manual_decision_import_only"
    assert manifest["external_approval_enforcement_support_level"] == "policy_required_decision_enforced"
    assert manifest["external_approval_submission_support_level"] == "local_outbox_submission_only"
    assert manifest["external_approval_submission_status_support_level"] == (
        "ledger_and_local_outbox_status"
    )
    assert manifest["external_approval_decision_refresh_support_level"] == "mock_http_manual_refresh_only"
    assert manifest["external_approval_adapter_support_level"] == "local_outbox_and_mock_http_submission"
    assert manifest["external_approval_submission_adapter"]["adapter_mode"] == "local_outbox"
    assert manifest["external_approval_submission_adapter"]["enabled"] is True
    assert manifest["external_approval_submission_adapter"]["supports_external_network"] is False
    assert manifest["external_approval_submission_adapter"]["adapter_support_level"] == (
        "local_outbox_and_mock_http_submission"
    )
    assert manifest["governance_support_level"] == "role_aware_policy_pack_enforced"
    assert manifest["environment_policy_pack_support_level"] == "suite_fixture_environment_selection"
    assert manifest["release_evidence_support_level"] == "contract_policy_redaction_checks"
    assert manifest["user_workflow_support_level"] == "manual_user_owned_consent_gate_only"
    assert manifest["user_plan_approval_support_level"] == "manual_active_plan_approval_only"
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
        "quant_data.run_eda_review",
        "quant_studio.run_model_readiness_check",
        "quant_studio.fit_candidate_model",
        "quant_monitoring.run_monitoring_review",
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
        "quant_data.run_eda_review",
        "quant_studio.run_model_readiness_check",
        "quant_studio.fit_candidate_model",
        "quant_monitoring.run_monitoring_review",
    ]
    assert manifest["supported_execution_capabilities"] == [
        "quant_studio.prepare_model_config_draft",
        "quant_documentation.inspect_package",
        "quant_documentation.create_draft_workspace",
        "quant_data.register_source_reference",
        "quant_data.create_eda_plan",
        "quant_data.run_eda_review",
        "quant_data.export_eda_handoff",
        "quant_studio.fit_candidate_model",
        "quant_studio.compare_candidate_runs",
        "quant_studio.create_documentation_package",
        "quant_documentation.draft_section",
        "quant_documentation.find_unsupported_claims",
        "quant_documentation.export_markdown_review_package",
        "quant_monitoring.inspect_bundle",
        "quant_monitoring.prepare_profile_draft",
        "quant_monitoring.run_monitoring_review",
        "quant_monitoring.create_feedback_signal",
    ]
    assert manifest["capability_discovery"]["supported_execution_capabilities"] == [
        "quant_studio.prepare_model_config_draft",
        "quant_documentation.inspect_package",
        "quant_documentation.create_draft_workspace",
        "quant_data.register_source_reference",
        "quant_data.create_eda_plan",
        "quant_data.run_eda_review",
        "quant_data.export_eda_handoff",
        "quant_studio.fit_candidate_model",
        "quant_studio.compare_candidate_runs",
        "quant_studio.create_documentation_package",
        "quant_documentation.draft_section",
        "quant_documentation.find_unsupported_claims",
        "quant_documentation.export_markdown_review_package",
        "quant_monitoring.inspect_bundle",
        "quant_monitoring.prepare_profile_draft",
        "quant_monitoring.run_monitoring_review",
        "quant_monitoring.create_feedback_signal",
    ]
    governance = manifest["governance_summary"]
    assert governance["policy_pack_id"] == "quant_agent_local_governance_policy_pack_v1"
    assert governance["environment"] == "local_development"
    assert governance["environment_policy_pack_support_level"] == "suite_fixture_environment_selection"
    assert governance["release_evidence_support_level"] == "contract_policy_redaction_checks"
    assert governance["source"].endswith(":environment_policy_pack_fixture")
    assert governance["actor_role"] == "local_developer_operator"
    assert governance["effective_actor_role"] == "local_developer_operator"
    assert governance["fallback_active"] is False
    assert "*" in governance["allowed_routes"]
    assert "*" in governance["allowed_capability_ids"]
    enforcement = manifest["external_approval_enforcement_summary"]
    assert enforcement["support_level"] == "policy_required_decision_enforced"
    assert "POST /executions" in enforcement["protected_routes"]
    assert "quant_studio.prepare_model_config_draft" in enforcement["protected_capability_ids"]
    assert enforcement["blocked"] is False


def test_governance_selects_team_staging_environment_policy_pack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QUANT_AGENT_GOVERNANCE_POLICY_PACK_PATH", raising=False)
    monkeypatch.delenv("QUANT_AGENT_ACTOR_ROLE", raising=False)
    monkeypatch.setenv("QUANT_AGENT_GOVERNANCE_ENVIRONMENT", "team_staging")
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    manifest = client.get("/runtime/manifest").json()

    governance = manifest["governance_summary"]
    assert governance["policy_pack_id"] == "quant_agent_team_staging_governance_policy_pack_v1"
    assert governance["environment"] == "team_staging"
    assert governance["actor_role"] == "approver"
    assert governance["effective_actor_role"] == "approver"
    assert governance["fallback_active"] is False
    assert governance["source"].endswith(":environment_policy_pack_fixture")
    assert "POST /plans" in governance["allowed_routes"]
    assert "POST /executions" in governance["denied_routes"]
    assert "quant_studio.prepare_model_config_draft" in governance["allowed_capability_ids"]
    enforcement = manifest["external_approval_enforcement_summary"]
    assert "draft_action_external_approval_gate" in enforcement["blocking_rule_ids"]
    assert "POST /retries" in enforcement["protected_routes"]


def test_governance_regulated_review_defaults_to_viewer_denials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QUANT_AGENT_GOVERNANCE_POLICY_PACK_PATH", raising=False)
    monkeypatch.delenv("QUANT_AGENT_ACTOR_ROLE", raising=False)
    monkeypatch.setenv("QUANT_AGENT_GOVERNANCE_ENVIRONMENT", "regulated_review")
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    manifest = client.get("/runtime/manifest").json()
    denied_response = client.post(
        "/plans",
        json={"user_goal": "Plan in regulated review.", "context_summary": _safe_lifecycle_context()},
    )

    governance = manifest["governance_summary"]
    assert governance["policy_pack_id"] == "quant_agent_regulated_review_governance_policy_pack_v1"
    assert governance["actor_role"] == "viewer"
    assert "GET /runs/{run_id}/support-bundle" in governance["allowed_routes"]
    assert "POST /plans" in governance["denied_routes"]
    assert denied_response.status_code == 422
    assert denied_response.json()["detail"]["errors"][0]["code"] == "governance_permission_denied"


def test_governance_explicit_policy_path_overrides_environment_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_path = tmp_path / "governance_policy_pack.json"
    _write_governance_policy_pack(policy_path, role_id="path_operator")
    monkeypatch.setenv("QUANT_AGENT_GOVERNANCE_POLICY_PACK_PATH", str(policy_path))
    monkeypatch.setenv("QUANT_AGENT_GOVERNANCE_ENVIRONMENT", "regulated_review")
    monkeypatch.delenv("QUANT_AGENT_ACTOR_ROLE", raising=False)
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    manifest = client.get("/runtime/manifest").json()

    governance = manifest["governance_summary"]
    assert governance["policy_pack_id"] == "test_governance_policy_pack"
    assert governance["environment"] == "regulated_review"
    assert governance["actor_role"] == "path_operator"
    assert governance["source"] == "QUANT_AGENT_GOVERNANCE_POLICY_PACK_PATH"


def test_governance_unknown_environment_uses_canonical_example_with_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QUANT_AGENT_GOVERNANCE_POLICY_PACK_PATH", raising=False)
    monkeypatch.delenv("QUANT_AGENT_ACTOR_ROLE", raising=False)
    monkeypatch.setenv("QUANT_AGENT_GOVERNANCE_ENVIRONMENT", "unknown_review_env")
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    manifest = client.get("/runtime/manifest").json()

    governance = manifest["governance_summary"]
    assert governance["policy_pack_id"] == "quant_agent_local_governance_policy_pack_v1"
    assert governance["environment"] == "unknown_review_env"
    assert governance["fallback_active"] is False
    assert governance["source"] in {"configured_path", "sibling_quant_suite", "QUANT_SUITE_ROOT"}
    assert any(
        item.get("code") == "environment_policy_pack_not_found"
        for item in governance["diagnostics"]
    )


def test_governance_viewer_can_read_but_mutating_routes_are_denied_and_ledgeder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = InMemoryLedger()
    monkeypatch.delenv("QUANT_AGENT_ACTOR_ROLE", raising=False)
    operator_client = TestClient(
        create_app(runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger))
    )
    plan_payload = _create_plan(operator_client)

    monkeypatch.setenv("QUANT_AGENT_ACTOR_ROLE", "viewer")
    viewer_client = TestClient(
        create_app(runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger))
    )
    status_response = viewer_client.get(f"/runs/{plan_payload['run_id']}")
    assert status_response.status_code == 200
    assert status_response.json()["governance_summary"]["actor_role"] == "viewer"

    denied_response = viewer_client.post(
        "/pauses",
        json={
            "run_id": plan_payload["run_id"],
            "pause_intent": "pause_run",
            "reason": "user_paused",
        },
    )

    assert denied_response.status_code == 422
    assert denied_response.json()["detail"]["errors"][0]["code"] == "governance_permission_denied"
    ledger_payload = viewer_client.get(f"/runs/{plan_payload['run_id']}/ledger").json()
    denial = ledger_payload["recovery_events"][-1]
    assert denial["event_type"] == "governance_permission_denied"
    assert denial["actor_role"] == "viewer"
    assert denial["denied_route"] == "POST /pauses"
    assert denial["execution_permitted"] is False


def test_governance_approver_cannot_execute_draft_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = InMemoryLedger()
    monkeypatch.delenv("QUANT_AGENT_ACTOR_ROLE", raising=False)
    operator_client = TestClient(
        create_app(runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger))
    )
    plan_payload, studio_step, _preview = _create_confirmed_studio_preview(operator_client)

    monkeypatch.setenv("QUANT_AGENT_ACTOR_ROLE", "approver")
    approver_client = TestClient(
        create_app(runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger))
    )
    denied_response = approver_client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )

    assert denied_response.status_code == 422
    assert denied_response.json()["detail"]["errors"][0]["code"] == "governance_permission_denied"
    ledger_payload = approver_client.get(f"/runs/{plan_payload['run_id']}/ledger").json()
    denial = ledger_payload["recovery_events"][-1]
    assert denial["actor_role"] == "approver"
    assert denial["denied_route"] == "POST /executions"
    assert denial["capability_id"] == "quant_studio.prepare_model_config_draft"


def test_governance_executor_can_execute_after_prior_gates_but_cannot_approve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = InMemoryLedger()
    app_client = FakePreflightAppClient()
    monkeypatch.delenv("QUANT_AGENT_ACTOR_ROLE", raising=False)
    monkeypatch.setenv("QUANT_AGENT_ACTOR_ID", "gate_actor")
    operator_client = TestClient(create_app(runtime_with_preflight_client(app_client, ledger=ledger)))
    plan_payload, studio_step, _preview = _create_confirmed_studio_preview(operator_client)

    monkeypatch.setenv("QUANT_AGENT_ACTOR_ROLE", "executor")
    monkeypatch.setenv("QUANT_AGENT_ACTOR_ID", "execution_actor")
    executor_runtime = runtime_with_preflight_client(app_client, ledger=ledger)
    executor_client = TestClient(create_app(executor_runtime))
    denied_approval = executor_client.post(
        "/user-plan-approvals",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "approve_user_plan",
            "plan_review_id": "review_not_used",
        },
    )
    execution_response = executor_client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )

    assert denied_approval.status_code == 422
    assert denied_approval.json()["detail"]["errors"][0]["code"] == "governance_permission_denied"
    assert execution_response.status_code == 200
    assert execution_response.json()["action_result"]["execution_status"] == "succeeded"


def test_governance_capability_denial_uses_recorded_plan_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_path = tmp_path / "governance_policy_pack.json"
    _write_governance_policy_pack(
        policy_path,
        denied_capability_ids=["quant_data.run_source_preflight"],
    )
    monkeypatch.setenv("QUANT_AGENT_GOVERNANCE_POLICY_PACK_PATH", str(policy_path))
    monkeypatch.delenv("QUANT_AGENT_ACTOR_ROLE", raising=False)
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))
    plan_payload = _create_plan_with_lifecycle_reference(client)
    source_step = _step_for_capability(plan_payload, "quant_data.run_source_preflight")

    denied_response = client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": source_step["step_id"]},
    )

    assert denied_response.status_code == 422
    issue = denied_response.json()["detail"]["errors"][0]
    assert issue["code"] == "governance_permission_denied"
    assert issue["capability_id"] == "quant_data.run_source_preflight"
    ledger_payload = client.get(f"/runs/{plan_payload['run_id']}/ledger").json()
    denial = ledger_payload["recovery_events"][-1]
    assert denial["event_type"] == "governance_permission_denied"
    assert denial["capability_id"] == "quant_data.run_source_preflight"


def test_governance_unknown_role_falls_back_to_local_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QUANT_AGENT_ACTOR_ROLE", "unknown_role")
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    manifest = client.get("/runtime/manifest").json()
    plan_response = client.post(
        "/plans",
        json={
            "user_goal": "Plan with fallback governance.",
            "context_summary": _safe_lifecycle_context(),
        },
    )

    assert manifest["governance_summary"]["actor_role"] == "unknown_role"
    assert manifest["governance_summary"]["effective_actor_role"] == "local_developer_operator"
    assert manifest["governance_summary"]["fallback_active"] is True
    assert plan_response.status_code == 200


def test_separation_of_duties_denies_same_actor_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_path = tmp_path / "governance_policy_pack.json"
    _write_governance_policy_pack(
        policy_path,
        role_id="workflow_operator",
        separation_of_duties_rules=[_blocking_sod_rule()],
    )
    monkeypatch.setenv("QUANT_AGENT_GOVERNANCE_POLICY_PACK_PATH", str(policy_path))
    monkeypatch.setenv("QUANT_AGENT_ACTOR_ROLE", "workflow_operator")
    monkeypatch.setenv("QUANT_AGENT_ACTOR_ID", "same_actor")
    ledger = InMemoryLedger()
    app_client = FakePreflightAppClient()
    client = TestClient(create_app(runtime_with_preflight_client(app_client, ledger=ledger)))
    plan_payload, studio_step = _prepare_user_owned_studio_execution(client)

    denied_response = client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )

    assert denied_response.status_code == 422
    issue = denied_response.json()["detail"]["errors"][0]
    assert issue["code"] == "governance_separation_of_duties_denied"
    assert issue["capability_id"] == "quant_studio.prepare_model_config_draft"
    assert app_client.execution_calls == []
    ledger_payload = client.get(f"/runs/{plan_payload['run_id']}/ledger").json()
    denial = ledger_payload["recovery_events"][-1]
    assert denial["event_type"] == "governance_separation_of_duties_denied"
    assert denial["governance_actor"]["actor_id"] == "same_actor"
    assert denial["reason"] == "same_actor_performed_prior_governance_gate"
    status_payload = client.get(f"/runs/{plan_payload['run_id']}").json()
    sod_summary = status_payload["separation_of_duties_summary"]
    assert sod_summary["blocked"] is True
    assert "POST /executions" in sod_summary["blocked_routes"]


def test_separation_of_duties_allows_distinct_non_exempt_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_path = tmp_path / "governance_policy_pack.json"
    _write_governance_policy_pack(
        policy_path,
        role_id="workflow_operator",
        separation_of_duties_rules=[_blocking_sod_rule()],
    )
    monkeypatch.setenv("QUANT_AGENT_GOVERNANCE_POLICY_PACK_PATH", str(policy_path))
    monkeypatch.setenv("QUANT_AGENT_ACTOR_ROLE", "workflow_operator")
    monkeypatch.setenv("QUANT_AGENT_ACTOR_ID", "approval_actor")
    ledger = InMemoryLedger()
    app_client = FakePreflightAppClient()
    approval_client = TestClient(create_app(runtime_with_preflight_client(app_client, ledger=ledger)))
    plan_payload, studio_step = _prepare_user_owned_studio_execution(approval_client)

    monkeypatch.setenv("QUANT_AGENT_ACTOR_ID", "execution_actor")
    execution_client = TestClient(create_app(runtime_with_preflight_client(app_client, ledger=ledger)))
    execution_response = execution_client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )

    assert execution_response.status_code == 200
    payload = execution_response.json()
    assert payload["action_result"]["execution_status"] == "succeeded"
    assert payload["action_request"]["governance_actor"]["actor_id"] == "execution_actor"
    assert payload["action_result"]["governance_actor"]["actor_id"] == "execution_actor"


def test_separation_of_duties_denies_non_exempt_execution_when_prior_actor_is_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_path = tmp_path / "governance_policy_pack.json"
    _write_governance_policy_pack(
        policy_path,
        role_id="workflow_operator",
        separation_of_duties_rules=[_blocking_sod_rule()],
    )
    monkeypatch.setenv("QUANT_AGENT_GOVERNANCE_POLICY_PACK_PATH", str(policy_path))
    monkeypatch.setenv("QUANT_AGENT_ACTOR_ROLE", "workflow_operator")
    monkeypatch.setenv("QUANT_AGENT_ACTOR_ID", "approval_actor")
    ledger = InMemoryLedger()
    app_client = FakePreflightAppClient()
    approval_client = TestClient(create_app(runtime_with_preflight_client(app_client, ledger=ledger)))
    plan_payload, studio_step = _prepare_user_owned_studio_execution(approval_client)

    entry = ledger.get(plan_payload["run_id"])
    assert entry is not None
    recovery_events = []
    for event in entry.recovery_events:
        event_copy = dict(event)
        event_copy.pop("governance_actor", None)
        recovery_events.append(event_copy)
    confirmation_records = []
    for record in entry.confirmation_records:
        record_copy = dict(record)
        record_copy.pop("governance_actor", None)
        confirmation_records.append(record_copy)
    stripped_entry = entry.model_copy(
        update={
            "recovery_events": recovery_events,
            "confirmation_records": confirmation_records,
        },
        deep=True,
    )
    ledger._entries = [
        stripped_entry if item.run_id == plan_payload["run_id"] else item
        for item in ledger.list_entries()
    ]

    monkeypatch.setenv("QUANT_AGENT_ACTOR_ID", "execution_actor")
    execution_client = TestClient(create_app(runtime_with_preflight_client(app_client, ledger=ledger)))
    denied_response = execution_client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )

    assert denied_response.status_code == 422
    issue = denied_response.json()["detail"]["errors"][0]
    assert issue["code"] == "governance_separation_of_duties_denied"
    assert app_client.execution_calls == []
    ledger_payload = execution_client.get(f"/runs/{plan_payload['run_id']}/ledger").json()
    denial = ledger_payload["recovery_events"][-1]
    assert denial["event_type"] == "governance_separation_of_duties_denied"
    assert denial["reason"] == "prior_gate_actor_unknown"


def test_user_owned_readiness_and_consent_gate_gated_actions() -> None:
    app_client = FakePreflightAppClient()
    runtime = runtime_with_preflight_client(app_client)
    client = TestClient(create_app(runtime))
    plan_payload = _create_user_owned_plan(client)
    source_step = _step_for_capability(plan_payload, "quant_data.run_source_preflight")

    blocked_preflight = client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": source_step["step_id"]},
    )
    assert blocked_preflight.status_code == 422
    assert blocked_preflight.json()["detail"]["errors"][0]["code"] == (
        "user_workflow_readiness_required"
    )
    assert app_client.calls == []
    initial_status = client.get(f"/runs/{plan_payload['run_id']}").json()
    assert initial_status["ownership_summary"]["ownership"] == "user_owned"
    assert initial_status["plan_review_summary"]["status"] == "not_reviewed"
    assert initial_status["plan_approval_summary"]["status"] == "not_approved"
    assert initial_status["readiness_summary"]["status"] == "not_checked"
    assert initial_status["consent_summary"]["status"] == "not_recorded"
    assert initial_status["allowed_user_owned_actions"] == ["check_user_owned_readiness"]

    readiness = _check_user_owned_readiness(client, plan_payload)
    assert readiness["ownership_summary"]["ownership"] == "user_owned"
    assert readiness["readiness_summary"]["status"] == "ready"
    assert readiness["readiness_summary"]["consent_required"] is True
    assert "quant_data.run_source_preflight" in readiness["readiness_summary"]["allowed_preflight_capabilities"]
    assert "quant_studio.prepare_model_config_draft" in readiness["readiness_summary"]["allowed_execution_capabilities"]
    status_after_readiness = client.get(f"/runs/{plan_payload['run_id']}").json()
    assert status_after_readiness["readiness_summary"]["status"] == "ready"
    assert status_after_readiness["consent_summary"]["status"] == "not_recorded"
    assert status_after_readiness["allowed_user_owned_actions"] == [
        "review_plan_assumptions"
    ]

    blocked_after_readiness = client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": source_step["step_id"]},
    )
    assert blocked_after_readiness.status_code == 422
    assert blocked_after_readiness.json()["detail"]["errors"][0]["code"] == (
        "user_plan_approval_required"
    )
    blocked_consent = client.post(
        "/user-workflow-consents",
        json={
            "run_id": plan_payload["run_id"],
            "consent_intent": "approve_user_owned_guided_execution",
            "consent_scope": "single_run_review_draft_actions",
        },
    )
    assert blocked_consent.status_code == 422
    assert blocked_consent.json()["detail"]["errors"][0]["code"] == (
        "user_plan_approval_required"
    )

    review = _review_user_plan(client, plan_payload)
    assert review["plan_review_summary"]["status"] == "reviewed"
    assert review["plan_review_summary"]["accepted_assumption_count"] == len(
        plan_payload["plan"]["assumptions"]
    )
    status_after_review = client.get(f"/runs/{plan_payload['run_id']}").json()
    assert status_after_review["allowed_user_owned_actions"] == ["approve_user_plan"]
    approval = _approve_user_plan(client, plan_payload, review)
    assert approval["plan_approval_summary"]["status"] == "approved"
    status_after_approval = client.get(f"/runs/{plan_payload['run_id']}").json()
    assert status_after_approval["plan_review_summary"]["status"] == "reviewed"
    assert status_after_approval["plan_approval_summary"]["status"] == "approved"
    assert status_after_approval["allowed_user_owned_actions"] == [
        "approve_user_owned_guided_execution"
    ]

    preflight = client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": source_step["step_id"]},
    )
    assert preflight.status_code == 200

    studio_step = _step_for_capability(plan_payload, "quant_studio.prepare_model_config_draft")
    blocked_confirmation = client.post(
        "/confirmations",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": studio_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    )
    assert blocked_confirmation.status_code == 422
    assert blocked_confirmation.json()["detail"]["errors"][0]["code"] == (
        "user_workflow_consent_required"
    )

    consent = _approve_user_owned_consent(client, plan_payload)
    assert consent["consent_summary"]["status"] == "consented"
    assert consent["consent_summary"]["consent_scope"] == "single_run_review_draft_actions"
    assert consent["consent_summary"]["execution_permitted"] is False
    status_after_consent = client.get(f"/runs/{plan_payload['run_id']}").json()
    assert status_after_consent["ownership_summary"]["ownership"] == "user_owned"
    assert status_after_consent["plan_review_summary"]["status"] == "reviewed"
    assert status_after_consent["plan_approval_summary"]["status"] == "approved"
    assert status_after_consent["readiness_summary"]["status"] == "ready"
    assert status_after_consent["consent_summary"]["status"] == "consented"
    assert "run_preflight" in status_after_consent["allowed_user_owned_actions"]
    assert "execute_step" in status_after_consent["allowed_user_owned_actions"]
    orchestration_after_consent = client.get(
        f"/runs/{plan_payload['run_id']}/orchestration"
    ).json()
    assert orchestration_after_consent["ownership_summary"]["ownership"] == "user_owned"
    assert orchestration_after_consent["plan_review_summary"]["status"] == "reviewed"
    assert orchestration_after_consent["plan_approval_summary"]["status"] == "approved"
    assert orchestration_after_consent["readiness_summary"]["status"] == "ready"
    assert orchestration_after_consent["consent_summary"]["status"] == "consented"

    confirmation = client.post(
        "/confirmations",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": studio_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    )
    assert confirmation.status_code == 200
    action_request = client.post(
        "/action-requests",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )
    assert action_request.status_code == 200
    execution = client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )
    assert execution.status_code == 200
    assert len(app_client.execution_calls) == 1

    entry = runtime.planner.ledger.list_entries()[0]
    assert [event["event_type"] for event in entry.recovery_events[:4]] == [
        "user_workflow_readiness",
        "user_plan_review",
        "user_plan_approval",
        "user_workflow_consent",
    ]
    runtime.contract_loader.validate_agent_contract_payload(
        entry.model_dump(mode="json"),
        "agent_execution_ledger.v1.schema.json",
    )
    listed = client.get("/runs").json()["runs"][0]
    assert listed["run_id"] == plan_payload["run_id"]
    assert listed["ownership_summary"]["ownership"] == "user_owned"
    assert listed["plan_review_summary"]["status"] == "reviewed"
    assert listed["plan_approval_summary"]["status"] == "approved"
    assert listed["readiness_summary"]["status"] == "ready"
    assert listed["consent_summary"]["status"] == "consented"


def test_full_user_owned_guided_draft_path_and_refreshed_gate_state() -> None:
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
    plan_payload = _create_user_owned_plan(client)

    _check_user_owned_readiness(client, plan_payload)
    _approve_user_plan(client, plan_payload)
    _approve_user_owned_consent(client, plan_payload)
    _run_source_preflight(client, plan_payload)
    _complete_studio_step(client, plan_payload)
    _complete_documentation_step(client, plan_payload)
    monitoring_preflight = _run_monitoring_preflight(client, plan_payload)

    assert monitoring_preflight["capability_id"] == "quant_monitoring.validate_bundle"
    assert [call["capability_id"] for call in app_client.calls] == [
        "quant_data.run_source_preflight",
        "quant_monitoring.validate_bundle",
    ]
    assert [call["capability_id"] for call in app_client.execution_calls] == [
        "quant_studio.prepare_model_config_draft",
        "quant_documentation.inspect_package",
        "quant_documentation.create_draft_workspace",
    ]

    status = client.get(f"/runs/{plan_payload['run_id']}").json()
    orchestration = client.get(f"/runs/{plan_payload['run_id']}/orchestration").json()
    assert status["run_state"] in {"completed", "completed_with_warnings"}
    assert status["ownership_summary"]["ownership"] == "user_owned"
    assert status["plan_review_summary"]["status"] == "reviewed"
    assert status["plan_approval_summary"]["status"] == "approved"
    assert status["readiness_summary"]["status"] == "ready"
    assert status["consent_summary"]["status"] == "consented"
    assert orchestration["ownership_summary"]["ownership"] == "user_owned"
    assert orchestration["plan_review_summary"]["status"] == "reviewed"
    assert orchestration["plan_approval_summary"]["status"] == "approved"
    assert orchestration["readiness_summary"]["status"] == "ready"
    assert orchestration["consent_summary"]["status"] == "consented"
    assert orchestration["ledger_summary"]["preflight_count"] == 2
    assert orchestration["ledger_summary"]["action_result_count"] == 3

    entry = runtime.planner.ledger.get(plan_payload["run_id"])
    assert entry is not None
    assert [event["event_type"] for event in entry.recovery_events[:4]] == [
        "user_workflow_readiness",
        "user_plan_review",
        "user_plan_approval",
        "user_workflow_consent",
    ]
    runtime.contract_loader.validate_agent_contract_payload(
        entry.model_dump(mode="json"),
        "agent_execution_ledger.v1.schema.json",
    )


def test_user_owned_gate_summaries_survive_durable_ledger_reload(tmp_path: Path) -> None:
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    first_ledger = FileBackedLedger(
        tmp_path,
        validate_contract=loader.validate_agent_contract_payload,
    )
    first_runtime = runtime_with_preflight_client(
        FakePreflightAppClient(),
        ledger=first_ledger,
    )
    first_client = TestClient(create_app(first_runtime))
    plan_payload = _create_user_owned_plan(first_client)
    _check_user_owned_readiness(first_client, plan_payload)
    _approve_user_plan(first_client, plan_payload)
    _approve_user_owned_consent(first_client, plan_payload)

    second_ledger = FileBackedLedger(
        tmp_path,
        validate_contract=loader.validate_agent_contract_payload,
    )
    second_runtime = runtime_with_preflight_client(
        FakePreflightAppClient(),
        ledger=second_ledger,
    )
    second_client = TestClient(create_app(second_runtime))

    status = second_client.get(f"/runs/{plan_payload['run_id']}")
    orchestration = second_client.get(f"/runs/{plan_payload['run_id']}/orchestration")
    listed = second_client.get("/runs")

    assert status.status_code == 200
    assert orchestration.status_code == 200
    assert listed.status_code == 200
    assert status.json()["ownership_summary"]["ownership"] == "user_owned"
    assert status.json()["plan_review_summary"]["status"] == "reviewed"
    assert status.json()["plan_approval_summary"]["status"] == "approved"
    assert status.json()["readiness_summary"]["status"] == "ready"
    assert status.json()["consent_summary"]["status"] == "consented"
    assert orchestration.json()["ownership_summary"]["ownership"] == "user_owned"
    assert orchestration.json()["plan_review_summary"]["status"] == "reviewed"
    assert orchestration.json()["plan_approval_summary"]["status"] == "approved"
    assert orchestration.json()["readiness_summary"]["status"] == "ready"
    assert orchestration.json()["consent_summary"]["status"] == "consented"
    assert listed.json()["runs"][0]["ownership_summary"]["ownership"] == "user_owned"
    assert listed.json()["runs"][0]["plan_review_summary"]["status"] == "reviewed"
    assert listed.json()["runs"][0]["plan_approval_summary"]["status"] == "approved"
    assert listed.json()["runs"][0]["readiness_summary"]["status"] == "ready"
    assert listed.json()["runs"][0]["consent_summary"]["status"] == "consented"


def test_paused_user_owned_runs_block_gate_changes_until_resumed() -> None:
    runtime = runtime_with_preflight_client(FakePreflightAppClient())
    client = TestClient(create_app(runtime))
    plan_payload = _create_user_owned_plan(client)
    _check_user_owned_readiness(client, plan_payload)
    _approve_user_plan(client, plan_payload)
    _approve_user_owned_consent(client, plan_payload)
    source_step = _step_for_capability(plan_payload, "quant_data.run_source_preflight")
    studio_step = _step_for_capability(plan_payload, "quant_studio.prepare_model_config_draft")

    pause = client.post(
        "/pauses",
        json={
            "run_id": plan_payload["run_id"],
            "pause_intent": "pause_run",
            "reason": "user_paused",
        },
    )
    status = client.get(f"/runs/{plan_payload['run_id']}")
    paused_readiness = client.post(
        "/user-workflow-readiness",
        json={
            "run_id": plan_payload["run_id"],
            "readiness_intent": "check_user_owned_readiness",
            "current_context_summary": _safe_user_owned_lifecycle_context(),
        },
    )
    paused_consent = client.post(
        "/user-workflow-consents",
        json={
            "run_id": plan_payload["run_id"],
            "consent_intent": "approve_user_owned_guided_execution",
            "consent_scope": "single_run_review_draft_actions",
        },
    )
    paused_preflight = client.post(
        "/preflights",
        json={"run_id": plan_payload["run_id"], "step_id": source_step["step_id"]},
    )
    paused_confirmation = client.post(
        "/confirmations",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": studio_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    )
    paused_action_request = client.post(
        "/action-requests",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )
    paused_execution = client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )
    paused_retry = client.post(
        "/retries",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": studio_step["step_id"],
            "retry_intent": "retry_failed_step",
        },
    )
    resume = client.post(
        "/resumptions",
        json={"run_id": plan_payload["run_id"], "resume_intent": "resume_run"},
    )
    refreshed_readiness = _check_user_owned_readiness(client, plan_payload)
    resumed_consent = _approve_user_owned_consent(client, plan_payload)
    resumed_preflight = _run_source_preflight(client, plan_payload)

    revision_plan = _create_user_owned_plan(client)
    revision_review = _review_user_plan(
        client,
        revision_plan,
        decision="revise",
        safe_note="Revise this assumption before user-owned approval.",
    )
    revision = client.post(
        "/plan-revisions",
        json={
            "run_id": revision_plan["run_id"],
            "revision_intent": "revise_plan",
            "reason": "user_requested",
            "current_context_summary": _safe_user_owned_lifecycle_context(),
        },
    )
    assert revision_review["plan_review_summary"]["status"] == "revision_requested"
    assert revision.status_code == 200
    assert client.post(
        "/pauses",
        json={
            "run_id": revision_plan["run_id"],
            "pause_intent": "pause_run",
            "reason": "user_paused",
        },
    ).status_code == 200
    paused_activation = client.post(
        "/plan-revision-activations",
        json={
            "run_id": revision_plan["run_id"],
            "revision_id": revision.json()["revision_id"],
            "activation_intent": "activate_plan_revision",
        },
    )

    assert pause.status_code == 200
    assert pause.json()["run_state"] == "paused"
    assert status.status_code == 200
    assert status.json()["run_state"] == "paused"
    assert status.json()["allowed_user_owned_actions"] == []
    assert paused_readiness.status_code == 200
    assert paused_readiness.json()["readiness_summary"]["status"] == "blocked"
    assert paused_readiness.json()["validation"]["status"] == "rejected"
    assert any(
        "Paused runs" in blocker
        for blocker in paused_readiness.json()["readiness_summary"]["blockers"]
    )
    assert paused_consent.status_code == 422
    assert paused_consent.json()["detail"]["errors"][0]["code"] == (
        "paused_run_user_workflow_consent"
    )
    assert paused_preflight.status_code == 422
    assert paused_preflight.json()["detail"]["errors"][0]["code"] == "paused_run_preflight"
    assert paused_confirmation.status_code == 422
    assert paused_confirmation.json()["detail"]["errors"][0]["code"] == "paused_run_confirmation"
    assert paused_action_request.status_code == 422
    assert paused_action_request.json()["detail"]["errors"][0]["code"] == (
        "paused_run_action_request"
    )
    assert paused_execution.status_code == 422
    assert paused_execution.json()["detail"]["errors"][0]["code"] == "paused_run_execution"
    assert paused_retry.status_code == 422
    assert paused_retry.json()["detail"]["errors"][0]["code"] == "paused_run_retry"
    assert paused_activation.status_code == 422
    assert paused_activation.json()["detail"]["errors"][0]["code"] == (
        "paused_run_revision_activation"
    )
    assert resume.status_code == 200
    assert refreshed_readiness["readiness_summary"]["status"] == "ready"
    assert resumed_consent["consent_summary"]["status"] == "consented"
    assert resumed_preflight["capability_id"] == "quant_data.run_source_preflight"


def test_user_owned_recovery_gate_state_and_exports_survive_durable_reload(tmp_path: Path) -> None:
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    first_ledger = FileBackedLedger(
        tmp_path,
        validate_contract=loader.validate_agent_contract_payload,
    )
    first_runtime = runtime_with_preflight_client(
        FakePreflightAppClient(),
        ledger=first_ledger,
    )
    first_client = TestClient(create_app(first_runtime))
    plan_payload = _create_user_owned_plan(first_client)
    _check_user_owned_readiness(first_client, plan_payload)
    _approve_user_plan(first_client, plan_payload)
    _approve_user_owned_consent(first_client, plan_payload)
    _run_source_preflight(first_client, plan_payload)
    assert first_client.post(
        "/pauses",
        json={
            "run_id": plan_payload["run_id"],
            "pause_intent": "pause_run",
            "reason": "user_paused",
        },
    ).status_code == 200
    assert first_client.post(
        "/resumptions",
        json={"run_id": plan_payload["run_id"], "resume_intent": "resume_run"},
    ).status_code == 200
    revalidation = first_client.post(
        "/run-revalidations",
        json={
            "run_id": plan_payload["run_id"],
            "revalidation_intent": "check_current_context",
            "current_context_summary": _safe_user_owned_lifecycle_context(),
        },
    )
    assert revalidation.status_code == 200

    second_ledger = FileBackedLedger(
        tmp_path,
        validate_contract=loader.validate_agent_contract_payload,
    )
    second_runtime = runtime_with_preflight_client(
        FakePreflightAppClient(),
        ledger=second_ledger,
    )
    second_client = TestClient(create_app(second_runtime))
    run_id = plan_payload["run_id"]
    status = second_client.get(f"/runs/{run_id}")
    orchestration = second_client.get(f"/runs/{run_id}/orchestration")
    listed = second_client.get("/runs")
    ledger_response = second_client.get(f"/runs/{run_id}/ledger")

    assert status.status_code == 200
    assert status.json()["ownership_summary"]["ownership"] == "user_owned"
    assert status.json()["plan_review_summary"]["status"] == "reviewed"
    assert status.json()["plan_approval_summary"]["status"] == "approved"
    assert status.json()["readiness_summary"]["status"] == "ready"
    assert status.json()["consent_summary"]["status"] == "consented"
    assert status.json()["run_state"] == "waiting_for_confirmation"
    assert status.json()["latest_recovery"]["event_type"] == "run_revalidation"
    assert orchestration.status_code == 200
    assert orchestration.json()["ownership_summary"]["ownership"] == "user_owned"
    assert orchestration.json()["current_step_id"] == _step_for_capability(
        plan_payload,
        "quant_studio.prepare_model_config_draft",
    )["step_id"]
    assert "confirm_step" in orchestration.json()["allowed_next_actions"]
    assert listed.status_code == 200
    listed_summary = listed.json()["runs"][0]
    assert listed_summary["ownership_summary"]["ownership"] == "user_owned"
    assert listed_summary["latest_recovery"]["event_type"] == "run_revalidation"
    assert ledger_response.status_code == 200
    ledger_payload = ledger_response.json()
    event_types = [event["event_type"] for event in ledger_payload["recovery_events"]]
    assert "pause" in event_types
    assert "resume" in event_types
    assert "run_revalidation" in event_types
    loader.validate_agent_contract_payload(
        ledger_payload,
        "agent_execution_ledger.v1.schema.json",
    )
    _assert_safe_ledger_export_text(ledger_response.text)


def test_user_owned_parent_child_recovery_exports_are_safe_after_cancel(tmp_path: Path) -> None:
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path,
        validate_contract=loader.validate_agent_contract_payload,
    )
    runtime = runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)
    client = TestClient(create_app(runtime))
    parent_plan = _create_user_owned_plan(client)
    parent_review = _review_user_plan(
        client,
        parent_plan,
        decision="revise",
        safe_note="Revise this assumption before user-owned gate approval.",
    )
    revision = client.post(
        "/plan-revisions",
        json={
            "run_id": parent_plan["run_id"],
            "revision_intent": "revise_plan",
            "reason": "user_requested",
            "current_context_summary": _safe_user_owned_lifecycle_context(),
        },
    )
    assert parent_review["plan_review_summary"]["status"] == "revision_requested"
    assert revision.status_code == 200
    activation = client.post(
        "/plan-revision-activations",
        json={
            "run_id": parent_plan["run_id"],
            "revision_id": revision.json()["revision_id"],
            "activation_intent": "activate_plan_revision",
        },
    )
    assert activation.status_code == 200
    child_run_id = activation.json()["child_run_id"]
    child_status = client.get(f"/runs/{child_run_id}").json()
    child_plan = {"run_id": child_run_id, "plan": child_status["plan"]}
    _check_user_owned_readiness(client, child_plan)
    _approve_user_plan(client, child_plan)
    _approve_user_owned_consent(client, child_plan)
    _run_source_preflight(client, child_plan)
    assert client.post(
        "/pauses",
        json={
            "run_id": child_run_id,
            "pause_intent": "pause_run",
            "reason": "user_paused",
        },
    ).status_code == 200
    assert client.post(
        "/resumptions",
        json={"run_id": child_run_id, "resume_intent": "resume_run"},
    ).status_code == 200
    cancelled = client.post(
        "/cancellations",
        json={
            "run_id": child_run_id,
            "cancellation_intent": "cancel_run",
            "reason": "user_cancelled",
        },
    )
    assert cancelled.status_code == 200

    reloaded_ledger = FileBackedLedger(
        tmp_path,
        validate_contract=loader.validate_agent_contract_payload,
    )
    reloaded_client = TestClient(
        create_app(runtime_with_preflight_client(FakePreflightAppClient(), ledger=reloaded_ledger))
    )
    parent_status = reloaded_client.get(f"/runs/{parent_plan['run_id']}")
    child_status = reloaded_client.get(f"/runs/{child_run_id}")
    child_orchestration = reloaded_client.get(f"/runs/{child_run_id}/orchestration")
    parent_ledger = reloaded_client.get(f"/runs/{parent_plan['run_id']}/ledger")
    child_ledger = reloaded_client.get(f"/runs/{child_run_id}/ledger")
    blocked_child_preflight = reloaded_client.post(
        "/preflights",
        json={
            "run_id": child_run_id,
            "step_id": _step_for_capability(child_status.json(), "quant_data.run_source_preflight")[
                "step_id"
            ],
        },
    )

    assert parent_status.status_code == 200
    assert parent_status.json()["plan_review_summary"]["status"] == "revision_requested"
    assert parent_status.json()["plan_approval_summary"]["status"] == "not_approved"
    assert parent_status.json()["child_run_ids"] == [child_run_id]
    assert child_status.status_code == 200
    assert child_status.json()["parent_run_id"] == parent_plan["run_id"]
    assert child_status.json()["run_state"] == "cancelled"
    assert child_status.json()["plan_review_summary"]["status"] == "reviewed"
    assert child_status.json()["plan_approval_summary"]["status"] == "approved"
    assert child_status.json()["readiness_summary"]["status"] == "ready"
    assert child_status.json()["consent_summary"]["status"] == "consented"
    assert child_orchestration.status_code == 200
    assert child_orchestration.json()["run_state"] == "cancelled"
    assert child_orchestration.json()["allowed_next_actions"] == []
    assert blocked_child_preflight.status_code == 422
    assert blocked_child_preflight.json()["detail"]["errors"][0]["code"] == (
        "cancelled_run_preflight"
    )
    for response in [parent_ledger, child_ledger]:
        assert response.status_code == 200
        loader.validate_agent_contract_payload(
            response.json(),
            "agent_execution_ledger.v1.schema.json",
        )
        _assert_safe_ledger_export_text(response.text)


def test_user_owned_phase8_certification_path_matches_expected_fixture(tmp_path: Path) -> None:
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path,
        validate_contract=loader.validate_agent_contract_payload,
    )
    app_client = FakePreflightAppClient(
        responses_by_capability={
            "quant_monitoring.validate_bundle": _valid_preflight_response(
                capability_id="quant_monitoring.validate_bundle",
                app_id="quant_monitoring",
            )
        }
    )
    runtime = runtime_with_preflight_client(app_client, ledger=ledger)
    client = TestClient(create_app(runtime))
    expected = _expected_user_owned_phase8_certification_fixture()

    certified_plan = _create_user_owned_plan_with_policy(
        client,
        policy={"provider_mode": "disabled_or_local_fallback"},
    )
    assert certified_plan["provider_metadata"]["provider_mode"] in expected["allowed_provider_modes"]
    assert certified_plan["provider_metadata"]["supports_execution"] is False
    assert [
        step["capability_id"]
        for step in certified_plan["plan"]["proposed_steps"]
    ] == expected["expected_capability_order"]

    review = _review_user_plan(client, certified_plan)
    approval = _approve_user_plan(client, certified_plan, review)
    readiness = _check_user_owned_readiness(client, certified_plan)
    consent = _approve_user_owned_consent(client, certified_plan)
    assert review["plan_review_summary"]["status"] == "reviewed"
    assert approval["plan_approval_summary"]["status"] == "approved"
    assert readiness["readiness_summary"]["allowed_preflight_capabilities"] == (
        expected["expected_guided_preflight_capabilities"]
    )
    assert readiness["readiness_summary"]["allowed_execution_capabilities"] == (
        expected["expected_guided_execution_capabilities"]
    )
    assert consent["consent_summary"]["status"] == "consented"

    _run_source_preflight(client, certified_plan)
    assert client.post(
        "/pauses",
        json={
            "run_id": certified_plan["run_id"],
            "pause_intent": "pause_run",
            "reason": "user_paused",
        },
    ).status_code == 200
    assert client.post(
        "/resumptions",
        json={"run_id": certified_plan["run_id"], "resume_intent": "resume_run"},
    ).status_code == 200
    stale_revalidation = client.post(
        "/run-revalidations",
        json={
            "run_id": certified_plan["run_id"],
            "revalidation_intent": "check_current_context",
            "current_context_summary": _safe_user_owned_lifecycle_context()
            | {
                "lifecycle_summary": {
                    **_safe_user_owned_lifecycle_context()["lifecycle_summary"],
                    "state": "ready_for_documentation",
                    "summary": "Lifecycle state changed after the user-owned plan was approved.",
                }
            },
        },
    )
    assert stale_revalidation.status_code == 200
    assert stale_revalidation.json()["stale_assumption_summary"]["status"] == "stale"
    assert stale_revalidation.json()["ledger_recorded"] is True

    _complete_studio_step(client, certified_plan)
    _complete_documentation_step(client, certified_plan)
    _run_monitoring_preflight(client, certified_plan)

    certified_status = client.get(f"/runs/{certified_plan['run_id']}")
    certified_orchestration = client.get(f"/runs/{certified_plan['run_id']}/orchestration")
    certified_ledger_response = client.get(f"/runs/{certified_plan['run_id']}/ledger")
    assert certified_status.status_code == 200
    assert certified_status.json()["run_state"] in expected["allowed_completed_run_states"]
    assert certified_status.json()["ownership_summary"]["ownership"] == expected["workflow_ownership"]
    assert certified_status.json()["plan_review_summary"]["status"] == "reviewed"
    assert certified_status.json()["plan_approval_summary"]["status"] == "approved"
    assert certified_status.json()["readiness_summary"]["status"] == "ready"
    assert certified_status.json()["consent_summary"]["status"] == "consented"
    assert certified_orchestration.status_code == 200
    assert certified_orchestration.json()["run_progress_summary"]["completed_steps"] >= 4
    assert certified_ledger_response.status_code == 200
    certified_ledger = certified_ledger_response.json()
    assert [record["capability_id"] for record in certified_ledger["preflight_records"]] == (
        expected["expected_guided_preflight_capabilities"]
    )
    assert [record["capability_id"] for record in certified_ledger["action_results"]] == (
        expected["expected_action_result_capabilities"]
    )
    assert [event["event_type"] for event in certified_ledger["recovery_events"][:4]] == (
        expected["expected_user_gate_event_order"]
    )
    event_types = [event["event_type"] for event in certified_ledger["recovery_events"]]
    for event_type in expected["required_recovery_event_types"]:
        assert event_type in event_types
    for collection_name, minimum_count in expected["minimum_completed_ledger_record_counts"].items():
        assert len(certified_ledger.get(collection_name, [])) >= minimum_count
    serialized_certified_ledger = json.dumps(certified_ledger, sort_keys=True)
    for term in expected["forbidden_terms"]:
        assert term not in serialized_certified_ledger
    loader.validate_agent_contract_payload(
        certified_ledger,
        "agent_execution_ledger.v1.schema.json",
    )
    _assert_safe_ledger_export_text(certified_ledger_response.text)

    reloaded_ledger = FileBackedLedger(
        tmp_path,
        validate_contract=loader.validate_agent_contract_payload,
    )
    reloaded_runtime = runtime_with_preflight_client(app_client, ledger=reloaded_ledger)
    reloaded_client = TestClient(create_app(reloaded_runtime))
    reloaded_status = reloaded_client.get(f"/runs/{certified_plan['run_id']}")
    reloaded_orchestration = reloaded_client.get(
        f"/runs/{certified_plan['run_id']}/orchestration"
    )
    reloaded_history = reloaded_client.get("/runs")
    assert reloaded_status.status_code == 200
    assert reloaded_status.json()["ownership_summary"]["ownership"] == "user_owned"
    assert reloaded_status.json()["plan_review_summary"]["status"] == "reviewed"
    assert reloaded_status.json()["plan_approval_summary"]["status"] == "approved"
    assert reloaded_status.json()["readiness_summary"]["status"] == "ready"
    assert reloaded_status.json()["consent_summary"]["status"] == "consented"
    assert reloaded_status.json()["latest_recovery"]["event_type"] == "run_revalidation"
    assert reloaded_orchestration.status_code == 200
    assert reloaded_orchestration.json()["run_state"] in expected["allowed_completed_run_states"]
    assert reloaded_history.status_code == 200
    assert certified_plan["run_id"] in {
        run["run_id"]
        for run in reloaded_history.json()["runs"]
    }

    parent_plan = _create_user_owned_plan(reloaded_client)
    parent_review = _review_user_plan(
        reloaded_client,
        parent_plan,
        decision="revise",
        safe_note="Revise this assumption before Phase 8 certification approval.",
    )
    assert parent_review["plan_review_summary"]["status"] == "revision_requested"
    revision = reloaded_client.post(
        "/plan-revisions",
        json={
            "run_id": parent_plan["run_id"],
            "revision_intent": "revise_plan",
            "reason": "user_requested",
            "current_context_summary": _safe_user_owned_lifecycle_context(),
        },
    )
    assert revision.status_code == 200
    activation = reloaded_client.post(
        "/plan-revision-activations",
        json={
            "run_id": parent_plan["run_id"],
            "revision_id": revision.json()["revision_id"],
            "activation_intent": "activate_plan_revision",
        },
    )
    assert activation.status_code == 200
    child_run_id = activation.json()["child_run_id"]
    child_status = reloaded_client.get(f"/runs/{child_run_id}")
    assert child_status.status_code == 200
    assert child_status.json()["parent_run_id"] == parent_plan["run_id"]
    assert child_status.json()["plan_review_summary"]["status"] == (
        expected["child_run_initial_gate_state"]["plan_review"]
    )
    assert child_status.json()["plan_approval_summary"]["status"] == (
        expected["child_run_initial_gate_state"]["plan_approval"]
    )
    assert child_status.json()["readiness_summary"]["status"] == (
        expected["child_run_initial_gate_state"]["readiness"]
    )
    assert child_status.json()["consent_summary"]["status"] == (
        expected["child_run_initial_gate_state"]["consent"]
    )
    child_plan = {"run_id": child_run_id, "plan": child_status.json()["plan"]}
    blocked_child_preflight = reloaded_client.post(
        "/preflights",
        json={
            "run_id": child_run_id,
            "step_id": _step_for_capability(child_plan, "quant_data.run_source_preflight")[
                "step_id"
            ],
        },
    )
    assert blocked_child_preflight.status_code == 422
    assert blocked_child_preflight.json()["detail"]["errors"][0]["code"] == (
        "user_workflow_readiness_required"
    )
    child_review = _review_user_plan(reloaded_client, child_plan)
    _approve_user_plan(reloaded_client, child_plan, child_review)
    _check_user_owned_readiness(reloaded_client, child_plan)
    _approve_user_owned_consent(reloaded_client, child_plan)
    assert _run_source_preflight(reloaded_client, child_plan)["capability_id"] == (
        "quant_data.run_source_preflight"
    )

    cancellation_plan = _create_user_owned_plan(reloaded_client)
    cancellation_review = _review_user_plan(reloaded_client, cancellation_plan)
    _approve_user_plan(reloaded_client, cancellation_plan, cancellation_review)
    _check_user_owned_readiness(reloaded_client, cancellation_plan)
    _approve_user_owned_consent(reloaded_client, cancellation_plan)
    _run_source_preflight(reloaded_client, cancellation_plan)
    cancellation = reloaded_client.post(
        "/cancellations",
        json={
            "run_id": cancellation_plan["run_id"],
            "cancellation_intent": "cancel_run",
            "reason": "user_cancelled",
        },
    )
    assert cancellation.status_code == 200
    assert cancellation.json()["run_state"] == expected["cancelled_run_expectations"]["run_state"]
    cancelled_step = _step_for_capability(
        cancellation_plan,
        "quant_monitoring.validate_bundle",
    )
    blocked_cancelled_preflight = reloaded_client.post(
        "/preflights",
        json={"run_id": cancellation_plan["run_id"], "step_id": cancelled_step["step_id"]},
    )
    assert blocked_cancelled_preflight.status_code == 422
    assert blocked_cancelled_preflight.json()["detail"]["errors"][0]["code"] == (
        expected["cancelled_run_expectations"]["blocked_preflight_error"]
    )

    final_ledger = FileBackedLedger(
        tmp_path,
        validate_contract=loader.validate_agent_contract_payload,
    )
    final_client = TestClient(
        create_app(runtime_with_preflight_client(app_client, ledger=final_ledger))
    )
    for run_id in [
        certified_plan["run_id"],
        parent_plan["run_id"],
        child_run_id,
        cancellation_plan["run_id"],
    ]:
        ledger_response = final_client.get(f"/runs/{run_id}/ledger")
        assert ledger_response.status_code == 200
        loader.validate_agent_contract_payload(
            ledger_response.json(),
            "agent_execution_ledger.v1.schema.json",
        )
        _assert_safe_ledger_export_text(ledger_response.text)
        serialized = json.dumps(ledger_response.json(), sort_keys=True)
        for term in expected["forbidden_terms"]:
            assert term not in serialized
    final_parent_status = final_client.get(f"/runs/{parent_plan['run_id']}").json()
    final_child_status = final_client.get(f"/runs/{child_run_id}").json()
    final_cancelled_status = final_client.get(f"/runs/{cancellation_plan['run_id']}").json()
    assert final_parent_status["child_run_ids"] == [child_run_id]
    assert final_parent_status["plan_review_summary"]["status"] == "revision_requested"
    assert final_child_status["parent_run_id"] == parent_plan["run_id"]
    assert final_child_status["plan_review_summary"]["status"] == "reviewed"
    assert final_child_status["plan_approval_summary"]["status"] == "approved"
    assert final_child_status["readiness_summary"]["status"] == "ready"
    assert final_child_status["consent_summary"]["status"] == "consented"
    assert final_cancelled_status["final_status"] == (
        expected["cancelled_run_expectations"]["final_status"]
    )
    assert final_cancelled_status["run_state"] == expected["cancelled_run_expectations"]["run_state"]


def test_user_plan_review_and_approval_validate_assumptions_and_block_revisions() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))
    plan_payload = _create_user_owned_plan(client)
    run_id = plan_payload["run_id"]

    missing = client.post(
        "/user-plan-reviews",
        json={
            "run_id": run_id,
            "review_intent": "review_plan_assumptions",
            "assumption_reviews": [{"assumption_index": 0, "decision": "accept"}],
            "current_context_summary": _safe_user_owned_lifecycle_context(),
        },
    )
    assert missing.status_code == 422
    assert missing.json()["detail"]["errors"][0]["code"] == "user_plan_assumption_review_count_mismatch"

    duplicate = client.post(
        "/user-plan-reviews",
        json={
            "run_id": run_id,
            "review_intent": "review_plan_assumptions",
            "assumption_reviews": [
                {"assumption_index": 0, "decision": "accept"},
                {"assumption_index": 0, "decision": "accept"},
            ],
            "current_context_summary": _safe_user_owned_lifecycle_context(),
        },
    )
    assert duplicate.status_code == 422
    assert duplicate.json()["detail"]["errors"][0]["code"] == "duplicate_user_plan_assumption_review"

    out_of_range = client.post(
        "/user-plan-reviews",
        json={
            "run_id": run_id,
            "review_intent": "review_plan_assumptions",
            "assumption_reviews": [
                {"assumption_index": 0, "decision": "accept"},
                {"assumption_index": 99, "decision": "accept"},
            ],
            "current_context_summary": _safe_user_owned_lifecycle_context(),
        },
    )
    assert out_of_range.status_code == 422
    assert out_of_range.json()["detail"]["errors"][0]["code"] == "invalid_user_plan_assumption_review_index"

    unsafe = client.post(
        "/user-plan-reviews",
        json={
            "run_id": run_id,
            "review_intent": "review_plan_assumptions",
            "assumption_reviews": [
                {"assumption_index": 0, "decision": "accept"},
                {
                    "assumption_index": 1,
                    "decision": "revise",
                    "safe_note": "Check C:\\private\\raw.csv before approval.",
                },
            ],
            "current_context_summary": _safe_user_owned_lifecycle_context(),
        },
    )
    assert unsafe.status_code == 422
    assert unsafe.json()["detail"]["errors"][0]["code"] == "unsafe_user_plan_review_record"
    assert "raw.csv" not in unsafe.text

    extra = client.post(
        "/user-plan-reviews",
        json={
            "run_id": run_id,
            "review_intent": "review_plan_assumptions",
            "assumption_reviews": [
                {"assumption_index": 0, "decision": "accept"},
                {"assumption_index": 1, "decision": "accept", "execution_permitted": True},
            ],
            "current_context_summary": _safe_user_owned_lifecycle_context(),
        },
    )
    assert extra.status_code == 422

    revision_review = _review_user_plan(
        client,
        plan_payload,
        decision="revise",
        safe_note="Revise this assumption using only safe summary evidence.",
    )
    assert revision_review["plan_review_summary"]["status"] == "revision_requested"
    blocked_approval = client.post(
        "/user-plan-approvals",
        json={
            "run_id": run_id,
            "approval_intent": "approve_user_plan",
            "plan_review_id": revision_review["plan_review_summary"]["plan_review_id"],
        },
    )
    assert blocked_approval.status_code == 422
    assert blocked_approval.json()["detail"]["errors"][0]["code"] == "user_plan_revision_requested"

    accepted_review = _review_user_plan(client, plan_payload)
    approval = _approve_user_plan(client, plan_payload, accepted_review)
    assert approval["plan_approval_summary"]["status"] == "approved"
    duplicate_approval = _approve_user_plan(client, plan_payload, accepted_review)
    assert duplicate_approval["plan_approval_summary"]["plan_approval_id"] == (
        approval["plan_approval_summary"]["plan_approval_id"]
    )


def test_user_owned_readiness_blocks_unknown_unsafe_and_extra_payload_fields() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))
    unknown_plan = _create_plan_with_lifecycle_reference(client)

    unknown = client.post(
        "/user-workflow-readiness",
        json={
            "run_id": unknown_plan["run_id"],
            "readiness_intent": "check_user_owned_readiness",
            "current_context_summary": _safe_lifecycle_context(),
        },
    )
    assert unknown.status_code == 200
    assert unknown.json()["ownership_summary"]["ownership"] == "unknown"
    assert unknown.json()["readiness_summary"]["status"] == "blocked"
    assert unknown.json()["validation"]["status"] == "rejected"

    user_plan = _create_user_owned_plan(client)
    unsafe = client.post(
        "/user-workflow-readiness",
        json={
            "run_id": user_plan["run_id"],
            "readiness_intent": "check_user_owned_readiness",
            "current_context_summary": {
                **_safe_user_owned_lifecycle_context(),
                "raw_path": "C:\\private\\user-data.csv",
            },
        },
    )
    assert unsafe.status_code == 422
    assert unsafe.json()["detail"]["errors"][0]["code"] == "unsafe_user_workflow_context"
    assert "user-data.csv" not in unsafe.text

    extra = client.post(
        "/user-workflow-consents",
        json={
            "run_id": user_plan["run_id"],
            "consent_intent": "approve_user_owned_guided_execution",
            "consent_scope": "single_run_review_draft_actions",
            "execution_permitted": True,
        },
    )
    assert extra.status_code == 422


def test_sample_owned_runs_do_not_require_user_workflow_gate_and_cannot_use_user_consent() -> None:
    app_client = FakePreflightAppClient()
    runtime = runtime_with_preflight_client(app_client)
    client = TestClient(create_app(runtime))
    sample_plan = client.post(
        "/plans",
        json={
            "user_goal": "Run the sample path.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    source_step = _step_for_capability(sample_plan, "quant_data.run_source_preflight")

    preflight = client.post(
        "/preflights",
        json={"run_id": sample_plan["run_id"], "step_id": source_step["step_id"]},
    )
    assert preflight.status_code == 200
    assert len(app_client.calls) == 1

    readiness = client.post(
        "/user-workflow-readiness",
        json={
            "run_id": sample_plan["run_id"],
            "readiness_intent": "check_user_owned_readiness",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )
    assert readiness.status_code == 200
    assert readiness.json()["ownership_summary"]["ownership"] == "sample_owned"
    assert readiness.json()["plan_review_summary"]["status"] == "not_required"
    assert readiness.json()["plan_approval_summary"]["status"] == "not_required"
    assert readiness.json()["readiness_summary"]["status"] == "sample_owned"

    sample_review = client.post(
        "/user-plan-reviews",
        json={
            "run_id": sample_plan["run_id"],
            "review_intent": "review_plan_assumptions",
            "assumption_reviews": [],
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )
    assert sample_review.status_code == 422
    assert sample_review.json()["detail"]["errors"][0]["code"] == "sample_plan_approval_not_required"

    consent = client.post(
        "/user-workflow-consents",
        json={
            "run_id": sample_plan["run_id"],
            "consent_intent": "approve_user_owned_guided_execution",
            "consent_scope": "single_run_review_draft_actions",
        },
    )
    assert consent.status_code == 422
    assert consent.json()["detail"]["errors"][0]["code"] == "sample_workflow_consent_not_required"


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
    assert stored_payload["ledger_integrity"]["status"] == "verified"
    assert stored_payload["ledger_integrity"]["algorithm"] == "sha256"
    assert stored_payload["ledger_integrity"]["sequence_number"] == 1
    journal_files = list((tmp_path / "integrity_journals").glob("*.jsonl"))
    assert len(journal_files) == 1
    assert stored_payload["ledger_integrity"]["payload_hash"] in journal_files[0].read_text(
        encoding="utf-8"
    )
    loader.validate_agent_contract_payload(stored_payload, "agent_execution_ledger.v1.schema.json")

    ledger_response = client.get(f"/runs/{plan_payload['run_id']}/ledger")
    assert ledger_response.status_code == 200
    exported = ledger_response.json()
    assert exported["run_id"] == plan_payload["run_id"]
    assert exported["data_policy"] == "summaries_and_references_only"
    assert str(tmp_path) not in ledger_response.text
    assert "raw_path" not in ledger_response.text
    assert exported["ledger_integrity"]["status"] == "verified"
    loader.validate_agent_contract_payload(exported, "agent_execution_ledger.v1.schema.json")

    status_response = client.get(f"/runs/{plan_payload['run_id']}")
    assert status_response.status_code == 200
    assert status_response.json()["ledger_integrity_summary"]["status"] == "verified"

    support_response = client.get(f"/runs/{plan_payload['run_id']}/support-bundle")
    assert support_response.status_code == 200
    support_bundle = support_response.json()
    assert support_bundle["run_id"] == plan_payload["run_id"]
    assert support_bundle["ledger_integrity_summary"]["status"] == "verified"
    assert support_bundle["redaction_report"]["raw_payloads_included"] is False
    dumped_bundle = json.dumps(support_bundle, sort_keys=True)
    assert str(tmp_path) not in dumped_bundle
    assert '"raw_path"' not in dumped_bundle
    assert "OPENAI_API_KEY" not in dumped_bundle
    assert "provider_prompt" not in dumped_bundle
    assert '"provider_response"' not in dumped_bundle
    loader.validate_agent_contract_payload(support_bundle, "agent_support_bundle.v1.schema.json")


def test_file_backed_ledger_loads_legacy_entries_as_unverified(tmp_path: Path) -> None:
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path,
        validate_contract=loader.validate_agent_contract_payload,
    )
    runtime = runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)
    client = TestClient(create_app(runtime))
    plan_payload = _create_plan_with_lifecycle_reference(client)
    ledger_file = next(tmp_path.glob("*.json"))
    payload = json.loads(ledger_file.read_text(encoding="utf-8"))
    payload.pop("ledger_integrity", None)
    ledger_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    reloaded = FileBackedLedger(
        tmp_path,
        validate_contract=loader.validate_agent_contract_payload,
    )

    entry = reloaded.get(plan_payload["run_id"])
    assert entry is not None
    assert entry.ledger_integrity is not None
    assert entry.ledger_integrity.status == "legacy_unverified"
    diagnostics = reloaded.diagnostics()
    assert diagnostics["loaded_entry_count"] == 1
    assert diagnostics["legacy_unverified_entry_count"] == 1
    assert diagnostics["invalid_entry_count"] == 0


def test_file_backed_ledger_ignores_tampered_integrity_payload(tmp_path: Path) -> None:
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path,
        validate_contract=loader.validate_agent_contract_payload,
    )
    runtime = runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)
    client = TestClient(create_app(runtime))
    plan_payload = _create_plan_with_lifecycle_reference(client)
    ledger_file = next(tmp_path.glob("*.json"))
    payload = json.loads(ledger_file.read_text(encoding="utf-8"))
    payload["user_goal_summary"] = "Tampered but schema-compatible summary."
    ledger_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    reloaded = FileBackedLedger(
        tmp_path,
        validate_contract=loader.validate_agent_contract_payload,
    )

    assert reloaded.get(plan_payload["run_id"]) is None
    diagnostics = reloaded.diagnostics()
    assert diagnostics["loaded_entry_count"] == 0
    assert diagnostics["invalid_entry_count"] == 1
    assert diagnostics["tampered_entry_count"] == 1


def test_support_bundle_route_is_readable_for_viewer_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path,
        validate_contract=loader.validate_agent_contract_payload,
    )
    local_runtime = runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)
    local_client = TestClient(create_app(local_runtime))
    plan_payload = _create_plan_with_lifecycle_reference(local_client)

    monkeypatch.setenv("QUANT_AGENT_ACTOR_ROLE", "viewer")
    viewer_runtime = runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)
    viewer_client = TestClient(create_app(viewer_runtime))

    support_response = viewer_client.get(f"/runs/{plan_payload['run_id']}/support-bundle")
    assert support_response.status_code == 200
    assert support_response.json()["governance_summary"]["effective_actor_role"] == "viewer"
    denied_response = viewer_client.post(
        "/pauses",
        json={
            "run_id": plan_payload["run_id"],
            "pause_intent": "pause_run",
            "reason": "user_paused",
        },
    )
    assert denied_response.status_code == 422
    assert denied_response.json()["detail"]["errors"][0]["code"] == "governance_permission_denied"


def test_external_approval_request_preview_validates_and_ledgers_idempotently(
    tmp_path: Path,
) -> None:
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path,
        validate_contract=loader.validate_agent_contract_payload,
    )
    runtime = runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)
    client = TestClient(create_app(runtime))
    plan_payload = _create_plan_with_lifecycle_reference(client)
    studio_step = _step_for_capability(
        plan_payload,
        "quant_studio.prepare_model_config_draft",
    )

    run_response = client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "run",
            "step_id": None,
        },
    )
    assert run_response.status_code == 200
    run_payload = run_response.json()
    run_package = run_payload["approval_request"]
    assert run_payload["run_id"] == plan_payload["run_id"]
    assert run_package["approval_scope"] == "run"
    assert run_package["external_submission_status"] == "not_submitted"
    assert run_package["support_bundle_reference"]["reference_type"] == "agent_support_bundle"
    assert run_package["redaction_report"]["raw_payloads_included"] is False
    loader.validate_agent_contract_payload(
        run_package,
        "agent_external_approval_request.v1.schema.json",
    )

    step_response = client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "step",
            "step_id": studio_step["step_id"],
        },
    )
    assert step_response.status_code == 200
    step_package = step_response.json()["approval_request"]
    assert step_package["approval_scope"] == "step"
    assert step_package["step_id"] == studio_step["step_id"]
    assert step_package["capability_id"] == "quant_studio.prepare_model_config_draft"
    loader.validate_agent_contract_payload(
        step_package,
        "agent_external_approval_request.v1.schema.json",
    )

    duplicate_response = client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "step",
            "step_id": studio_step["step_id"],
        },
    )
    assert duplicate_response.status_code == 200
    assert duplicate_response.json()["approval_request"]["approval_request_id"] == (
        step_package["approval_request_id"]
    )

    entry = ledger.get(plan_payload["run_id"])
    assert entry is not None
    approval_events = [
        event
        for event in entry.recovery_events
        if event.get("event_type") == "external_approval_request_preview"
    ]
    assert len(approval_events) == 2
    loader.validate_agent_contract_payload(
        entry.model_dump(mode="json"),
        "agent_execution_ledger.v1.schema.json",
    )
    dumped_package = json.dumps(step_package, sort_keys=True)
    assert str(tmp_path) not in dumped_package
    assert "OPENAI_API_KEY" not in dumped_package
    assert "provider_prompt" not in dumped_package
    assert '"provider_response"' not in dumped_package


def test_external_approval_request_preview_governance_denies_viewer_and_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path,
        validate_contract=loader.validate_agent_contract_payload,
    )
    operator_runtime = runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)
    operator_client = TestClient(create_app(operator_runtime))
    plan_payload = _create_plan_with_lifecycle_reference(operator_client)

    for role in ["viewer", "executor"]:
        monkeypatch.setenv("QUANT_AGENT_ACTOR_ROLE", role)
        restricted_runtime = runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)
        restricted_client = TestClient(create_app(restricted_runtime))
        response = restricted_client.post(
            "/external-approval-requests",
            json={
                "run_id": plan_payload["run_id"],
                "approval_intent": "preview_external_approval_request",
                "approval_scope": "run",
                "step_id": None,
            },
        )
        assert response.status_code == 422
        assert response.json()["detail"]["errors"][0]["code"] == "governance_permission_denied"

    ledger_payload = operator_client.get(f"/runs/{plan_payload['run_id']}/ledger").json()
    denial_events = [
        event
        for event in ledger_payload["recovery_events"]
        if event.get("event_type") == "governance_permission_denied"
    ]
    assert {event["actor_role"] for event in denial_events} == {"viewer", "executor"}


def test_external_approval_request_preview_allows_approver_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path,
        validate_contract=loader.validate_agent_contract_payload,
    )
    operator_runtime = runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)
    operator_client = TestClient(create_app(operator_runtime))
    plan_payload = _create_plan_with_lifecycle_reference(operator_client)

    monkeypatch.setenv("QUANT_AGENT_ACTOR_ROLE", "approver")
    approver_runtime = runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)
    approver_client = TestClient(create_app(approver_runtime))
    response = approver_client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "run",
            "step_id": None,
        },
    )

    assert response.status_code == 200
    approval_request = response.json()["approval_request"]
    assert approval_request["requester"]["effective_actor_role"] == "approver"
    loader.validate_agent_contract_payload(
        approval_request,
        "agent_external_approval_request.v1.schema.json",
    )


def test_external_approval_request_preview_rejects_unknown_terminal_and_bad_requests(
    tmp_path: Path,
) -> None:
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path,
        validate_contract=loader.validate_agent_contract_payload,
    )
    runtime = runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)
    client = TestClient(create_app(runtime))
    plan_payload = _create_plan_with_lifecycle_reference(client)

    missing_run = client.post(
        "/external-approval-requests",
        json={
            "run_id": "run_missing",
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "run",
            "step_id": None,
        },
    )
    assert missing_run.status_code == 422
    assert missing_run.json()["detail"]["errors"][0]["code"] == "unknown_run"

    unknown_step = client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "step",
            "step_id": "step_missing",
        },
    )
    assert unknown_step.status_code == 422
    assert unknown_step.json()["detail"]["errors"][0]["code"] == "unknown_step"

    extra_field = client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "run",
            "step_id": None,
            "execution_permitted": True,
        },
    )
    assert extra_field.status_code == 422

    cancellation = client.post(
        "/cancellations",
        json={
            "run_id": plan_payload["run_id"],
            "cancellation_intent": "cancel_run",
            "reason": "user_cancelled",
        },
    )
    assert cancellation.status_code == 200
    terminal_response = client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "run",
            "step_id": None,
        },
    )
    assert terminal_response.status_code == 422
    assert terminal_response.json()["detail"]["errors"][0]["code"] == (
        "terminal_run_external_approval_request"
    )


def _external_approval_decision(
    approval_request: dict[str, Any],
    *,
    decision_status: str = "approved",
    decision_id: str | None = None,
) -> dict[str, Any]:
    approval_request_id = str(approval_request["approval_request_id"])
    return {
        "schema_version": "1.0",
        "data_policy": "summaries_and_references_only",
        "approval_decision_id": decision_id or f"decision_{approval_request_id}_{decision_status}",
        "approval_request_id": approval_request_id,
        "run_id": approval_request["run_id"],
        "step_id": approval_request.get("step_id"),
        "capability_id": approval_request.get("capability_id"),
        "decision_status": decision_status,
        "decided_by": {
            "actor_id": "external_approver",
            "actor_role": "approver",
        },
        "decided_at_utc": "2026-06-23T12:00:00Z",
        "decision_summary": {
            "summary": f"Manual external approval decision is {decision_status}.",
            "advisory_only": True,
        },
        "evidence_references": [
            {
                "reference_type": "external_approval_request",
                "reference_id": approval_request_id,
                "summary": "Decision references the redacted approval request package.",
            }
        ],
        "redaction_report": {
            "data_policy": "summaries_and_references_only",
            "raw_payloads_included": False,
            "unsafe_issue_count": 0,
        },
        "validation": {
            "status": "valid",
            "errors": [],
            "warnings": [],
        },
        "execution_permitted": False,
    }


def test_external_approval_decision_import_validates_ledgers_and_is_advisory(
    tmp_path: Path,
) -> None:
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path,
        validate_contract=loader.validate_agent_contract_payload,
    )
    runtime = runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)
    client = TestClient(create_app(runtime))
    plan_payload = _create_plan_with_lifecycle_reference(client)
    studio_step = _step_for_capability(plan_payload, "quant_studio.prepare_model_config_draft")

    run_preview = client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "run",
            "step_id": None,
        },
    )
    assert run_preview.status_code == 200
    step_preview = client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "step",
            "step_id": studio_step["step_id"],
        },
    )
    assert step_preview.status_code == 200

    run_request = run_preview.json()["approval_request"]
    step_request = step_preview.json()["approval_request"]
    imported_statuses: list[str] = []
    for index, decision_status in enumerate(["approved", "rejected", "needs_changes", "expired"], start=1):
        approval_request = run_request if decision_status in {"approved", "expired"} else step_request
        decision = _external_approval_decision(
            approval_request,
            decision_status=decision_status,
            decision_id=f"decision_{index}_{decision_status}",
        )
        response = client.post(
            "/external-approval-decisions",
            json={
                "run_id": plan_payload["run_id"],
                "decision_intent": "import_external_approval_decision",
                "approval_decision": decision,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        imported_statuses.append(payload["approval_decision"]["decision_status"])
        assert payload["approval_request_id"] == approval_request["approval_request_id"]
        assert payload["external_approval_summary"]["enforcement_mode"] == "advisory_only"
        assert payload["external_approval_summary"]["execution_permitted"] is False
        assert payload["run_status"]["external_approval_summary"]["decision_count"] == index
        assert payload["orchestration"]["external_approval_summary"]["decision_count"] == index
        loader.validate_agent_contract_payload(
            payload["approval_decision"],
            "agent_external_approval_decision.v1.schema.json",
        )

    assert imported_statuses == ["approved", "rejected", "needs_changes", "expired"]
    status_response = client.get(f"/runs/{plan_payload['run_id']}")
    assert status_response.status_code == 200
    assert status_response.json()["external_approval_summary"]["decision_count"] == 4
    assert status_response.json()["external_approval_summary"]["status"] == "expired"
    orchestration_response = client.get(f"/runs/{plan_payload['run_id']}/orchestration")
    assert orchestration_response.status_code == 200
    assert orchestration_response.json()["external_approval_summary"]["decision_count"] == 4

    entry = ledger.get(plan_payload["run_id"])
    assert entry is not None
    decision_events = [
        event
        for event in entry.recovery_events
        if event.get("event_type") == "external_approval_decision_import"
    ]
    assert len(decision_events) == 4
    assert all(event["execution_permitted"] is False for event in decision_events)
    assert all(event["advisory_only"] is True for event in decision_events)
    loader.validate_agent_contract_payload(
        entry.model_dump(mode="json"),
        "agent_execution_ledger.v1.schema.json",
    )
    dumped_ledger = json.dumps(entry.model_dump(mode="json"), sort_keys=True)
    assert str(tmp_path) not in dumped_ledger
    assert "OPENAI_API_KEY" not in dumped_ledger
    assert "provider_prompt" not in dumped_ledger
    assert '"provider_response"' not in dumped_ledger


def test_external_approval_decision_import_is_idempotent_and_rejects_conflicts(
    tmp_path: Path,
) -> None:
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path,
        validate_contract=loader.validate_agent_contract_payload,
    )
    runtime = runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)
    client = TestClient(create_app(runtime))
    plan_payload = _create_plan_with_lifecycle_reference(client)
    preview = client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "run",
            "step_id": None,
        },
    )
    assert preview.status_code == 200
    approval_request = preview.json()["approval_request"]
    decision = _external_approval_decision(
        approval_request,
        decision_status="approved",
        decision_id="decision_idempotent",
    )
    first = client.post(
        "/external-approval-decisions",
        json={
            "run_id": plan_payload["run_id"],
            "decision_intent": "import_external_approval_decision",
            "approval_decision": decision,
        },
    )
    second = client.post(
        "/external-approval-decisions",
        json={
            "run_id": plan_payload["run_id"],
            "decision_intent": "import_external_approval_decision",
            "approval_decision": decision,
        },
    )
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["approval_decision"]["approval_decision_id"] == "decision_idempotent"

    conflict = copy.deepcopy(decision)
    conflict["decision_summary"]["summary"] = "Conflicting decision text."
    conflict_response = client.post(
        "/external-approval-decisions",
        json={
            "run_id": plan_payload["run_id"],
            "decision_intent": "import_external_approval_decision",
            "approval_decision": conflict,
        },
    )
    assert conflict_response.status_code == 422
    assert conflict_response.json()["detail"]["errors"][0]["code"] == (
        "conflicting_external_approval_decision"
    )
    entry = ledger.get(plan_payload["run_id"])
    assert entry is not None
    assert [
        event.get("event_type")
        for event in entry.recovery_events
        if event.get("event_type") == "external_approval_decision_import"
    ] == ["external_approval_decision_import"]


def test_external_approval_decision_import_governance_denies_viewer_and_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path,
        validate_contract=loader.validate_agent_contract_payload,
    )
    operator_runtime = runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)
    operator_client = TestClient(create_app(operator_runtime))
    plan_payload = _create_plan_with_lifecycle_reference(operator_client)
    preview = operator_client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "run",
            "step_id": None,
        },
    )
    assert preview.status_code == 200
    decision = _external_approval_decision(preview.json()["approval_request"])

    for role in ["viewer", "executor"]:
        monkeypatch.setenv("QUANT_AGENT_ACTOR_ROLE", role)
        restricted_runtime = runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)
        restricted_client = TestClient(create_app(restricted_runtime))
        response = restricted_client.post(
            "/external-approval-decisions",
            json={
                "run_id": plan_payload["run_id"],
                "decision_intent": "import_external_approval_decision",
                "approval_decision": decision,
            },
        )
        assert response.status_code == 422
        assert response.json()["detail"]["errors"][0]["code"] == "governance_permission_denied"

    monkeypatch.setenv("QUANT_AGENT_ACTOR_ROLE", "approver")
    approver_runtime = runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)
    approver_client = TestClient(create_app(approver_runtime))
    response = approver_client.post(
        "/external-approval-decisions",
        json={
            "run_id": plan_payload["run_id"],
            "decision_intent": "import_external_approval_decision",
            "approval_decision": decision,
        },
    )
    assert response.status_code == 200
    assert response.json()["approval_decision"]["decision_status"] == "approved"


def test_external_approval_decision_import_rejects_bad_requests_and_payloads(
    tmp_path: Path,
) -> None:
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path,
        validate_contract=loader.validate_agent_contract_payload,
    )
    runtime = runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)
    client = TestClient(create_app(runtime))
    plan_payload = _create_plan_with_lifecycle_reference(client)
    preview = client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "run",
            "step_id": None,
        },
    )
    assert preview.status_code == 200
    approval_request = preview.json()["approval_request"]
    decision = _external_approval_decision(approval_request)

    no_preview_decision = copy.deepcopy(decision)
    no_preview_decision["approval_request_id"] = "missing_request"
    missing_preview = client.post(
        "/external-approval-decisions",
        json={
            "run_id": plan_payload["run_id"],
            "decision_intent": "import_external_approval_decision",
            "approval_decision": no_preview_decision,
        },
    )
    assert missing_preview.status_code == 422
    assert missing_preview.json()["detail"]["errors"][0]["code"] == "missing_external_approval_request_preview"

    run_mismatch = copy.deepcopy(decision)
    run_mismatch["run_id"] = "other_run"
    mismatch_response = client.post(
        "/external-approval-decisions",
        json={
            "run_id": plan_payload["run_id"],
            "decision_intent": "import_external_approval_decision",
            "approval_decision": run_mismatch,
        },
    )
    assert mismatch_response.status_code == 422
    assert mismatch_response.json()["detail"]["errors"][0]["code"] == (
        "external_approval_decision_run_mismatch"
    )

    permitted = copy.deepcopy(decision)
    permitted["execution_permitted"] = True
    permitted_response = client.post(
        "/external-approval-decisions",
        json={
            "run_id": plan_payload["run_id"],
            "decision_intent": "import_external_approval_decision",
            "approval_decision": permitted,
        },
    )
    assert permitted_response.status_code == 422
    assert permitted_response.json()["detail"]["errors"][0]["code"] == (
        "external_approval_decision_execution_permitted"
    )

    unsafe = copy.deepcopy(decision)
    unsafe["decision_summary"]["raw_path"] = "C:\\raw\\approval.json"
    unsafe_response = client.post(
        "/external-approval-decisions",
        json={
            "run_id": plan_payload["run_id"],
            "decision_intent": "import_external_approval_decision",
            "approval_decision": unsafe,
        },
    )
    assert unsafe_response.status_code == 422
    assert unsafe_response.json()["detail"]["errors"][0]["code"] == (
        "unsafe_external_approval_decision_payload"
    )

    extra = client.post(
        "/external-approval-decisions",
        json={
            "run_id": plan_payload["run_id"],
            "decision_intent": "import_external_approval_decision",
            "approval_decision": decision,
            "execution_permitted": True,
        },
    )
    assert extra.status_code == 422

    cancellation = client.post(
        "/cancellations",
        json={
            "run_id": plan_payload["run_id"],
            "cancellation_intent": "cancel_run",
            "reason": "user_cancelled",
        },
    )
    assert cancellation.status_code == 200
    terminal_response = client.post(
        "/external-approval-decisions",
        json={
            "run_id": plan_payload["run_id"],
            "decision_intent": "import_external_approval_decision",
            "approval_decision": decision,
        },
    )
    assert terminal_response.status_code == 422
    assert terminal_response.json()["detail"]["errors"][0]["code"] == (
        "terminal_run_external_approval_decision"
    )


def test_external_approval_submission_local_outbox_idempotent_and_contract_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outbox_dir = tmp_path / "approval_outbox"
    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_ADAPTER", "local_outbox")
    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_OUTBOX_DIR", str(outbox_dir))
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path / "ledgers",
        validate_contract=loader.validate_agent_contract_payload,
    )
    runtime = runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)
    client = TestClient(create_app(runtime))
    plan_payload = _create_plan_with_lifecycle_reference(client)
    preview = client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "run",
            "step_id": None,
        },
    )
    assert preview.status_code == 200
    approval_request_id = preview.json()["approval_request"]["approval_request_id"]

    first = client.post(
        "/external-approval-submissions",
        json={
            "run_id": plan_payload["run_id"],
            "submission_intent": "submit_external_approval_request",
            "approval_request_id": approval_request_id,
        },
    )
    second = client.post(
        "/external-approval-submissions",
        json={
            "run_id": plan_payload["run_id"],
            "submission_intent": "submit_external_approval_request",
            "approval_request_id": approval_request_id,
        },
    )
    assert first.status_code == 200
    assert second.status_code == 200
    payload = first.json()
    duplicate_payload = second.json()
    assert duplicate_payload["external_approval_submission"]["external_approval_submission_id"] == (
        payload["external_approval_submission"]["external_approval_submission_id"]
    )
    assert payload["external_approval_submission"]["submission_status"] == "submitted"
    assert payload["external_approval_submission"]["execution_permitted"] is False
    assert payload["external_approval_submission"]["adapter_summary"]["adapter_mode"] == "local_outbox"
    assert payload["external_approval_submission"]["adapter_summary"]["supports_external_network"] is False
    assert payload["external_approval_submission"]["adapter_summary"]["adapter_support_level"] == (
        "local_outbox_and_mock_http_submission"
    )
    assert payload["external_approval_submission"]["adapter_delivery_summary"]["adapter_delivery_status"] == (
        "submitted"
    )
    assert payload["external_approval_submission"]["submission_reference"]["reference_type"] == (
        "local_outbox_submission"
    )
    assert payload["run_status"]["external_approval_summary"]["status"] == "submitted"
    assert payload["orchestration"]["external_approval_summary"]["submission_count"] == 1
    loader.validate_agent_contract_payload(
        payload["external_approval_submission"],
        "agent_external_approval_submission.v1.schema.json",
    )

    outbox_files = sorted(outbox_dir.glob("*.json"))
    assert len(outbox_files) == 1
    outbox_payload = json.loads(outbox_files[0].read_text(encoding="utf-8"))
    loader.validate_agent_contract_payload(
        outbox_payload,
        "agent_external_approval_submission.v1.schema.json",
    )
    entry = ledger.get(plan_payload["run_id"])
    assert entry is not None
    submission_events = [
        event
        for event in entry.recovery_events
        if event.get("event_type") == "external_approval_submission"
    ]
    assert len(submission_events) == 1
    assert submission_events[0]["execution_permitted"] is False
    loader.validate_agent_contract_payload(
        entry.model_dump(mode="json"),
        "agent_execution_ledger.v1.schema.json",
    )
    serialized = json.dumps(
        {
            "manifest": client.get("/runtime/manifest").json(),
            "response": payload,
            "ledger": entry.model_dump(mode="json"),
            "outbox": outbox_payload,
        },
        sort_keys=True,
    )
    assert str(outbox_dir) not in serialized
    for unsafe_term in [
        "C:\\",
        "/Users/",
        "http://",
        "https://",
        "OPENAI_API_KEY",
        "sk-test",
        "paste-your-openai-api-key",
        "\"provider_response\"",
        "\"app_payload\"",
        "\"links\"",
        "\"query\"",
    ]:
        assert unsafe_term not in serialized


def test_external_approval_submission_mock_http_adapter_is_safe_and_contract_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_ADAPTER", "mock_http")
    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_MOCK_BASE_URL", "http://127.0.0.1:8895")
    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_TIMEOUT_SECONDS", "7")
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path / "ledgers",
        validate_contract=loader.validate_agent_contract_payload,
    )
    captured_requests: list[dict[str, Any]] = []

    class MockResponse:
        def __enter__(self) -> "MockResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "accepted": True,
                    "external_reference_id": "mock_external_ref_001",
                    "status": "submitted",
                    "received_at_utc": "2026-06-23T17:00:00Z",
                    "warnings": ["Review package accepted by mock adapter."],
                },
                sort_keys=True,
            ).encode("utf-8")

    def fake_urlopen(request: Any, timeout: int) -> MockResponse:
        captured_requests.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "body": json.loads(request.data.decode("utf-8")),
                "headers": dict(request.headers),
            }
        )
        return MockResponse()

    monkeypatch.setattr(external_approval_module.urllib.request, "urlopen", fake_urlopen)

    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)))
    plan_payload = _create_plan_with_lifecycle_reference(client)
    preview = client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "run",
            "step_id": None,
        },
    )
    assert preview.status_code == 200
    approval_request = preview.json()["approval_request"]

    submitted = client.post(
        "/external-approval-submissions",
        json={
            "run_id": plan_payload["run_id"],
            "submission_intent": "submit_external_approval_request",
            "approval_request_id": approval_request["approval_request_id"],
        },
    )
    duplicate = client.post(
        "/external-approval-submissions",
        json={
            "run_id": plan_payload["run_id"],
            "submission_intent": "submit_external_approval_request",
            "approval_request_id": approval_request["approval_request_id"],
        },
    )
    assert submitted.status_code == 200
    assert duplicate.status_code == 200
    assert len(captured_requests) == 1
    assert captured_requests[0]["url"] == "http://127.0.0.1:8895/api/external-approval/submissions"
    assert captured_requests[0]["timeout"] == 7
    assert list(captured_requests[0]["body"].keys()) == ["submission"]

    payload = submitted.json()
    submission = payload["external_approval_submission"]
    assert duplicate.json()["external_approval_submission"]["external_approval_submission_id"] == (
        submission["external_approval_submission_id"]
    )
    assert submission["adapter_summary"]["adapter_mode"] == "mock_http"
    assert submission["adapter_summary"]["server_side_http"] is True
    assert submission["adapter_summary"]["supports_external_network"] is False
    assert submission["adapter_summary"]["safe_endpoint_label"] == "mock_external_approval_submission_endpoint"
    assert submission["submission_reference"]["reference_type"] == "mock_http_submission"
    assert submission["submission_reference"]["reference_id"] == "mock_external_ref_001"
    assert submission["adapter_delivery_summary"]["adapter_delivery_status"] == "submitted"
    assert submission["adapter_delivery_summary"]["external_reference_id"] == "mock_external_ref_001"
    loader.validate_agent_contract_payload(
        submission,
        "agent_external_approval_submission.v1.schema.json",
    )

    status = client.get(f"/runs/{plan_payload['run_id']}/external-approval-submissions")
    assert status.status_code == 200
    status_payload = status.json()
    assert status_payload["submissions"][0]["adapter_mode"] == "mock_http"
    assert status_payload["submissions"][0]["outbox_status"] == "not_checked"
    assert status_payload["submissions"][0]["adapter_delivery_status"] == "submitted"

    decision = _external_approval_decision(approval_request)
    imported = client.post(
        "/external-approval-decisions",
        json={
            "run_id": plan_payload["run_id"],
            "decision_intent": "import_external_approval_decision",
            "approval_decision": decision,
        },
    )
    assert imported.status_code == 200
    latest_decision = imported.json()["external_approval_summary"]["latest_matching_decision"]
    assert latest_decision["matched_submission_reference"]["adapter_mode"] == "mock_http"
    assert latest_decision["matched_submission_reference"]["adapter_delivery_status"] == "submitted"

    manifest = client.get("/runtime/manifest").json()
    assert manifest["external_approval_submission_adapter"]["adapter_mode"] == "mock_http"
    assert manifest["external_approval_submission_adapter"]["safe_endpoint_label"] == (
        "mock_external_approval_submission_endpoint"
    )
    assert manifest["external_approval_submission_adapter"]["timeout_seconds"] == 7

    entry = ledger.get(plan_payload["run_id"])
    assert entry is not None
    loader.validate_agent_contract_payload(
        entry.model_dump(mode="json"),
        "agent_execution_ledger.v1.schema.json",
    )
    serialized = json.dumps(
        {
            "manifest": manifest,
            "response": payload,
            "status": status_payload,
            "decision": imported.json(),
            "ledger": entry.model_dump(mode="json"),
        },
        sort_keys=True,
    )
    assert "http://127.0.0.1:8895" not in serialized
    for unsafe_term in [
        "Authorization",
        "OPENAI_API_KEY",
        "sk-test",
        "\"raw_response\"",
        "\"provider_response\"",
        "\"app_payload\"",
        "\"headers\"",
    ]:
        assert unsafe_term not in serialized


def test_external_approval_decision_refresh_mock_http_pending_and_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_ADAPTER", "mock_http")
    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_MOCK_BASE_URL", "http://127.0.0.1:8895")
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path / "ledgers",
        validate_contract=loader.validate_agent_contract_payload,
    )
    captured_requests: list[dict[str, Any]] = []
    refresh_count = 0
    approval_request: dict[str, Any] = {}

    class MockResponse:
        def __init__(self, payload: Any) -> None:
            self._payload = payload

        def __enter__(self) -> "MockResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(self._payload, sort_keys=True).encode("utf-8")

    def fake_urlopen(request: Any, timeout: int) -> MockResponse:
        nonlocal refresh_count
        body = json.loads(request.data.decode("utf-8"))
        captured_requests.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "body": body,
            }
        )
        if request.full_url.endswith("/api/external-approval/submissions"):
            return MockResponse(
                {
                    "accepted": True,
                    "external_reference_id": "mock_refresh_ref_001",
                    "status": "submitted",
                    "received_at_utc": "2026-06-23T17:00:00Z",
                    "warnings": [],
                }
            )
        refresh_count += 1
        if refresh_count == 1:
            return MockResponse(
                {
                    "decision_available": False,
                    "status": "pending",
                    "checked_at_utc": "2026-06-23T17:01:00Z",
                    "warnings": ["Decision is still pending in the mock adapter."],
                }
            )
        return MockResponse(
            {
                "decision_available": True,
                "status": "decision_available",
                "approval_decision": _external_approval_decision(
                    approval_request,
                    decision_id="mock_http_refresh_decision_001",
                ),
                "checked_at_utc": "2026-06-23T17:02:00Z",
                "warnings": ["Decision returned by the mock adapter."],
            }
        )

    monkeypatch.setattr(external_approval_module.urllib.request, "urlopen", fake_urlopen)

    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)))
    plan_payload = _create_plan_with_lifecycle_reference(client)
    manifest = client.get("/runtime/manifest").json()
    assert "POST /external-approval-decision-refreshes" in manifest["supported_routes"]
    assert manifest["external_approval_decision_refresh_support_level"] == "mock_http_manual_refresh_only"

    preview = client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "run",
            "step_id": None,
        },
    )
    assert preview.status_code == 200
    approval_request = preview.json()["approval_request"]
    submission = client.post(
        "/external-approval-submissions",
        json={
            "run_id": plan_payload["run_id"],
            "submission_intent": "submit_external_approval_request",
            "approval_request_id": approval_request["approval_request_id"],
        },
    )
    assert submission.status_code == 200

    pending = client.post(
        "/external-approval-decision-refreshes",
        json={
            "run_id": plan_payload["run_id"],
            "decision_refresh_intent": "refresh_external_approval_decision",
            "approval_request_id": approval_request["approval_request_id"],
        },
    )
    assert pending.status_code == 200
    pending_payload = pending.json()
    assert pending_payload["decision_refresh"]["status"] == "pending"
    assert pending_payload["decision_refresh"]["decision_available"] is False
    assert pending_payload["approval_decision"] is None
    assert pending_payload["external_approval_summary"]["decision_refresh_count"] == 1
    assert pending_payload["external_approval_summary"]["decision_count"] == 0
    assert captured_requests[-1]["url"] == "http://127.0.0.1:8895/api/external-approval/decisions/refresh"
    assert list(captured_requests[-1]["body"].keys()) == ["decision_lookup"]
    assert captured_requests[-1]["body"]["decision_lookup"]["approval_request_id"] == (
        approval_request["approval_request_id"]
    )

    available = client.post(
        "/external-approval-decision-refreshes",
        json={
            "run_id": plan_payload["run_id"],
            "decision_refresh_intent": "refresh_external_approval_decision",
            "approval_request_id": approval_request["approval_request_id"],
        },
    )
    assert available.status_code == 200
    available_payload = available.json()
    assert available_payload["decision_refresh"]["status"] == "decision_available"
    assert available_payload["decision_refresh"]["decision_available"] is True
    assert available_payload["approval_decision"]["decision_status"] == "approved"
    assert available_payload["external_approval_summary"]["status"] == "approved"
    assert available_payload["external_approval_summary"]["latest_matching_decision"]["decision_source"] == (
        "mock_http_refresh"
    )
    assert available_payload["external_approval_summary"]["latest_matching_decision_refresh"][
        "adapter_refresh_status"
    ] == "decision_available"

    duplicate = client.post(
        "/external-approval-decision-refreshes",
        json={
            "run_id": plan_payload["run_id"],
            "decision_refresh_intent": "refresh_external_approval_decision",
            "approval_request_id": approval_request["approval_request_id"],
        },
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["approval_decision"]["approval_decision_id"] == "mock_http_refresh_decision_001"

    entry = ledger.get(plan_payload["run_id"])
    assert entry is not None
    refresh_events = [
        event
        for event in entry.recovery_events
        if event.get("event_type") == "external_approval_decision_refresh"
    ]
    decision_events = [
        event
        for event in entry.recovery_events
        if event.get("event_type") == "external_approval_decision_import"
    ]
    assert [event["status"] for event in refresh_events] == ["pending", "decision_available"]
    assert [event["decision_source"] for event in decision_events] == ["mock_http_refresh"]
    loader.validate_agent_contract_payload(
        entry.model_dump(mode="json"),
        "agent_execution_ledger.v1.schema.json",
    )

    status = client.get(f"/runs/{plan_payload['run_id']}/external-approval-submissions")
    assert status.status_code == 200
    assert status.json()["submissions"][0]["latest_decision_refresh"]["status"] == "decision_available"
    assert status.json()["submissions"][0]["latest_matching_decision"]["decision_status"] == "approved"

    serialized = json.dumps(
        {
            "manifest": manifest,
            "pending": pending_payload,
            "available": available_payload,
            "duplicate": duplicate.json(),
            "status": status.json(),
            "ledger": entry.model_dump(mode="json"),
        },
        sort_keys=True,
    )
    assert "http://127.0.0.1:8895" not in serialized
    for unsafe_term in [
        "Authorization",
        "OPENAI_API_KEY",
        "sk-test",
        "\"raw_response\"",
        "\"headers\"",
        "\"provider_response\"",
        "\"app_payload\"",
    ]:
        assert unsafe_term not in serialized


@pytest.mark.parametrize(
    ("response_payload", "expected_code"),
    [
        ("not json", "external_approval_submission_adapter_invalid_response"),
        ({"accepted": False, "status": "rejected"}, "external_approval_submission_adapter_rejected"),
        (
            {"accepted": True, "external_reference_id": "ref", "status": "submitted", "raw_url": "http://unsafe.example"},
            "unsafe_external_approval_adapter_response",
        ),
    ],
)
def test_external_approval_submission_mock_http_bad_responses_are_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response_payload: Any,
    expected_code: str,
) -> None:
    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_ADAPTER", "mock_http")
    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_MOCK_BASE_URL", "http://127.0.0.1:8895")
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path / "ledgers",
        validate_contract=loader.validate_agent_contract_payload,
    )

    class MockResponse:
        def __enter__(self) -> "MockResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            if isinstance(response_payload, str):
                return response_payload.encode("utf-8")
            return json.dumps(response_payload, sort_keys=True).encode("utf-8")

    monkeypatch.setattr(
        external_approval_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: MockResponse(),
    )
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)))
    plan_payload = _create_plan_with_lifecycle_reference(client)
    preview = client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "run",
            "step_id": None,
        },
    )
    assert preview.status_code == 200
    response = client.post(
        "/external-approval-submissions",
        json={
            "run_id": plan_payload["run_id"],
            "submission_intent": "submit_external_approval_request",
            "approval_request_id": preview.json()["approval_request"]["approval_request_id"],
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["code"] == expected_code
    assert "http://unsafe.example" not in json.dumps(response.json(), sort_keys=True)
    entry = ledger.get(plan_payload["run_id"])
    assert entry is not None
    assert [
        event
        for event in entry.recovery_events
        if event.get("event_type") == "external_approval_submission"
    ] == []


def test_external_approval_submission_mock_http_unreachable_and_invalid_config_are_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_ADAPTER", "mock_http")
    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_MOCK_BASE_URL", "http://127.0.0.1:8895")
    monkeypatch.setattr(
        external_approval_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            external_approval_module.urllib.error.URLError("connection failed for secret endpoint")
        ),
    )
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path / "ledgers",
        validate_contract=loader.validate_agent_contract_payload,
    )
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)))
    plan_payload = _create_plan_with_lifecycle_reference(client)
    preview = client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "run",
            "step_id": None,
        },
    )
    assert preview.status_code == 200
    failed = client.post(
        "/external-approval-submissions",
        json={
            "run_id": plan_payload["run_id"],
            "submission_intent": "submit_external_approval_request",
            "approval_request_id": preview.json()["approval_request"]["approval_request_id"],
        },
    )
    assert failed.status_code == 422
    assert failed.json()["detail"]["errors"][0]["code"] == "external_approval_submission_adapter_failed"
    assert "secret endpoint" not in json.dumps(failed.json(), sort_keys=True)

    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_MOCK_BASE_URL", "not a url")
    invalid_client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)))
    manifest = invalid_client.get("/runtime/manifest").json()
    assert manifest["external_approval_submission_adapter"]["adapter_mode"] == "mock_http"
    assert manifest["external_approval_submission_adapter"]["enabled"] is False
    assert manifest["external_approval_submission_adapter"]["disabled_reason"] == "invalid_mock_base_url"
    assert "not a url" not in json.dumps(manifest, sort_keys=True)


@pytest.mark.parametrize(
    ("response_payload", "expected_code"),
    [
        ("not json", "external_approval_decision_refresh_adapter_invalid_response"),
        (
            {"decision_available": False, "status": "waiting"},
            "external_approval_decision_refresh_adapter_invalid_response",
        ),
        (
            {"decision_available": False, "status": "pending", "raw_url": "http://unsafe.example"},
            "unsafe_external_approval_decision_refresh_adapter_response",
        ),
        (
            {"decision_available": True, "status": "decision_available", "approval_decision": {"not": "valid"}},
            "external_approval_decision_contract_validation_failed",
        ),
    ],
)
def test_external_approval_decision_refresh_mock_http_bad_responses_are_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response_payload: Any,
    expected_code: str,
) -> None:
    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_ADAPTER", "mock_http")
    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_MOCK_BASE_URL", "http://127.0.0.1:8895")
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path / "ledgers",
        validate_contract=loader.validate_agent_contract_payload,
    )

    class MockResponse:
        def __init__(self, payload: Any) -> None:
            self._payload = payload

        def __enter__(self) -> "MockResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            if isinstance(self._payload, str):
                return self._payload.encode("utf-8")
            return json.dumps(self._payload, sort_keys=True).encode("utf-8")

    def fake_urlopen(request: Any, timeout: int) -> MockResponse:
        if request.full_url.endswith("/api/external-approval/submissions"):
            return MockResponse(
                {
                    "accepted": True,
                    "external_reference_id": "mock_bad_refresh_ref",
                    "status": "submitted",
                    "received_at_utc": "2026-06-23T17:00:00Z",
                    "warnings": [],
                }
            )
        return MockResponse(response_payload)

    monkeypatch.setattr(external_approval_module.urllib.request, "urlopen", fake_urlopen)
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)))
    plan_payload = _create_plan_with_lifecycle_reference(client)
    preview = client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "run",
            "step_id": None,
        },
    )
    assert preview.status_code == 200
    approval_request = preview.json()["approval_request"]
    submitted = client.post(
        "/external-approval-submissions",
        json={
            "run_id": plan_payload["run_id"],
            "submission_intent": "submit_external_approval_request",
            "approval_request_id": approval_request["approval_request_id"],
        },
    )
    assert submitted.status_code == 200
    response = client.post(
        "/external-approval-decision-refreshes",
        json={
            "run_id": plan_payload["run_id"],
            "decision_refresh_intent": "refresh_external_approval_decision",
            "approval_request_id": approval_request["approval_request_id"],
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["code"] == expected_code
    serialized = json.dumps(response.json(), sort_keys=True)
    assert "http://unsafe.example" not in serialized
    assert "http://127.0.0.1:8895" not in serialized
    entry = ledger.get(plan_payload["run_id"])
    assert entry is not None
    assert [
        event
        for event in entry.recovery_events
        if event.get("event_type") == "external_approval_decision_refresh"
    ] == []
    assert [
        event
        for event in entry.recovery_events
        if event.get("event_type") == "external_approval_decision_import"
    ] == []


def test_external_approval_decision_refresh_rejects_local_outbox_and_denied_roles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_ADAPTER", "local_outbox")
    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_OUTBOX_DIR", str(tmp_path / "outbox"))
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path / "ledgers",
        validate_contract=loader.validate_agent_contract_payload,
    )
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)))
    plan_payload = _create_plan_with_lifecycle_reference(client)
    preview = client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "run",
            "step_id": None,
        },
    )
    assert preview.status_code == 200
    approval_request_id = preview.json()["approval_request"]["approval_request_id"]
    submitted = client.post(
        "/external-approval-submissions",
        json={
            "run_id": plan_payload["run_id"],
            "submission_intent": "submit_external_approval_request",
            "approval_request_id": approval_request_id,
        },
    )
    assert submitted.status_code == 200
    unsupported = client.post(
        "/external-approval-decision-refreshes",
        json={
            "run_id": plan_payload["run_id"],
            "decision_refresh_intent": "refresh_external_approval_decision",
            "approval_request_id": approval_request_id,
        },
    )
    assert unsupported.status_code == 422
    assert unsupported.json()["detail"]["errors"][0]["code"] == (
        "external_approval_decision_refresh_adapter_unsupported"
    )

    denied_policy_path = tmp_path / "viewer_policy.json"
    _write_governance_policy_pack(
        denied_policy_path,
        role_id="viewer",
        allowed_routes=["GET /runtime/manifest"],
        denied_routes=["POST /external-approval-decision-refreshes"],
    )
    monkeypatch.setenv("QUANT_AGENT_GOVERNANCE_POLICY_PACK_PATH", str(denied_policy_path))
    monkeypatch.setenv("QUANT_AGENT_ACTOR_ROLE", "viewer")
    denied_client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)))
    denied = denied_client.post(
        "/external-approval-decision-refreshes",
        json={
            "run_id": plan_payload["run_id"],
            "decision_refresh_intent": "refresh_external_approval_decision",
            "approval_request_id": approval_request_id,
        },
    )
    assert denied.status_code == 422
    assert denied.json()["detail"]["errors"][0]["code"] == "governance_permission_denied"


def test_external_approval_submission_status_links_outbox_and_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outbox_dir = tmp_path / "approval_outbox"
    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_ADAPTER", "local_outbox")
    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_OUTBOX_DIR", str(outbox_dir))
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path / "ledgers",
        validate_contract=loader.validate_agent_contract_payload,
    )
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)))
    plan_payload = _create_plan_with_lifecycle_reference(client)

    empty_status = client.get(f"/runs/{plan_payload['run_id']}/external-approval-submissions")
    assert empty_status.status_code == 200
    empty_payload = empty_status.json()
    assert empty_payload["submissions"] == []
    assert empty_payload["external_approval_summary"]["outbox_status"] == "not_checked"
    assert empty_payload["ledger_recorded"] is False

    preview = client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "run",
            "step_id": None,
        },
    )
    assert preview.status_code == 200
    approval_request = preview.json()["approval_request"]
    submission = client.post(
        "/external-approval-submissions",
        json={
            "run_id": plan_payload["run_id"],
            "submission_intent": "submit_external_approval_request",
            "approval_request_id": approval_request["approval_request_id"],
        },
    )
    assert submission.status_code == 200

    submitted_status = client.get(f"/runs/{plan_payload['run_id']}/external-approval-submissions")
    assert submitted_status.status_code == 200
    submitted_payload = submitted_status.json()
    assert len(submitted_payload["submissions"]) == 1
    submitted_summary = submitted_payload["submissions"][0]
    assert submitted_summary["approval_request_id"] == approval_request["approval_request_id"]
    assert submitted_summary["submission_status"] == "submitted"
    assert submitted_summary["outbox_status"] == "present"
    assert submitted_summary["adapter_mode"] == "local_outbox"
    assert submitted_summary["latest_matching_decision"] is None
    assert submitted_summary["ledger_integrity_summary"]["status"] == "verified"
    assert submitted_payload["external_approval_summary"]["latest_submission"]["outbox_status"] == "present"
    assert submitted_payload["external_approval_summary"]["outbox_status"] == "present"

    decision = _external_approval_decision(approval_request)
    imported = client.post(
        "/external-approval-decisions",
        json={
            "run_id": plan_payload["run_id"],
            "decision_intent": "import_external_approval_decision",
            "approval_decision": decision,
        },
    )
    assert imported.status_code == 200
    imported_payload = imported.json()
    latest_decision = imported_payload["external_approval_summary"]["latest_matching_decision"]
    assert latest_decision["decision_status"] == "approved"
    assert latest_decision["submission_status"] == "submitted"
    assert latest_decision["matched_submission_reference"]["outbox_status"] == "present"

    linked_status = client.get(f"/runs/{plan_payload['run_id']}/external-approval-submissions")
    assert linked_status.status_code == 200
    linked_summary = linked_status.json()["submissions"][0]
    assert linked_summary["latest_matching_decision"]["decision_status"] == "approved"
    assert linked_summary["latest_matching_decision"]["matched_submission_reference"][
        "external_approval_submission_id"
    ] == linked_summary["external_approval_submission_id"]

    support_bundle = client.get(f"/runs/{plan_payload['run_id']}/support-bundle")
    assert support_bundle.status_code == 200
    bundle_summary = support_bundle.json()["run_status"]["external_approval_summary"]
    assert bundle_summary["latest_submission"]["outbox_status"] == "present"
    assert bundle_summary["latest_matching_decision"]["decision_status"] == "approved"

    outbox_files = sorted(outbox_dir.glob("*.json"))
    assert len(outbox_files) == 1
    outbox_files[0].unlink()
    missing_status = client.get(f"/runs/{plan_payload['run_id']}/external-approval-submissions")
    assert missing_status.status_code == 200
    assert missing_status.json()["submissions"][0]["outbox_status"] == "missing"
    assert missing_status.json()["external_approval_summary"]["outbox_status"] == "missing"

    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_ADAPTER", "disabled")
    disabled_status = client.get(f"/runs/{plan_payload['run_id']}/external-approval-submissions")
    assert disabled_status.status_code == 200
    assert disabled_status.json()["submissions"][0]["outbox_status"] == "disabled"

    serialized = json.dumps(
        {
            "linked_status": linked_status.json(),
            "support_bundle": support_bundle.json(),
            "missing_status": missing_status.json(),
            "disabled_status": disabled_status.json(),
        },
        sort_keys=True,
    )
    assert str(outbox_dir) not in serialized
    for unsafe_term in [
        "C:\\",
        "/Users/",
        "http://",
        "https://",
        "OPENAI_API_KEY",
        "sk-test",
        "\"provider_response\"",
        "\"app_payload\"",
        "\"raw_path\"",
    ]:
        assert unsafe_term not in serialized


def test_external_approval_submission_status_governance_read_access_and_denial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_OUTBOX_DIR", str(tmp_path / "outbox"))
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path / "ledgers",
        validate_contract=loader.validate_agent_contract_payload,
    )
    operator_client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)))
    plan_payload = _create_plan_with_lifecycle_reference(operator_client)
    preview = operator_client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "run",
            "step_id": None,
        },
    )
    assert preview.status_code == 200
    submitted = operator_client.post(
        "/external-approval-submissions",
        json={
            "run_id": plan_payload["run_id"],
            "submission_intent": "submit_external_approval_request",
            "approval_request_id": preview.json()["approval_request"]["approval_request_id"],
        },
    )
    assert submitted.status_code == 200

    monkeypatch.setenv("QUANT_AGENT_ACTOR_ROLE", "viewer")
    viewer_client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)))
    viewer_response = viewer_client.get(f"/runs/{plan_payload['run_id']}/external-approval-submissions")
    assert viewer_response.status_code == 200
    assert viewer_response.json()["submissions"][0]["outbox_status"] == "present"

    policy_path = tmp_path / "deny_submission_status_policy.json"
    _write_governance_policy_pack(
        policy_path,
        role_id="status_reader",
        denied_routes=["GET /runs/{run_id}/external-approval-submissions"],
    )
    monkeypatch.setenv("QUANT_AGENT_GOVERNANCE_POLICY_PACK_PATH", str(policy_path))
    monkeypatch.setenv("QUANT_AGENT_ACTOR_ROLE", "status_reader")
    denied_client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)))
    denied_response = denied_client.get(f"/runs/{plan_payload['run_id']}/external-approval-submissions")
    assert denied_response.status_code == 422
    assert denied_response.json()["detail"]["errors"][0]["code"] == "governance_permission_denied"


def test_external_approval_submission_disabled_adapter_rejects_without_path_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outbox_dir = tmp_path / "disabled_outbox"
    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_ADAPTER", "disabled")
    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_OUTBOX_DIR", str(outbox_dir))
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path / "ledgers",
        validate_contract=loader.validate_agent_contract_payload,
    )
    runtime = runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)
    client = TestClient(create_app(runtime))
    plan_payload = _create_plan_with_lifecycle_reference(client)
    preview = client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "run",
            "step_id": None,
        },
    )
    assert preview.status_code == 200
    response = client.post(
        "/external-approval-submissions",
        json={
            "run_id": plan_payload["run_id"],
            "submission_intent": "submit_external_approval_request",
            "approval_request_id": preview.json()["approval_request"]["approval_request_id"],
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["code"] == (
        "external_approval_submission_adapter_disabled"
    )
    manifest = client.get("/runtime/manifest").json()
    assert manifest["external_approval_submission_adapter"]["adapter_mode"] == "disabled"
    assert manifest["external_approval_submission_adapter"]["enabled"] is False
    assert str(outbox_dir) not in json.dumps(manifest, sort_keys=True)
    assert not outbox_dir.exists()


def test_external_approval_submission_adapter_failure_is_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outbox_file = tmp_path / "outbox_file"
    outbox_file.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_ADAPTER", "local_outbox")
    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_OUTBOX_DIR", str(outbox_file))
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path / "ledgers",
        validate_contract=loader.validate_agent_contract_payload,
    )
    runtime = runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)
    client = TestClient(create_app(runtime))
    plan_payload = _create_plan_with_lifecycle_reference(client)
    preview = client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "run",
            "step_id": None,
        },
    )
    assert preview.status_code == 200

    response = client.post(
        "/external-approval-submissions",
        json={
            "run_id": plan_payload["run_id"],
            "submission_intent": "submit_external_approval_request",
            "approval_request_id": preview.json()["approval_request"]["approval_request_id"],
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["code"] == "external_approval_submission_adapter_failed"
    assert str(outbox_file) not in json.dumps(response.json(), sort_keys=True)
    entry = ledger.get(plan_payload["run_id"])
    assert entry is not None
    assert [
        event
        for event in entry.recovery_events
        if event.get("event_type") == "external_approval_submission"
    ] == []


def test_external_approval_submission_governance_denies_viewer_and_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_OUTBOX_DIR", str(tmp_path / "outbox"))
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path / "ledgers",
        validate_contract=loader.validate_agent_contract_payload,
    )
    operator_client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)))
    plan_payload = _create_plan_with_lifecycle_reference(operator_client)
    preview = operator_client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "run",
            "step_id": None,
        },
    )
    assert preview.status_code == 200
    approval_request_id = preview.json()["approval_request"]["approval_request_id"]

    for role in ["viewer", "executor"]:
        monkeypatch.setenv("QUANT_AGENT_ACTOR_ROLE", role)
        restricted_client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)))
        response = restricted_client.post(
            "/external-approval-submissions",
            json={
                "run_id": plan_payload["run_id"],
                "submission_intent": "submit_external_approval_request",
                "approval_request_id": approval_request_id,
            },
        )
        assert response.status_code == 422
        assert response.json()["detail"]["errors"][0]["code"] == "governance_permission_denied"

    monkeypatch.setenv("QUANT_AGENT_ACTOR_ROLE", "approver")
    approver_client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)))
    response = approver_client.post(
        "/external-approval-submissions",
        json={
            "run_id": plan_payload["run_id"],
            "submission_intent": "submit_external_approval_request",
            "approval_request_id": approval_request_id,
        },
    )
    assert response.status_code == 200
    assert response.json()["external_approval_submission"]["submission_status"] == "submitted"


def test_external_approval_submission_rejects_missing_request_decision_and_extra_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_OUTBOX_DIR", str(tmp_path / "outbox"))
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path / "ledgers",
        validate_contract=loader.validate_agent_contract_payload,
    )
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)))
    plan_payload = _create_plan_with_lifecycle_reference(client)

    missing_preview = client.post(
        "/external-approval-submissions",
        json={
            "run_id": plan_payload["run_id"],
            "submission_intent": "submit_external_approval_request",
            "approval_request_id": "missing_request",
        },
    )
    assert missing_preview.status_code == 422
    assert missing_preview.json()["detail"]["errors"][0]["code"] == (
        "missing_external_approval_request_preview"
    )

    preview = client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "run",
            "step_id": None,
        },
    )
    assert preview.status_code == 200
    approval_request = preview.json()["approval_request"]
    extra = client.post(
        "/external-approval-submissions",
        json={
            "run_id": plan_payload["run_id"],
            "submission_intent": "submit_external_approval_request",
            "approval_request_id": approval_request["approval_request_id"],
            "external_approval_submission": approval_request,
        },
    )
    assert extra.status_code == 422

    decision = _external_approval_decision(approval_request)
    imported_decision = client.post(
        "/external-approval-decisions",
        json={
            "run_id": plan_payload["run_id"],
            "decision_intent": "import_external_approval_decision",
            "approval_decision": decision,
        },
    )
    assert imported_decision.status_code == 200
    assert imported_decision.json()["external_approval_summary"]["latest_matching_decision"][
        "submission_status"
    ] == "not_submitted"
    entry_after_decision = ledger.get(plan_payload["run_id"])
    assert entry_after_decision is not None
    decision_event = next(
        event
        for event in entry_after_decision.recovery_events
        if event.get("event_type") == "external_approval_decision_import"
    )
    assert decision_event["submission_status"] == "not_submitted"
    assert decision_event["matched_submission_reference"] is None
    after_decision = client.post(
        "/external-approval-submissions",
        json={
            "run_id": plan_payload["run_id"],
            "submission_intent": "submit_external_approval_request",
            "approval_request_id": approval_request["approval_request_id"],
        },
    )
    assert after_decision.status_code == 422
    assert after_decision.json()["detail"]["errors"][0]["code"] == (
        "external_approval_decision_already_imported"
    )

    cancelled_plan = _create_plan_with_lifecycle_reference(client)
    cancelled_preview = client.post(
        "/external-approval-requests",
        json={
            "run_id": cancelled_plan["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "run",
            "step_id": None,
        },
    )
    assert cancelled_preview.status_code == 200
    cancellation = client.post(
        "/cancellations",
        json={
            "run_id": cancelled_plan["run_id"],
            "cancellation_intent": "cancel_run",
            "reason": "user_cancelled",
        },
    )
    assert cancellation.status_code == 200
    terminal = client.post(
        "/external-approval-submissions",
        json={
            "run_id": cancelled_plan["run_id"],
            "submission_intent": "submit_external_approval_request",
            "approval_request_id": cancelled_preview.json()["approval_request"]["approval_request_id"],
        },
    )
    assert terminal.status_code == 422
    assert terminal.json()["detail"]["errors"][0]["code"] == (
        "terminal_run_external_approval_submission"
    )


def test_external_approval_enforcement_blocks_until_approved_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_path = tmp_path / "governance_policy_pack.json"
    _write_governance_policy_pack(
        policy_path,
        role_id="workflow_operator",
        external_approval_rules=[_blocking_external_approval_rule()],
    )
    monkeypatch.setenv("QUANT_AGENT_GOVERNANCE_POLICY_PACK_PATH", str(policy_path))
    monkeypatch.setenv("QUANT_AGENT_ACTOR_ROLE", "workflow_operator")
    monkeypatch.setenv("QUANT_AGENT_ACTOR_ID", "workflow_actor")
    ledger = InMemoryLedger()
    app_client = FakePreflightAppClient()
    client = TestClient(create_app(runtime_with_preflight_client(app_client, ledger=ledger)))
    plan_payload, studio_step = _prepare_user_owned_studio_execution(client)

    denied = client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )
    assert denied.status_code == 422
    issue = denied.json()["detail"]["errors"][0]
    assert issue["code"] == "external_approval_required"
    assert issue["capability_id"] == "quant_studio.prepare_model_config_draft"
    assert app_client.execution_calls == []
    ledger_payload = client.get(f"/runs/{plan_payload['run_id']}/ledger").json()
    denial = ledger_payload["recovery_events"][-1]
    assert denial["event_type"] == "external_approval_enforcement_denied"
    assert denial["reason"] == "missing_external_approval_request"
    assert denial["execution_permitted"] is False

    preview = client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "run",
            "step_id": None,
        },
    )
    assert preview.status_code == 200
    still_denied = client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )
    assert still_denied.status_code == 422
    assert still_denied.json()["detail"]["errors"][0]["code"] == "external_approval_required"

    decision = _external_approval_decision(preview.json()["approval_request"])
    import_response = client.post(
        "/external-approval-decisions",
        json={
            "run_id": plan_payload["run_id"],
            "decision_intent": "import_external_approval_decision",
            "approval_decision": decision,
        },
    )
    assert import_response.status_code == 200

    execution = client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )
    assert execution.status_code == 200
    assert execution.json()["action_result"]["execution_status"] == "succeeded"
    status_payload = client.get(f"/runs/{plan_payload['run_id']}").json()
    enforcement_summary = status_payload["external_approval_enforcement_summary"]
    assert enforcement_summary["blocked"] is False
    assert enforcement_summary["latest_decision"]["approval_decision_status"] == "approved"


def test_external_approval_enforcement_accepts_step_level_approved_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_path = tmp_path / "governance_policy_pack.json"
    _write_governance_policy_pack(
        policy_path,
        role_id="workflow_operator",
        external_approval_rules=[_blocking_external_approval_rule()],
    )
    monkeypatch.setenv("QUANT_AGENT_GOVERNANCE_POLICY_PACK_PATH", str(policy_path))
    monkeypatch.setenv("QUANT_AGENT_ACTOR_ROLE", "workflow_operator")
    ledger = InMemoryLedger()
    app_client = FakePreflightAppClient()
    client = TestClient(create_app(runtime_with_preflight_client(app_client, ledger=ledger)))
    plan_payload, studio_step = _prepare_user_owned_studio_execution(client)

    preview = client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "step",
            "step_id": studio_step["step_id"],
        },
    )
    assert preview.status_code == 200
    decision = _external_approval_decision(preview.json()["approval_request"])
    import_response = client.post(
        "/external-approval-decisions",
        json={
            "run_id": plan_payload["run_id"],
            "decision_intent": "import_external_approval_decision",
            "approval_decision": decision,
        },
    )
    assert import_response.status_code == 200

    execution = client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )
    assert execution.status_code == 200
    assert execution.json()["action_result"]["execution_status"] == "succeeded"


@pytest.mark.parametrize("decision_status", ["rejected", "needs_changes", "expired"])
def test_external_approval_enforcement_blocks_non_approved_decisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision_status: str,
) -> None:
    policy_path = tmp_path / "governance_policy_pack.json"
    _write_governance_policy_pack(
        policy_path,
        role_id="workflow_operator",
        external_approval_rules=[_blocking_external_approval_rule()],
    )
    monkeypatch.setenv("QUANT_AGENT_GOVERNANCE_POLICY_PACK_PATH", str(policy_path))
    monkeypatch.setenv("QUANT_AGENT_ACTOR_ROLE", "workflow_operator")
    ledger = InMemoryLedger()
    app_client = FakePreflightAppClient()
    client = TestClient(create_app(runtime_with_preflight_client(app_client, ledger=ledger)))
    plan_payload, studio_step = _prepare_user_owned_studio_execution(client)
    preview = client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "run",
            "step_id": None,
        },
    )
    assert preview.status_code == 200
    decision = _external_approval_decision(
        preview.json()["approval_request"],
        decision_status=decision_status,
    )
    import_response = client.post(
        "/external-approval-decisions",
        json={
            "run_id": plan_payload["run_id"],
            "decision_intent": "import_external_approval_decision",
            "approval_decision": decision,
        },
    )
    assert import_response.status_code == 200

    denied = client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )
    assert denied.status_code == 422
    issue = denied.json()["detail"]["errors"][0]
    assert issue["code"] == "external_approval_decision_denied"
    assert issue["capability_id"] == "quant_studio.prepare_model_config_draft"
    assert app_client.execution_calls == []
    status_payload = client.get(f"/runs/{plan_payload['run_id']}").json()
    enforcement_summary = status_payload["external_approval_enforcement_summary"]
    assert enforcement_summary["blocked"] is True
    assert enforcement_summary["latest_decision"]["approval_decision_status"] == decision_status


def test_external_approval_enforcement_protects_retry_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_path = tmp_path / "governance_policy_pack.json"
    _write_governance_policy_pack(
        policy_path,
        role_id="workflow_operator",
        external_approval_rules=[_blocking_external_approval_rule()],
    )
    monkeypatch.setenv("QUANT_AGENT_GOVERNANCE_POLICY_PACK_PATH", str(policy_path))
    monkeypatch.setenv("QUANT_AGENT_ACTOR_ROLE", "workflow_operator")
    ledger = InMemoryLedger()
    app_client = FakePreflightAppClient()
    client = TestClient(create_app(runtime_with_preflight_client(app_client, ledger=ledger)))
    plan_payload, studio_step = _prepare_user_owned_studio_execution(client)

    denied = client.post(
        "/retries",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": studio_step["step_id"],
            "retry_intent": "retry_failed_step",
        },
    )

    assert denied.status_code == 422
    issue = denied.json()["detail"]["errors"][0]
    assert issue["code"] == "external_approval_required"
    assert issue["capability_id"] == "quant_studio.prepare_model_config_draft"
    ledger_payload = client.get(f"/runs/{plan_payload['run_id']}/ledger").json()
    denial = ledger_payload["recovery_events"][-1]
    assert denial["event_type"] == "external_approval_enforcement_denied"
    assert denial["denied_route"] == "POST /retries"


def test_phase9_certification_local_development_workflow_and_exports_remain_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QUANT_AGENT_GOVERNANCE_POLICY_PACK_PATH", raising=False)
    monkeypatch.setenv("QUANT_AGENT_GOVERNANCE_ENVIRONMENT", "local_development")
    monkeypatch.setenv("QUANT_AGENT_ACTOR_ROLE", "local_developer_operator")
    monkeypatch.setenv("QUANT_AGENT_ACTOR_ID", "local_user")
    fixture = _expected_phase9_governance_certification_fixture()
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path / "ledgers",
        validate_contract=loader.validate_agent_contract_payload,
    )
    app_client = FakePreflightAppClient()
    client = TestClient(create_app(runtime_with_preflight_client(app_client, ledger=ledger)))

    manifest = client.get("/runtime/manifest").json()
    for key, expected_value in fixture["required_support_levels"].items():
        assert manifest[key] == expected_value
    assert manifest["governance_summary"]["environment"] == "local_development"
    enforcement_summary = manifest["external_approval_enforcement_summary"]
    assert enforcement_summary["actor_exempt"] is True
    assert enforcement_summary["audit_only_rule_ids"] == ["draft_action_external_approval_gate"]

    plan_payload, studio_step = _prepare_user_owned_studio_execution(client)
    execution = client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )
    assert execution.status_code == 200
    assert execution.json()["action_result"]["execution_status"] == "succeeded"

    ledger_payload = client.get(f"/runs/{plan_payload['run_id']}/ledger").json()
    support_bundle_payload = client.get(f"/runs/{plan_payload['run_id']}/support-bundle").json()
    serialized_evidence = json.dumps(
        {"ledger": ledger_payload, "support_bundle": support_bundle_payload},
        sort_keys=True,
    )
    for unsafe_term in [
        "C:\\",
        "/Users/",
        "http://",
        "https://",
        "OPENAI_API_KEY",
        "sk-test",
        "paste-your-openai-api-key",
        "do-not-ledger",
        "\"links\"",
        "\"query\"",
    ]:
        assert unsafe_term not in serialized_evidence
    loader.validate_agent_contract_payload(ledger_payload, "agent_execution_ledger.v1.schema.json")
    loader.validate_agent_contract_payload(support_bundle_payload, "agent_support_bundle.v1.schema.json")


def test_phase9_certification_team_staging_requires_approval_before_executor_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = InMemoryLedger()
    app_client = FakePreflightAppClient()
    approver_client = _client_for_governance_environment(
        monkeypatch,
        ledger=ledger,
        app_client=app_client,
        environment="team_staging",
        actor_role="approver",
        actor_id="approval_actor",
    )

    plan_payload = _create_user_owned_plan(approver_client)
    _check_user_owned_readiness(approver_client, plan_payload)
    _approve_user_plan(approver_client, plan_payload)
    _approve_user_owned_consent(approver_client, plan_payload)

    executor_client = _client_for_governance_environment(
        monkeypatch,
        ledger=ledger,
        app_client=app_client,
        environment="team_staging",
        actor_role="executor",
        actor_id="execution_actor",
    )
    _run_source_preflight(executor_client, plan_payload)

    studio_step = _step_for_capability(plan_payload, "quant_studio.prepare_model_config_draft")
    approver_client = _client_for_governance_environment(
        monkeypatch,
        ledger=ledger,
        app_client=app_client,
        environment="team_staging",
        actor_role="approver",
        actor_id="approval_actor",
    )
    confirmation = approver_client.post(
        "/confirmations",
        json={
            "run_id": plan_payload["run_id"],
            "step_id": studio_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    )
    assert confirmation.status_code == 200
    action_request = executor_client.post(
        "/action-requests",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )
    assert action_request.status_code == 200

    denied_before_approval = executor_client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )
    assert denied_before_approval.status_code == 422
    assert denied_before_approval.json()["detail"]["errors"][0]["code"] == "external_approval_required"
    assert app_client.execution_calls == []

    approver_client = _client_for_governance_environment(
        monkeypatch,
        ledger=ledger,
        app_client=app_client,
        environment="team_staging",
        actor_role="approver",
        actor_id="approval_actor",
    )
    approval_package = approver_client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "step",
            "step_id": studio_step["step_id"],
        },
    )
    assert approval_package.status_code == 200
    decision = _external_approval_decision(approval_package.json()["approval_request"])
    imported_decision = approver_client.post(
        "/external-approval-decisions",
        json={
            "run_id": plan_payload["run_id"],
            "decision_intent": "import_external_approval_decision",
            "approval_decision": decision,
        },
    )
    assert imported_decision.status_code == 200

    executor_client = _client_for_governance_environment(
        monkeypatch,
        ledger=ledger,
        app_client=app_client,
        environment="team_staging",
        actor_role="executor",
        actor_id="execution_actor",
    )
    execution = executor_client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )
    assert execution.status_code == 200
    assert execution.json()["action_result"]["execution_status"] == "succeeded"
    assert len(app_client.execution_calls) == 1

    status_payload = executor_client.get(f"/runs/{plan_payload['run_id']}").json()
    assert status_payload["governance_summary"]["policy_pack_id"] == (
        "quant_agent_team_staging_governance_policy_pack_v1"
    )
    assert status_payload["external_approval_enforcement_summary"]["latest_decision"][
        "approval_decision_status"
    ] == "approved"

    viewer_client = _client_for_governance_environment(
        monkeypatch,
        ledger=ledger,
        app_client=app_client,
        environment="team_staging",
        actor_role="viewer",
        actor_id="audit_viewer",
    )
    support_bundle = viewer_client.get(f"/runs/{plan_payload['run_id']}/support-bundle")
    assert support_bundle.status_code == 200
    serialized_support_bundle = json.dumps(support_bundle.json(), sort_keys=True)
    for unsafe_term in [
        "C:\\",
        "/Users/",
        "http://",
        "https://",
        "OPENAI_API_KEY",
        "sk-test",
        "paste-your-openai-api-key",
        "do-not-ledger",
        "\"links\"",
        "\"query\"",
    ]:
        assert unsafe_term not in serialized_support_bundle
    assert "external_approval_enforcement_denied" in serialized_support_bundle
    assert "external_approval_decision_import" in serialized_support_bundle
    QuantSuiteContractLoader(QUANT_SUITE_ROOT).validate_agent_contract_payload(
        support_bundle.json(),
        "agent_support_bundle.v1.schema.json",
    )


def test_phase9_certification_regulated_review_viewer_is_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = InMemoryLedger()
    app_client = FakePreflightAppClient()
    local_client = _client_for_governance_environment(
        monkeypatch,
        ledger=ledger,
        app_client=app_client,
        environment="local_development",
        actor_role="local_developer_operator",
        actor_id="local_user",
    )
    plan_payload = _create_user_owned_plan(local_client)

    viewer_client = _client_for_governance_environment(
        monkeypatch,
        ledger=ledger,
        app_client=app_client,
        environment="regulated_review",
        actor_role="viewer",
        actor_id="regulated_viewer",
    )
    run_status = viewer_client.get(f"/runs/{plan_payload['run_id']}")
    assert run_status.status_code == 200
    assert run_status.json()["governance_summary"]["policy_pack_id"] == (
        "quant_agent_regulated_review_governance_policy_pack_v1"
    )
    support_bundle = viewer_client.get(f"/runs/{plan_payload['run_id']}/support-bundle")
    assert support_bundle.status_code == 200

    denied_pause = viewer_client.post(
        "/pauses",
        json={
            "run_id": plan_payload["run_id"],
            "pause_intent": "pause_run",
            "reason": "user_paused",
        },
    )
    assert denied_pause.status_code == 422
    assert denied_pause.json()["detail"]["errors"][0]["code"] == "governance_permission_denied"
    ledger_payload = viewer_client.get(f"/runs/{plan_payload['run_id']}/ledger").json()
    denial = ledger_payload["recovery_events"][-1]
    assert denial["event_type"] == "governance_permission_denied"
    assert denial["denied_route"] == "POST /pauses"
    assert denial["execution_permitted"] is False
    serialized_ledger = json.dumps(ledger_payload, sort_keys=True)
    for unsafe_term in [
        "C:\\",
        "/Users/",
        "http://",
        "https://",
        "OPENAI_API_KEY",
        "sk-test",
        "paste-your-openai-api-key",
        "do-not-ledger",
        "\"links\"",
        "\"query\"",
    ]:
        assert unsafe_term not in serialized_ledger


def test_phase9_certification_sod_denial_precedes_external_approval_enforcement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_path = tmp_path / "governance_policy_pack.json"
    _write_governance_policy_pack(
        policy_path,
        role_id="workflow_operator",
        separation_of_duties_rules=[_blocking_sod_rule()],
        external_approval_rules=[_blocking_external_approval_rule()],
    )
    monkeypatch.setenv("QUANT_AGENT_GOVERNANCE_POLICY_PACK_PATH", str(policy_path))
    monkeypatch.setenv("QUANT_AGENT_ACTOR_ROLE", "workflow_operator")
    monkeypatch.setenv("QUANT_AGENT_ACTOR_ID", "same_actor")
    ledger = InMemoryLedger()
    app_client = FakePreflightAppClient()
    client = TestClient(create_app(runtime_with_preflight_client(app_client, ledger=ledger)))
    plan_payload, studio_step = _prepare_user_owned_studio_execution(client)

    denied = client.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )
    assert denied.status_code == 422
    issue = denied.json()["detail"]["errors"][0]
    assert issue["code"] == "governance_separation_of_duties_denied"
    assert app_client.execution_calls == []
    ledger_payload = client.get(f"/runs/{plan_payload['run_id']}/ledger").json()
    denial = ledger_payload["recovery_events"][-1]
    assert denial["event_type"] == "governance_separation_of_duties_denied"
    assert denial["event_type"] != "external_approval_enforcement_denied"
    assert denial["execution_permitted"] is False


def test_phase10_certification_local_outbox_chain_and_exports_are_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _expected_phase10_external_approval_adapter_fixture()
    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_ADAPTER", "local_outbox")
    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_OUTBOX_DIR", str(tmp_path / "outbox"))
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path / "ledgers",
        validate_contract=loader.validate_agent_contract_payload,
    )
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient(), ledger=ledger)))

    manifest = client.get("/runtime/manifest").json()
    for key, expected_value in fixture["required_support_levels"].items():
        assert manifest[key] == expected_value
    assert manifest["external_approval_submission_adapter"]["adapter_mode"] == "local_outbox"

    plan_payload = _create_plan_with_lifecycle_reference(client)
    preview = client.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "run",
            "step_id": None,
        },
    )
    assert preview.status_code == 200
    approval_request = preview.json()["approval_request"]
    submitted = client.post(
        "/external-approval-submissions",
        json={
            "run_id": plan_payload["run_id"],
            "submission_intent": "submit_external_approval_request",
            "approval_request_id": approval_request["approval_request_id"],
        },
    )
    assert submitted.status_code == 200
    status = client.get(f"/runs/{plan_payload['run_id']}/external-approval-submissions")
    assert status.status_code == 200
    assert status.json()["submissions"][0]["adapter_mode"] == "local_outbox"
    assert status.json()["submissions"][0]["outbox_status"] == "present"

    decision = _external_approval_decision(approval_request)
    imported = client.post(
        "/external-approval-decisions",
        json={
            "run_id": plan_payload["run_id"],
            "decision_intent": "import_external_approval_decision",
            "approval_decision": decision,
        },
    )
    assert imported.status_code == 200
    linked_status = client.get(f"/runs/{plan_payload['run_id']}/external-approval-submissions").json()
    assert linked_status["submissions"][0]["latest_matching_decision"]["decision_status"] == "approved"
    assert linked_status["submissions"][0]["latest_matching_decision"]["submission_status"] == "submitted"

    ledger_payload = client.get(f"/runs/{plan_payload['run_id']}/ledger").json()
    support_bundle = client.get(f"/runs/{plan_payload['run_id']}/support-bundle").json()
    loader.validate_agent_contract_payload(ledger_payload, "agent_execution_ledger.v1.schema.json")
    loader.validate_agent_contract_payload(support_bundle, "agent_support_bundle.v1.schema.json")
    serialized = json.dumps(
        {
            "manifest": manifest,
            "submission": submitted.json(),
            "status": linked_status,
            "ledger": ledger_payload,
            "support_bundle": support_bundle,
        },
        sort_keys=True,
    )
    for unsafe_term in [
        "C:\\",
        "/Users/",
        "http://",
        "https://",
        "OPENAI_API_KEY",
        "sk-test",
        "paste-your-openai-api-key",
        "Authorization",
        "\"headers\"",
        "\"raw_response\"",
        "\"app_payload\"",
        "\"links\"",
        "\"query\"",
    ]:
        assert unsafe_term not in serialized


def test_phase10_certification_mock_http_refresh_and_enforcement_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _expected_phase10_external_approval_adapter_fixture()
    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_ADAPTER", "mock_http")
    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_MOCK_BASE_URL", "http://127.0.0.1:8895")
    monkeypatch.setenv("QUANT_AGENT_EXTERNAL_APPROVAL_TIMEOUT_SECONDS", "6")
    loader = QuantSuiteContractLoader(QUANT_SUITE_ROOT)
    ledger = FileBackedLedger(
        tmp_path / "ledgers",
        validate_contract=loader.validate_agent_contract_payload,
    )
    app_client = FakePreflightAppClient()
    approval_requests_by_id: dict[str, dict[str, Any]] = {}
    decision_status_by_request_id: dict[str, str] = {}
    refresh_counts_by_request_id: dict[str, int] = {}
    captured_adapter_calls: list[dict[str, Any]] = []

    class MockResponse:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def __enter__(self) -> "MockResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(self._payload, sort_keys=True).encode("utf-8")

    def fake_urlopen(request: Any, timeout: int) -> MockResponse:
        body = json.loads(request.data.decode("utf-8"))
        captured_adapter_calls.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "body_keys": sorted(body.keys()),
            }
        )
        if request.full_url.endswith("/api/external-approval/submissions"):
            submission = body["submission"]
            return MockResponse(
                {
                    "accepted": True,
                    "external_reference_id": f"mock_ref_{submission['approval_request_id']}",
                    "status": "submitted",
                    "received_at_utc": "2026-06-23T17:00:00Z",
                    "warnings": [],
                }
            )
        lookup = body["decision_lookup"]
        approval_request_id = str(lookup["approval_request_id"])
        refresh_count = refresh_counts_by_request_id.get(approval_request_id, 0) + 1
        refresh_counts_by_request_id[approval_request_id] = refresh_count
        if refresh_count == 1:
            return MockResponse(
                {
                    "decision_available": False,
                    "status": "pending",
                    "checked_at_utc": "2026-06-23T17:01:00Z",
                    "warnings": ["Decision is pending in the mock adapter."],
                }
            )
        decision_status = decision_status_by_request_id[approval_request_id]
        return MockResponse(
            {
                "decision_available": True,
                "status": "decision_available",
                "approval_decision": _external_approval_decision(
                    approval_requests_by_id[approval_request_id],
                    decision_status=decision_status,
                    decision_id=f"mock_refresh_{approval_request_id}_{decision_status}",
                ),
                "checked_at_utc": "2026-06-23T17:02:00Z",
                "warnings": [f"Decision is {decision_status} in the mock adapter."],
            }
        )

    monkeypatch.setattr(external_approval_module.urllib.request, "urlopen", fake_urlopen)

    def client_for(role: str, actor_id: str) -> TestClient:
        return _client_for_governance_environment(
            monkeypatch,
            ledger=ledger,
            app_client=app_client,
            environment="team_staging",
            actor_role=role,
            actor_id=actor_id,
        )

    def prepare_governed_studio_step(label: str) -> tuple[dict[str, Any], dict[str, Any], TestClient, TestClient]:
        approver = client_for("approver", f"approver_{label}")
        plan_payload = _create_user_owned_plan(approver)
        _check_user_owned_readiness(approver, plan_payload)
        _approve_user_plan(approver, plan_payload)
        _approve_user_owned_consent(approver, plan_payload)
        preflight_executor = client_for("executor", f"preflight_executor_{label}")
        _run_source_preflight(preflight_executor, plan_payload)
        studio_step = _step_for_capability(plan_payload, "quant_studio.prepare_model_config_draft")
        confirmation = approver.post(
            "/confirmations",
            json={
                "run_id": plan_payload["run_id"],
                "step_id": studio_step["step_id"],
                "confirmation_intent": "approve_plan_step",
            },
        )
        assert confirmation.status_code == 200
        preview_executor = client_for("executor", f"preview_executor_{label}")
        action_request = preview_executor.post(
            "/action-requests",
            json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
        )
        assert action_request.status_code == 200
        execution_executor = client_for("executor", f"execution_executor_{label}")
        denied = execution_executor.post(
            "/executions",
            json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
        )
        assert denied.status_code == 422
        assert denied.json()["detail"]["errors"][0]["code"] == "external_approval_required"
        return plan_payload, studio_step, approver, execution_executor

    manifest = client_for("approver", "cert_manifest_actor").get("/runtime/manifest").json()
    for key, expected_value in fixture["required_support_levels"].items():
        assert manifest[key] == expected_value
    assert manifest["external_approval_submission_adapter"]["adapter_mode"] == "mock_http"
    assert manifest["external_approval_submission_adapter"]["timeout_seconds"] == 6

    plan_payload, studio_step, approver, executor = prepare_governed_studio_step("approved")
    package = approver.post(
        "/external-approval-requests",
        json={
            "run_id": plan_payload["run_id"],
            "approval_intent": "preview_external_approval_request",
            "approval_scope": "step",
            "step_id": studio_step["step_id"],
        },
    )
    assert package.status_code == 200
    approval_request = package.json()["approval_request"]
    approval_requests_by_id[approval_request["approval_request_id"]] = approval_request
    decision_status_by_request_id[approval_request["approval_request_id"]] = "approved"
    submitted = approver.post(
        "/external-approval-submissions",
        json={
            "run_id": plan_payload["run_id"],
            "submission_intent": "submit_external_approval_request",
            "approval_request_id": approval_request["approval_request_id"],
        },
    )
    assert submitted.status_code == 200
    pending = approver.post(
        "/external-approval-decision-refreshes",
        json={
            "run_id": plan_payload["run_id"],
            "decision_refresh_intent": "refresh_external_approval_decision",
            "approval_request_id": approval_request["approval_request_id"],
        },
    )
    assert pending.status_code == 200
    assert pending.json()["decision_refresh"]["status"] == "pending"
    assert pending.json()["approval_decision"] is None
    assert executor.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    ).status_code == 422
    available = approver.post(
        "/external-approval-decision-refreshes",
        json={
            "run_id": plan_payload["run_id"],
            "decision_refresh_intent": "refresh_external_approval_decision",
            "approval_request_id": approval_request["approval_request_id"],
        },
    )
    assert available.status_code == 200
    assert available.json()["approval_decision"]["decision_status"] == "approved"
    execution = executor.post(
        "/executions",
        json={"run_id": plan_payload["run_id"], "step_id": studio_step["step_id"]},
    )
    assert execution.status_code == 200
    assert execution.json()["action_result"]["execution_status"] == "succeeded"

    for decision_status in ["rejected", "needs_changes", "expired"]:
        blocked_plan, blocked_step, blocked_approver, blocked_executor = prepare_governed_studio_step(decision_status)
        blocked_package = blocked_approver.post(
            "/external-approval-requests",
            json={
                "run_id": blocked_plan["run_id"],
                "approval_intent": "preview_external_approval_request",
                "approval_scope": "step",
                "step_id": blocked_step["step_id"],
            },
        )
        assert blocked_package.status_code == 200
        blocked_request = blocked_package.json()["approval_request"]
        approval_requests_by_id[blocked_request["approval_request_id"]] = blocked_request
        decision_status_by_request_id[blocked_request["approval_request_id"]] = decision_status
        blocked_submission = blocked_approver.post(
            "/external-approval-submissions",
            json={
                "run_id": blocked_plan["run_id"],
                "submission_intent": "submit_external_approval_request",
                "approval_request_id": blocked_request["approval_request_id"],
            },
        )
        assert blocked_submission.status_code == 200
        assert blocked_approver.post(
            "/external-approval-decision-refreshes",
            json={
                "run_id": blocked_plan["run_id"],
                "decision_refresh_intent": "refresh_external_approval_decision",
                "approval_request_id": blocked_request["approval_request_id"],
            },
        ).status_code == 200
        refreshed = blocked_approver.post(
            "/external-approval-decision-refreshes",
            json={
                "run_id": blocked_plan["run_id"],
                "decision_refresh_intent": "refresh_external_approval_decision",
                "approval_request_id": blocked_request["approval_request_id"],
            },
        )
        assert refreshed.status_code == 200
        assert refreshed.json()["approval_decision"]["decision_status"] == decision_status
        blocked_execution = blocked_executor.post(
            "/executions",
            json={"run_id": blocked_plan["run_id"], "step_id": blocked_step["step_id"]},
        )
        assert blocked_execution.status_code == 422
        assert blocked_execution.json()["detail"]["errors"][0]["code"] == (
            "external_approval_decision_denied"
        )

    viewer = client_for("viewer", "cert_viewer")
    status_payload = viewer.get(f"/runs/{plan_payload['run_id']}/external-approval-submissions").json()
    ledger_payload = viewer.get(f"/runs/{plan_payload['run_id']}/ledger").json()
    support_bundle = viewer.get(f"/runs/{plan_payload['run_id']}/support-bundle").json()
    assert status_payload["submissions"][0]["latest_decision_refresh"]["status"] == "decision_available"
    assert status_payload["submissions"][0]["latest_matching_decision"]["decision_status"] == "approved"
    loader.validate_agent_contract_payload(ledger_payload, "agent_execution_ledger.v1.schema.json")
    loader.validate_agent_contract_payload(support_bundle, "agent_support_bundle.v1.schema.json")
    serialized = json.dumps(
        {
            "manifest": manifest,
            "status": status_payload,
            "ledger": ledger_payload,
            "support_bundle": support_bundle,
            "pending": pending.json(),
            "available": available.json(),
        },
        sort_keys=True,
    )
    for unsafe_term in [
        "C:\\",
        "/Users/",
        "http://",
        "https://",
        "OPENAI_API_KEY",
        "sk-test",
        "paste-your-openai-api-key",
        "Authorization",
        "\"headers\"",
        "\"raw_response\"",
        "\"provider_response\"",
        "\"app_payload\"",
        "\"links\"",
        "\"query\"",
    ]:
        assert unsafe_term not in serialized
    assert {call["body_keys"][0] for call in captured_adapter_calls} == {"decision_lookup", "submission"}


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


def test_user_requested_plan_revision_from_revise_review_activates_child_with_fresh_gates() -> None:
    runtime = runtime_with_preflight_client(FakePreflightAppClient())
    client = TestClient(create_app(runtime))
    parent_plan = _create_user_owned_plan(client)
    parent_run_id = parent_plan["run_id"]
    parent_plan_id = parent_plan["plan"]["plan_id"]
    parent_entry = runtime.planner.ledger.get(parent_run_id)
    assert parent_entry is not None
    original_parent_plan = copy.deepcopy(parent_entry.plan_snapshot)

    _check_user_owned_readiness(client, parent_plan)
    revision_review = _review_user_plan(
        client,
        parent_plan,
        decision="revise",
        safe_note="Revise this assumption using the refreshed user-owned summary evidence.",
    )
    assert revision_review["plan_review_summary"]["status"] == "revision_requested"
    parent_status_after_review = client.get(f"/runs/{parent_run_id}")
    assert parent_status_after_review.status_code == 200
    assert parent_status_after_review.json()["allowed_user_owned_actions"] == ["revise_plan"]

    blocked_approval = client.post(
        "/user-plan-approvals",
        json={
            "run_id": parent_run_id,
            "approval_intent": "approve_user_plan",
            "plan_review_id": revision_review["plan_review_summary"]["plan_review_id"],
        },
    )
    assert blocked_approval.status_code == 422
    assert blocked_approval.json()["detail"]["errors"][0]["code"] == "user_plan_revision_requested"
    blocked_consent = client.post(
        "/user-workflow-consents",
        json={
            "run_id": parent_run_id,
            "consent_intent": "approve_user_owned_guided_execution",
            "consent_scope": "single_run_review_draft_actions",
        },
    )
    assert blocked_consent.status_code == 422
    assert blocked_consent.json()["detail"]["errors"][0]["code"] == "user_plan_approval_required"

    revision_request = {
        "run_id": parent_run_id,
        "revision_intent": "revise_plan",
        "reason": "user_requested",
        "current_context_summary": _safe_user_owned_lifecycle_context(),
    }
    revision = client.post("/plan-revisions", json=revision_request)
    duplicate_revision = client.post("/plan-revisions", json=revision_request)
    assert revision.status_code == 200
    assert duplicate_revision.status_code == 200
    revision_payload = revision.json()
    assert duplicate_revision.json()["revision_id"] == revision_payload["revision_id"]
    assert revision_payload["parent_plan_id"] == parent_plan_id
    requested = revision_payload["revision_event"]["blocker_summary"]["requested_assumption_revisions"]
    assert requested["status"] == "revision_requested"
    assert requested["plan_review_id"] == revision_review["plan_review_summary"]["plan_review_id"]
    assert requested["revise_assumption_count"] == len(parent_plan["plan"]["assumptions"])
    assert requested["revision_notes"][0]["safe_note"] == (
        "Revise this assumption using the refreshed user-owned summary evidence."
    )
    assert revision_payload["revised_plan"]["revision_source_run_id"] == parent_run_id
    assert revision_payload["revision_event"]["execution_permitted"] is False
    assert "raw_path" not in revision.text

    activation = client.post(
        "/plan-revision-activations",
        json={
            "run_id": parent_run_id,
            "revision_id": revision_payload["revision_id"],
            "activation_intent": "activate_plan_revision",
        },
    )
    assert activation.status_code == 200
    activation_payload = activation.json()
    child_run_id = activation_payload["child_run_id"]
    assert child_run_id != parent_run_id
    assert activation_payload["activation_event"]["active_plan_replaced"] is False
    assert activation_payload["child_orchestration"]["parent_run_id"] == parent_run_id

    parent_status = client.get(f"/runs/{parent_run_id}")
    child_status = client.get(f"/runs/{child_run_id}")
    assert parent_status.status_code == 200
    assert parent_status.json()["plan"] == original_parent_plan
    assert parent_status.json()["plan_review_summary"]["status"] == "revision_requested"
    assert parent_status.json()["plan_approval_summary"]["status"] == "not_approved"
    assert child_status.status_code == 200
    assert child_status.json()["parent_run_id"] == parent_run_id
    assert child_status.json()["plan"]["plan_id"] == revision_payload["revised_plan"]["plan_id"]
    assert child_status.json()["plan_review_summary"]["status"] == "not_reviewed"
    assert child_status.json()["plan_approval_summary"]["status"] == "not_approved"
    assert child_status.json()["readiness_summary"]["status"] == "not_checked"
    assert child_status.json()["consent_summary"]["status"] == "not_recorded"

    child_preflight_step = _step_for_capability(
        child_status.json(),
        "quant_data.run_source_preflight",
    )
    child_preflight_before_gates = client.post(
        "/preflights",
        json={"run_id": child_run_id, "step_id": child_preflight_step["step_id"]},
    )
    assert child_preflight_before_gates.status_code == 422
    assert child_preflight_before_gates.json()["detail"]["errors"][0]["code"] == "user_workflow_readiness_required"

    child_plan = {"run_id": child_run_id, "plan": child_status.json()["plan"]}
    _check_user_owned_readiness(client, child_plan)
    child_review = _review_user_plan(client, child_plan)
    child_approval = _approve_user_plan(client, child_plan, child_review)
    assert child_approval["plan_approval_summary"]["status"] == "approved"
    _approve_user_owned_consent(client, child_plan)
    child_preflight = client.post(
        "/preflights",
        json={"run_id": child_run_id, "step_id": child_preflight_step["step_id"]},
    )
    assert child_preflight.status_code == 200

    recorded_parent = runtime.planner.ledger.get(parent_run_id)
    recorded_child = runtime.planner.ledger.get(child_run_id)
    assert recorded_parent is not None
    assert recorded_child is not None
    runtime.contract_loader.validate_agent_contract_payload(
        recorded_parent.model_dump(mode="json"),
        "agent_execution_ledger.v1.schema.json",
    )
    runtime.contract_loader.validate_agent_contract_payload(
        recorded_child.model_dump(mode="json"),
        "agent_execution_ledger.v1.schema.json",
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
    assert discovery["supported_preflight_capabilities"] == [
        "quant_data.run_source_preflight",
        "quant_data.run_eda_review",
        "quant_studio.run_model_readiness_check",
        "quant_studio.fit_candidate_model",
    ]
    assert discovery["supported_execution_capabilities"] == [
        "quant_studio.prepare_model_config_draft",
        "quant_documentation.inspect_package",
        "quant_documentation.create_draft_workspace",
        "quant_data.register_source_reference",
        "quant_data.create_eda_plan",
        "quant_data.run_eda_review",
        "quant_data.export_eda_handoff",
        "quant_studio.fit_candidate_model",
        "quant_studio.compare_candidate_runs",
        "quant_studio.create_documentation_package",
        "quant_documentation.draft_section",
        "quant_documentation.find_unsupported_claims",
        "quant_documentation.export_markdown_review_package",
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
    assert payload["supported_preflight_capabilities"] == [
        "quant_monitoring.validate_bundle",
        "quant_studio.run_model_readiness_check",
        "quant_studio.fit_candidate_model",
        "quant_monitoring.run_monitoring_review",
    ]
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
    assert discovery["supported_preflight_capabilities"] == [
        "quant_monitoring.validate_bundle",
        "quant_studio.run_model_readiness_check",
        "quant_studio.fit_candidate_model",
        "quant_monitoring.run_monitoring_review",
    ]
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
    assert discovery["supported_preflight_capabilities"] == [
        "quant_studio.run_model_readiness_check",
        "quant_studio.fit_candidate_model",
    ]
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
    assert discovery["supported_preflight_capabilities"] == [
        "quant_monitoring.validate_bundle",
        "quant_studio.run_model_readiness_check",
        "quant_studio.fit_candidate_model",
        "quant_monitoring.run_monitoring_review",
    ]
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
    assert payload["run_state"] == "ready_for_execution_preview"
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
    assert action_request["action_input"]["package_summary"] == documentation_step["action_input"]["package_summary"]
    assert action_request["action_input"]["documentation_package_summary"] == {
        "reference_type": "documentation_package_summary",
        "reference_id": "documentation_package_summary_test",
        "label": "Documentation package summary",
        "stored_in": "quant_agent_ledger",
    }
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
    assert len(entry.action_results) == 3
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
    assert len(entry.action_requests) == 6
    assert len(entry.action_results) == 3


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
    assert studio_completed_status.json()["run_state"] == "ready_for_execution_preview"
    assert "preview_action_request" in studio_completed_status.json()["allowed_next_actions"]
    assert studio_completed_status.json()["latest_action_result"]["execution_status"] == "succeeded"

    _complete_documentation_inspection_step(client, plan_payload)
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
    assert payload["run_state"] == "ready_for_execution_preview"
    assert payload["orchestration"]["run_state"] == "ready_for_execution_preview"
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


def test_demo_narrative_follows_expected_sample_fixture_and_reset_state() -> None:
    responses_by_capability = {
        "quant_monitoring.validate_bundle": _valid_preflight_response(
            capability_id="quant_monitoring.validate_bundle",
            app_id="quant_monitoring",
        )
    }
    app_client = FakePreflightAppClient(responses_by_capability=responses_by_capability)
    runtime = runtime_with_preflight_client(app_client)
    client = TestClient(create_app(runtime))
    expected = _expected_demo_narrative_fixture()
    plan = client.post(
        "/plans",
        json={
            "user_goal": "Run the replayable Credit PD sample demo path.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()

    initial = client.get(f"/runs/{plan['run_id']}/demo-narrative")
    assert initial.status_code == 200
    initial_payload = initial.json()
    assert initial_payload["demo_status"] == "in_progress"
    assert [section["section_id"] for section in initial_payload["narrative_sections"]] == (
        expected["expected_section_ids"]
    )
    assert initial_payload["sample_eligibility"]["sample_workspace_id"] == expected["sample_workspace_id"]

    _run_source_preflight(client, plan)
    _complete_studio_step(client, plan)
    _complete_documentation_step(client, plan)
    monitoring_step = _step_for_capability(plan, "quant_monitoring.validate_bundle")
    monitoring_preflight = client.post(
        "/preflights",
        json={"run_id": plan["run_id"], "step_id": monitoring_step["step_id"]},
    )
    assert monitoring_preflight.status_code == 200

    completed = client.get(f"/runs/{plan['run_id']}/demo-narrative")
    assert completed.status_code == 200
    completed_payload = completed.json()
    assert completed_payload["demo_status"] == "completed"
    assert completed_payload["orchestration"]["run_state"] == "completed"
    assert completed_payload["run_progress_summary"]["completed_steps"] == 5
    assert completed_payload["run_progress_summary"]["informational_steps"] == 0
    assert completed_payload["safety_summary"]["ledger_record_counts"]["preflight_records"] == 2
    assert completed_payload["safety_summary"]["ledger_record_counts"]["action_results"] == 3

    reset_preview = client.post(
        "/sample-reset-previews",
        json={
            "run_id": plan["run_id"],
            "reset_intent": "preview_sample_reset",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    reset = client.post(
        "/sample-resets",
        json={
            "run_id": plan["run_id"],
            "reset_intent": "reset_sample_demo",
            "reset_preview_id": reset_preview["reset_preview_id"],
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )
    assert reset.status_code == 200

    narrative = client.get(f"/runs/{plan['run_id']}/demo-narrative")
    assert narrative.status_code == 200
    payload = narrative.json()
    serialized = json.dumps(payload, sort_keys=True)
    assert payload["demo_status"] == "sample_reset"
    assert payload["orchestration"]["run_state"] == "sample_reset"
    assert [section["section_id"] for section in payload["narrative_sections"]] == (
        expected["expected_section_ids"]
    )
    sections = {section["section_id"]: section for section in payload["narrative_sections"]}
    assert sections["data_preflight"]["status"] == "ready"
    assert sections["studio_draft"]["status"] == "succeeded"
    assert sections["documentation_draft"]["status"] == "succeeded"
    assert sections["monitoring_preflight"]["status"] == "ready"
    assert sections["sample_reset"]["status"] == "sample_reset"
    assert sections["safety_boundaries"]["status"] == "enforced"
    for key, value in expected["expected_safety_summary"].items():
        assert payload["safety_summary"][key] == value
    for key in expected["expected_ledger_summary_fields"]:
        assert key in payload["ledger_summary"]
    for term in expected["forbidden_terms"]:
        assert term not in serialized

    entry = runtime.planner.ledger.get(plan["run_id"])
    assert entry is not None
    runtime.contract_loader.validate_agent_contract_payload(
        entry.model_dump(mode="json"),
        "agent_execution_ledger.v1.schema.json",
    )


def test_sample_demo_certification_path_is_repeatable_after_reset() -> None:
    responses_by_capability = {
        "quant_monitoring.validate_bundle": _valid_preflight_response(
            capability_id="quant_monitoring.validate_bundle",
            app_id="quant_monitoring",
        )
    }
    app_client = FakePreflightAppClient(responses_by_capability=responses_by_capability)
    runtime = runtime_with_preflight_client(app_client)
    client = TestClient(create_app(runtime))
    expected = _expected_demo_certification_fixture()

    first_plan = client.post(
        "/plans",
        json={
            "user_goal": "Certify the repeatable Credit PD sample demo path.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    first_capability_order = [
        step["capability_id"]
        for step in first_plan["plan"]["proposed_steps"]
    ]
    assert first_capability_order == expected["expected_capability_order"]

    _run_source_preflight(client, first_plan)
    studio_step, _ = _complete_studio_step(client, first_plan)
    documentation_step, _ = _complete_documentation_step(client, first_plan)
    _run_monitoring_preflight(client, first_plan)

    completed_narrative = client.get(f"/runs/{first_plan['run_id']}/demo-narrative")
    assert completed_narrative.status_code == 200
    assert completed_narrative.json()["demo_status"] == "completed"

    reset_preview = client.post(
        "/sample-reset-previews",
        json={
            "run_id": first_plan["run_id"],
            "reset_intent": "preview_sample_reset",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    reset = client.post(
        "/sample-resets",
        json={
            "run_id": first_plan["run_id"],
            "reset_intent": "reset_sample_demo",
            "reset_preview_id": reset_preview["reset_preview_id"],
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )
    assert reset.status_code == 200
    assert reset.json()["reset_status"] == "reset"

    first_entry = runtime.planner.ledger.get(first_plan["run_id"])
    assert first_entry is not None
    first_ledger = first_entry.model_dump(mode="json")
    assert first_ledger["final_status"] == expected["expected_terminal_final_status"]
    for collection_name, minimum_count in expected["minimum_ledger_record_counts"].items():
        assert len(first_ledger.get(collection_name, [])) >= minimum_count
    serialized_ledger = json.dumps(first_ledger, sort_keys=True)
    for term in expected["forbidden_terms"]:
        assert term not in serialized_ledger
    runtime.contract_loader.validate_agent_contract_payload(
        first_ledger,
        "agent_execution_ledger.v1.schema.json",
    )
    reset_record_counts = {
        name: len(first_ledger.get(name, []))
        for name in (
            "preflight_records",
            "confirmation_records",
            "action_requests",
            "action_results",
            "cancellation_events",
        )
    }

    old_run_preflight = client.post(
        "/preflights",
        json={
            "run_id": first_plan["run_id"],
            "step_id": _step_for_capability(first_plan, "quant_monitoring.validate_bundle")["step_id"],
        },
    )
    old_run_confirmation = client.post(
        "/confirmations",
        json={
            "run_id": first_plan["run_id"],
            "step_id": studio_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    )
    old_run_preview = client.post(
        "/action-requests",
        json={"run_id": first_plan["run_id"], "step_id": documentation_step["step_id"]},
    )
    old_run_execution = client.post(
        "/executions",
        json={"run_id": first_plan["run_id"], "step_id": documentation_step["step_id"]},
    )
    old_run_autopilot = client.post(
        "/autopilot-steps",
        json={
            "run_id": first_plan["run_id"],
            "autopilot_intent": "advance_sample_autopilot_one_step",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )
    old_run_revision = client.post(
        "/plan-revisions",
        json={
            "run_id": first_plan["run_id"],
            "revision_intent": "revise_plan",
            "reason": "user_requested",
            "current_context_summary": _safe_sample_lifecycle_context(),
        },
    )
    blocked_statuses = {
        "POST /preflights": old_run_preflight.status_code,
        "POST /confirmations": old_run_confirmation.status_code,
        "POST /action-requests": old_run_preview.status_code,
        "POST /executions": old_run_execution.status_code,
        "POST /autopilot-steps": old_run_autopilot.status_code,
        "POST /plan-revisions": old_run_revision.status_code,
    }
    assert set(blocked_statuses) == set(expected["old_run_non_mutating_routes_after_reset"])
    assert all(status in {200, 422} for status in blocked_statuses.values())
    post_reset_entry = runtime.planner.ledger.get(first_plan["run_id"])
    assert post_reset_entry is not None
    assert post_reset_entry.final_status == "sample_reset"
    post_reset_ledger = post_reset_entry.model_dump(mode="json")
    assert {
        name: len(post_reset_ledger.get(name, []))
        for name in reset_record_counts
    } == reset_record_counts
    runtime.contract_loader.validate_agent_contract_payload(
        post_reset_ledger,
        "agent_execution_ledger.v1.schema.json",
    )

    reset_narrative = client.get(f"/runs/{first_plan['run_id']}/demo-narrative").json()
    assert reset_narrative["demo_status"] == "sample_reset"
    assert reset_narrative["orchestration"]["run_state"] == "sample_reset"

    second_plan = client.post(
        "/plans",
        json={
            "user_goal": "Run the Credit PD sample demo again after reset.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    assert second_plan["run_id"] != first_plan["run_id"]
    assert [
        step["capability_id"]
        for step in second_plan["plan"]["proposed_steps"]
    ] == expected["expected_capability_order"]
    assert runtime.planner.ledger.get(first_plan["run_id"]) is not None
    second_entry = runtime.planner.ledger.get(second_plan["run_id"])
    assert second_entry is not None
    assert second_entry.final_status == expected["second_run_expectations"]["final_status"]

    second_narrative = client.get(f"/runs/{second_plan['run_id']}/demo-narrative")
    assert second_narrative.status_code == 200
    second_payload = second_narrative.json()
    assert second_payload["demo_status"] == expected["expected_second_run_demo_status"]
    assert second_payload["sample_eligibility"]["sample_workspace_id"] == expected["sample_workspace_id"]
    assert second_payload["ledger_summary"]["preflight_count"] == 0
    assert second_payload["ledger_summary"]["action_result_count"] == 0


def test_demo_narrative_reports_user_owned_blocked_and_unsafe_ledgers() -> None:
    runtime = runtime_with_preflight_client(FakePreflightAppClient())
    client = TestClient(create_app(runtime))

    unknown = client.get("/runs/run_missing/demo-narrative")
    user_plan = _create_plan_with_lifecycle_reference(client)
    user_owned = client.get(f"/runs/{user_plan['run_id']}/demo-narrative")
    blocked_plan = client.post(
        "/plans",
        json={
            "user_goal": "Preview a non-allowlisted sample narrative.",
            "context_summary": _safe_sample_lifecycle_context(sample_workspace_id="monitoring_drift_review"),
        },
    ).json()
    blocked = client.get(f"/runs/{blocked_plan['run_id']}/demo-narrative")
    unsafe_plan = client.post(
        "/plans",
        json={
            "user_goal": "Preview unsafe ledger rejection.",
            "context_summary": _safe_sample_lifecycle_context(),
        },
    ).json()
    unsafe_entry = runtime.planner.ledger.get(unsafe_plan["run_id"])
    assert unsafe_entry is not None
    runtime.planner.ledger._entries[-1] = unsafe_entry.model_copy(  # noqa: SLF001 - corrupts test ledger intentionally.
        update={"safe_artifact_map": [{"reference_type": "raw_file", "raw_path": "C:\\private\\rows.csv"}]},
        deep=True,
    )
    unsafe = client.get(f"/runs/{unsafe_plan['run_id']}/demo-narrative")

    assert unknown.status_code == 422
    assert unknown.json()["detail"]["errors"][0]["code"] == "unknown_run"
    assert user_owned.status_code == 200
    assert user_owned.json()["demo_status"] == "not_sample_demo"
    assert user_owned.json()["validation"]["errors"][0]["code"] == "sample_demo_narrative_not_available"
    assert blocked.status_code == 200
    assert blocked.json()["demo_status"] == "blocked"
    assert blocked.json()["validation"]["errors"][0]["code"] == "sample_demo_narrative_blocked"
    assert unsafe.status_code == 422
    assert unsafe.json()["detail"]["errors"][0]["code"] == "unsafe_demo_ledger"
    assert "private\\rows.csv" not in unsafe.text


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
    app_client = FakePreflightAppClient(
        responses_by_capability={
            "quant_monitoring.validate_bundle": _valid_preflight_response(),
        }
    )
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


def test_manifest_advertises_scoped_workflow_runner() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    response = client.get("/runtime/manifest")

    assert response.status_code == 200
    payload = response.json()
    assert payload["workflow_run_support_level"] == "scoped_template_guided_manual_and_until_blocked"
    assert payload["workflow_scope_resolution_support_level"] == "deterministic_prompt_to_workflow_scope_v1"
    assert payload["workflow_template_support_level"] == "canonical_suite_workflow_templates_v1"
    assert payload["long_running_action_support_level"] == "ledgered_running_app_action_reference_v1"
    assert "POST /workflow-scope-resolutions" in payload["supported_routes"]
    assert "POST /workflow-runs" in payload["supported_routes"]
    assert "POST /workflow-runs/{run_id}/advance" in payload["supported_routes"]
    assert "app_workflow" in payload["supported_workflow_scopes"]


def test_workflow_scope_resolution_resolves_full_lifecycle_prompt() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    response = client.post(
        "/workflow-scope-resolutions",
        json={
            "goal": "Run the full workflow from Data to Studio to Documentation to Monitoring.",
            "context_summary": _safe_lifecycle_context(lifecycle_summary="Lifecycle starts from safe summaries."),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ledger_recorded"] is False
    assert payload["resolution_summary"]["strategy"] == "full_lifecycle_keywords"
    assert payload["resolved_request"]["workflow_scope"] == "full_lifecycle"
    assert payload["resolved_request"]["source_app"] == "quant_suite"
    assert payload["workflow_scope"]["selected_template_ids"] == ["full_lifecycle_default"]
    assert payload["workflow_scope"]["selected_capability_ids"] == _full_lifecycle_capability_ids()


def test_workflow_scope_resolution_resolves_app_steps_1_to_5_prompt() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    response = client.post(
        "/workflow-scope-resolutions",
        json={
            "goal": "Run Quant Studio steps 1 through 5.",
            "context_summary": _safe_lifecycle_context(),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["resolved_request"]["workflow_scope"] == "app_workflow"
    assert payload["resolved_request"]["source_app"] == "quant_studio"
    assert payload["workflow_scope"]["selected_template_ids"] == ["quant_studio_steps_1_5"]
    assert payload["workflow_scope"]["selected_capability_ids"] == [
        "quant_studio.run_model_readiness_check",
        "quant_studio.prepare_model_config_draft",
        "quant_studio.fit_candidate_model",
        "quant_studio.compare_candidate_runs",
        "quant_studio.create_documentation_package",
    ]


def test_workflow_scope_resolution_resolves_stage_range_prompt() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    response = client.post(
        "/workflow-scope-resolutions",
        json={
            "goal": "Run Quant Studio steps 2-4 only.",
            "context_summary": _safe_lifecycle_context(),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["resolved_request"]["workflow_scope"] == "stage_range"
    assert payload["resolved_request"]["source_app"] == "quant_studio"
    assert payload["resolved_request"]["start_stage"] == "2"
    assert payload["resolved_request"]["end_stage"] == "4"
    assert payload["workflow_scope"]["selected_capability_ids"] == [
        "quant_studio.prepare_model_config_draft",
        "quant_studio.fit_candidate_model",
        "quant_studio.compare_candidate_runs",
    ]


def test_workflow_scope_resolution_resolves_capability_set_prompt() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    response = client.post(
        "/workflow-scope-resolutions",
        json={
            "goal": "Just validate the Monitoring bundle.",
            "context_summary": _safe_lifecycle_context(lifecycle_state="monitoring_ready"),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["resolved_request"]["workflow_scope"] == "capability_set"
    assert payload["resolved_request"]["requested_capability_ids"] == ["quant_monitoring.validate_bundle"]
    assert payload["workflow_scope"]["selected_template_ids"] == []
    assert payload["workflow_scope"]["selected_capability_ids"] == ["quant_monitoring.validate_bundle"]


def test_workflow_scope_resolution_resolves_studio_fit_shorthand_to_safe_stage_range() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    response = client.post(
        "/workflow-scope-resolutions",
        json={
            "goal": "Fit a model from this Studio handoff.",
            "source_app": "quant_studio",
            "context_summary": _safe_lifecycle_context(),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["resolution_summary"]["strategy"] == "studio_fit_stage_range"
    assert payload["resolved_request"]["workflow_scope"] == "stage_range"
    assert payload["resolved_request"]["start_stage"] == "lifecycle_handoff_intake"
    assert payload["resolved_request"]["end_stage"] == "candidate_model_fit"
    assert payload["workflow_scope"]["selected_capability_ids"] == [
        "quant_studio.run_model_readiness_check",
        "quant_studio.prepare_model_config_draft",
        "quant_studio.fit_candidate_model",
    ]


def test_workflow_scope_resolution_rejects_unknown_stage() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    response = client.post(
        "/workflow-scope-resolutions",
        json={
            "goal": "Run Quant Studio step 9.",
            "context_summary": _safe_lifecycle_context(),
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["code"] == "workflow_stage_not_found"


def test_workflow_scope_resolution_rejects_unknown_quant_app() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    response = client.post(
        "/workflow-scope-resolutions",
        json={
            "goal": "Run Quant Pricing steps 1-5.",
            "context_summary": _safe_lifecycle_context(),
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["code"] == "workflow_scope_unknown_app"


def test_workflow_scope_resolution_rejects_unsafe_prompt_content() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    response = client.post(
        "/workflow-scope-resolutions",
        json={
            "goal": "Run Quant Studio steps 1-5 using C:\\Users\\matth\\Desktop\\secret.csv.",
            "context_summary": _safe_lifecycle_context(),
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["code"] == "unsafe_workflow_scope_resolution_request"
    assert "secret.csv" not in response.text


def test_workflow_scope_resolution_rejects_extra_browser_payload_fields() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    response = client.post(
        "/workflow-scope-resolutions",
        json={
            "goal": "Run Quant Studio steps 1-5.",
            "context_summary": _safe_lifecycle_context(),
            "action_payload": {"execution_permitted": True},
        },
    )

    assert response.status_code == 422


def test_workflow_run_studio_scope_uses_all_enabled_capabilities_without_gaps() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    response = client.post(
        "/workflow-runs",
        json={
            "goal": "Run only Quant Studio steps 1 through 5.",
            "workflow_scope": "app_workflow",
            "source_app": "quant_studio",
            "context_summary": _safe_lifecycle_context(),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    scope = payload["workflow_scope"]
    assert scope["selected_template_ids"] == ["quant_studio_steps_1_5"]
    assert scope["selected_capability_ids"] == [
        "quant_studio.run_model_readiness_check",
        "quant_studio.prepare_model_config_draft",
        "quant_studio.fit_candidate_model",
        "quant_studio.compare_candidate_runs",
        "quant_studio.create_documentation_package",
    ]
    assert scope["omitted_capability_ids"] == []
    assert scope["workflow_gaps"] == []
    assert [
        step["capability_id"] for step in payload["plan"]["proposed_steps"]
    ] == scope["selected_capability_ids"]


def test_workflow_run_data_scope_uses_all_enabled_capabilities_without_gaps() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    response = client.post(
        "/workflow-runs",
        json={
            "goal": "Run only Quant Data steps 1 through 5.",
            "workflow_scope": "app_workflow",
            "source_app": "quant_data",
            "context_summary": _safe_lifecycle_context(lifecycle_state="data_review"),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    scope = payload["workflow_scope"]
    assert scope["selected_template_ids"] == ["quant_data_steps_1_5"]
    assert scope["selected_capability_ids"] == [
        "quant_data.register_source_reference",
        "quant_data.run_source_preflight",
        "quant_data.create_eda_plan",
        "quant_data.run_eda_review",
        "quant_data.export_eda_handoff",
    ]
    assert scope["omitted_capability_ids"] == []
    assert scope["workflow_gaps"] == []
    assert [step["capability_id"] for step in payload["plan"]["proposed_steps"]] == scope["selected_capability_ids"]


def test_workflow_run_documentation_scope_uses_all_enabled_capabilities_without_gaps() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    response = client.post(
        "/workflow-runs",
        json={
            "goal": "Run only Quant Documentation steps 1 through 5.",
            "workflow_scope": "app_workflow",
            "source_app": "quant_documentation",
            "context_summary": _safe_lifecycle_context(
                lifecycle_state="documentation_review",
                lifecycle_summary="Lifecycle has safe documentation package summaries.",
            )
            | {"package_summary": _safe_documentation_package_summary()},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    scope = payload["workflow_scope"]
    assert scope["selected_template_ids"] == ["quant_documentation_steps_1_5"]
    assert scope["selected_capability_ids"] == [
        "quant_documentation.inspect_package",
        "quant_documentation.create_draft_workspace",
        "quant_documentation.draft_section",
        "quant_documentation.find_unsupported_claims",
        "quant_documentation.export_markdown_review_package",
    ]
    assert scope["omitted_capability_ids"] == []
    assert scope["workflow_gaps"] == []
    assert [step["capability_id"] for step in payload["plan"]["proposed_steps"]] == scope["selected_capability_ids"]


def test_workflow_run_monitoring_scope_uses_all_enabled_capabilities_without_gaps() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    response = client.post(
        "/workflow-runs",
        json={
            "goal": "Run only Quant Monitoring steps 1 through 5.",
            "workflow_scope": "app_workflow",
            "source_app": "quant_monitoring",
            "context_summary": _safe_lifecycle_context(
                lifecycle_state="monitoring_ready",
                lifecycle_summary="Lifecycle has safe monitoring bundle summaries.",
            ),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    scope = payload["workflow_scope"]
    assert scope["selected_template_ids"] == ["quant_monitoring_steps_1_5"]
    assert scope["selected_capability_ids"] == [
        "quant_monitoring.inspect_bundle",
        "quant_monitoring.prepare_profile_draft",
        "quant_monitoring.validate_bundle",
        "quant_monitoring.run_monitoring_review",
        "quant_monitoring.create_feedback_signal",
    ]
    assert scope["omitted_capability_ids"] == []
    assert scope["workflow_gaps"] == []
    assert [step["capability_id"] for step in payload["plan"]["proposed_steps"]] == scope["selected_capability_ids"]


def test_workflow_run_full_lifecycle_uses_complete_chain_without_gaps() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))
    context = _safe_lifecycle_context(lifecycle_summary="Lifecycle starts from safe source summaries.")
    context.pop("target_summary")
    context.pop("package_summary")
    context.pop("bundle_summary")

    response = client.post(
        "/workflow-runs",
        json={
            "goal": "Run the full Data to Studio to Documentation to Monitoring workflow.",
            "workflow_scope": "full_lifecycle",
            "context_summary": context,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    scope = payload["workflow_scope"]
    expected = _full_lifecycle_capability_ids()
    assert scope["selected_template_ids"] == ["full_lifecycle_default"]
    assert scope["selected_capability_ids"] == expected
    assert scope["omitted_capability_ids"] == []
    assert scope["workflow_gaps"] == []
    assert [step["capability_id"] for step in payload["plan"]["proposed_steps"]] == expected
    assert payload["plan"]["status"] == "valid"


def test_workflow_run_stage_range_limits_selected_app_steps() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    response = client.post(
        "/workflow-runs",
        json={
            "goal": "Only prepare the Studio model config draft.",
            "workflow_scope": "stage_range",
            "source_app": "quant_studio",
            "start_stage": "model_config_readiness",
            "end_stage": "model_config_readiness",
            "context_summary": _safe_lifecycle_context(),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["workflow_scope"]["selected_capability_ids"] == [
        "quant_studio.prepare_model_config_draft"
    ]
    assert payload["workflow_scope"]["workflow_gaps"] == []


@pytest.mark.parametrize(
    ("source_app", "start_stage", "end_stage", "expected_template", "expected_capabilities", "context_summary"),
    [
        (
            "quant_data",
            "eda_plan",
            "handoff_export",
            "quant_data_steps_1_5",
            [
                "quant_data.create_eda_plan",
                "quant_data.run_eda_review",
                "quant_data.export_eda_handoff",
            ],
            _safe_lifecycle_context(lifecycle_state="data_review"),
        ),
        (
            "quant_studio",
            "model_config_readiness",
            "candidate_comparison",
            "quant_studio_steps_1_5",
            [
                "quant_studio.prepare_model_config_draft",
                "quant_studio.fit_candidate_model",
                "quant_studio.compare_candidate_runs",
            ],
            _safe_lifecycle_context(),
        ),
        (
            "quant_documentation",
            "draft_workspace",
            "citation_claim_review",
            "quant_documentation_steps_1_5",
            [
                "quant_documentation.create_draft_workspace",
                "quant_documentation.draft_section",
                "quant_documentation.find_unsupported_claims",
            ],
            _safe_lifecycle_context(lifecycle_state="documentation_review")
            | {"package_summary": _safe_documentation_package_summary()},
        ),
        (
            "quant_monitoring",
            "bundle_dataset_validation",
            "bundle_dataset_validation",
            "quant_monitoring_steps_1_5",
            ["quant_monitoring.validate_bundle"],
            _safe_lifecycle_context(lifecycle_state="monitoring_ready"),
        ),
    ],
)
def test_workflow_run_stage_range_certifies_requested_partial_scopes(
    source_app: str,
    start_stage: str,
    end_stage: str,
    expected_template: str,
    expected_capabilities: list[str],
    context_summary: dict[str, Any],
) -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    response = client.post(
        "/workflow-runs",
        json={
            "goal": f"Run only {source_app} from {start_stage} through {end_stage}.",
            "workflow_scope": "stage_range",
            "source_app": source_app,
            "start_stage": start_stage,
            "end_stage": end_stage,
            "context_summary": context_summary,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    scope = payload["workflow_scope"]
    assert scope["selected_template_ids"] == [expected_template]
    assert scope["selected_capability_ids"] == expected_capabilities
    assert scope["omitted_capability_ids"] == []
    assert scope["workflow_gaps"] == []
    assert [step["capability_id"] for step in payload["plan"]["proposed_steps"]] == expected_capabilities


def test_workflow_run_stage_range_rejects_unknown_stage() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    response = client.post(
        "/workflow-runs",
        json={
            "goal": "Run an unknown Studio slice.",
            "workflow_scope": "stage_range",
            "source_app": "quant_studio",
            "start_stage": "not_a_stage",
            "end_stage": "candidate_model_fit",
            "context_summary": _safe_lifecycle_context(),
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["code"] == "workflow_stage_not_found"


def test_workflow_run_capability_set_selects_only_requested_capabilities() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))
    requested = [
        "quant_documentation.inspect_package",
        "quant_monitoring.validate_bundle",
    ]

    response = client.post(
        "/workflow-runs",
        json={
            "goal": "Inspect documentation and validate the monitoring bundle only.",
            "workflow_scope": "capability_set",
            "requested_capability_ids": requested,
            "context_summary": _safe_lifecycle_context(lifecycle_state="monitoring_ready")
            | {"package_summary": _safe_documentation_package_summary()},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    scope = payload["workflow_scope"]
    assert scope["selected_template_ids"] == []
    assert scope["requested_capability_ids"] == requested
    assert scope["selected_capability_ids"] == requested
    assert scope["workflow_gaps"] == []
    assert [step["capability_id"] for step in payload["plan"]["proposed_steps"]] == requested


def test_workflow_run_capability_set_rejects_browser_supplied_action_payloads() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    response = client.post(
        "/workflow-runs",
        json={
            "goal": "Run selected actions.",
            "workflow_scope": "capability_set",
            "requested_capability_ids": ["quant_monitoring.validate_bundle"],
            "context_summary": _safe_lifecycle_context(lifecycle_state="monitoring_ready"),
            "action_payload": {"execution_permitted": True},
        },
    )

    assert response.status_code == 422


def test_capability_set_advance_until_blocked_completes_only_selected_preflight() -> None:
    app_client = FakePreflightAppClient()
    client = TestClient(create_app(runtime_with_preflight_client(app_client)))
    created = client.post(
        "/workflow-runs",
        json={
            "goal": "Just validate the Monitoring bundle.",
            "workflow_scope": "capability_set",
            "requested_capability_ids": ["quant_monitoring.validate_bundle"],
            "context_summary": _safe_lifecycle_context(lifecycle_state="monitoring_ready"),
        },
    ).json()

    response = client.post(
        f"/workflow-runs/{created['run_id']}/advance-until-blocked",
        json={"advance_intent": "advance_workflow_until_blocked"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["advance_status"] == "completed"
    assert payload["completed_action_count"] == 1
    assert [call["capability_id"] for call in app_client.calls] == ["quant_monitoring.validate_bundle"]
    assert app_client.execution_calls == []
    assert payload["orchestration"]["current_step_id"] is None


def test_stage_range_mid_workflow_missing_handoff_blocks_without_inventing_input() -> None:
    app_client = FakePreflightAppClient()
    client = TestClient(create_app(runtime_with_preflight_client(app_client)))
    created = client.post(
        "/workflow-runs",
        json={
            "goal": "Fit a candidate model without running the earlier Studio steps.",
            "workflow_scope": "stage_range",
            "source_app": "quant_studio",
            "start_stage": "candidate_model_fit",
            "end_stage": "candidate_model_fit",
            "context_summary": _safe_lifecycle_context(),
        },
    ).json()

    response = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["code"] == "missing_workflow_handoff_reference"
    assert app_client.calls == []


def test_workflow_advance_until_blocked_stops_at_manual_confirmation() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))
    created = client.post(
        "/workflow-runs",
        json={
            "goal": "Run only Quant Studio steps 1 through 5.",
            "workflow_scope": "app_workflow",
            "source_app": "quant_studio",
            "context_summary": _safe_lifecycle_context(),
        },
    ).json()

    response = client.post(
        f"/workflow-runs/{created['run_id']}/advance-until-blocked",
        json={"advance_intent": "advance_workflow_until_blocked"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["advance_status"] == "manual_confirmation_required"
    assert payload["completed_action_count"] == 1
    assert payload["run_state"] == "waiting_for_confirmation"
    assert payload["last_result"]["capability_id"] == "quant_studio.prepare_model_config_draft"


def test_data_workflow_advance_until_blocked_stops_at_source_registration_confirmation() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))
    created = client.post(
        "/workflow-runs",
        json={
            "goal": "Run only Quant Data steps 1 through 5.",
            "workflow_scope": "app_workflow",
            "source_app": "quant_data",
            "context_summary": _safe_lifecycle_context(lifecycle_state="data_review"),
        },
    ).json()

    response = client.post(
        f"/workflow-runs/{created['run_id']}/advance-until-blocked",
        json={"advance_intent": "advance_workflow_until_blocked"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["advance_status"] == "manual_confirmation_required"
    assert payload["completed_action_count"] == 0
    assert payload["run_state"] == "waiting_for_confirmation"
    assert payload["last_result"]["capability_id"] == "quant_data.register_source_reference"


def test_documentation_workflow_advance_until_blocked_stops_at_draft_workspace_confirmation() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))
    created = client.post(
        "/workflow-runs",
        json={
            "goal": "Run only Quant Documentation steps 1 through 5.",
            "workflow_scope": "app_workflow",
            "source_app": "quant_documentation",
            "context_summary": _safe_lifecycle_context(lifecycle_state="documentation_review")
            | {"package_summary": _safe_documentation_package_summary()},
        },
    ).json()

    response = client.post(
        f"/workflow-runs/{created['run_id']}/advance-until-blocked",
        json={"advance_intent": "advance_workflow_until_blocked"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["advance_status"] == "manual_confirmation_required"
    assert payload["completed_action_count"] == 2
    assert payload["run_state"] == "waiting_for_confirmation"
    assert payload["last_result"]["capability_id"] == "quant_documentation.create_draft_workspace"


def test_monitoring_workflow_advance_until_blocked_stops_at_profile_draft_confirmation() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))
    created = client.post(
        "/workflow-runs",
        json={
            "goal": "Run only Quant Monitoring steps 1 through 5.",
            "workflow_scope": "app_workflow",
            "source_app": "quant_monitoring",
            "context_summary": _safe_lifecycle_context(lifecycle_state="monitoring_ready"),
        },
    ).json()

    response = client.post(
        f"/workflow-runs/{created['run_id']}/advance-until-blocked",
        json={"advance_intent": "advance_workflow_until_blocked"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["advance_status"] == "manual_confirmation_required"
    assert payload["completed_action_count"] == 2
    assert payload["run_state"] == "waiting_for_confirmation"
    assert payload["last_result"]["capability_id"] == "quant_monitoring.prepare_profile_draft"


def test_studio_scoped_workflow_hands_draft_reference_to_fit_preflight() -> None:
    app_client = FakePreflightAppClient(
        responses_by_capability={
            "quant_studio.run_model_readiness_check": _valid_preflight_response(
                capability_id="quant_studio.run_model_readiness_check",
                app_id="quant_studio",
            ),
            "quant_studio.fit_candidate_model": _valid_preflight_response(
                capability_id="quant_studio.fit_candidate_model",
                app_id="quant_studio",
            ),
        }
    )
    client = TestClient(create_app(runtime_with_preflight_client(app_client)))
    created = client.post(
        "/workflow-runs",
        json={
            "goal": "Run only Quant Studio steps 1 through 5.",
            "workflow_scope": "app_workflow",
            "source_app": "quant_studio",
            "context_summary": _safe_lifecycle_context(),
        },
    ).json()
    steps = created["plan"]["proposed_steps"]
    draft_step = next(step for step in steps if step["capability_id"] == "quant_studio.prepare_model_config_draft")
    fit_step = next(step for step in steps if step["capability_id"] == "quant_studio.fit_candidate_model")
    app_client.execution_responses_by_capability["quant_studio.prepare_model_config_draft"] = _valid_action_result(
        capability_id="quant_studio.prepare_model_config_draft",
        step_id=draft_step["step_id"],
    )

    first_advance = client.post(
        f"/workflow-runs/{created['run_id']}/advance-until-blocked",
        json={"advance_intent": "advance_workflow_until_blocked"},
    )
    assert first_advance.status_code == 200
    assert first_advance.json()["advance_status"] == "manual_confirmation_required"

    confirmation = client.post(
        "/confirmations",
        json={
            "run_id": created["run_id"],
            "step_id": draft_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    )
    assert confirmation.status_code == 200
    preview = client.post(
        "/action-requests",
        json={"run_id": created["run_id"], "step_id": draft_step["step_id"]},
    )
    assert preview.status_code == 200
    execution = client.post(
        "/executions",
        json={"run_id": created["run_id"], "step_id": draft_step["step_id"]},
    )
    assert execution.status_code == 200

    fit_preflight = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )

    assert fit_preflight.status_code == 200
    payload = fit_preflight.json()
    assert payload["advance_status"] == "advanced"
    assert payload["selected_action"] == "run_preflight"
    assert payload["step_id"] == fit_step["step_id"]
    assert app_client.calls[-1]["capability_id"] == "quant_studio.fit_candidate_model"
    assert app_client.calls[-1]["payload"]["action_input"]["model_config_draft"]["reference_type"] == "model_config_draft"


def test_data_scoped_workflow_hands_source_and_preflight_references_forward() -> None:
    app_client = FakePreflightAppClient()
    client = TestClient(create_app(runtime_with_preflight_client(app_client)))
    created = client.post(
        "/workflow-runs",
        json={
            "goal": "Run only Quant Data steps 1 through 5.",
            "workflow_scope": "app_workflow",
            "source_app": "quant_data",
            "context_summary": _safe_lifecycle_context(lifecycle_state="data_review"),
        },
    ).json()
    steps = created["plan"]["proposed_steps"]
    register_step = next(step for step in steps if step["capability_id"] == "quant_data.register_source_reference")
    preflight_step = next(step for step in steps if step["capability_id"] == "quant_data.run_source_preflight")
    eda_plan_step = next(step for step in steps if step["capability_id"] == "quant_data.create_eda_plan")

    first_advance = client.post(
        f"/workflow-runs/{created['run_id']}/advance-until-blocked",
        json={"advance_intent": "advance_workflow_until_blocked"},
    )
    assert first_advance.status_code == 200
    assert first_advance.json()["advance_status"] == "manual_confirmation_required"

    confirmation = client.post(
        "/confirmations",
        json={
            "run_id": created["run_id"],
            "step_id": register_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    )
    assert confirmation.status_code == 200

    preview = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert preview.status_code == 200
    assert preview.json()["selected_action"] == "preview_action_request"

    execution = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert execution.status_code == 200
    assert execution.json()["selected_action"] == "execute_step"
    assert app_client.execution_calls[-1]["capability_id"] == "quant_data.register_source_reference"

    source_preflight = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert source_preflight.status_code == 200
    assert source_preflight.json()["step_id"] == preflight_step["step_id"]
    assert source_preflight.json()["selected_action"] == "run_preflight"
    assert app_client.calls[-1]["capability_id"] == "quant_data.run_source_preflight"
    assert app_client.calls[-1]["payload"]["action_input"]["source_reference"]["reference_type"] == "source_reference"

    eda_plan_block = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert eda_plan_block.status_code == 200
    assert eda_plan_block.json()["advance_status"] == "manual_confirmation_required"
    assert eda_plan_block.json()["capability_id"] == "quant_data.create_eda_plan"

    eda_plan_confirmation = client.post(
        "/confirmations",
        json={
            "run_id": created["run_id"],
            "step_id": eda_plan_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    )
    assert eda_plan_confirmation.status_code == 200

    eda_plan_preview = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert eda_plan_preview.status_code == 200
    assert eda_plan_preview.json()["selected_action"] == "preview_action_request"
    assert (
        eda_plan_preview.json()["delegated_result"]["result"]["action_request"]["action_input"]["preflight_summary"][
            "reference_type"
        ]
        == "preflight_summary"
    )


def test_documentation_scoped_workflow_hands_documentation_references_forward() -> None:
    app_client = FakePreflightAppClient()
    client = TestClient(create_app(runtime_with_preflight_client(app_client)))
    created = client.post(
        "/workflow-runs",
        json={
            "goal": "Run only Quant Documentation steps 1 through 5.",
            "workflow_scope": "app_workflow",
            "source_app": "quant_documentation",
            "context_summary": _safe_lifecycle_context(lifecycle_state="documentation_review")
            | {"package_summary": _safe_documentation_package_summary()},
        },
    ).json()
    steps = created["plan"]["proposed_steps"]
    draft_workspace_step = next(
        step for step in steps if step["capability_id"] == "quant_documentation.create_draft_workspace"
    )
    draft_section_step = next(step for step in steps if step["capability_id"] == "quant_documentation.draft_section")
    export_step = next(
        step for step in steps if step["capability_id"] == "quant_documentation.export_markdown_review_package"
    )

    inspect_preview = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert inspect_preview.status_code == 200
    assert inspect_preview.json()["selected_action"] == "preview_action_request"
    inspect_execution = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert inspect_execution.status_code == 200
    assert inspect_execution.json()["selected_action"] == "execute_step"

    draft_workspace_block = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert draft_workspace_block.status_code == 200
    assert draft_workspace_block.json()["advance_status"] == "manual_confirmation_required"

    confirmation = client.post(
        "/confirmations",
        json={
            "run_id": created["run_id"],
            "step_id": draft_workspace_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    )
    assert confirmation.status_code == 200
    draft_workspace_preview = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert draft_workspace_preview.status_code == 200
    assert draft_workspace_preview.json()["selected_action"] == "preview_action_request"
    assert (
        draft_workspace_preview.json()["delegated_result"]["result"]["action_request"]["action_input"][
            "documentation_package_summary"
        ]["reference_type"]
        == "documentation_package_summary"
    )
    draft_workspace_execution = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert draft_workspace_execution.status_code == 200
    assert draft_workspace_execution.json()["selected_action"] == "execute_step"

    draft_section_block = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert draft_section_block.status_code == 200
    assert draft_section_block.json()["advance_status"] == "manual_confirmation_required"
    assert draft_section_block.json()["capability_id"] == "quant_documentation.draft_section"

    section_confirmation = client.post(
        "/confirmations",
        json={
            "run_id": created["run_id"],
            "step_id": draft_section_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    )
    assert section_confirmation.status_code == 200
    draft_section_preview = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert draft_section_preview.status_code == 200
    assert (
        draft_section_preview.json()["delegated_result"]["result"]["action_request"]["action_input"][
            "documentation_draft"
        ]["reference_type"]
        == "documentation_draft"
    )
    draft_section_execution = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert draft_section_execution.status_code == 200

    claim_review_preview = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert claim_review_preview.status_code == 200
    assert (
        claim_review_preview.json()["delegated_result"]["result"]["action_request"]["action_input"]["draft_section"][
            "reference_type"
        ]
        == "draft_section"
    )
    claim_review_execution = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert claim_review_execution.status_code == 200

    export_block = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert export_block.status_code == 200
    assert export_block.json()["advance_status"] == "manual_confirmation_required"

    export_confirmation = client.post(
        "/confirmations",
        json={
            "run_id": created["run_id"],
            "step_id": export_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    )
    assert export_confirmation.status_code == 200
    export_preview = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert export_preview.status_code == 200
    assert (
        export_preview.json()["delegated_result"]["result"]["action_request"]["action_input"]["claim_review_summary"][
            "reference_type"
        ]
        == "claim_review_summary"
    )


def test_monitoring_scoped_workflow_hands_monitoring_references_forward() -> None:
    app_client = FakePreflightAppClient()
    client = TestClient(create_app(runtime_with_preflight_client(app_client)))
    created = client.post(
        "/workflow-runs",
        json={
            "goal": "Run only Quant Monitoring steps 1 through 5.",
            "workflow_scope": "app_workflow",
            "source_app": "quant_monitoring",
            "context_summary": _safe_lifecycle_context(lifecycle_state="monitoring_ready"),
        },
    ).json()
    steps = created["plan"]["proposed_steps"]
    profile_step = next(
        step for step in steps if step["capability_id"] == "quant_monitoring.prepare_profile_draft"
    )
    review_step = next(
        step for step in steps if step["capability_id"] == "quant_monitoring.run_monitoring_review"
    )
    feedback_step = next(
        step for step in steps if step["capability_id"] == "quant_monitoring.create_feedback_signal"
    )

    inspect_preview = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert inspect_preview.status_code == 200
    assert inspect_preview.json()["selected_action"] == "preview_action_request"
    inspect_execution = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert inspect_execution.status_code == 200
    assert inspect_execution.json()["selected_action"] == "execute_step"

    profile_block = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert profile_block.status_code == 200
    assert profile_block.json()["advance_status"] == "manual_confirmation_required"

    profile_confirmation = client.post(
        "/confirmations",
        json={
            "run_id": created["run_id"],
            "step_id": profile_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    )
    assert profile_confirmation.status_code == 200
    profile_preview = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert profile_preview.status_code == 200
    assert profile_preview.json()["selected_action"] == "preview_action_request"
    assert (
        profile_preview.json()["delegated_result"]["result"]["action_request"]["action_input"]["bundle_summary"][
            "reference_type"
        ]
        == "bundle_summary"
    )
    profile_execution = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert profile_execution.status_code == 200
    assert profile_execution.json()["selected_action"] == "execute_step"

    validation_preflight = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert validation_preflight.status_code == 200
    assert validation_preflight.json()["selected_action"] == "run_preflight"
    assert app_client.calls[-1]["capability_id"] == "quant_monitoring.validate_bundle"
    assert (
        app_client.calls[-1]["payload"]["action_input"]["monitoring_profile_draft"]["reference_type"]
        == "monitoring_profile_draft"
    )

    review_preflight = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert review_preflight.status_code == 200
    assert review_preflight.json()["selected_action"] == "run_preflight"
    assert app_client.calls[-1]["capability_id"] == "quant_monitoring.run_monitoring_review"
    assert (
        app_client.calls[-1]["payload"]["action_input"]["bundle_validation_summary"]["reference_type"]
        == "bundle_validation_summary"
    )

    review_block = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert review_block.status_code == 200
    assert review_block.json()["advance_status"] == "manual_confirmation_required"
    assert review_block.json()["capability_id"] == "quant_monitoring.run_monitoring_review"

    review_confirmation = client.post(
        "/confirmations",
        json={
            "run_id": created["run_id"],
            "step_id": review_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    )
    assert review_confirmation.status_code == 200
    review_preview = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert review_preview.status_code == 200
    assert (
        review_preview.json()["delegated_result"]["result"]["action_request"]["action_input"][
            "bundle_validation_summary"
        ]["reference_type"]
        == "bundle_validation_summary"
    )
    review_execution = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert review_execution.status_code == 200

    feedback_block = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert feedback_block.status_code == 200
    assert feedback_block.json()["advance_status"] == "manual_confirmation_required"
    assert feedback_block.json()["capability_id"] == "quant_monitoring.create_feedback_signal"

    feedback_confirmation = client.post(
        "/confirmations",
        json={
            "run_id": created["run_id"],
            "step_id": feedback_step["step_id"],
            "confirmation_intent": "approve_plan_step",
        },
    )
    assert feedback_confirmation.status_code == 200
    feedback_preview = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert feedback_preview.status_code == 200
    assert (
        feedback_preview.json()["delegated_result"]["result"]["action_request"]["action_input"]["monitoring_run"][
            "reference_type"
        ]
        == "monitoring_run"
    )


def test_full_lifecycle_workflow_advances_across_apps_with_safe_handoffs() -> None:
    app_client = FakePreflightAppClient()
    client = TestClient(create_app(runtime_with_preflight_client(app_client)))
    context = _safe_lifecycle_context(lifecycle_summary="Lifecycle starts from safe source summaries.")
    context.pop("target_summary")
    context.pop("package_summary")
    context.pop("bundle_summary")
    created = client.post(
        "/workflow-runs",
        json={
            "goal": "Run the full Data to Studio to Documentation to Monitoring workflow.",
            "workflow_scope": "full_lifecycle",
            "context_summary": context,
        },
    ).json()

    statuses: list[str] = []
    for _index in range(90):
        response = client.post(
            f"/workflow-runs/{created['run_id']}/advance",
            json={"advance_intent": "advance_workflow_one_step"},
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        statuses.append(payload["advance_status"])
        if payload["advance_status"] == "completed":
            break
        if payload["advance_status"] == "manual_confirmation_required":
            confirmation = client.post(
                "/confirmations",
                json={
                    "run_id": created["run_id"],
                    "step_id": payload["step_id"],
                    "confirmation_intent": "approve_plan_step",
                },
            )
            assert confirmation.status_code == 200, confirmation.text
            continue
        assert payload["advance_status"] == "advanced"
    else:
        pytest.fail("Full lifecycle workflow did not complete within the expected advance budget.")

    assert statuses[-1] == "completed"
    assert any(call["capability_id"] == "quant_data.run_eda_review" for call in app_client.calls)
    assert app_client.execution_calls[-1]["capability_id"] == "quant_monitoring.create_feedback_signal"

    studio_readiness_call = next(
        call for call in app_client.calls if call["capability_id"] == "quant_studio.run_model_readiness_check"
    )
    studio_readiness_input = studio_readiness_call["payload"]["action_input"]
    assert studio_readiness_input["eda_handoff"]["reference_type"] == "eda_handoff"
    assert studio_readiness_input["target_summary"].startswith("Target summary from EDA handoff")

    documentation_inspect_call = next(
        call for call in app_client.execution_calls if call["capability_id"] == "quant_documentation.inspect_package"
    )
    documentation_input = documentation_inspect_call["payload"]["action_request"]["action_input"]
    assert documentation_input["documentation_package"]["reference_type"] == "documentation_package"
    assert documentation_input["package_summary"]["documentation_package_id"] == "documentation_package_test"

    monitoring_inspect_call = next(
        call for call in app_client.execution_calls if call["capability_id"] == "quant_monitoring.inspect_bundle"
    )
    monitoring_input = monitoring_inspect_call["payload"]["action_request"]["action_input"]
    assert monitoring_input["monitoring_bundle"]["reference_type"] == "monitoring_bundle"
    assert monitoring_input["bundle_summary"]["bundle_id"] == "monitoring_bundle_test"


def test_workflow_runner_ledgers_running_action_result_and_stops_idempotently() -> None:
    running_result = _valid_action_result(
        "running",
        capability_id="quant_documentation.inspect_package",
        app_id="quant_documentation",
        step_id="step_1",
    )
    running_result["output_references"] = []
    running_result["app_run_reference"] = {
        "reference_type": "app_run",
        "reference_id": "documentation_inspect_pending_test",
        "status": "running",
        "label": "Documentation package inspection",
    }
    running_result["recommended_next_step"] = {
        "label": "Refresh workflow status before advancing this step again.",
        "target_app": "quant_agent",
        "review_only": True,
    }
    app_client = FakePreflightAppClient(
        execution_responses_by_capability={"quant_documentation.inspect_package": running_result}
    )
    client = TestClient(create_app(runtime_with_preflight_client(app_client)))
    created = client.post(
        "/workflow-runs",
        json={
            "goal": "Run only Quant Documentation steps 1 through 5.",
            "workflow_scope": "app_workflow",
            "source_app": "quant_documentation",
            "context_summary": _safe_lifecycle_context(lifecycle_state="documentation_review")
            | {"package_summary": _safe_documentation_package_summary()},
        },
    ).json()
    step_id = created["plan"]["proposed_steps"][0]["step_id"]

    preview = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    execution = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )

    assert preview.status_code == 200
    assert preview.json()["selected_action"] == "preview_action_request"
    assert execution.status_code == 200
    execution_payload = execution.json()
    assert execution_payload["selected_action"] == "execute_step"
    assert execution_payload["delegated_result"]["result"]["action_result"]["execution_status"] == "running"
    assert execution_payload["run_state"] == "running"
    assert execution_payload["orchestration"]["steps"][0]["status"] == "running"
    assert execution_payload["orchestration"]["steps"][0]["allowed_actions"] == []
    assert (
        execution_payload["orchestration"]["steps"][0]["latest_action_result_reference"]["app_run_reference"][
            "reference_id"
        ]
        == "documentation_inspect_pending_test"
    )
    assert len(app_client.execution_calls) == 1

    duplicate_execution = client.post(
        "/executions",
        json={"run_id": created["run_id"], "step_id": step_id},
    )
    assert duplicate_execution.status_code == 200
    assert duplicate_execution.json()["action_result"]["execution_status"] == "running"
    assert duplicate_execution.json()["run_state"] == "running"
    assert len(app_client.execution_calls) == 1

    blocked = client.post(
        f"/workflow-runs/{created['run_id']}/advance",
        json={"advance_intent": "advance_workflow_one_step"},
    )
    assert blocked.status_code == 200
    assert blocked.json()["advance_status"] == "blocked"
    assert blocked.json()["delegated_result"]["step_status"] == "running"
    assert len(app_client.execution_calls) == 1


def test_data_preflight_blocks_before_source_registration_confirmation() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))
    created = client.post(
        "/workflow-runs",
        json={
            "goal": "Run only Quant Data steps 1 through 5.",
            "workflow_scope": "app_workflow",
            "source_app": "quant_data",
            "context_summary": _safe_lifecycle_context(lifecycle_state="data_review"),
        },
    ).json()
    preflight_step = next(
        step for step in created["plan"]["proposed_steps"] if step["capability_id"] == "quant_data.run_source_preflight"
    )

    response = client.post(
        "/preflights",
        json={"run_id": created["run_id"], "step_id": preflight_step["step_id"]},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["code"] == "orchestration_step_not_ready"


def test_documentation_workflow_blocks_when_required_handoff_reference_is_missing() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))
    created = client.post(
        "/workflow-runs",
        json={
            "goal": "Run only Quant Documentation steps 1 through 5.",
            "workflow_scope": "app_workflow",
            "source_app": "quant_documentation",
            "context_summary": _safe_lifecycle_context(lifecycle_state="documentation_review")
            | {"package_summary": _safe_documentation_package_summary()},
        },
    ).json()
    draft_section_step = next(
        step for step in created["plan"]["proposed_steps"] if step["capability_id"] == "quant_documentation.draft_section"
    )

    response = client.post(
        "/action-requests",
        json={"run_id": created["run_id"], "step_id": draft_section_step["step_id"]},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["code"] == "orchestration_step_not_ready"


def test_monitoring_workflow_blocks_when_required_handoff_reference_is_missing() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))
    created = client.post(
        "/workflow-runs",
        json={
            "goal": "Run only Quant Monitoring steps 1 through 5.",
            "workflow_scope": "app_workflow",
            "source_app": "quant_monitoring",
            "context_summary": _safe_lifecycle_context(lifecycle_state="monitoring_ready"),
        },
    ).json()
    review_step = next(
        step for step in created["plan"]["proposed_steps"] if step["capability_id"] == "quant_monitoring.run_monitoring_review"
    )

    response = client.post(
        "/action-requests",
        json={"run_id": created["run_id"], "step_id": review_step["step_id"]},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["code"] == "orchestration_step_not_ready"


def test_workflow_run_rejects_unknown_capability_set_member() -> None:
    client = TestClient(create_app(runtime_with_preflight_client(FakePreflightAppClient())))

    response = client.post(
        "/workflow-runs",
        json={
            "goal": "Run selected actions.",
            "workflow_scope": "capability_set",
            "requested_capability_ids": ["quant_studio.not_a_real_capability"],
            "context_summary": {
                "target_summary": {"label": "Credit PD model target", "status": "ready"}
            },
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["code"] == "unknown_workflow_capability"
