from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from quant_agent_runtime.contracts import QuantSuiteContractLoader
from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.models import (
    AgentSupportBundleResult,
    ExternalApprovalDecisionImportRequest,
    ExternalApprovalDecisionImportResult,
    ExternalApprovalPreviewRequest,
    ExternalApprovalPreviewResult,
    ExternalApprovalSubmissionRequest,
    ExternalApprovalSubmissionListResult,
    ExternalApprovalSubmissionResult,
    ExternalApprovalSubmissionSummary,
    LedgerEntry,
    LedgerIntegritySummary,
    PlanValidationResult,
    RunOrchestrationResult,
    RunStatusResult,
    ValidationIssue,
)
from quant_agent_runtime.redaction import find_unsafe_payload_issues
from quant_agent_runtime.validation.errors import RuntimeValidationError


EXTERNAL_APPROVAL_REQUEST_CONTRACT_SCHEMA = "agent_external_approval_request.v1.schema.json"
EXTERNAL_APPROVAL_DECISION_CONTRACT_SCHEMA = "agent_external_approval_decision.v1.schema.json"
EXTERNAL_APPROVAL_SUBMISSION_CONTRACT_SCHEMA = "agent_external_approval_submission.v1.schema.json"
EXTERNAL_APPROVAL_SUPPORT_LEVEL = "manual_approval_package_preview_only"
EXTERNAL_APPROVAL_DECISION_SUPPORT_LEVEL = "manual_decision_import_only"
EXTERNAL_APPROVAL_SUBMISSION_SUPPORT_LEVEL = "local_outbox_submission_only"
EXTERNAL_APPROVAL_SUBMISSION_STATUS_SUPPORT_LEVEL = "ledger_and_local_outbox_status"
EXTERNAL_APPROVAL_EVENT_TYPE = "external_approval_request_preview"
EXTERNAL_APPROVAL_DECISION_EVENT_TYPE = "external_approval_decision_import"
EXTERNAL_APPROVAL_SUBMISSION_EVENT_TYPE = "external_approval_submission"
TERMINAL_APPROVAL_DECISION_FINAL_STATUSES = {
    "cancelled",
    "sample_reset",
    "completed",
    "completed_with_warnings",
    "failed_terminal",
}
SUPPORTED_EXTERNAL_APPROVAL_ADAPTERS = {"local_outbox", "disabled"}
DEFAULT_EXTERNAL_APPROVAL_ADAPTER = "local_outbox"
LOCAL_OUTBOX_SAFE_LABEL = "quant_agent_external_approval_outbox"


class ExternalApprovalService:
    def __init__(
        self,
        *,
        ledger: InMemoryLedger,
        contract_loader: QuantSuiteContractLoader,
        governance: Any | None = None,
    ) -> None:
        self._ledger = ledger
        self._contract_loader = contract_loader
        self._governance = governance

    def preview_request(
        self,
        request: ExternalApprovalPreviewRequest,
        *,
        run_status: RunStatusResult,
        orchestration: RunOrchestrationResult,
        support_bundle: AgentSupportBundleResult,
    ) -> ExternalApprovalPreviewResult:
        entry = self._ledger.get(request.run_id)
        if entry is None:
            raise _rejected("unknown_run", "No recorded run was found for the requested run_id.")
        _reject_unsafe_ledger(entry)
        plan_id = _active_plan_id(entry)
        if plan_id is None:
            raise _rejected(
                "missing_plan_snapshot",
                "The recorded run is missing a contract-valid active plan snapshot.",
            )
        if entry.final_status in {"cancelled", "sample_reset"}:
            raise _rejected(
                "terminal_run_external_approval_request",
                "Cancelled and reset runs cannot produce external approval request previews.",
            )
        if request.approval_scope == "run" and request.step_id is not None:
            raise _rejected(
                "unsupported_approval_scope",
                "Run-level approval previews must not include a step_id.",
                step_id=request.step_id,
            )
        step = None
        if request.approval_scope == "step":
            if not request.step_id:
                raise _rejected(
                    "missing_approval_step",
                    "Step-level approval previews require a step_id.",
                )
            step = _plan_step(entry, request.step_id)
            if step is None:
                raise _rejected(
                    "unknown_step",
                    "No recorded plan step was found for the requested step_id.",
                    step_id=request.step_id,
                )

        if support_bundle.validation.status != "valid":
            raise _rejected(
                "missing_support_bundle_evidence",
                "A valid support bundle summary is required before approval package preview.",
                step_id=request.step_id,
                capability_id=_step_capability_id(step),
            )

        governance_summary = _governance_summary(self._governance, request.run_id, run_status)
        policy_pack_id = str(governance_summary.get("policy_pack_id") or "unknown_policy_pack")
        fingerprint = _approval_fingerprint(
            entry=entry,
            approval_scope=request.approval_scope,
            step_id=request.step_id,
            plan_id=plan_id,
            policy_pack_id=policy_pack_id,
        )
        existing = _existing_preview(entry, fingerprint)
        if existing is not None:
            approval_request = existing.get("approval_request_snapshot")
            if isinstance(approval_request, dict):
                self._validate_approval_request(
                    approval_request,
                    step_id=request.step_id,
                    capability_id=_step_capability_id(step),
                )
                return ExternalApprovalPreviewResult(
                    run_id=request.run_id,
                    step_id=request.step_id,
                    approval_request=approval_request,
                    run_status=run_status,
                    orchestration=orchestration,
                    validation=PlanValidationResult(status="valid"),
                    ledger_recorded=True,
                )

        approval_request = _approval_request_payload(
            request=request,
            entry=entry,
            step=step,
            plan_id=plan_id,
            governance_summary=governance_summary,
            separation_of_duties_summary=_sod_summary(self._governance, request.run_id, run_status),
            run_status=run_status,
            orchestration=orchestration,
            support_bundle=support_bundle,
            fingerprint=fingerprint,
        )
        self._validate_approval_request(
            approval_request,
            step_id=request.step_id,
            capability_id=_step_capability_id(step),
        )

        event = {
            "recovery_event_id": f"external_approval_preview_{uuid4().hex[:12]}",
            "event_type": EXTERNAL_APPROVAL_EVENT_TYPE,
            "status": "previewed",
            "approval_intent": request.approval_intent,
            "approval_scope": request.approval_scope,
            "run_id": request.run_id,
            "step_id": request.step_id,
            "capability_id": approval_request.get("capability_id"),
            "plan_id": plan_id,
            "policy_pack_id": policy_pack_id,
            "approval_request_id": approval_request["approval_request_id"],
            "approval_request_fingerprint": fingerprint,
            "approval_request_snapshot": approval_request,
            "created_by": "local_user",
            "created_at_utc": _utc_now_label(),
            "external_submission_status": "not_submitted",
            "execution_permitted": False,
        }
        _reject_unsafe_event(event, step_id=request.step_id, capability_id=_step_capability_id(step))

        try:
            self._ledger.append_recovery_event(request.run_id, event)
        except ValueError as exc:
            raise _rejected(
                "unsafe_external_approval_request_record",
                "The external approval request preview could not be safely ledgered.",
                step_id=request.step_id,
                capability_id=_step_capability_id(step),
            ) from exc

        return ExternalApprovalPreviewResult(
            run_id=request.run_id,
            step_id=request.step_id,
            approval_request=approval_request,
            run_status=run_status,
            orchestration=orchestration,
            validation=PlanValidationResult(status="valid"),
            ledger_recorded=True,
        )

    def import_decision(
        self,
        request: ExternalApprovalDecisionImportRequest,
        *,
        run_status: RunStatusResult,
        orchestration: RunOrchestrationResult,
    ) -> ExternalApprovalDecisionImportResult:
        entry = self._ledger.get(request.run_id)
        if entry is None:
            raise _rejected("unknown_run", "No recorded run was found for the requested run_id.")
        _reject_unsafe_ledger(entry, code="unsafe_ledger_external_approval_decision")
        if _active_plan_id(entry) is None:
            raise _rejected(
                "missing_plan_snapshot",
                "The recorded run is missing a contract-valid active plan snapshot.",
            )
        if entry.final_status in TERMINAL_APPROVAL_DECISION_FINAL_STATUSES:
            raise _rejected(
                "terminal_run_external_approval_decision",
                "Terminal runs cannot import external approval decisions.",
            )

        approval_decision = _json_clone(request.approval_decision)
        self._validate_approval_decision(approval_decision)
        if approval_decision.get("run_id") != request.run_id:
            raise _rejected(
                "external_approval_decision_run_mismatch",
                "The approval decision run_id does not match the requested run_id.",
            )
        approval_request_id = str(approval_decision.get("approval_request_id") or "")
        request_event = _matching_request_event(entry, approval_request_id)
        if request_event is None:
            raise _rejected(
                "missing_external_approval_request_preview",
                "The decision does not match a ledgered external approval request preview.",
            )
        approval_request = request_event.get("approval_request_snapshot")
        if not isinstance(approval_request, dict):
            raise _rejected(
                "malformed_external_approval_request_preview",
                "The ledgered external approval request preview is malformed.",
            )
        self._validate_approval_request(
            approval_request,
            step_id=_nullable_str(approval_request.get("step_id")),
            capability_id=_nullable_str(approval_request.get("capability_id")),
        )
        _assert_decision_matches_request(approval_decision, approval_request)

        decision_id = str(approval_decision.get("approval_decision_id") or "")
        decision_fingerprint = _decision_fingerprint(approval_decision)
        existing = _existing_decision_import(entry, decision_id)
        if existing is not None:
            if existing.get("approval_decision_fingerprint") != decision_fingerprint:
                raise _rejected(
                    "conflicting_external_approval_decision",
                    "A different decision payload was already imported for this approval_decision_id.",
                    step_id=_nullable_str(approval_decision.get("step_id")),
                    capability_id=_nullable_str(approval_decision.get("capability_id")),
                )
            existing_decision = existing.get("approval_decision_snapshot")
            if isinstance(existing_decision, dict):
                summary = external_approval_summary_for_entry(entry)
                run_status = run_status.model_copy(update={"external_approval_summary": summary})
                orchestration = orchestration.model_copy(update={"external_approval_summary": summary})
                return ExternalApprovalDecisionImportResult(
                    run_id=request.run_id,
                    step_id=_nullable_str(existing_decision.get("step_id")),
                    approval_request_id=approval_request_id,
                    approval_decision=existing_decision,
                    external_approval_summary=summary,
                    run_status=run_status,
                    orchestration=orchestration,
                    validation=PlanValidationResult(status="valid"),
                    ledger_recorded=True,
                )

        governance_summary = _governance_summary(self._governance, request.run_id, run_status)
        matched_submission = _latest_matching_submission_event(entry, approval_request_id)
        matched_submission_reference = (
            _submission_reference_for_decision(matched_submission)
            if matched_submission is not None
            else None
        )
        event = {
            "recovery_event_id": f"external_approval_decision_{uuid4().hex[:12]}",
            "event_type": EXTERNAL_APPROVAL_DECISION_EVENT_TYPE,
            "status": "imported",
            "decision_intent": request.decision_intent,
            "approval_request_id": approval_request_id,
            "approval_decision_id": decision_id,
            "approval_decision_status": approval_decision.get("decision_status"),
            "submission_status": "submitted" if matched_submission_reference is not None else "not_submitted",
            "matched_submission_reference": matched_submission_reference,
            "approval_scope": approval_request.get("approval_scope"),
            "run_id": request.run_id,
            "step_id": _nullable_str(approval_decision.get("step_id")),
            "capability_id": _nullable_str(approval_decision.get("capability_id")),
            "policy_pack_id": request_event.get("policy_pack_id"),
            "approval_request_fingerprint": request_event.get("approval_request_fingerprint"),
            "approval_decision_fingerprint": decision_fingerprint,
            "decided_by": _safe_record(approval_decision.get("decided_by")),
            "decision_summary": _safe_record(approval_decision.get("decision_summary")),
            "approval_decision_snapshot": approval_decision,
            "imported_by": {
                "actor_id": governance_summary.get("actor_id") or "local_user",
                "actor_role": governance_summary.get("actor_role") or "unknown_role",
                "effective_actor_role": governance_summary.get("effective_actor_role") or "unknown_role",
            },
            "imported_at_utc": _utc_now_label(),
            "advisory_only": True,
            "execution_permitted": False,
        }
        _reject_unsafe_event(
            event,
            step_id=_nullable_str(approval_decision.get("step_id")),
            capability_id=_nullable_str(approval_decision.get("capability_id")),
            code="unsafe_external_approval_decision_record",
        )

        try:
            recorded_entry = self._ledger.append_recovery_event(request.run_id, event)
        except ValueError as exc:
            raise _rejected(
                "unsafe_external_approval_decision_record",
                "The external approval decision could not be safely ledgered.",
                step_id=_nullable_str(approval_decision.get("step_id")),
                capability_id=_nullable_str(approval_decision.get("capability_id")),
            ) from exc

        summary = external_approval_summary_for_entry(recorded_entry)
        run_status = run_status.model_copy(update={"external_approval_summary": summary})
        orchestration = orchestration.model_copy(update={"external_approval_summary": summary})
        return ExternalApprovalDecisionImportResult(
            run_id=request.run_id,
            step_id=_nullable_str(approval_decision.get("step_id")),
            approval_request_id=approval_request_id,
            approval_decision=approval_decision,
            external_approval_summary=summary,
            run_status=run_status,
            orchestration=orchestration,
            validation=PlanValidationResult(status="valid"),
            ledger_recorded=True,
        )

    def _validate_approval_request(
        self,
        approval_request: dict[str, Any],
        *,
        step_id: str | None,
        capability_id: str | None,
    ) -> None:
        unsafe_issues = find_unsafe_payload_issues(approval_request, root="external_approval_request")
        if unsafe_issues:
            raise RuntimeValidationError(
                PlanValidationResult(
                    status="rejected",
                    errors=[
                        issue.model_copy(
                            update={
                                "code": "unsafe_external_approval_request_payload",
                                "step_id": step_id,
                                "capability_id": capability_id,
                            }
                        )
                        for issue in unsafe_issues
                    ],
                )
            )
        try:
            self._contract_loader.validate_agent_contract_payload(
                approval_request,
                EXTERNAL_APPROVAL_REQUEST_CONTRACT_SCHEMA,
            )
        except Exception as exc:
            raise _rejected(
                "external_approval_request_contract_validation_failed",
                "The generated external approval request did not validate against the canonical contract.",
                step_id=step_id,
                capability_id=capability_id,
            ) from exc

    def _validate_approval_decision(self, approval_decision: dict[str, Any]) -> None:
        unsafe_issues = find_unsafe_payload_issues(approval_decision, root="external_approval_decision")
        if unsafe_issues:
            raise RuntimeValidationError(
                PlanValidationResult(
                    status="rejected",
                    errors=[
                        issue.model_copy(update={"code": "unsafe_external_approval_decision_payload"})
                        for issue in unsafe_issues
                    ],
                )
            )
        try:
            self._contract_loader.validate_agent_contract_payload(
                approval_decision,
                EXTERNAL_APPROVAL_DECISION_CONTRACT_SCHEMA,
            )
        except Exception as exc:
            raise _rejected(
                "external_approval_decision_contract_validation_failed",
                "The imported external approval decision did not validate against the canonical contract.",
            ) from exc
        if approval_decision.get("execution_permitted") is not False:
            raise _rejected(
                "external_approval_decision_execution_permitted",
                "Imported external approval decisions are advisory and must not permit execution in this slice.",
                step_id=_nullable_str(approval_decision.get("step_id")),
                capability_id=_nullable_str(approval_decision.get("capability_id")),
            )
        validation = approval_decision.get("validation")
        if not isinstance(validation, dict) or validation.get("status") != "valid":
            raise _rejected(
                "external_approval_decision_validation_not_valid",
                "Imported external approval decisions must include a valid validation summary.",
                step_id=_nullable_str(approval_decision.get("step_id")),
                capability_id=_nullable_str(approval_decision.get("capability_id")),
            )


class ExternalApprovalSubmissionService:
    def __init__(
        self,
        *,
        ledger: InMemoryLedger,
        contract_loader: QuantSuiteContractLoader,
        governance: Any | None = None,
    ) -> None:
        self._ledger = ledger
        self._contract_loader = contract_loader
        self._governance = governance

    def submit_request(
        self,
        request: ExternalApprovalSubmissionRequest,
        *,
        run_status: RunStatusResult,
        orchestration: RunOrchestrationResult,
    ) -> ExternalApprovalSubmissionResult:
        entry = self._ledger.get(request.run_id)
        if entry is None:
            raise _rejected("unknown_run", "No recorded run was found for the requested run_id.")
        _reject_unsafe_ledger(entry, code="unsafe_ledger_external_approval_submission")
        plan_id = _active_plan_id(entry)
        if plan_id is None:
            raise _rejected(
                "missing_plan_snapshot",
                "The recorded run is missing a contract-valid active plan snapshot.",
            )
        if entry.final_status in TERMINAL_APPROVAL_DECISION_FINAL_STATUSES:
            raise _rejected(
                "terminal_run_external_approval_submission",
                "Terminal runs cannot submit external approval packages.",
            )
        request_event = _matching_request_event(entry, request.approval_request_id)
        if request_event is None:
            raise _rejected(
                "missing_external_approval_request_preview",
                "The submission does not match a ledgered external approval request preview.",
            )
        approval_request = request_event.get("approval_request_snapshot")
        if not isinstance(approval_request, dict):
            raise _rejected(
                "malformed_external_approval_request_preview",
                "The ledgered external approval request preview is malformed.",
            )
        _validate_contract_payload(
            contract_loader=self._contract_loader,
            payload=approval_request,
            schema_name=EXTERNAL_APPROVAL_REQUEST_CONTRACT_SCHEMA,
            code="external_approval_request_contract_validation_failed",
            message="The ledgered external approval request did not validate against the canonical contract.",
            step_id=_nullable_str(approval_request.get("step_id")),
            capability_id=_nullable_str(approval_request.get("capability_id")),
        )
        if _latest_matching_decision_event(entry, request.approval_request_id) is not None:
            raise _rejected(
                "external_approval_decision_already_imported",
                "Approval packages cannot be submitted after a matching decision has already been imported.",
                step_id=_nullable_str(approval_request.get("step_id")),
                capability_id=_nullable_str(approval_request.get("capability_id")),
            )
        adapter_summary = external_approval_submission_adapter_status()
        if not adapter_summary["enabled"]:
            raise _rejected(
                "external_approval_submission_adapter_disabled",
                "External approval submission adapter is disabled.",
                step_id=_nullable_str(approval_request.get("step_id")),
                capability_id=_nullable_str(approval_request.get("capability_id")),
            )

        governance_summary = _governance_summary(self._governance, request.run_id, run_status)
        fingerprint = _submission_fingerprint(
            entry=entry,
            approval_request_event=request_event,
            plan_id=plan_id,
            governance_summary=governance_summary,
        )
        existing = _existing_submission(entry, fingerprint)
        if existing is not None:
            submission = existing.get("external_approval_submission_snapshot")
            if isinstance(submission, dict):
                self._validate_submission(submission)
                summary = external_approval_summary_for_entry(entry)
                run_status = run_status.model_copy(update={"external_approval_summary": summary})
                orchestration = orchestration.model_copy(update={"external_approval_summary": summary})
                return ExternalApprovalSubmissionResult(
                    run_id=request.run_id,
                    step_id=_nullable_str(submission.get("step_id")),
                    approval_request_id=request.approval_request_id,
                    external_approval_submission=submission,
                    run_status=run_status,
                    orchestration=orchestration,
                    validation=PlanValidationResult(status="valid"),
                    ledger_recorded=True,
                )

        submission = _submission_payload(
            request=request,
            approval_request=approval_request,
            request_event=request_event,
            governance_summary=governance_summary,
            adapter_summary=adapter_summary,
            fingerprint=fingerprint,
        )
        self._validate_submission(submission)
        try:
            _write_local_outbox_submission(submission)
        except OSError as exc:
            raise _rejected(
                "external_approval_submission_adapter_failed",
                "External approval submission adapter could not write the redacted local outbox item.",
                step_id=_nullable_str(submission.get("step_id")),
                capability_id=_nullable_str(submission.get("capability_id")),
            ) from exc

        event = {
            "recovery_event_id": f"external_approval_submission_{uuid4().hex[:12]}",
            "event_type": EXTERNAL_APPROVAL_SUBMISSION_EVENT_TYPE,
            "status": "submitted",
            "submission_intent": request.submission_intent,
            "run_id": request.run_id,
            "step_id": _nullable_str(submission.get("step_id")),
            "capability_id": _nullable_str(submission.get("capability_id")),
            "approval_request_id": request.approval_request_id,
            "approval_scope": submission.get("approval_scope"),
            "policy_pack_id": governance_summary.get("policy_pack_id"),
            "approval_request_fingerprint": request_event.get("approval_request_fingerprint"),
            "external_approval_submission_id": submission["external_approval_submission_id"],
            "external_approval_submission_fingerprint": fingerprint,
            "adapter_summary": submission["adapter_summary"],
            "submission_reference": submission["submission_reference"],
            "external_approval_submission_snapshot": submission,
            "submitted_by": {
                "actor_id": governance_summary.get("actor_id") or "local_user",
                "actor_role": governance_summary.get("actor_role") or "unknown_role",
                "effective_actor_role": governance_summary.get("effective_actor_role") or "unknown_role",
            },
            "submitted_at_utc": submission["submitted_at_utc"],
            "execution_permitted": False,
        }
        _reject_unsafe_event(
            event,
            step_id=_nullable_str(submission.get("step_id")),
            capability_id=_nullable_str(submission.get("capability_id")),
            code="unsafe_external_approval_submission_record",
        )
        try:
            recorded_entry = self._ledger.append_recovery_event(request.run_id, event)
        except ValueError as exc:
            raise _rejected(
                "unsafe_external_approval_submission_record",
                "The external approval submission could not be safely ledgered.",
                step_id=_nullable_str(submission.get("step_id")),
                capability_id=_nullable_str(submission.get("capability_id")),
            ) from exc

        summary = external_approval_summary_for_entry(recorded_entry)
        run_status = run_status.model_copy(update={"external_approval_summary": summary})
        orchestration = orchestration.model_copy(update={"external_approval_summary": summary})
        return ExternalApprovalSubmissionResult(
            run_id=request.run_id,
            step_id=_nullable_str(submission.get("step_id")),
            approval_request_id=request.approval_request_id,
            external_approval_submission=submission,
            run_status=run_status,
            orchestration=orchestration,
            validation=PlanValidationResult(status="valid"),
            ledger_recorded=True,
        )

    def list_submissions(self, run_id: str) -> ExternalApprovalSubmissionListResult:
        entry = self._ledger.get(run_id)
        if entry is None:
            raise _rejected("unknown_run", "No recorded run was found for the requested run_id.")
        _reject_unsafe_ledger(entry, code="unsafe_ledger_external_approval_submission_status")

        summaries: list[ExternalApprovalSubmissionSummary] = []
        integrity_summary = _ledger_integrity_summary(entry)
        for event in entry.recovery_events:
            if not isinstance(event, dict):
                continue
            if event.get("event_type") != EXTERNAL_APPROVAL_SUBMISSION_EVENT_TYPE:
                continue
            summary = _external_approval_submission_summary(event, entry, integrity_summary)
            summaries.append(summary)

        return ExternalApprovalSubmissionListResult(
            run_id=run_id,
            submissions=summaries,
            external_approval_summary=external_approval_summary_for_entry(entry),
            validation=PlanValidationResult(status="valid"),
            ledger_recorded=False,
        )

    def _validate_submission(self, submission: dict[str, Any]) -> None:
        _validate_contract_payload(
            contract_loader=self._contract_loader,
            payload=submission,
            schema_name=EXTERNAL_APPROVAL_SUBMISSION_CONTRACT_SCHEMA,
            code="external_approval_submission_contract_validation_failed",
            message="The generated external approval submission did not validate against the canonical contract.",
            step_id=_nullable_str(submission.get("step_id")),
            capability_id=_nullable_str(submission.get("capability_id")),
        )
        if submission.get("execution_permitted") is not False:
            raise _rejected(
                "external_approval_submission_execution_permitted",
                "External approval submissions must not permit execution.",
                step_id=_nullable_str(submission.get("step_id")),
                capability_id=_nullable_str(submission.get("capability_id")),
            )


def external_approval_summary_for_entry(entry: LedgerEntry) -> dict[str, Any]:
    request_events = [
        event
        for event in entry.recovery_events
        if isinstance(event, dict) and event.get("event_type") == EXTERNAL_APPROVAL_EVENT_TYPE
    ]
    decision_events = [
        event
        for event in entry.recovery_events
        if isinstance(event, dict) and event.get("event_type") == EXTERNAL_APPROVAL_DECISION_EVENT_TYPE
    ]
    submission_events = [
        event
        for event in entry.recovery_events
        if isinstance(event, dict) and event.get("event_type") == EXTERNAL_APPROVAL_SUBMISSION_EVENT_TYPE
    ]
    latest_request = request_events[-1] if request_events else None
    latest_submission = submission_events[-1] if submission_events else None
    latest_decision = decision_events[-1] if decision_events else None
    latest_request_summary = _approval_request_event_summary(latest_request) if latest_request else None
    latest_submission_summary = _approval_submission_event_summary(latest_submission) if latest_submission else None
    latest_decision_summary = _approval_decision_event_summary(latest_decision) if latest_decision else None
    matching_decision_event = None
    if latest_submission is not None:
        matching_decision_event = _latest_matching_decision_event(
            entry,
            str(latest_submission.get("approval_request_id") or ""),
        )
    elif latest_request is not None:
        matching_decision_event = _latest_matching_decision_event(
            entry,
            str(latest_request.get("approval_request_id") or ""),
        )
    latest_matching_decision_summary = (
        _approval_decision_event_summary(matching_decision_event)
        if matching_decision_event is not None
        else None
    )
    outbox_status = (
        str(latest_submission_summary.get("outbox_status") or "not_checked")
        if latest_submission_summary
        else "not_checked"
    )
    status = "not_requested"
    if latest_request_summary:
        status = "request_previewed"
    if latest_submission_summary:
        status = "submitted"
    if latest_decision_summary:
        status = str(latest_decision_summary.get("decision_status") or "decision_imported")
    return {
        "status": status,
        "request_count": len(request_events),
        "submission_count": len(submission_events),
        "decision_count": len(decision_events),
        "latest_request": latest_request_summary,
        "latest_submission": latest_submission_summary,
        "latest_decision": latest_decision_summary,
        "latest_matching_decision": latest_matching_decision_summary,
        "outbox_status": outbox_status,
        "enforcement_mode": "advisory_only",
        "execution_permitted": False,
    }


def external_approval_submission_adapter_status() -> dict[str, Any]:
    configured_mode = os.environ.get("QUANT_AGENT_EXTERNAL_APPROVAL_ADAPTER", DEFAULT_EXTERNAL_APPROVAL_ADAPTER)
    mode = configured_mode.strip().lower() or DEFAULT_EXTERNAL_APPROVAL_ADAPTER
    diagnostics: list[dict[str, Any]] = []
    if mode not in SUPPORTED_EXTERNAL_APPROVAL_ADAPTERS:
        diagnostics.append(
            {
                "code": "unsupported_external_approval_adapter",
                "message": "Configured external approval adapter is not supported; submissions are disabled.",
            }
        )
        return {
            "adapter_mode": mode,
            "enabled": False,
            "supports_external_network": False,
            "safe_storage_label": LOCAL_OUTBOX_SAFE_LABEL,
            "disabled_reason": "unsupported_adapter",
            "diagnostics": diagnostics,
        }
    if mode == "disabled":
        return {
            "adapter_mode": mode,
            "enabled": False,
            "supports_external_network": False,
            "safe_storage_label": LOCAL_OUTBOX_SAFE_LABEL,
            "disabled_reason": "adapter_disabled",
            "diagnostics": diagnostics,
        }
    return {
        "adapter_mode": "local_outbox",
        "enabled": True,
        "supports_external_network": False,
        "safe_storage_label": LOCAL_OUTBOX_SAFE_LABEL,
        "disabled_reason": None,
        "diagnostics": diagnostics,
    }


def _approval_request_payload(
    *,
    request: ExternalApprovalPreviewRequest,
    entry: LedgerEntry,
    step: dict[str, Any] | None,
    plan_id: str,
    governance_summary: dict[str, Any],
    separation_of_duties_summary: dict[str, Any],
    run_status: RunStatusResult,
    orchestration: RunOrchestrationResult,
    support_bundle: AgentSupportBundleResult,
    fingerprint: str,
) -> dict[str, Any]:
    step_id = request.step_id if request.approval_scope == "step" else None
    capability_id = _step_capability_id(step)
    requester = {
        "actor_id": governance_summary.get("actor_id") or "local_user",
        "actor_role": governance_summary.get("actor_role") or "unknown_role",
        "effective_actor_role": governance_summary.get("effective_actor_role") or "unknown_role",
        "policy_pack_id": governance_summary.get("policy_pack_id") or "unknown_policy_pack",
    }
    return {
        "schema_version": "1.0",
        "data_policy": "summaries_and_references_only",
        "approval_request_id": f"external_approval_request_{fingerprint[:16]}",
        "run_id": request.run_id,
        "step_id": step_id,
        "capability_id": capability_id,
        "approval_scope": request.approval_scope,
        "requester": requester,
        "governance_summary": {
            "environment": governance_summary.get("environment") or "local_development",
            "policy_pack_id": governance_summary.get("policy_pack_id") or "unknown_policy_pack",
            "actor_role": governance_summary.get("actor_role") or "unknown_role",
            "effective_actor_role": governance_summary.get("effective_actor_role") or "unknown_role",
            "fallback_active": bool(governance_summary.get("fallback_active")),
        },
        "separation_of_duties_summary": {
            "support_level": separation_of_duties_summary.get("support_level") or "not_available",
            "actor_exempt": bool(separation_of_duties_summary.get("actor_exempt")),
            "active_rule_ids": _safe_list(separation_of_duties_summary.get("active_rule_ids")),
            "blocked": bool(separation_of_duties_summary.get("blocked")),
            "blocked_routes": _safe_list(separation_of_duties_summary.get("blocked_routes")),
        },
        "run_status_summary": {
            "run_state": run_status.run_state,
            "final_status": run_status.final_status,
            "user_goal_summary": run_status.user_goal_summary,
            "allowed_next_actions": run_status.allowed_next_actions,
        },
        "run_progress_summary": run_status.run_progress_summary.model_dump(mode="json"),
        "orchestration_summary": _orchestration_summary(orchestration, step_id=step_id),
        "ledger_integrity_summary": (
            support_bundle.ledger_integrity_summary.model_dump(mode="json")
            if support_bundle.ledger_integrity_summary is not None
            else {"status": "not_available", "algorithm": None, "sequence_number": 0, "payload_hash": None}
        ),
        "support_bundle_reference": {
            "reference_type": "agent_support_bundle",
            "bundle_id": support_bundle.bundle_id,
            "run_id": support_bundle.run_id,
            "data_policy": support_bundle.data_policy,
            "redaction_status": "safe",
            "validation_status": support_bundle.validation.status,
        },
        "evidence_references": _evidence_references(
            entry=entry,
            plan_id=plan_id,
            step=step,
            support_bundle=support_bundle,
        ),
        "redaction_report": {
            "data_policy": "summaries_and_references_only",
            "raw_payloads_included": False,
            "unsafe_issue_count": 0,
            "excluded_categories": [
                "raw_rows",
                "raw_paths",
                "urls",
                "bucket_names",
                "secrets",
                "credentials",
                "raw_prompts",
                "raw_provider_responses",
                "raw_app_payloads",
            ],
        },
        "validation": {
            "status": "valid",
            "errors": [],
            "warnings": [],
        },
        "external_submission_status": "not_submitted",
    }


def _submission_payload(
    *,
    request: ExternalApprovalSubmissionRequest,
    approval_request: dict[str, Any],
    request_event: dict[str, Any],
    governance_summary: dict[str, Any],
    adapter_summary: dict[str, Any],
    fingerprint: str,
) -> dict[str, Any]:
    submission_id = f"external_approval_submission_{fingerprint[:16]}"
    support_bundle_reference = _safe_record(approval_request.get("support_bundle_reference"))
    return {
        "schema_version": "1.0",
        "data_policy": "summaries_and_references_only",
        "external_approval_submission_id": submission_id,
        "run_id": request.run_id,
        "approval_request_id": request.approval_request_id,
        "step_id": _nullable_str(approval_request.get("step_id")),
        "capability_id": _nullable_str(approval_request.get("capability_id")),
        "approval_scope": str(approval_request.get("approval_scope") or "run"),
        "governance_summary": {
            "environment": governance_summary.get("environment") or "local_development",
            "policy_pack_id": governance_summary.get("policy_pack_id") or "unknown_policy_pack",
            "actor_id": governance_summary.get("actor_id") or "local_user",
            "actor_role": governance_summary.get("actor_role") or "unknown_role",
            "effective_actor_role": governance_summary.get("effective_actor_role") or "unknown_role",
            "fallback_active": bool(governance_summary.get("fallback_active")),
        },
        "adapter_summary": {
            "adapter_mode": adapter_summary.get("adapter_mode") or "local_outbox",
            "enabled": bool(adapter_summary.get("enabled")),
            "supports_external_network": False,
            "safe_storage_label": LOCAL_OUTBOX_SAFE_LABEL,
            "disabled_reason": adapter_summary.get("disabled_reason"),
        },
        "submission_status": "submitted",
        "submission_reference": {
            "reference_type": "local_outbox_submission",
            "reference_id": submission_id,
            "label": "Local redacted approval outbox item",
        },
        "submitted_at_utc": _utc_now_label(),
        "approval_request_reference": {
            "reference_type": "external_approval_request_preview",
            "reference_id": request.approval_request_id,
            "data_policy": "summaries_and_references_only",
            "status": request_event.get("status") or "previewed",
        },
        "support_bundle_reference": {
            "reference_type": "agent_support_bundle",
            "bundle_id": support_bundle_reference.get("bundle_id") or "support_bundle_not_available",
            "run_id": support_bundle_reference.get("run_id") or request.run_id,
            "data_policy": "summaries_and_references_only",
            "redaction_status": support_bundle_reference.get("redaction_status") or "safe",
        },
        "redaction_report": {
            "data_policy": "summaries_and_references_only",
            "raw_payloads_included": False,
            "unsafe_issue_count": 0,
            "excluded_categories": [
                "raw_rows",
                "raw_paths",
                "urls",
                "bucket_names",
                "secrets",
                "credentials",
                "raw_prompts",
                "raw_provider_responses",
                "raw_app_payloads",
            ],
        },
        "validation": {
            "status": "valid",
            "errors": [],
            "warnings": [],
        },
        "execution_permitted": False,
    }


def _orchestration_summary(
    orchestration: RunOrchestrationResult,
    *,
    step_id: str | None,
) -> dict[str, Any]:
    steps = [
        {
            "step_id": step.step_id,
            "capability_id": step.capability_id,
            "app_id": step.app_id,
            "title": step.title,
            "status": step.status,
            "is_current": step.is_current,
            "required_gate": step.required_gate,
            "allowed_actions": step.allowed_actions,
        }
        for step in orchestration.steps
        if step_id is None or step.step_id == step_id
    ]
    return {
        "run_id": orchestration.run_id,
        "plan_id": orchestration.plan_id,
        "current_step_id": orchestration.current_step_id,
        "allowed_next_actions": orchestration.allowed_next_actions,
        "step_count": len(orchestration.steps),
        "selected_step_id": step_id,
        "steps": steps,
    }


def _evidence_references(
    *,
    entry: LedgerEntry,
    plan_id: str,
    step: dict[str, Any] | None,
    support_bundle: AgentSupportBundleResult,
) -> list[dict[str, Any]]:
    references = [
        {
            "reference_type": "plan_snapshot",
            "reference_id": plan_id,
            "summary": "Active plan snapshot is available in the durable agent ledger.",
        },
        {
            "reference_type": "agent_support_bundle",
            "reference_id": support_bundle.bundle_id,
            "summary": "Redacted support bundle was generated from durable run evidence.",
        },
        {
            "reference_type": "ledger_integrity",
            "reference_id": support_bundle.ledger_integrity_summary.payload_hash or "ledger_hash_not_available",
            "summary": f"Ledger integrity status is {support_bundle.ledger_integrity_summary.status}.",
        },
    ]
    if step is not None:
        references.append(
            {
                "reference_type": "plan_step",
                "reference_id": str(step.get("step_id") or ""),
                "summary": str(step.get("title") or step.get("capability_id") or "Selected plan step"),
            }
        )
    if entry.preflight_records:
        references.append(
            {
                "reference_type": "preflight_records",
                "reference_id": f"preflight_count_{len(entry.preflight_records)}",
                "summary": "Safe app-owned preflight summaries are present in the ledger.",
            }
        )
    if entry.action_results:
        references.append(
            {
                "reference_type": "action_results",
                "reference_id": f"action_result_count_{len(entry.action_results)}",
                "summary": "Safe app-owned action result summaries are present in the ledger.",
            }
        )
    return references


def _governance_summary(
    governance: Any | None,
    run_id: str,
    run_status: RunStatusResult,
) -> dict[str, Any]:
    if governance is not None:
        return governance.run_summary(run_id).model_dump(mode="json")
    if run_status.governance_summary is not None:
        return run_status.governance_summary.model_dump(mode="json")
    return {
        "policy_pack_id": "unknown_policy_pack",
        "environment": "local_development",
        "actor_id": "local_user",
        "actor_role": "local_developer_operator",
        "effective_actor_role": "local_developer_operator",
        "fallback_active": True,
    }


def _sod_summary(governance: Any | None, run_id: str, run_status: RunStatusResult) -> dict[str, Any]:
    if governance is not None:
        return governance.separation_of_duties_run_summary(run_id).model_dump(mode="json")
    if run_status.separation_of_duties_summary is not None:
        return run_status.separation_of_duties_summary.model_dump(mode="json")
    return {
        "support_level": "not_available",
        "actor_exempt": False,
        "active_rule_ids": [],
        "blocked": False,
        "blocked_routes": [],
    }


def _approval_fingerprint(
    *,
    entry: LedgerEntry,
    approval_scope: str,
    step_id: str | None,
    plan_id: str,
    policy_pack_id: str,
) -> str:
    payload = entry.model_dump(mode="json")
    payload.pop("ledger_integrity", None)
    payload["recovery_events"] = [
        event
        for event in payload.get("recovery_events", [])
        if not (isinstance(event, dict) and event.get("event_type") == EXTERNAL_APPROVAL_EVENT_TYPE)
    ]
    canonical = {
        "run_id": entry.run_id,
        "approval_scope": approval_scope,
        "step_id": step_id,
        "plan_id": plan_id,
        "policy_pack_id": policy_pack_id,
        "ledger_evidence": payload,
    }
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _existing_preview(entry: LedgerEntry, fingerprint: str) -> dict[str, Any] | None:
    for event in reversed(entry.recovery_events):
        if not isinstance(event, dict):
            continue
        if (
            event.get("event_type") == EXTERNAL_APPROVAL_EVENT_TYPE
            and event.get("approval_request_fingerprint") == fingerprint
            and event.get("status") == "previewed"
        ):
            return event
    return None


def _matching_request_event(entry: LedgerEntry, approval_request_id: str) -> dict[str, Any] | None:
    if not approval_request_id:
        return None
    for event in reversed(entry.recovery_events):
        if not isinstance(event, dict):
            continue
        if (
            event.get("event_type") == EXTERNAL_APPROVAL_EVENT_TYPE
            and event.get("status") == "previewed"
            and event.get("approval_request_id") == approval_request_id
        ):
            return event
    return None


def _submission_fingerprint(
    *,
    entry: LedgerEntry,
    approval_request_event: dict[str, Any],
    plan_id: str,
    governance_summary: dict[str, Any],
) -> str:
    canonical = {
        "run_id": entry.run_id,
        "plan_id": plan_id,
        "approval_request_id": approval_request_event.get("approval_request_id"),
        "approval_request_fingerprint": approval_request_event.get("approval_request_fingerprint"),
        "policy_pack_id": governance_summary.get("policy_pack_id"),
        "adapter_mode": "local_outbox",
    }
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _existing_submission(entry: LedgerEntry, fingerprint: str) -> dict[str, Any] | None:
    for event in reversed(entry.recovery_events):
        if not isinstance(event, dict):
            continue
        if (
            event.get("event_type") == EXTERNAL_APPROVAL_SUBMISSION_EVENT_TYPE
            and event.get("external_approval_submission_fingerprint") == fingerprint
            and event.get("status") == "submitted"
        ):
            return event
    return None


def _existing_decision_import(entry: LedgerEntry, approval_decision_id: str) -> dict[str, Any] | None:
    if not approval_decision_id:
        return None
    for event in reversed(entry.recovery_events):
        if not isinstance(event, dict):
            continue
        if (
            event.get("event_type") == EXTERNAL_APPROVAL_DECISION_EVENT_TYPE
            and event.get("approval_decision_id") == approval_decision_id
            and event.get("status") == "imported"
        ):
            return event
    return None


def _latest_matching_decision_event(entry: LedgerEntry, approval_request_id: str) -> dict[str, Any] | None:
    for event in reversed(entry.recovery_events):
        if not isinstance(event, dict):
            continue
        if (
            event.get("event_type") == EXTERNAL_APPROVAL_DECISION_EVENT_TYPE
            and event.get("approval_request_id") == approval_request_id
            and event.get("status") == "imported"
        ):
            return event
    return None


def _latest_matching_submission_event(entry: LedgerEntry, approval_request_id: str) -> dict[str, Any] | None:
    if not approval_request_id:
        return None
    for event in reversed(entry.recovery_events):
        if not isinstance(event, dict):
            continue
        if (
            event.get("event_type") == EXTERNAL_APPROVAL_SUBMISSION_EVENT_TYPE
            and event.get("approval_request_id") == approval_request_id
            and event.get("status") == "submitted"
        ):
            return event
    return None


def _decision_fingerprint(approval_decision: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            approval_decision,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _assert_decision_matches_request(
    approval_decision: dict[str, Any],
    approval_request: dict[str, Any],
) -> None:
    expected_step_id = _nullable_str(approval_request.get("step_id"))
    expected_capability_id = _nullable_str(approval_request.get("capability_id"))
    decision_step_id = _nullable_str(approval_decision.get("step_id"))
    decision_capability_id = _nullable_str(approval_decision.get("capability_id"))
    if approval_decision.get("run_id") != approval_request.get("run_id"):
        raise _rejected(
            "external_approval_decision_request_mismatch",
            "The approval decision run does not match the ledgered approval request.",
            step_id=decision_step_id,
            capability_id=decision_capability_id,
        )
    if approval_decision.get("approval_request_id") != approval_request.get("approval_request_id"):
        raise _rejected(
            "external_approval_decision_request_mismatch",
            "The approval decision request id does not match the ledgered approval request.",
            step_id=decision_step_id,
            capability_id=decision_capability_id,
        )
    if decision_step_id != expected_step_id or decision_capability_id != expected_capability_id:
        raise _rejected(
            "external_approval_decision_request_mismatch",
            "The approval decision step or capability does not match the ledgered approval request.",
            step_id=decision_step_id,
            capability_id=decision_capability_id,
        )


def _approval_request_event_summary(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_type": EXTERNAL_APPROVAL_EVENT_TYPE,
        "status": event.get("status"),
        "approval_request_id": event.get("approval_request_id"),
        "approval_scope": event.get("approval_scope"),
        "step_id": event.get("step_id"),
        "capability_id": event.get("capability_id"),
        "policy_pack_id": event.get("policy_pack_id"),
        "external_submission_status": event.get("external_submission_status"),
        "created_at_utc": event.get("created_at_utc"),
        "execution_permitted": False,
    }


def _approval_decision_event_summary(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_type": EXTERNAL_APPROVAL_DECISION_EVENT_TYPE,
        "status": event.get("status"),
        "approval_request_id": event.get("approval_request_id"),
        "approval_decision_id": event.get("approval_decision_id"),
        "decision_status": event.get("approval_decision_status"),
        "submission_status": event.get("submission_status") or "not_submitted",
        "matched_submission_reference": _safe_record(event.get("matched_submission_reference")),
        "approval_scope": event.get("approval_scope"),
        "step_id": event.get("step_id"),
        "capability_id": event.get("capability_id"),
        "imported_at_utc": event.get("imported_at_utc"),
        "advisory_only": True,
        "execution_permitted": False,
    }


def _approval_submission_event_summary(event: dict[str, Any]) -> dict[str, Any]:
    adapter_summary = _safe_record(event.get("adapter_summary"))
    return {
        "event_type": EXTERNAL_APPROVAL_SUBMISSION_EVENT_TYPE,
        "status": event.get("status"),
        "approval_request_id": event.get("approval_request_id"),
        "external_approval_submission_id": event.get("external_approval_submission_id"),
        "approval_scope": event.get("approval_scope"),
        "step_id": event.get("step_id"),
        "capability_id": event.get("capability_id"),
        "policy_pack_id": event.get("policy_pack_id"),
        "adapter_mode": adapter_summary.get("adapter_mode") or "local_outbox",
        "adapter_summary": adapter_summary,
        "submission_reference": _safe_record(event.get("submission_reference")),
        "outbox_status": _outbox_status_for_submission_event(event),
        "submitted_at_utc": event.get("submitted_at_utc"),
        "execution_permitted": False,
    }


def _external_approval_submission_summary(
    event: dict[str, Any],
    entry: LedgerEntry,
    integrity_summary: LedgerIntegritySummary,
) -> ExternalApprovalSubmissionSummary:
    submission_snapshot = event.get("external_approval_submission_snapshot")
    submission = submission_snapshot if isinstance(submission_snapshot, dict) else {}
    approval_request_id = str(event.get("approval_request_id") or submission.get("approval_request_id") or "")
    adapter_summary = _safe_record(event.get("adapter_summary") or submission.get("adapter_summary"))
    matching_decision = _latest_matching_decision_event(entry, approval_request_id)
    return ExternalApprovalSubmissionSummary(
        external_approval_submission_id=str(
            event.get("external_approval_submission_id")
            or submission.get("external_approval_submission_id")
            or ""
        ),
        approval_request_id=approval_request_id,
        approval_scope=str(event.get("approval_scope") or submission.get("approval_scope") or "run"),
        step_id=_nullable_str(event.get("step_id") or submission.get("step_id")),
        capability_id=_nullable_str(event.get("capability_id") or submission.get("capability_id")),
        adapter_mode=str(adapter_summary.get("adapter_mode") or "local_outbox"),
        submission_status=str(event.get("status") or submission.get("submission_status") or "submitted"),
        outbox_status=_outbox_status_for_submission_event(event),
        submitted_at_utc=_nullable_str(event.get("submitted_at_utc") or submission.get("submitted_at_utc")),
        submission_reference=_safe_record(event.get("submission_reference") or submission.get("submission_reference")),
        latest_matching_decision=(
            _approval_decision_event_summary(matching_decision)
            if matching_decision is not None
            else None
        ),
        validation=PlanValidationResult(status="valid"),
        ledger_integrity_summary=integrity_summary,
    )


def _submission_reference_for_decision(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "reference_type": "external_approval_submission",
        "external_approval_submission_id": event.get("external_approval_submission_id"),
        "approval_request_id": event.get("approval_request_id"),
        "submission_status": event.get("status") or "submitted",
        "adapter_mode": _safe_record(event.get("adapter_summary")).get("adapter_mode") or "local_outbox",
        "submitted_at_utc": event.get("submitted_at_utc"),
        "outbox_status": _outbox_status_for_submission_event(event),
    }


def _outbox_status_for_submission_event(event: dict[str, Any]) -> str:
    adapter_summary = _safe_record(event.get("adapter_summary"))
    adapter_mode = str(adapter_summary.get("adapter_mode") or "local_outbox")
    if adapter_mode == "disabled":
        return "disabled"
    current_adapter = external_approval_submission_adapter_status()
    if not current_adapter.get("enabled"):
        return "disabled"
    if adapter_mode != "local_outbox":
        return "not_checked"
    try:
        return "present" if _outbox_file_for_submission_event(event).is_file() else "missing"
    except OSError:
        return "not_checked"


def _outbox_file_for_submission_event(event: dict[str, Any]) -> Path:
    submission_id = str(event.get("external_approval_submission_id") or "external_submission")
    file_stem = _safe_file_stem(submission_id)
    return _local_outbox_dir() / f"{file_stem}.json"


def _ledger_integrity_summary(entry: LedgerEntry) -> LedgerIntegritySummary:
    if entry.ledger_integrity is not None:
        return LedgerIntegritySummary.model_validate(entry.ledger_integrity.model_dump(mode="json"))
    return LedgerIntegritySummary(
        status="not_available",
        diagnostics=[
            {
                "code": "ledger_integrity_not_available",
                "message": "The ledger entry does not include file-backed integrity metadata.",
            }
        ],
    )


def _plan_step(entry: LedgerEntry, step_id: str) -> dict[str, Any] | None:
    snapshot = entry.plan_snapshot if isinstance(entry.plan_snapshot, dict) else {}
    steps = snapshot.get("proposed_steps")
    if not isinstance(steps, list):
        return None
    for step in steps:
        if isinstance(step, dict) and step.get("step_id") == step_id:
            return step
    return None


def _active_plan_id(entry: LedgerEntry) -> str | None:
    snapshot = entry.plan_snapshot if isinstance(entry.plan_snapshot, dict) else {}
    plan_id = snapshot.get("plan_id")
    if isinstance(plan_id, str) and plan_id:
        return plan_id
    return None


def _step_capability_id(step: dict[str, Any] | None) -> str | None:
    if not isinstance(step, dict):
        return None
    capability_id = step.get("capability_id")
    return capability_id if isinstance(capability_id, str) and capability_id else None


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_record(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _nullable_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _json_clone(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=True))


def _validate_contract_payload(
    *,
    contract_loader: QuantSuiteContractLoader,
    payload: dict[str, Any],
    schema_name: str,
    code: str,
    message: str,
    step_id: str | None = None,
    capability_id: str | None = None,
) -> None:
    unsafe_issues = find_unsafe_payload_issues(payload, root=schema_name)
    if unsafe_issues:
        raise RuntimeValidationError(
            PlanValidationResult(
                status="rejected",
                errors=[
                    issue.model_copy(
                        update={
                            "code": code.replace("contract_validation_failed", "unsafe_payload"),
                            "step_id": step_id,
                            "capability_id": capability_id,
                        }
                    )
                    for issue in unsafe_issues
                ],
            )
        )
    try:
        contract_loader.validate_agent_contract_payload(payload, schema_name)
    except Exception as exc:
        raise _rejected(
            code,
            message,
            step_id=step_id,
            capability_id=capability_id,
        ) from exc


def _write_local_outbox_submission(submission: dict[str, Any]) -> None:
    outbox_dir = _local_outbox_dir()
    outbox_dir.mkdir(parents=True, exist_ok=True)
    submission_id = str(submission.get("external_approval_submission_id") or "external_submission")
    file_stem = _safe_file_stem(submission_id)
    path = outbox_dir / f"{file_stem}.json"
    temp_path = outbox_dir / f"{file_stem}.tmp"
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(submission, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temp_path.replace(path)


def _safe_file_stem(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _local_outbox_dir() -> Path:
    configured = os.environ.get("QUANT_AGENT_EXTERNAL_APPROVAL_OUTBOX_DIR")
    if configured:
        return Path(configured)
    return Path.home() / ".quant_agent" / "external_approval_outbox"


def _reject_unsafe_ledger(
    entry: LedgerEntry,
    *,
    code: str = "unsafe_ledger_external_approval_request",
) -> None:
    payload = entry.model_dump(mode="json")
    unsafe_issues = find_unsafe_payload_issues(payload, root="ledger")
    if unsafe_issues:
        raise RuntimeValidationError(
            PlanValidationResult(
                status="rejected",
                errors=[
                    issue.model_copy(update={"code": code})
                    for issue in unsafe_issues
                ],
            )
        )


def _reject_unsafe_event(
    event: dict[str, Any],
    *,
    step_id: str | None,
    capability_id: str | None,
    code: str = "unsafe_external_approval_request_record",
) -> None:
    unsafe_issues = find_unsafe_payload_issues(event, root="external_approval_event")
    if unsafe_issues:
        raise RuntimeValidationError(
            PlanValidationResult(
                status="rejected",
                errors=[
                    issue.model_copy(
                        update={
                            "code": code,
                            "step_id": step_id,
                            "capability_id": capability_id,
                        }
                    )
                    for issue in unsafe_issues
                ],
            )
        )


def _utc_now_label() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _rejected(
    code: str,
    message: str,
    *,
    step_id: str | None = None,
    capability_id: str | None = None,
) -> RuntimeValidationError:
    return RuntimeValidationError(
        PlanValidationResult(
            status="rejected",
            errors=[
                ValidationIssue(
                    code=code,
                    message=message,
                    step_id=step_id,
                    capability_id=capability_id,
                )
            ],
        )
    )
