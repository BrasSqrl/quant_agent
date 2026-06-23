from __future__ import annotations

from typing import Any
from uuid import uuid4

from quant_agent_runtime.app_clients import AgentAppClient, AppClientError
from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.models import (
    LedgerEntry,
    PlanValidationResult,
    SampleAutopilotEligibility,
    SampleResetPreviewRequest,
    SampleResetPreviewResult,
    SampleResetRequest,
    SampleResetResult,
    ValidationIssue,
)
from quant_agent_runtime.orchestration import orchestration_for_entry
from quant_agent_runtime.redaction import find_unsafe_payload_issues
from quant_agent_runtime.run_state import run_state_for_entry
from quant_agent_runtime.sample_autopilot import (
    SampleAutopilotEvaluation,
    SampleAutopilotPreviewService,
    _reject_unsafe_autopilot_event,
    _sample_ownership_summary,
    _safe_label,
    _safe_string_list,
    _utc_now_label,
    _rejected,
)
from quant_agent_runtime.validation.errors import RuntimeValidationError


class SampleResetService:
    def __init__(
        self,
        *,
        ledger: InMemoryLedger,
        app_client: AgentAppClient,
        sample_workspace_root: Any | None = None,
        allowlist: set[str] | None = None,
    ) -> None:
        self._ledger = ledger
        self._app_client = app_client
        self._eligibility = SampleAutopilotPreviewService(
            ledger=ledger,
            sample_workspace_root=sample_workspace_root,
            allowlist=allowlist,
        )

    def preview_reset(self, request: SampleResetPreviewRequest) -> SampleResetPreviewResult:
        evaluation = self._evaluate_for_reset(request.run_id, request.current_context_summary)
        blockers = _reset_state_blockers(evaluation.entry)
        eligibility = _eligibility_with_reset_blockers(evaluation.eligibility, blockers)
        reset_preview_id = f"sample_reset_preview_{uuid4().hex[:12]}"
        validation = _validation_for_blockers(
            "sample_reset_preview_blocked",
            "The recorded run is not eligible for sample-owned reset preview.",
            eligibility.blockers,
        )
        event = {
            "recovery_event_id": reset_preview_id,
            "event_type": "sample_reset_preview",
            "status": "previewed" if eligibility.eligible else "blocked",
            "reset_intent": request.reset_intent,
            "run_id": request.run_id,
            "plan_id": evaluation.orchestration.plan_id,
            "created_by": "local_user",
            "created_at_utc": _utc_now_label(),
            "sample_ownership_summary": _sample_ownership_summary(eligibility),
            "reset_boundary_summary": _reset_boundary_summary(eligibility),
            "blockers": eligibility.blockers,
            "warnings": eligibility.warnings,
            "execution_permitted": False,
        }
        _reject_unsafe_autopilot_event(event)
        try:
            recorded_entry = self._ledger.append_recovery_event(request.run_id, event)
        except ValueError as exc:
            raise _rejected(
                "unsafe_sample_reset_preview_record",
                "The sample reset preview could not be safely ledgered.",
            ) from exc

        orchestration = orchestration_for_entry(recorded_entry)
        return SampleResetPreviewResult(
            run_id=request.run_id,
            reset_preview_id=reset_preview_id,
            sample_eligibility=eligibility,
            reset_boundary_summary=_reset_boundary_summary(eligibility),
            run_progress_summary=orchestration.run_progress_summary,
            orchestration=orchestration,
            validation=validation,
            ledger_recorded=True,
        )

    def reset_sample(self, request: SampleResetRequest) -> SampleResetResult:
        evaluation = self._evaluate_for_reset(request.run_id, request.current_context_summary)
        existing = _existing_reset_event(evaluation.entry, request.reset_preview_id)
        if existing is not None and existing.get("status") == "reset":
            return self._result_from_existing_reset(
                request=request,
                evaluation=evaluation,
                event=existing,
            )

        blockers = [
            *_reset_state_blockers(evaluation.entry),
            *_reset_preview_blockers(evaluation.entry, request.reset_preview_id, evaluation.eligibility),
        ]
        eligibility = _eligibility_with_reset_blockers(evaluation.eligibility, blockers)
        if blockers or not eligibility.eligible:
            return self._record_reset_event(
                request=request,
                evaluation=evaluation,
                eligibility=eligibility,
                reset_status="blocked",
                blockers=eligibility.blockers,
                validation=_validation_for_blockers(
                    "sample_reset_blocked",
                    "The recorded run is not eligible for sample-owned reset.",
                    eligibility.blockers,
                ),
            )

        try:
            app_response = self._app_client.reset_sample_workspaces()
        except AppClientError as exc:
            status = "app_unavailable" if exc.status_code == 503 else "app_rejected"
            return self._record_reset_event(
                request=request,
                evaluation=evaluation,
                eligibility=eligibility,
                reset_status=status,
                blockers=[f"app_status_{exc.status_code}"],
                validation=PlanValidationResult(
                    status="rejected",
                    errors=[
                        ValidationIssue(
                            code="sample_reset_app_unavailable"
                            if status == "app_unavailable"
                            else "sample_reset_app_rejected",
                            message="Quant Studio could not complete the sample-owned reset.",
                        )
                    ],
                ),
            )

        unsafe_issues = _unsafe_reset_app_response_issues(app_response)
        reset_summary = _reset_app_result_summary(app_response)
        if unsafe_issues or reset_summary.get("status") != "reset":
            return self._record_reset_event(
                request=request,
                evaluation=evaluation,
                eligibility=eligibility,
                reset_status="app_rejected",
                reset_result=reset_summary,
                blockers=["unsafe_app_response" if unsafe_issues else "unexpected_app_reset_status"],
                validation=PlanValidationResult(
                    status="rejected",
                    errors=[
                        ValidationIssue(
                            code="unsafe_sample_reset_app_response"
                            if unsafe_issues
                            else "sample_reset_app_rejected",
                            message="Quant Studio returned a reset response that could not be safely accepted.",
                        )
                    ],
                ),
            )

        return self._record_reset_event(
            request=request,
            evaluation=evaluation,
            eligibility=eligibility,
            reset_status="reset",
            reset_result=reset_summary,
            final=True,
        )

    def _evaluate_for_reset(
        self,
        run_id: str,
        current_context_summary: dict[str, Any],
    ) -> SampleAutopilotEvaluation:
        return self._eligibility.evaluate_eligibility(
            run_id=run_id,
            current_context_summary=current_context_summary,
            include_run_state_checks=False,
        )

    def _record_reset_event(
        self,
        *,
        request: SampleResetRequest,
        evaluation: SampleAutopilotEvaluation,
        eligibility: SampleAutopilotEligibility,
        reset_status: str,
        reset_result: dict[str, Any] | None = None,
        blockers: list[str] | None = None,
        warnings: list[str] | None = None,
        validation: PlanValidationResult | None = None,
        final: bool = False,
    ) -> SampleResetResult:
        event = {
            "recovery_event_id": f"sample_reset_{uuid4().hex[:12]}",
            "event_type": "sample_reset",
            "status": reset_status,
            "reset_intent": request.reset_intent,
            "reset_preview_id": request.reset_preview_id,
            "run_id": request.run_id,
            "plan_id": evaluation.orchestration.plan_id,
            "created_by": "local_user",
            "created_at_utc": _utc_now_label(),
            "sample_ownership_summary": _sample_ownership_summary(eligibility),
            "reset_boundary_summary": _reset_boundary_summary(eligibility),
            "reset_result_summary": reset_result,
            "blockers": blockers or [],
            "warnings": warnings or eligibility.warnings,
            "sample_owned_only": True,
            "execution_permitted": False,
        }
        _reject_unsafe_autopilot_event(event)
        try:
            recorded_entry = (
                self._ledger.append_sample_reset_event(request.run_id, event)
                if final
                else self._ledger.append_recovery_event(request.run_id, event)
            )
        except ValueError as exc:
            raise _rejected(
                "unsafe_sample_reset_record",
                "The sample reset event could not be safely ledgered.",
            ) from exc

        orchestration = orchestration_for_entry(recorded_entry)
        return SampleResetResult(
            run_id=request.run_id,
            reset_preview_id=request.reset_preview_id,
            reset_status=reset_status,
            reset_event=event,
            reset_result=reset_result,
            sample_eligibility=eligibility,
            reset_boundary_summary=_reset_boundary_summary(eligibility),
            run_progress_summary=orchestration.run_progress_summary,
            orchestration=orchestration,
            validation=validation or PlanValidationResult(status="valid"),
            ledger_recorded=True,
        )

    def _result_from_existing_reset(
        self,
        *,
        request: SampleResetRequest,
        evaluation: SampleAutopilotEvaluation,
        event: dict[str, Any],
    ) -> SampleResetResult:
        orchestration = orchestration_for_entry(evaluation.entry)
        reset_result = event.get("reset_result_summary")
        return SampleResetResult(
            run_id=request.run_id,
            reset_preview_id=request.reset_preview_id,
            reset_status="reset",
            reset_event=event,
            reset_result=reset_result if isinstance(reset_result, dict) else None,
            sample_eligibility=evaluation.eligibility,
            reset_boundary_summary=_reset_boundary_summary(evaluation.eligibility),
            run_progress_summary=orchestration.run_progress_summary,
            orchestration=orchestration,
            validation=PlanValidationResult(status="valid"),
            ledger_recorded=True,
        )


def _reset_state_blockers(entry: LedgerEntry) -> list[str]:
    run_state = run_state_for_entry(entry)
    blockers: list[str] = []
    if run_state == "paused":
        blockers.append("Paused runs must be resumed before sample-owned reset.")
    if run_state == "running":
        blockers.append("Running runs cannot be reset.")
    if run_state == "sample_reset" or entry.final_status == "sample_reset":
        blockers.append("The sample-owned demo run has already been reset.")
    return blockers


def _reset_preview_blockers(
    entry: LedgerEntry,
    reset_preview_id: str,
    eligibility: SampleAutopilotEligibility,
) -> list[str]:
    preview = _reset_preview_event(entry, reset_preview_id)
    if preview is None:
        return ["A matching ledgered reset preview is required before reset."]
    if preview.get("status") != "previewed":
        return ["The matching reset preview was blocked and cannot be used for reset."]
    summary = preview.get("sample_ownership_summary")
    preview_sample_id = summary.get("sample_workspace_id") if isinstance(summary, dict) else None
    if preview_sample_id != eligibility.sample_workspace_id:
        return ["The reset preview sample ownership does not match current context."]
    return []


def _reset_preview_event(entry: LedgerEntry, reset_preview_id: str) -> dict[str, Any] | None:
    for event in reversed(entry.recovery_events):
        if not isinstance(event, dict):
            continue
        if event.get("event_type") == "sample_reset_preview" and event.get("recovery_event_id") == reset_preview_id:
            return event
    return None


def _existing_reset_event(entry: LedgerEntry, reset_preview_id: str) -> dict[str, Any] | None:
    for event in reversed(entry.recovery_events):
        if not isinstance(event, dict):
            continue
        if event.get("event_type") == "sample_reset" and event.get("reset_preview_id") == reset_preview_id:
            return event
    return None


def _eligibility_with_reset_blockers(
    eligibility: SampleAutopilotEligibility,
    blockers: list[str],
) -> SampleAutopilotEligibility:
    all_blockers = [*eligibility.blockers, *blockers]
    return eligibility.model_copy(
        update={
            "eligible": not all_blockers,
            "status": "eligible" if not all_blockers else "blocked",
            "blockers": all_blockers,
        },
        deep=True,
    )


def _reset_boundary_summary(eligibility: SampleAutopilotEligibility) -> dict[str, Any]:
    labels = eligibility.safe_labels
    return {
        "sample_workspace_id": eligibility.sample_workspace_id,
        "sample_label": eligibility.sample_label,
        "lifecycle_id": eligibility.lifecycle_id,
        "sample_owned": eligibility.sample_owned,
        "sample_owned_only": eligibility.reset_boundary_available,
        "reset_boundary_available": eligibility.reset_boundary_available,
        "allowed_delete_scopes": _safe_string_list(labels.get("allowed_delete_scopes")),
        "protected_scopes": _safe_string_list(labels.get("protected_scopes")),
    }


def _reset_app_result_summary(payload: dict[str, Any]) -> dict[str, Any]:
    warnings = payload.get("warnings")
    warning_count = len(warnings) if isinstance(warnings, list) else 0
    return {
        "result_type": "sample_workspace_reset",
        "status": _safe_label(payload.get("status")) or "unknown",
        "deleted_lifecycle_ids": _safe_string_list(payload.get("deleted_lifecycle_ids")),
        "deleted_lifecycle_count": len(_safe_string_list(payload.get("deleted_lifecycle_ids"))),
        "warning_count": warning_count,
        "warning_labels": [f"warning_{index + 1}" for index in range(min(warning_count, 12))],
    }


def _unsafe_reset_app_response_issues(payload: dict[str, Any]) -> list[ValidationIssue]:
    scan_payload = {
        key: value
        for key, value in payload.items()
        if key != "lifecycle_response"
    }
    return find_unsafe_payload_issues(scan_payload, root="sample_reset_app_response")


def _validation_for_blockers(
    code: str,
    message: str,
    blockers: list[str],
) -> PlanValidationResult:
    if not blockers:
        return PlanValidationResult(status="valid")
    return PlanValidationResult(
        status="rejected",
        errors=[
            ValidationIssue(
                code=code,
                message=message,
            )
        ],
        warnings=[
            ValidationIssue(
                code="sample_reset_blocker",
                message=blocker,
            )
            for blocker in blockers[:8]
        ],
    )
