from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.models import (
    CancellationRequest,
    CancellationResult,
    LedgerEntry,
    PauseRequest,
    PauseResult,
    PlanValidationResult,
    ResumptionRequest,
    ResumptionResult,
    RunListResult,
    RunSummary,
    RunStatusResult,
    ValidationIssue,
)
from quant_agent_runtime.capability_discovery import CapabilityDiscoveryService
from quant_agent_runtime.orchestration import (
    latest_recovery_event_type,
    ledger_summary,
    orchestration_for_entry,
)
from quant_agent_runtime.redaction import find_unsafe_payload_issues
from quant_agent_runtime.run_state import run_state_for_entry
from quant_agent_runtime.user_workflow import user_workflow_summaries_for_entry
from quant_agent_runtime.validation.errors import RuntimeValidationError


_TERMINAL_RUN_STATES = {"completed", "completed_with_warnings", "failed_terminal", "sample_reset"}
_TERMINAL_OR_CANCELLED_RUN_STATES = {*_TERMINAL_RUN_STATES, "cancelled"}


class RunStatusService:
    def __init__(
        self,
        *,
        ledger: InMemoryLedger,
        capability_discovery: CapabilityDiscoveryService | None = None,
        governance: Any | None = None,
    ) -> None:
        self._ledger = ledger
        self._capability_discovery = capability_discovery
        self._governance = governance

    def get_run_status(self, run_id: str) -> RunStatusResult:
        entry = self._ledger.get(run_id)
        if entry is None:
            raise _rejected("unknown_run", "No recorded run was found for the requested run_id.")
        return _status_result(entry, governance=self._governance)

    def list_runs(
        self,
        *,
        lifecycle_id: str | None = None,
        app_id: str | None = None,
        capability_id: str | None = None,
        final_status: str | None = None,
        limit: int = 50,
    ) -> RunListResult:
        bounded_limit = min(max(limit, 1), 200)
        summaries: list[RunSummary] = []
        for entry in reversed(self._ledger.list_entries()):
            summary = _run_summary(entry, governance=self._governance)
            if lifecycle_id and summary.lifecycle_id != lifecycle_id:
                continue
            if app_id and app_id not in summary.app_ids:
                continue
            if capability_id and capability_id not in summary.capability_ids:
                continue
            if final_status and summary.final_status != final_status:
                continue
            summaries.append(summary)
            if len(summaries) >= bounded_limit:
                break
        return RunListResult(
            runs=summaries,
            count=len(summaries),
            validation=PlanValidationResult(status="valid"),
        )

    def get_ledger_entry(self, run_id: str) -> LedgerEntry:
        entry = self._ledger.get(run_id)
        if entry is None:
            raise _rejected("unknown_run", "No recorded run was found for the requested run_id.")
        return entry

    def cancel_run(self, request: CancellationRequest) -> CancellationResult:
        entry = self._ledger.get(request.run_id)
        if entry is None:
            raise _rejected("unknown_run", "No recorded run was found for the requested run_id.")

        existing = _latest(entry.cancellation_events)
        if existing is not None:
            return CancellationResult(
                run_id=request.run_id,
                run_state="cancelled",
                cancellation=existing,
                final_status="cancelled",
                allowed_next_actions=[],
                validation=PlanValidationResult(status="valid"),
                ledger_recorded=True,
            )

        state = run_state_for_entry(entry)
        if state in _TERMINAL_RUN_STATES:
            raise _rejected(
                "terminal_run_cancellation",
                "The recorded run is already terminal and cannot be cancelled.",
            )

        reason = request.reason.strip()
        cancellation = {
            "cancellation_id": f"cancel_{uuid4().hex[:12]}",
            "status": "cancelled",
            "reason": reason,
            "cancellation_intent": request.cancellation_intent,
            "cancelled_by": "local_user",
            "cancelled_at_utc": _utc_now_label(),
            "execution_permitted": False,
        }
        unsafe_issues = find_unsafe_payload_issues(cancellation, root="cancellation")
        if unsafe_issues:
            raise RuntimeValidationError(
                PlanValidationResult(
                    status="rejected",
                    errors=[
                        issue.model_copy(update={"code": "unsafe_cancellation_record"})
                        for issue in unsafe_issues
                    ],
                )
            )

        try:
            recorded_entry = self._ledger.append_cancellation_event(request.run_id, cancellation)
        except ValueError as exc:
            raise _rejected(
                "unsafe_cancellation_record",
                "The cancellation record could not be safely ledgered.",
            ) from exc

        return CancellationResult(
            run_id=request.run_id,
            run_state=run_state_for_entry(recorded_entry),
            cancellation=cancellation,
            final_status=recorded_entry.final_status,
            allowed_next_actions=[],
            validation=PlanValidationResult(status="valid"),
            ledger_recorded=True,
        )

    def pause_run(self, request: PauseRequest) -> PauseResult:
        entry = self._ledger.get(request.run_id)
        if entry is None:
            raise _rejected("unknown_run", "No recorded run was found for the requested run_id.")

        existing_pause = _latest_unresumed_pause(entry)
        if existing_pause is not None:
            return PauseResult(
                run_id=request.run_id,
                run_state="paused",
                pause_event=existing_pause,
                final_status=entry.final_status,
                allowed_next_actions=["resume_run", "cancel_run"],
                validation=PlanValidationResult(status="valid"),
                ledger_recorded=True,
            )

        state = run_state_for_entry(entry)
        if state in _TERMINAL_OR_CANCELLED_RUN_STATES:
            raise _rejected(
                "terminal_run_pause",
                "The recorded run is already terminal or cancelled and cannot be paused.",
            )

        pause_event = {
            "recovery_event_id": f"recovery_{uuid4().hex[:12]}",
            "event_type": "pause",
            "status": "paused",
            "reason": request.reason.strip(),
            "pause_intent": request.pause_intent,
            "paused_by": "local_user",
            "paused_at_utc": _utc_now_label(),
            "execution_permitted": False,
        }
        _reject_unsafe_recovery_event(pause_event)

        try:
            recorded_entry = self._ledger.append_recovery_event(request.run_id, pause_event)
        except ValueError as exc:
            raise _rejected(
                "unsafe_recovery_record",
                "The pause record could not be safely ledgered.",
            ) from exc

        return PauseResult(
            run_id=request.run_id,
            run_state=run_state_for_entry(recorded_entry),
            pause_event=pause_event,
            final_status=recorded_entry.final_status,
            allowed_next_actions=orchestration_for_entry(recorded_entry).allowed_next_actions,
            validation=PlanValidationResult(status="valid"),
            ledger_recorded=True,
        )

    def resume_run(self, request: ResumptionRequest) -> ResumptionResult:
        entry = self._ledger.get(request.run_id)
        if entry is None:
            raise _rejected("unknown_run", "No recorded run was found for the requested run_id.")

        state = run_state_for_entry(entry)
        if state in _TERMINAL_OR_CANCELLED_RUN_STATES:
            raise _rejected(
                "terminal_run_resumption",
                "The recorded run is terminal or cancelled and cannot be resumed.",
            )
        if state != "paused" or _latest_unresumed_pause(entry) is None:
            raise _rejected(
                "run_not_paused",
                "The recorded run is not paused and cannot be resumed.",
            )

        draft_resume = {
            "recovery_event_id": f"recovery_{uuid4().hex[:12]}",
            "event_type": "resume",
            "status": "resumed",
            "resume_intent": request.resume_intent,
            "resumed_by": "local_user",
            "resumed_at_utc": _utc_now_label(),
            "revalidation_summary": {},
            "execution_permitted": False,
        }
        candidate = entry.model_copy(
            update={"recovery_events": [*entry.recovery_events, draft_resume]},
            deep=True,
        )
        orchestration = orchestration_for_entry(candidate)
        revalidation_summary = self._resume_revalidation_summary(candidate, orchestration)
        resume_event = {
            **draft_resume,
            "revalidation_summary": revalidation_summary,
        }
        _reject_unsafe_recovery_event(resume_event)

        try:
            recorded_entry = self._ledger.append_recovery_event(request.run_id, resume_event)
        except ValueError as exc:
            raise _rejected(
                "unsafe_recovery_record",
                "The resume record could not be safely ledgered.",
            ) from exc

        recorded_orchestration = orchestration_for_entry(recorded_entry)
        return ResumptionResult(
            run_id=request.run_id,
            run_state=recorded_orchestration.run_state,
            resumption_event=resume_event,
            final_status=recorded_entry.final_status,
            orchestration=recorded_orchestration,
            allowed_next_actions=recorded_orchestration.allowed_next_actions,
            validation=PlanValidationResult(status="valid"),
            ledger_recorded=True,
        )

    def _resume_revalidation_summary(
        self,
        entry: LedgerEntry,
        orchestration: Any,
    ) -> dict[str, Any]:
        current = next((step for step in orchestration.steps if step.is_current), None)
        if current is None:
            raise _rejected(
                "resume_without_current_step",
                "No current orchestration step was available after resume revalidation.",
            )

        capability = _capability_snapshot(entry, current.capability_id, current.app_id)
        if capability is None:
            raise _rejected(
                "stale_resume_capability_snapshot",
                "The current step capability was not found in the ledger capability snapshot.",
            )
        if str(capability.get("risk_tier") or "") != current.risk_tier:
            raise _rejected(
                "stale_resume_capability_snapshot",
                "The current step risk tier no longer matches the ledger capability snapshot.",
            )
        if capability.get("enabled", True) is not True:
            raise _rejected(
                "stale_resume_capability_snapshot",
                "The current step capability is disabled in the ledger capability snapshot.",
            )

        capability_available = True
        app_available = True
        if self._capability_discovery is not None and (
            current.preflight_required or current.execution_supported
        ):
            discovery = self._capability_discovery.discover()
            app_available = not discovery.app_is_unavailable(current.app_id)
            capability_available = (
                discovery.supports_preflight(current.capability_id)
                if current.preflight_required
                else discovery.supports_execution(current.capability_id)
            )
            if not app_available:
                raise _rejected(
                    "resume_app_unavailable",
                    "The current step owning app is not currently advertising agent capabilities.",
                )
            if not capability_available:
                raise _rejected(
                    "resume_capability_unavailable",
                    "The current step capability is not currently reconciled as available.",
                )

        return {
            "status": "valid",
            "current_step_id": current.step_id,
            "current_capability_id": current.capability_id,
            "current_app_id": current.app_id,
            "current_step_status": current.status,
            "run_state_after_resume": orchestration.run_state,
            "capability_snapshot_valid": True,
            "capability_discovery_valid": capability_available,
            "app_available": app_available,
        }


def _status_result(entry: LedgerEntry, *, governance: Any | None = None) -> RunStatusResult:
    orchestration = orchestration_for_entry(entry)
    user_workflow = user_workflow_summaries_for_entry(entry, run_state=orchestration.run_state)
    return RunStatusResult(
        run_id=entry.run_id,
        parent_run_id=entry.parent_run_id,
        parent_plan_id=entry.parent_plan_id,
        activated_revision_id=entry.activated_revision_id,
        child_run_ids=entry.child_run_ids,
        run_state=orchestration.run_state,
        final_status=entry.final_status,
        user_goal_summary=entry.user_goal_summary,
        plan=entry.plan_snapshot,
        latest_preflight=_latest(entry.preflight_records),
        latest_confirmation=_latest(entry.confirmation_records),
        latest_action_request=_latest(entry.action_requests),
        latest_action_result=_latest(entry.action_results),
        latest_recovery=_latest(entry.recovery_events),
        latest_cancellation=_latest(entry.cancellation_events),
        ledger_summary=_ledger_summary(entry),
        run_progress_summary=orchestration.run_progress_summary,
        stale_assumption_summary=orchestration.stale_assumption_summary,
        ownership_summary=user_workflow["ownership_summary"],
        plan_review_summary=user_workflow["plan_review_summary"],
        plan_approval_summary=user_workflow["plan_approval_summary"],
        readiness_summary=user_workflow["readiness_summary"],
        consent_summary=user_workflow["consent_summary"],
        allowed_user_owned_actions=user_workflow["allowed_user_owned_actions"],
        allowed_next_actions=orchestration.allowed_next_actions,
        governance_summary=governance.run_summary(entry.run_id) if governance is not None else None,
        separation_of_duties_summary=(
            governance.separation_of_duties_run_summary(entry.run_id) if governance is not None else None
        ),
        validation=PlanValidationResult(status="valid"),
    )


def _run_summary(entry: LedgerEntry, *, governance: Any | None = None) -> RunSummary:
    run_state = run_state_for_entry(entry)
    user_workflow = user_workflow_summaries_for_entry(entry, run_state=run_state)
    return RunSummary(
        run_id=entry.run_id,
        parent_run_id=entry.parent_run_id,
        parent_plan_id=entry.parent_plan_id,
        activated_revision_id=entry.activated_revision_id,
        child_run_ids=entry.child_run_ids,
        run_state=run_state,
        final_status=entry.final_status,
        user_goal_summary=entry.user_goal_summary,
        lifecycle_id=_lifecycle_id(entry),
        app_ids=sorted(_app_ids(entry)),
        capability_ids=sorted(_capability_ids(entry)),
        latest_action_result=_latest(entry.action_results),
        latest_recovery=_latest(entry.recovery_events),
        latest_cancellation=_latest(entry.cancellation_events),
        latest_event_at_utc=_latest_event_at_utc(entry),
        ledger_summary=_ledger_summary(entry),
        ownership_summary=user_workflow["ownership_summary"],
        plan_review_summary=user_workflow["plan_review_summary"],
        plan_approval_summary=user_workflow["plan_approval_summary"],
        readiness_summary=user_workflow["readiness_summary"],
        consent_summary=user_workflow["consent_summary"],
        allowed_user_owned_actions=user_workflow["allowed_user_owned_actions"],
        governance_summary=governance.run_summary(entry.run_id) if governance is not None else None,
        separation_of_duties_summary=(
            governance.separation_of_duties_run_summary(entry.run_id) if governance is not None else None
        ),
    )


def _ledger_summary(entry: LedgerEntry) -> dict[str, Any]:
    return ledger_summary(entry)


def _lifecycle_id(entry: LedgerEntry) -> str | None:
    context = entry.context_preview.context if entry.context_preview else {}
    lifecycle = context.get("lifecycle_summary") if isinstance(context, dict) else None
    if isinstance(lifecycle, dict):
        value = lifecycle.get("lifecycle_id")
        if isinstance(value, str) and value:
            return value
    for record in reversed(entry.action_requests):
        if not isinstance(record, dict):
            continue
        reference = record.get("lifecycle_state_reference")
        if isinstance(reference, dict):
            value = reference.get("lifecycle_id")
            if isinstance(value, str) and value:
                return value
    return None


def _app_ids(entry: LedgerEntry) -> set[str]:
    items: set[str] = set()
    snapshot = entry.plan_snapshot if isinstance(entry.plan_snapshot, dict) else {}
    steps = snapshot.get("proposed_steps")
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, dict) and isinstance(step.get("app_id"), str):
                items.add(step["app_id"])
    for records in [
        entry.preflight_records,
        entry.action_requests,
        entry.action_results,
    ]:
        for record in records:
            if isinstance(record, dict) and isinstance(record.get("app_id"), str):
                items.add(record["app_id"])
    return items


def _capability_ids(entry: LedgerEntry) -> set[str]:
    items: set[str] = set()
    snapshot = entry.plan_snapshot if isinstance(entry.plan_snapshot, dict) else {}
    steps = snapshot.get("proposed_steps")
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, dict) and isinstance(step.get("capability_id"), str):
                items.add(step["capability_id"])
    for records in [
        entry.preflight_records,
        entry.confirmation_records,
        entry.action_requests,
        entry.action_results,
    ]:
        for record in records:
            if isinstance(record, dict) and isinstance(record.get("capability_id"), str):
                items.add(record["capability_id"])
    return items


def _latest_event_at_utc(entry: LedgerEntry) -> str | None:
    candidates: list[str] = []
    for records in [
        entry.confirmation_records,
        entry.action_requests,
        entry.recovery_events,
        entry.cancellation_events,
    ]:
        for record in records:
            if not isinstance(record, dict):
                continue
            for key in [
                "confirmed_at_utc",
                "requested_at_utc",
                "paused_at_utc",
                "resumed_at_utc",
                "activated_at_utc",
                "checked_at_utc",
                "consented_at_utc",
                "cancelled_at_utc",
            ]:
                value = record.get(key)
                if isinstance(value, str) and value:
                    candidates.append(value)
    return sorted(candidates)[-1] if candidates else None


def _latest_unresumed_pause(entry: LedgerEntry) -> dict[str, Any] | None:
    for record in reversed(entry.recovery_events):
        if not isinstance(record, dict):
            continue
        event_type = record.get("event_type")
        if event_type == "resume":
            return None
        if event_type == "pause":
            return record
    return None


def _capability_snapshot(entry: LedgerEntry, capability_id: str, app_id: str) -> dict[str, Any] | None:
    for capability in entry.capability_snapshot:
        if (
            isinstance(capability, dict)
            and capability.get("capability_id") == capability_id
            and capability.get("app_id") == app_id
        ):
            return capability
    return None


def _latest(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    for record in reversed(records):
        if isinstance(record, dict):
            return record
    return None


def _reject_unsafe_recovery_event(record: dict[str, Any]) -> None:
    unsafe_issues = find_unsafe_payload_issues(record, root="recovery")
    if unsafe_issues:
        raise RuntimeValidationError(
            PlanValidationResult(
                status="rejected",
                errors=[
                    issue.model_copy(update={"code": "unsafe_recovery_record"})
                    for issue in unsafe_issues
                ],
            )
        )


def _utc_now_label() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _rejected(code: str, message: str) -> RuntimeValidationError:
    return RuntimeValidationError(
        PlanValidationResult(
            status="rejected",
            errors=[ValidationIssue(code=code, message=message)],
        )
    )
