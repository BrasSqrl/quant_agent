from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.models import (
    CancellationRequest,
    CancellationResult,
    LedgerEntry,
    PlanValidationResult,
    RunListResult,
    RunSummary,
    RunStatusResult,
    ValidationIssue,
)
from quant_agent_runtime.orchestration import ledger_summary, orchestration_for_entry
from quant_agent_runtime.redaction import find_unsafe_payload_issues
from quant_agent_runtime.run_state import run_state_for_entry
from quant_agent_runtime.validation.errors import RuntimeValidationError


_TERMINAL_RUN_STATES = {"completed", "completed_with_warnings", "failed_terminal"}


class RunStatusService:
    def __init__(self, *, ledger: InMemoryLedger) -> None:
        self._ledger = ledger

    def get_run_status(self, run_id: str) -> RunStatusResult:
        entry = self._ledger.get(run_id)
        if entry is None:
            raise _rejected("unknown_run", "No recorded run was found for the requested run_id.")
        return _status_result(entry)

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
            summary = _run_summary(entry)
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


def _status_result(entry: LedgerEntry) -> RunStatusResult:
    state = run_state_for_entry(entry)
    return RunStatusResult(
        run_id=entry.run_id,
        run_state=state,
        final_status=entry.final_status,
        user_goal_summary=entry.user_goal_summary,
        plan=entry.plan_snapshot,
        latest_preflight=_latest(entry.preflight_records),
        latest_confirmation=_latest(entry.confirmation_records),
        latest_action_request=_latest(entry.action_requests),
        latest_action_result=_latest(entry.action_results),
        latest_cancellation=_latest(entry.cancellation_events),
        ledger_summary=_ledger_summary(entry),
        allowed_next_actions=orchestration_for_entry(entry).allowed_next_actions,
        validation=PlanValidationResult(status="valid"),
    )


def _run_summary(entry: LedgerEntry) -> RunSummary:
    return RunSummary(
        run_id=entry.run_id,
        run_state=run_state_for_entry(entry),
        final_status=entry.final_status,
        user_goal_summary=entry.user_goal_summary,
        lifecycle_id=_lifecycle_id(entry),
        app_ids=sorted(_app_ids(entry)),
        capability_ids=sorted(_capability_ids(entry)),
        latest_action_result=_latest(entry.action_results),
        latest_event_at_utc=_latest_event_at_utc(entry),
        ledger_summary=_ledger_summary(entry),
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
        entry.cancellation_events,
    ]:
        for record in records:
            if not isinstance(record, dict):
                continue
            for key in ["confirmed_at_utc", "requested_at_utc", "cancelled_at_utc"]:
                value = record.get(key)
                if isinstance(value, str) and value:
                    candidates.append(value)
    return sorted(candidates)[-1] if candidates else None


def _latest(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    for record in reversed(records):
        if isinstance(record, dict):
            return record
    return None


def _utc_now_label() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _rejected(code: str, message: str) -> RuntimeValidationError:
    return RuntimeValidationError(
        PlanValidationResult(
            status="rejected",
            errors=[ValidationIssue(code=code, message=message)],
        )
    )
