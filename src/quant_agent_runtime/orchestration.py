from __future__ import annotations

from typing import Any

from quant_agent_runtime.capability_discovery import SUPPORTED_EXECUTION_CAPABILITIES
from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.models import (
    LedgerEntry,
    OrchestrationStepSummary,
    OrchestrationStepStatus,
    PlanValidationResult,
    RunOrchestrationResult,
    RunState,
    ValidationIssue,
)
from quant_agent_runtime.validation.errors import RuntimeValidationError


_COMPLETE_STATUSES = {"completed", "completed_with_warnings", "informational", "unsupported"}
_TERMINAL_RUN_STATES = {"completed", "completed_with_warnings", "failed_terminal", "cancelled"}


class OrchestrationService:
    def __init__(self, *, ledger: InMemoryLedger) -> None:
        self._ledger = ledger

    def get_run_orchestration(self, run_id: str) -> RunOrchestrationResult:
        entry = self._ledger.get(run_id)
        if entry is None:
            raise _rejected("unknown_run", "No recorded run was found for the requested run_id.")
        return orchestration_for_entry(entry)


def orchestration_for_entry(entry: LedgerEntry) -> RunOrchestrationResult:
    snapshot = entry.plan_snapshot if isinstance(entry.plan_snapshot, dict) else {}
    raw_steps = snapshot.get("proposed_steps")
    steps_payload = raw_steps if isinstance(raw_steps, list) else []
    run_is_cancelled = entry.final_status == "cancelled" or bool(entry.cancellation_events)
    run_is_paused = latest_recovery_event_type(entry) == "pause"
    plan_blocked = _plan_is_blocked(snapshot)

    summaries: list[OrchestrationStepSummary] = []
    dependency_blocked = run_is_cancelled or run_is_paused or plan_blocked
    current_step_id: str | None = None
    blocking_label: str | None = None

    for raw_step in steps_payload:
        if not isinstance(raw_step, dict):
            continue
        summary = _step_summary(
            entry,
            raw_step,
            dependency_blocked=dependency_blocked,
            blocking_label=blocking_label,
            run_is_cancelled=run_is_cancelled,
            run_is_paused=run_is_paused,
        )
        if current_step_id is None and _is_current_status(summary.status):
            current_step_id = summary.step_id
            summary = summary.model_copy(update={"is_current": True})
        summaries.append(summary)
        if summary.status not in _COMPLETE_STATUSES:
            dependency_blocked = True
            blocking_label = summary.title or summary.step_id

    run_state = _run_state_from_summaries(
        entry=entry,
        plan_blocked=plan_blocked,
        current_step=next((step for step in summaries if step.is_current), None),
        summaries=summaries,
    )
    allowed_next_actions = _run_allowed_actions(run_state, summaries)
    return RunOrchestrationResult(
        run_id=entry.run_id,
        parent_run_id=entry.parent_run_id,
        parent_plan_id=entry.parent_plan_id,
        activated_revision_id=entry.activated_revision_id,
        child_run_ids=entry.child_run_ids,
        run_state=run_state,
        final_status=entry.final_status,
        plan_id=snapshot.get("plan_id") if isinstance(snapshot.get("plan_id"), str) else None,
        current_step_id=current_step_id,
        steps=summaries,
        allowed_next_actions=allowed_next_actions,
        ledger_summary=ledger_summary(entry),
        validation=PlanValidationResult(status="valid"),
    )


def run_state_from_orchestration(entry: LedgerEntry) -> RunState:
    return orchestration_for_entry(entry).run_state


def ensure_step_action_allowed(entry: LedgerEntry, step_id: str, action: str) -> None:
    orchestration = orchestration_for_entry(entry)
    if orchestration.run_state == "paused":
        raise _rejected(
            "orchestration_run_paused",
            "The recorded run is paused and must be resumed before gated actions can continue.",
            step_id=step_id,
        )
    if orchestration.run_state in _TERMINAL_RUN_STATES:
        raise _rejected(
            "orchestration_run_terminal",
            "The recorded run is terminal or cancelled and cannot accept more gated actions.",
            step_id=step_id,
        )
    target = next((step for step in orchestration.steps if step.step_id == step_id), None)
    if target is None:
        raise _rejected(
            "unknown_step",
            "No recorded plan step was found for the requested step_id.",
            step_id=step_id,
        )
    if target.status == "not_ready":
        raise _rejected(
            "orchestration_step_not_ready",
            target.blocker_reason
            or "Earlier plan steps must satisfy their required gates before this step can proceed.",
            step_id=step_id,
            capability_id=target.capability_id,
        )
    if orchestration.current_step_id and orchestration.current_step_id != step_id:
        raise _rejected(
            "orchestration_step_not_current",
            "Only the current orchestration step can accept this gated action.",
            step_id=step_id,
            capability_id=target.capability_id,
        )
    if action not in target.allowed_actions:
        raise _rejected(
            "orchestration_action_not_allowed",
            "The requested action is not allowed for the current orchestration step state.",
            step_id=step_id,
            capability_id=target.capability_id,
        )


def ledger_summary(entry: LedgerEntry) -> dict[str, Any]:
    return {
        "data_policy": entry.data_policy,
        "provider_mode": entry.provider_mode,
        "parent_run_id": entry.parent_run_id,
        "parent_plan_id": entry.parent_plan_id,
        "activated_revision_id": entry.activated_revision_id,
        "child_run_count": len(entry.child_run_ids),
        "preflight_count": len(entry.preflight_records),
        "confirmation_count": len(entry.confirmation_records),
        "action_request_count": len(entry.action_requests),
        "action_result_count": len(entry.action_results),
        "cancellation_count": len(entry.cancellation_events),
        "recovery_event_count": len(entry.recovery_events),
        "policy_rejection_count": len(entry.policy_rejections),
        "safe_artifact_count": len(entry.safe_artifact_map),
        "ledger_recorded": True,
    }


def _step_summary(
    entry: LedgerEntry,
    step: dict[str, Any],
    *,
    dependency_blocked: bool,
    blocking_label: str | None,
    run_is_cancelled: bool,
    run_is_paused: bool,
) -> OrchestrationStepSummary:
    step_id = str(step.get("step_id") or "")
    capability_id = str(step.get("capability_id") or "")
    app_id = str(step.get("app_id") or "")
    title = str(step.get("title") or capability_id or step_id)
    risk_tier = str(step.get("risk_tier") or "")
    preflight_required = bool(step.get("preflight_required"))
    confirmation_required = bool(step.get("requires_confirmation")) or _confirmation_required(
        entry,
        step_id,
        capability_id,
    )
    execution_supported = capability_id in SUPPORTED_EXECUTION_CAPABILITIES

    latest_preflight = _latest_preflight(entry, step_id, capability_id, app_id)
    latest_confirmation = _latest_confirmation(entry, step_id, capability_id)
    latest_preview = _latest_action_request(entry, step_id, capability_id, app_id, False)
    latest_action_result = _latest_action_result(entry, step_id, capability_id, app_id)

    if run_is_cancelled:
        status: OrchestrationStepStatus = "cancelled"
        required_gate = None
        blocker_reason = "The run has been cancelled."
    elif run_is_paused:
        status = "not_ready"
        required_gate = "resume"
        blocker_reason = "The run is paused and must be resumed before gated actions can continue."
    elif dependency_blocked:
        status = "not_ready"
        required_gate = "dependency"
        blocker_reason = (
            f"Complete or resolve the earlier step: {blocking_label}."
            if blocking_label
            else "The recorded plan is blocked before this step can proceed."
        )
    else:
        status, required_gate, blocker_reason = _ready_status(
            preflight_required=preflight_required,
            confirmation_required=confirmation_required,
            execution_supported=execution_supported,
            latest_preflight=latest_preflight,
            latest_confirmation=latest_confirmation,
            latest_preview=latest_preview,
            latest_action_result=latest_action_result,
        )

    allowed_actions = _step_allowed_actions(status)
    return OrchestrationStepSummary(
        step_id=step_id,
        capability_id=capability_id,
        app_id=app_id,
        title=title,
        risk_tier=risk_tier,
        status=status,
        preflight_required=preflight_required,
        confirmation_required=confirmation_required,
        execution_supported=execution_supported,
        required_gate=required_gate,
        blocker_reason=blocker_reason,
        latest_preflight_reference=_preflight_reference(latest_preflight),
        latest_confirmation_reference=_confirmation_reference(latest_confirmation),
        latest_action_request_reference=_action_request_reference(latest_preview),
        latest_action_result_reference=_action_result_reference(latest_action_result),
        allowed_actions=allowed_actions,
    )


def _ready_status(
    *,
    preflight_required: bool,
    confirmation_required: bool,
    execution_supported: bool,
    latest_preflight: dict[str, Any] | None,
    latest_confirmation: dict[str, Any] | None,
    latest_preview: dict[str, Any] | None,
    latest_action_result: dict[str, Any] | None,
) -> tuple[OrchestrationStepStatus, str | None, str | None]:
    if latest_action_result is not None:
        status = str(latest_action_result.get("execution_status") or "")
        if status == "succeeded":
            return "completed", None, None
        if status == "succeeded_with_warnings":
            return "completed_with_warnings", None, None
        if status == "failed_recoverable":
            return "failed_recoverable", "recovery", "The latest action result is recoverable but retry is not available in this slice."
        if status == "failed_terminal":
            return "failed_terminal", "recovery", "The latest action result is terminal for this run."

    if preflight_required:
        if latest_preflight is None:
            return "needs_preflight", "preflight", "Run app-owned preflight before continuing."
        blockers = latest_preflight.get("blockers")
        if latest_preflight.get("status") == "blocked" or (isinstance(blockers, list) and blockers):
            return "preflight_blocked", "preflight", "The latest app-owned preflight has blockers."
        if latest_preflight.get("status") not in {"ready", "warning"}:
            return "needs_preflight", "preflight", "The latest app-owned preflight is not ready."

    if not execution_supported:
        if preflight_required:
            return (
                "completed_with_warnings"
                if latest_preflight and latest_preflight.get("status") == "warning"
                else "completed",
                None,
                None,
            )
        return "informational", None, None

    if confirmation_required and latest_confirmation is None:
        return "needs_confirmation", "confirmation", "Record explicit confirmation before continuing."
    if latest_preview is None:
        return "ready_for_action_request", "action_request", "Preview the action request before execution."
    return "ready_for_execution", "execution", "The step is ready for guarded execution."


def _run_state_from_summaries(
    *,
    entry: LedgerEntry,
    plan_blocked: bool,
    current_step: OrchestrationStepSummary | None,
    summaries: list[OrchestrationStepSummary],
) -> RunState:
    if entry.final_status == "cancelled" or entry.cancellation_events:
        return "cancelled"
    if latest_recovery_event_type(entry) == "pause":
        return "paused"
    if plan_blocked:
        return "waiting_for_input"
    if current_step is None:
        if summaries:
            if any(step.status == "completed_with_warnings" for step in summaries):
                return "completed_with_warnings"
            if any(step.status == "completed" for step in summaries):
                return "completed"
        return "planned"
    if current_step.status == "preflight_blocked":
        return "preflight_blocked"
    if current_step.status == "needs_confirmation":
        return "waiting_for_confirmation"
    if current_step.status in {"ready_for_action_request", "ready_for_execution"}:
        return "ready_for_execution_preview"
    if current_step.status == "failed_recoverable":
        return "failed_recoverable"
    if current_step.status == "failed_terminal":
        return "failed_terminal"
    return "planned"


def _run_allowed_actions(run_state: RunState, summaries: list[OrchestrationStepSummary]) -> list[str]:
    if run_state in _TERMINAL_RUN_STATES:
        return []
    if run_state == "paused":
        return ["resume_run", "cancel_run"]
    current = next((step for step in summaries if step.is_current), None)
    actions = ["cancel_run"]
    if current is not None:
        actions.extend(current.allowed_actions)
    return actions


def _step_allowed_actions(status: str) -> list[str]:
    if status in {"needs_preflight", "preflight_blocked"}:
        return ["run_preflight"]
    if status == "needs_confirmation":
        return ["confirm_step"]
    if status == "ready_for_action_request":
        return ["preview_action_request"]
    if status == "ready_for_execution":
        return ["preview_action_request", "execute_step"]
    return []


def _is_current_status(status: str) -> bool:
    return status not in _COMPLETE_STATUSES and status != "not_ready"


def latest_recovery_event_type(entry: LedgerEntry) -> str | None:
    for record in reversed(entry.recovery_events):
        if not isinstance(record, dict):
            continue
        event_type = record.get("event_type")
        if event_type in {"pause", "resume"}:
            return event_type
    return None


def _plan_is_blocked(snapshot: dict[str, Any]) -> bool:
    missing_inputs = snapshot.get("missing_inputs")
    return snapshot.get("status") == "blocked" or (
        isinstance(missing_inputs, list) and len(missing_inputs) > 0
    )


def _confirmation_required(entry: LedgerEntry, step_id: str, capability_id: str) -> bool:
    snapshot = entry.plan_snapshot if isinstance(entry.plan_snapshot, dict) else {}
    required = snapshot.get("required_confirmations")
    if not isinstance(required, list):
        return False
    return any(
        isinstance(item, dict)
        and item.get("step_id") == step_id
        and item.get("capability_id") == capability_id
        for item in required
    )


def _latest_preflight(
    entry: LedgerEntry,
    step_id: str,
    capability_id: str,
    app_id: str,
) -> dict[str, Any] | None:
    for record in reversed(entry.preflight_records):
        if not isinstance(record, dict):
            continue
        if (
            record.get("step_id") == step_id
            or (
                record.get("capability_id") == capability_id
                and record.get("app_id") == app_id
            )
        ):
            return record
    return None


def _latest_confirmation(entry: LedgerEntry, step_id: str, capability_id: str) -> dict[str, Any] | None:
    for record in reversed(entry.confirmation_records):
        if not isinstance(record, dict):
            continue
        if (
            record.get("step_id") == step_id
            and record.get("capability_id") == capability_id
            and record.get("status") == "confirmed"
        ):
            return record
    return None


def _latest_action_request(
    entry: LedgerEntry,
    step_id: str,
    capability_id: str,
    app_id: str,
    execution_permitted: bool,
) -> dict[str, Any] | None:
    for record in reversed(entry.action_requests):
        if not isinstance(record, dict):
            continue
        if (
            record.get("agent_run_id") == entry.run_id
            and record.get("step_id") == step_id
            and record.get("capability_id") == capability_id
            and record.get("app_id") == app_id
            and record.get("execution_permitted") is execution_permitted
        ):
            return record
    return None


def _latest_action_result(
    entry: LedgerEntry,
    step_id: str,
    capability_id: str,
    app_id: str,
) -> dict[str, Any] | None:
    for record in reversed(entry.action_results):
        if not isinstance(record, dict):
            continue
        if (
            record.get("step_id") == step_id
            and record.get("capability_id") == capability_id
            and record.get("app_id") == app_id
        ):
            return record
    return None


def _preflight_reference(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "preflight_id": record.get("preflight_id"),
        "status": record.get("status"),
        "capability_id": record.get("capability_id"),
        "app_id": record.get("app_id"),
        "blocker_count": len(record.get("blockers")) if isinstance(record.get("blockers"), list) else 0,
        "warning_count": len(record.get("warnings")) if isinstance(record.get("warnings"), list) else 0,
    }


def _confirmation_reference(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "confirmation_id": record.get("confirmation_id"),
        "status": record.get("status"),
        "capability_id": record.get("capability_id"),
        "confirmed_by": record.get("confirmed_by"),
        "confirmed_at_utc": record.get("confirmed_at_utc"),
        "execution_permitted": record.get("execution_permitted"),
    }


def _action_request_reference(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "step_id": record.get("step_id"),
        "capability_id": record.get("capability_id"),
        "app_id": record.get("app_id"),
        "idempotency_key": record.get("idempotency_key"),
        "execution_permitted": record.get("execution_permitted"),
        "requested_at_utc": record.get("requested_at_utc"),
    }


def _action_result_reference(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "action_run_id": record.get("action_run_id"),
        "step_id": record.get("step_id"),
        "capability_id": record.get("capability_id"),
        "app_id": record.get("app_id"),
        "execution_status": record.get("execution_status"),
        "retry_allowed": record.get("retry_allowed"),
        "completed_at_utc": record.get("completed_at_utc"),
    }


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
