from __future__ import annotations

import copy
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from quant_agent_runtime.app_clients import AgentAppClient, AppClientError
from quant_agent_runtime.execution import (
    ExecutionService,
    _capability_snapshot,
    _failure_action_result,
    _first_error_code,
    _latest_execution_action_request,
    _plan_step,
)
from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.models import (
    LedgerEntry,
    PlanValidationResult,
    RetryRequest,
    RetryResult,
    ValidationIssue,
)
from quant_agent_runtime.orchestration import (
    ensure_step_action_allowed,
    orchestration_for_entry,
)
from quant_agent_runtime.redaction import find_unsafe_payload_issues
from quant_agent_runtime.run_state import run_state_for_entry
from quant_agent_runtime.validation.errors import RuntimeValidationError


class RetryService:
    def __init__(
        self,
        *,
        ledger: InMemoryLedger,
        execution: ExecutionService,
        app_client: AgentAppClient,
    ) -> None:
        self._ledger = ledger
        self._execution = execution
        self._app_client = app_client

    def retry_step(self, request: RetryRequest) -> RetryResult:
        entry = self._ledger.get(request.run_id)
        if entry is None:
            raise _rejected("unknown_run", "No recorded run was found for the requested run_id.")

        step = _plan_step(entry, request.step_id)
        if step is None:
            raise _rejected(
                "unknown_step",
                "No recorded plan step was found for the requested step_id.",
                step_id=request.step_id,
            )

        capability_id = str(step.get("capability_id") or "")
        app_id = str(step.get("app_id") or "")
        existing_retry = _existing_retry_result(entry, request.step_id, capability_id, app_id)
        if existing_retry is not None:
            retry_event, action_request, action_result = existing_retry
            self._execution._validate_action_request(action_request, request.step_id, capability_id)
            self._execution._validate_action_result(action_result, step)
            return _retry_result(
                entry=entry,
                step_id=request.step_id,
                capability_id=capability_id,
                retry_event=retry_event,
                action_request=action_request,
                action_result=action_result,
            )

        run_state = run_state_for_entry(entry)
        if run_state == "cancelled":
            raise _rejected(
                "cancelled_run_retry",
                "The recorded run is cancelled and cannot retry failed steps.",
                step_id=request.step_id,
                capability_id=capability_id or None,
            )
        if run_state == "paused":
            raise _rejected(
                "paused_run_retry",
                "The recorded run is paused and must be resumed before retry.",
                step_id=request.step_id,
                capability_id=capability_id or None,
            )
        if run_state in {"completed", "completed_with_warnings", "failed_terminal"}:
            raise _rejected(
                "terminal_run_retry",
                "The recorded run is terminal and cannot retry failed steps.",
                step_id=request.step_id,
                capability_id=capability_id or None,
            )

        ensure_step_action_allowed(entry, request.step_id, "retry_failed_step")
        failed_result = _latest_action_result(entry, request.step_id, capability_id, app_id)
        if failed_result is None:
            raise _rejected(
                "missing_retryable_failure",
                "No ledgered action result was found for the requested retry step.",
                step_id=request.step_id,
                capability_id=capability_id or None,
            )
        if failed_result.get("execution_status") != "failed_recoverable":
            raise _rejected(
                "non_recoverable_retry_result",
                "Only recoverable failed action results can be retried.",
                step_id=request.step_id,
                capability_id=capability_id or None,
            )
        if failed_result.get("retry_allowed") is not True:
            raise _rejected(
                "retry_not_allowed",
                "The latest recoverable action result does not allow retry.",
                step_id=request.step_id,
                capability_id=capability_id or None,
            )

        _validate_retry_capability_snapshot(entry, step, request.step_id, capability_id, app_id)
        self._execution._validate_execution_gate(entry, step, request.step_id, capability_id, app_id)

        prior_action_request = _latest_execution_action_request(
            entry,
            request.step_id,
            capability_id,
            app_id,
        )
        if prior_action_request is None:
            raise _rejected(
                "missing_retry_action_request",
                "The retryable action result does not have a matching ledgered execution request.",
                step_id=request.step_id,
                capability_id=capability_id or None,
            )
        self._execution._validate_action_request(prior_action_request, request.step_id, capability_id)
        action_request = _retry_action_request(
            prior_action_request,
            failed_action_run_id=str(failed_result.get("action_run_id") or ""),
            retry_intent=request.retry_intent,
        )
        self._execution._validate_action_request(action_request, request.step_id, capability_id)

        try:
            action_result = self._app_client.execute_action(
                app_id=app_id,
                capability_id=capability_id,
                payload={"action_request": action_request},
            )
        except AppClientError as exc:
            action_result = _failure_action_result(
                action_request=action_request,
                execution_status="failed_recoverable",
                error_code="app_unavailable" if exc.status_code == 503 else "app_execution_error",
                error_message="The owning execution app could not complete the retry attempt.",
                retry_allowed=False,
            )
        except Exception:
            action_result = _failure_action_result(
                action_request=action_request,
                execution_status="failed_recoverable",
                error_code="app_execution_error",
                error_message="The owning execution app could not complete the retry attempt.",
                retry_allowed=False,
            )

        try:
            self._execution._validate_action_result(action_result, step)
        except RuntimeValidationError as exc:
            action_result = _failure_action_result(
                action_request=action_request,
                execution_status="failed_terminal",
                error_code=_first_error_code(exc, fallback="invalid_app_action_result"),
                error_message="The owning execution app returned a retry result that could not be safely accepted.",
                retry_allowed=False,
            )
            self._execution._validate_action_result(action_result, step)

        retry_event = _retry_event(
            request=request,
            failed_result=failed_result,
            action_request=action_request,
            action_result=action_result,
            capability_id=capability_id,
            app_id=app_id,
        )
        _reject_unsafe_retry_event(retry_event)

        try:
            recorded_entry = self._ledger.append_recovery_event(request.run_id, retry_event)
            recorded_entry = self._ledger.append_action_request(request.run_id, action_request)
            recorded_entry = self._ledger.append_action_result(request.run_id, action_result)
        except ValueError as exc:
            raise _rejected(
                "unsafe_retry_ledger_record",
                "The retry event, request, or result could not be safely ledgered.",
                step_id=request.step_id,
                capability_id=capability_id or None,
            ) from exc

        return _retry_result(
            entry=recorded_entry,
            step_id=request.step_id,
            capability_id=capability_id,
            retry_event=retry_event,
            action_request=action_request,
            action_result=action_result,
        )


def _retry_result(
    *,
    entry: LedgerEntry,
    step_id: str,
    capability_id: str,
    retry_event: dict[str, Any],
    action_request: dict[str, Any],
    action_result: dict[str, Any],
) -> RetryResult:
    orchestration = orchestration_for_entry(entry)
    return RetryResult(
        run_id=entry.run_id,
        step_id=step_id,
        capability_id=capability_id,
        retry_event=retry_event,
        action_request=action_request,
        action_result=action_result,
        run_state=orchestration.run_state,
        orchestration=orchestration,
        validation=PlanValidationResult(status="valid"),
        ledger_recorded=True,
    )


def _retry_action_request(
    prior_action_request: dict[str, Any],
    *,
    failed_action_run_id: str,
    retry_intent: str,
) -> dict[str, Any]:
    action_request = copy.deepcopy(prior_action_request)
    action_request["requested_at_utc"] = _utc_now_label()
    action_request["execution_permitted"] = True
    action_request["execution_request"] = True
    action_request["retry_request"] = True
    action_request["retry_intent"] = retry_intent
    action_request["retry_source_action_run_id"] = failed_action_run_id
    action_request["idempotency_key"] = f"retry_{failed_action_run_id}"
    return action_request


def _retry_event(
    *,
    request: RetryRequest,
    failed_result: dict[str, Any],
    action_request: dict[str, Any],
    action_result: dict[str, Any],
    capability_id: str,
    app_id: str,
) -> dict[str, Any]:
    retry_id = f"retry_{uuid4().hex[:12]}"
    return {
        "recovery_event_id": retry_id,
        "event_type": "retry",
        "status": "retried",
        "retry_intent": request.retry_intent,
        "run_id": request.run_id,
        "step_id": request.step_id,
        "capability_id": capability_id,
        "app_id": app_id,
        "failed_action_run_id": failed_result.get("action_run_id"),
        "failed_execution_status": failed_result.get("execution_status"),
        "retry_action_request_idempotency_key": action_request.get("idempotency_key"),
        "retry_action_run_id": action_result.get("action_run_id"),
        "retry_execution_status": action_result.get("execution_status"),
        "retried_by": "local_user",
        "retried_at_utc": _utc_now_label(),
        "execution_permitted": False,
    }


def _existing_retry_result(
    entry: LedgerEntry,
    step_id: str,
    capability_id: str,
    app_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    for event in reversed(entry.recovery_events):
        if not isinstance(event, dict):
            continue
        if (
            event.get("event_type") != "retry"
            or event.get("step_id") != step_id
            or event.get("capability_id") != capability_id
            or event.get("app_id") != app_id
        ):
            continue
        action_request = _action_request_by_idempotency(
            entry,
            str(event.get("retry_action_request_idempotency_key") or ""),
        )
        action_result = _action_result_by_run_id(entry, str(event.get("retry_action_run_id") or ""))
        if action_request is None or action_result is None:
            raise _rejected(
                "malformed_retry_record",
                "The existing retry record is missing its action request or result reference.",
                step_id=step_id,
                capability_id=capability_id or None,
            )
        return event, action_request, action_result
    return None


def _action_request_by_idempotency(entry: LedgerEntry, idempotency_key: str) -> dict[str, Any] | None:
    if not idempotency_key:
        return None
    for record in reversed(entry.action_requests):
        if isinstance(record, dict) and record.get("idempotency_key") == idempotency_key:
            return record
    return None


def _action_result_by_run_id(entry: LedgerEntry, action_run_id: str) -> dict[str, Any] | None:
    if not action_run_id:
        return None
    for record in reversed(entry.action_results):
        if isinstance(record, dict) and record.get("action_run_id") == action_run_id:
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


def _validate_retry_capability_snapshot(
    entry: LedgerEntry,
    step: dict[str, Any],
    step_id: str,
    capability_id: str,
    app_id: str,
) -> None:
    capability = _capability_snapshot(entry, capability_id, app_id)
    if capability is None:
        raise _rejected(
            "stale_retry_capability_snapshot",
            "The recorded plan step capability was not found in the ledger capability snapshot.",
            step_id=step_id,
            capability_id=capability_id or None,
        )
    if str(capability.get("risk_tier") or "") != str(step.get("risk_tier") or ""):
        raise _rejected(
            "stale_retry_capability_snapshot",
            "The recorded plan step risk tier no longer matches the ledger capability snapshot.",
            step_id=step_id,
            capability_id=capability_id or None,
        )
    if capability.get("enabled", True) is not True:
        raise _rejected(
            "stale_retry_capability_snapshot",
            "The recorded plan step capability is disabled in the ledger capability snapshot.",
            step_id=step_id,
            capability_id=capability_id or None,
        )


def _reject_unsafe_retry_event(event: dict[str, Any]) -> None:
    unsafe_issues = find_unsafe_payload_issues(event, root="retry_event")
    if unsafe_issues:
        raise RuntimeValidationError(
            PlanValidationResult(
                status="rejected",
                errors=[
                    issue.model_copy(update={"code": "unsafe_retry_event"})
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
