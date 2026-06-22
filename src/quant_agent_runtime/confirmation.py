from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.models import (
    ConfirmationRequest,
    ConfirmationResult,
    LedgerEntry,
    PlanValidationResult,
    ValidationIssue,
)
from quant_agent_runtime.redaction import find_unsafe_payload_issues
from quant_agent_runtime.run_state import run_state_for_entry
from quant_agent_runtime.validation.errors import RuntimeValidationError


CONFIRMATION_INTENT_APPROVE_PLAN_STEP = "approve_plan_step"


class ConfirmationService:
    def __init__(self, *, ledger: InMemoryLedger) -> None:
        self._ledger = ledger

    def create_confirmation(self, request: ConfirmationRequest) -> ConfirmationResult:
        entry = self._ledger.get(request.run_id)
        if entry is None:
            raise _rejected("unknown_run", "No recorded plan was found for the requested run_id.")

        if request.confirmation_intent != CONFIRMATION_INTENT_APPROVE_PLAN_STEP:
            raise _rejected(
                "unsupported_confirmation_intent",
                "Only approve_plan_step confirmation intent is supported; execution is not permitted.",
                step_id=request.step_id,
            )

        step = _plan_step(entry, request.step_id)
        if step is None:
            raise _rejected(
                "unknown_step",
                "No recorded plan step was found for the requested step_id.",
                step_id=request.step_id,
            )

        plan_state = run_state_for_entry(entry)
        if plan_state == "waiting_for_input":
            raise _rejected(
                "blocked_plan_confirmation",
                "The recorded plan is blocked by missing inputs and cannot be confirmed.",
                step_id=request.step_id,
                capability_id=str(step.get("capability_id") or "") or None,
            )
        if plan_state == "preflight_blocked":
            raise _rejected(
                "preflight_blocked_confirmation",
                "The recorded run has a blocked preflight and cannot be confirmed.",
                step_id=request.step_id,
                capability_id=str(step.get("capability_id") or "") or None,
            )
        if plan_state == "cancelled":
            raise _rejected(
                "cancelled_run_confirmation",
                "The recorded run is cancelled and cannot be confirmed.",
                step_id=request.step_id,
                capability_id=str(step.get("capability_id") or "") or None,
            )

        capability_id = str(step.get("capability_id") or "")
        app_id = str(step.get("app_id") or "")
        capability = _capability_snapshot(entry, capability_id, app_id)
        if capability is None:
            raise _rejected(
                "stale_confirmation_capability_snapshot",
                "The recorded plan step capability was not found in the ledger capability snapshot.",
                step_id=request.step_id,
                capability_id=capability_id or None,
            )
        if str(capability.get("risk_tier") or "") != str(step.get("risk_tier") or ""):
            raise _rejected(
                "stale_confirmation_capability_snapshot",
                "The recorded plan step risk tier no longer matches the ledger capability snapshot.",
                step_id=request.step_id,
                capability_id=capability_id or None,
            )
        if capability.get("enabled", True) is not True:
            raise _rejected(
                "stale_confirmation_capability_snapshot",
                "The recorded plan step capability is disabled in the ledger capability snapshot.",
                step_id=request.step_id,
                capability_id=capability_id or None,
            )

        required_confirmation = _required_confirmation(entry, request.step_id, capability_id)
        if required_confirmation is None and not bool(step.get("requires_confirmation")):
            raise _rejected(
                "confirmation_not_required",
                "The recorded plan step does not require confirmation.",
                step_id=request.step_id,
                capability_id=capability_id or None,
            )
        if _already_confirmed(entry, request.step_id, capability_id):
            raise _rejected(
                "duplicate_confirmation",
                "The recorded plan step has already been confirmed.",
                step_id=request.step_id,
                capability_id=capability_id or None,
            )

        reason = (
            str(required_confirmation.get("reason"))
            if isinstance(required_confirmation, dict) and required_confirmation.get("reason")
            else "Policy requires explicit confirmation before this step can execute."
        )
        confirmation = {
            "confirmation_id": f"confirmation_{uuid4().hex[:12]}",
            "step_id": request.step_id,
            "capability_id": capability_id,
            "status": "confirmed",
            "reason": reason,
            "confirmation_intent": CONFIRMATION_INTENT_APPROVE_PLAN_STEP,
            "confirmed_by": "local_user",
            "confirmed_at_utc": _utc_now_label(),
            "execution_permitted": False,
        }
        unsafe_issues = find_unsafe_payload_issues(confirmation, root="confirmation")
        if unsafe_issues:
            raise RuntimeValidationError(
                PlanValidationResult(
                    status="rejected",
                    errors=[
                        issue.model_copy(
                            update={
                                "code": "unsafe_confirmation_record",
                                "step_id": request.step_id,
                                "capability_id": capability_id or None,
                            }
                        )
                        for issue in unsafe_issues
                    ],
                )
            )

        try:
            recorded_entry = self._ledger.append_confirmation_record(request.run_id, confirmation)
        except ValueError as exc:
            raise _rejected(
                "unsafe_confirmation_record",
                "The confirmation record could not be safely ledgered.",
                step_id=request.step_id,
                capability_id=capability_id or None,
            ) from exc

        return ConfirmationResult(
            run_id=request.run_id,
            step_id=request.step_id,
            capability_id=capability_id,
            confirmation=confirmation,
            run_state=run_state_for_entry(recorded_entry),
            validation=PlanValidationResult(status="valid"),
            ledger_recorded=True,
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


def _capability_snapshot(
    entry: LedgerEntry,
    capability_id: str,
    app_id: str,
) -> dict[str, Any] | None:
    for capability in entry.capability_snapshot:
        if not isinstance(capability, dict):
            continue
        if capability.get("capability_id") == capability_id and capability.get("app_id") == app_id:
            return capability
    return None


def _required_confirmation(
    entry: LedgerEntry,
    step_id: str,
    capability_id: str,
) -> dict[str, Any] | None:
    snapshot = entry.plan_snapshot if isinstance(entry.plan_snapshot, dict) else {}
    required = snapshot.get("required_confirmations")
    if not isinstance(required, list):
        return None
    for item in required:
        if not isinstance(item, dict):
            continue
        if item.get("step_id") == step_id and item.get("capability_id") == capability_id:
            return item
    return None


def _already_confirmed(entry: LedgerEntry, step_id: str, capability_id: str) -> bool:
    for record in entry.confirmation_records:
        if not isinstance(record, dict):
            continue
        if (
            record.get("step_id") == step_id
            and record.get("capability_id") == capability_id
            and record.get("status") == "confirmed"
        ):
            return True
    return False


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
