from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.models import (
    PlanValidationResult,
    RunRevalidationRequest,
    RunRevalidationResult,
    StaleAssumptionSummary,
    ValidationIssue,
)
from quant_agent_runtime.orchestration import orchestration_for_entry
from quant_agent_runtime.progress import safe_section_labels
from quant_agent_runtime.redaction import find_unsafe_payload_issues, sanitize_value
from quant_agent_runtime.run_state import run_state_for_entry
from quant_agent_runtime.validation.errors import RuntimeValidationError


_TERMINAL_OR_CANCELLED_RUN_STATES = {
    "cancelled",
    "completed",
    "completed_with_warnings",
    "failed_terminal",
}


class RunRevalidationService:
    def __init__(self, *, ledger: InMemoryLedger) -> None:
        self._ledger = ledger

    def check_current_context(self, request: RunRevalidationRequest) -> RunRevalidationResult:
        entry = self._ledger.get(request.run_id)
        if entry is None:
            raise _rejected("unknown_run", "No recorded run was found for the requested run_id.")

        run_state = run_state_for_entry(entry)
        if run_state in _TERMINAL_OR_CANCELLED_RUN_STATES:
            raise _rejected(
                "terminal_run_revalidation",
                "The recorded run is terminal or cancelled and cannot be revalidated.",
            )

        sanitized_context, redaction = sanitize_value(
            request.current_context_summary,
            path="current_context_summary",
        )
        if not isinstance(sanitized_context, dict):
            sanitized_context = {}
        if redaction.redacted:
            raise RuntimeValidationError(
                PlanValidationResult(
                    status="rejected",
                    errors=[
                        ValidationIssue(
                            code="unsafe_revalidation_context",
                            message="The revalidation context included unsafe fields or values.",
                        )
                    ],
                )
            )

        checked_at = _utc_now_label()
        stale_summary = _stale_assumption_summary(
            _original_context(entry),
            sanitized_context,
            checked_at_utc=checked_at,
        )
        current_orchestration = orchestration_for_entry(entry)
        progress = current_orchestration.run_progress_summary
        revalidation_event = {
            "recovery_event_id": f"revalidation_{uuid4().hex[:12]}",
            "event_type": "run_revalidation",
            "status": stale_summary.status,
            "revalidation_intent": request.revalidation_intent,
            "run_id": entry.run_id,
            "plan_id": current_orchestration.plan_id,
            "checked_by": "local_user",
            "checked_at_utc": checked_at,
            "changed_sections": stale_summary.changed_sections,
            "added_sections": stale_summary.added_sections,
            "missing_current_sections": stale_summary.missing_current_sections,
            "current_context_section_count": len(stale_summary.current_sections),
            "original_context_section_count": len(stale_summary.original_sections),
            "context_fingerprint": _context_fingerprint(sanitized_context),
            "stale_assumption_summary": stale_summary.model_dump(mode="json"),
            "run_progress_summary": progress.model_dump(mode="json"),
            "execution_permitted": False,
        }
        _reject_unsafe_revalidation_event(revalidation_event)

        try:
            recorded_entry = self._ledger.append_recovery_event(request.run_id, revalidation_event)
        except ValueError as exc:
            raise _rejected(
                "unsafe_revalidation_record",
                "The run revalidation record could not be safely ledgered.",
            ) from exc

        recorded_orchestration = orchestration_for_entry(recorded_entry)
        return RunRevalidationResult(
            run_id=request.run_id,
            run_progress_summary=recorded_orchestration.run_progress_summary,
            stale_assumption_summary=stale_summary,
            orchestration=recorded_orchestration,
            validation=PlanValidationResult(status="valid"),
            ledger_recorded=True,
        )


def _stale_assumption_summary(
    original_context: dict[str, Any],
    current_context: dict[str, Any],
    *,
    checked_at_utc: str,
) -> StaleAssumptionSummary:
    original_sections = safe_section_labels(original_context)
    current_sections = safe_section_labels(current_context)
    if not current_context:
        return StaleAssumptionSummary(
            status="insufficient_context",
            current_context_provided=False,
            state_changed_since_planning=False,
            original_sections=original_sections,
            current_sections=current_sections,
            warnings=["No current sanitized lifecycle context was provided for revalidation."],
            revalidation_required=True,
            checked_at_utc=checked_at_utc,
        )

    original_keys = set(original_sections)
    current_keys = set(current_sections)
    shared = sorted(original_keys & current_keys)
    changed_sections = [
        key
        for key in shared
        if _section_fingerprint(original_context.get(key)) != _section_fingerprint(current_context.get(key))
    ]
    added_sections = sorted(current_keys - original_keys)
    missing_current_sections = sorted(original_keys - current_keys)
    if missing_current_sections:
        status = "insufficient_context"
    elif changed_sections or added_sections:
        status = "stale"
    else:
        status = "fresh"
    warnings = _stale_warnings(
        status=status,
        changed_sections=changed_sections,
        added_sections=added_sections,
        missing_current_sections=missing_current_sections,
    )
    return StaleAssumptionSummary(
        status=status,
        current_context_provided=True,
        state_changed_since_planning=status == "stale",
        changed_sections=changed_sections,
        added_sections=added_sections,
        missing_current_sections=missing_current_sections,
        original_sections=original_sections,
        current_sections=current_sections,
        warnings=warnings,
        revalidation_required=status in {"stale", "insufficient_context"},
        checked_at_utc=checked_at_utc,
    )


def _original_context(entry: Any) -> dict[str, Any]:
    preview = entry.context_preview
    if preview is None or not isinstance(preview.context, dict):
        return {}
    return preview.context


def _stale_warnings(
    *,
    status: str,
    changed_sections: list[str],
    added_sections: list[str],
    missing_current_sections: list[str],
) -> list[str]:
    if status == "fresh":
        return []
    warnings: list[str] = []
    if changed_sections:
        warnings.append("Current sanitized context differs from the original planning context.")
    if added_sections:
        warnings.append("Current sanitized context includes sections not present during planning.")
    if missing_current_sections:
        warnings.append("Current sanitized context is missing sections that were present during planning.")
    return warnings


def _context_fingerprint(context: dict[str, Any]) -> str:
    return _stable_hash(
        {
            key: _section_fingerprint(value)
            for key, value in sorted(context.items(), key=lambda item: str(item[0]))
        }
    )


def _section_fingerprint(value: Any) -> str:
    return _stable_hash(value)


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _reject_unsafe_revalidation_event(record: dict[str, Any]) -> None:
    unsafe_issues = find_unsafe_payload_issues(record, root="revalidation")
    if unsafe_issues:
        raise RuntimeValidationError(
            PlanValidationResult(
                status="rejected",
                errors=[
                    issue.model_copy(update={"code": "unsafe_revalidation_record"})
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
