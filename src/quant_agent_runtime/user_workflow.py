from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from quant_agent_runtime.capability_discovery import SUPPORTED_EXECUTION_CAPABILITIES
from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.models import (
    LedgerEntry,
    PlanValidationResult,
    UserWorkflowConsentRequest,
    UserWorkflowConsentResult,
    UserWorkflowConsentSummary,
    UserWorkflowOwnershipSummary,
    UserWorkflowReadinessRequest,
    UserWorkflowReadinessResult,
    UserWorkflowReadinessSummary,
    ValidationIssue,
)
from quant_agent_runtime.orchestration import orchestration_for_entry
from quant_agent_runtime.redaction import find_unsafe_payload_issues, sanitize_value
from quant_agent_runtime.run_state import run_state_for_entry
from quant_agent_runtime.validation.errors import RuntimeValidationError


READINESS_EVENT_TYPE = "user_workflow_readiness"
CONSENT_EVENT_TYPE = "user_workflow_consent"
READINESS_INTENT = "check_user_owned_readiness"
CONSENT_INTENT = "approve_user_owned_guided_execution"
CONSENT_SCOPE = "single_run_review_draft_actions"
TERMINAL_OR_CANCELLED_STATES = {
    "cancelled",
    "sample_reset",
    "completed",
    "completed_with_warnings",
    "failed_terminal",
}


class UserWorkflowService:
    def __init__(self, *, ledger: InMemoryLedger) -> None:
        self._ledger = ledger

    def check_readiness(self, request: UserWorkflowReadinessRequest) -> UserWorkflowReadinessResult:
        entry = self._entry(request.run_id)
        _sanitized_current_context(request.current_context_summary)

        ownership = ownership_summary_for_entry(entry)
        orchestration = orchestration_for_entry(entry)
        readiness = _readiness_summary(entry, ownership, request.readiness_intent)
        event = {
            "recovery_event_id": f"user_workflow_readiness_{uuid4().hex[:12]}",
            "event_type": READINESS_EVENT_TYPE,
            "status": readiness.status,
            "readiness_intent": request.readiness_intent,
            "run_id": entry.run_id,
            "ownership": ownership.ownership,
            "lifecycle_id": ownership.lifecycle_id,
            "sample_workspace_id": ownership.sample_workspace_id,
            "sample_owned": ownership.sample_owned,
            "consent_required": readiness.consent_required,
            "allowed_preflight_capabilities": readiness.allowed_preflight_capabilities,
            "allowed_execution_capabilities": readiness.allowed_execution_capabilities,
            "blockers": readiness.blockers,
            "warnings": readiness.warnings,
            "checked_by": "local_user",
            "checked_at_utc": readiness.checked_at_utc,
            "data_policy": readiness.data_policy,
            "execution_permitted": False,
        }
        _reject_unsafe_event(event, code="unsafe_user_workflow_readiness_record")

        try:
            recorded_entry = self._ledger.append_recovery_event(request.run_id, event)
        except ValueError as exc:
            raise _rejected(
                "unsafe_user_workflow_readiness_record",
                "The user-owned workflow readiness record could not be safely ledgered.",
            ) from exc

        return UserWorkflowReadinessResult(
            run_id=request.run_id,
            ownership_summary=ownership_summary_for_entry(recorded_entry),
            readiness_summary=latest_readiness_summary(recorded_entry),
            consent_summary=latest_consent_summary(recorded_entry),
            run_state=run_state_for_entry(recorded_entry),
            orchestration=orchestration_for_entry(recorded_entry),
            validation=PlanValidationResult(status="valid" if readiness.status != "blocked" else "rejected"),
            ledger_recorded=True,
        )

    def approve_consent(self, request: UserWorkflowConsentRequest) -> UserWorkflowConsentResult:
        entry = self._entry(request.run_id)
        ownership = ownership_summary_for_entry(entry)
        if ownership.ownership == "sample_owned":
            raise _rejected(
                "sample_workflow_consent_not_required",
                "Sample-owned demo runs do not use the user-owned workflow consent gate.",
            )
        if ownership.ownership != "user_owned":
            raise _rejected(
                "unknown_run_ownership",
                "The recorded run does not include explicit user-owned lifecycle ownership markers.",
            )

        readiness = latest_readiness_summary(entry)
        if readiness.status != "ready":
            raise _rejected(
                "user_workflow_readiness_required",
                "A successful user-owned workflow readiness check is required before consent.",
            )

        run_state = run_state_for_entry(entry)
        if run_state in TERMINAL_OR_CANCELLED_STATES:
            raise _rejected(
                "terminal_run_user_workflow_consent",
                "The recorded run is terminal or cancelled and cannot record user-owned consent.",
            )

        existing = latest_consent_summary(entry)
        if existing.status == "consented" and existing.consent_scope == request.consent_scope:
            return UserWorkflowConsentResult(
                run_id=request.run_id,
                ownership_summary=ownership,
                readiness_summary=readiness,
                consent_summary=existing,
                run_state=run_state,
                orchestration=orchestration_for_entry(entry),
                validation=PlanValidationResult(status="valid"),
                ledger_recorded=True,
            )

        consented_at = _utc_now_label()
        event = {
            "recovery_event_id": f"user_workflow_consent_{uuid4().hex[:12]}",
            "event_type": CONSENT_EVENT_TYPE,
            "status": "consented",
            "consent_intent": request.consent_intent,
            "consent_scope": request.consent_scope,
            "run_id": entry.run_id,
            "ownership": ownership.ownership,
            "lifecycle_id": ownership.lifecycle_id,
            "allowed_execution_capabilities": readiness.allowed_execution_capabilities,
            "allowed_preflight_capabilities": readiness.allowed_preflight_capabilities,
            "consented_by": "local_user",
            "consented_at_utc": consented_at,
            "execution_permitted": False,
        }
        _reject_unsafe_event(event, code="unsafe_user_workflow_consent_record")

        try:
            recorded_entry = self._ledger.append_recovery_event(request.run_id, event)
        except ValueError as exc:
            raise _rejected(
                "unsafe_user_workflow_consent_record",
                "The user-owned workflow consent record could not be safely ledgered.",
            ) from exc

        return UserWorkflowConsentResult(
            run_id=request.run_id,
            ownership_summary=ownership_summary_for_entry(recorded_entry),
            readiness_summary=latest_readiness_summary(recorded_entry),
            consent_summary=latest_consent_summary(recorded_entry),
            run_state=run_state_for_entry(recorded_entry),
            orchestration=orchestration_for_entry(recorded_entry),
            validation=PlanValidationResult(status="valid"),
            ledger_recorded=True,
        )

    def _entry(self, run_id: str) -> LedgerEntry:
        entry = self._ledger.get(run_id)
        if entry is None:
            raise _rejected("unknown_run", "No recorded run was found for the requested run_id.")
        return entry


def ownership_summary_for_entry(entry: LedgerEntry) -> UserWorkflowOwnershipSummary:
    lifecycle = _lifecycle_summary(entry)
    if lifecycle is None:
        return UserWorkflowOwnershipSummary(
            ownership="unknown",
            blockers=["The ledgered context preview is missing a structured lifecycle summary."],
        )

    lifecycle_id = _safe_string(lifecycle.get("lifecycle_id"))
    sample_marker_present = "sample_workspace" in lifecycle
    sample_marker = lifecycle.get("sample_workspace") if sample_marker_present else None
    explicit_ownership = _safe_string(lifecycle.get("ownership"))
    warnings: list[str] = []
    blockers: list[str] = []
    safe_labels: dict[str, Any] = {}
    if lifecycle_id:
        safe_labels["lifecycle_id"] = lifecycle_id

    if isinstance(sample_marker, dict):
        sample_workspace = sample_marker.get("sample_workspace") is True
        sample_owned = sample_marker.get("sample_owned") is True
        sample_workspace_id = _safe_string(sample_marker.get("sample_workspace_id"))
        if sample_workspace_id:
            safe_labels["sample_workspace_id"] = sample_workspace_id
        if sample_workspace and sample_owned:
            return UserWorkflowOwnershipSummary(
                ownership="sample_owned",
                lifecycle_id=lifecycle_id,
                sample_workspace_id=sample_workspace_id,
                sample_owned=True,
                sample_workspace=True,
                warnings=warnings,
                safe_labels=safe_labels,
            )
        if lifecycle_id:
            if sample_workspace:
                warnings.append("Lifecycle has a sample workspace marker that is not sample-owned.")
            return UserWorkflowOwnershipSummary(
                ownership="user_owned",
                lifecycle_id=lifecycle_id,
                sample_workspace_id=sample_workspace_id,
                sample_owned=False,
                sample_workspace=sample_workspace,
                warnings=warnings,
                safe_labels=safe_labels,
            )

    if sample_marker_present or explicit_ownership == "user_owned":
        if not lifecycle_id:
            blockers.append("A lifecycle_id is required to classify a user-owned workflow.")
            return UserWorkflowOwnershipSummary(
                ownership="unknown",
                blockers=blockers,
                warnings=warnings,
                safe_labels=safe_labels,
            )
        return UserWorkflowOwnershipSummary(
            ownership="user_owned",
            lifecycle_id=lifecycle_id,
            sample_owned=False,
            sample_workspace=False,
            warnings=warnings,
            safe_labels=safe_labels,
        )

    blockers.append("No sample-owned or user-owned lifecycle marker was found.")
    return UserWorkflowOwnershipSummary(
        ownership="unknown",
        lifecycle_id=lifecycle_id,
        blockers=blockers,
        warnings=warnings,
        safe_labels=safe_labels,
    )


def latest_readiness_summary(entry: LedgerEntry) -> UserWorkflowReadinessSummary:
    for event in reversed(entry.recovery_events):
        if not isinstance(event, dict) or event.get("event_type") != READINESS_EVENT_TYPE:
            continue
        return UserWorkflowReadinessSummary(
            status=str(event.get("status") or "not_checked"),  # type: ignore[arg-type]
            readiness_intent=_safe_string(event.get("readiness_intent")),
            consent_required=event.get("consent_required") is True,
            allowed_preflight_capabilities=_safe_string_list(event.get("allowed_preflight_capabilities")),
            allowed_execution_capabilities=_safe_string_list(event.get("allowed_execution_capabilities")),
            blockers=_safe_string_list(event.get("blockers")),
            warnings=_safe_string_list(event.get("warnings")),
            checked_at_utc=_safe_string(event.get("checked_at_utc")),
        )
    ownership = ownership_summary_for_entry(entry)
    if ownership.ownership == "sample_owned":
        return UserWorkflowReadinessSummary(
            status="sample_owned",
            consent_required=False,
            warnings=["Sample-owned demo runs use sample-specific gates, not user-owned consent."],
        )
    return UserWorkflowReadinessSummary(
        status="not_checked",
        consent_required=ownership.ownership == "user_owned",
        allowed_preflight_capabilities=_preflight_capabilities(entry),
        allowed_execution_capabilities=_execution_capabilities(entry),
    )


def latest_consent_summary(entry: LedgerEntry) -> UserWorkflowConsentSummary:
    ownership = ownership_summary_for_entry(entry)
    if ownership.ownership == "sample_owned":
        return UserWorkflowConsentSummary(
            status="not_required",
            warnings=["Sample-owned demo runs use sample-specific gates, not user-owned consent."],
        )
    for event in reversed(entry.recovery_events):
        if not isinstance(event, dict) or event.get("event_type") != CONSENT_EVENT_TYPE:
            continue
        return UserWorkflowConsentSummary(
            status="consented",
            consent_intent=_safe_string(event.get("consent_intent")),
            consent_scope=_safe_string(event.get("consent_scope")),
            consented_by=_safe_string(event.get("consented_by")),
            consented_at_utc=_safe_string(event.get("consented_at_utc")),
            execution_permitted=False,
        )
    return UserWorkflowConsentSummary(
        status="not_recorded",
        blockers=(
            ["Approve guided user-owned workflow consent before confirmation, preview, run, or retry."]
            if ownership.ownership == "user_owned"
            else []
        ),
    )


def ensure_user_workflow_readiness(
    entry: LedgerEntry,
    *,
    step_id: str | None = None,
    capability_id: str | None = None,
) -> None:
    ownership = ownership_summary_for_entry(entry)
    if ownership.ownership != "user_owned":
        return
    readiness = latest_readiness_summary(entry)
    if readiness.status != "ready":
        raise _rejected(
            "user_workflow_readiness_required",
            "User-owned workflow preflight requires a successful readiness check.",
            step_id=step_id,
            capability_id=capability_id,
        )


def ensure_user_workflow_consent(
    entry: LedgerEntry,
    *,
    step_id: str | None = None,
    capability_id: str | None = None,
) -> None:
    ownership = ownership_summary_for_entry(entry)
    if ownership.ownership != "user_owned":
        return
    ensure_user_workflow_readiness(entry, step_id=step_id, capability_id=capability_id)
    consent = latest_consent_summary(entry)
    if consent.status != "consented" or consent.consent_scope != CONSENT_SCOPE:
        raise _rejected(
            "user_workflow_consent_required",
            "User-owned workflow confirmation, action request preview, execution, and retry require run-level consent.",
            step_id=step_id,
            capability_id=capability_id,
        )


def _readiness_summary(
    entry: LedgerEntry,
    ownership: UserWorkflowOwnershipSummary,
    readiness_intent: str,
) -> UserWorkflowReadinessSummary:
    checked_at = _utc_now_label()
    preflight_capabilities = _preflight_capabilities(entry)
    execution_capabilities = _execution_capabilities(entry)
    warnings = list(ownership.warnings)
    blockers = list(ownership.blockers)
    run_state = run_state_for_entry(entry)
    if ownership.ownership == "sample_owned":
        return UserWorkflowReadinessSummary(
            status="sample_owned",
            readiness_intent=readiness_intent,
            consent_required=False,
            allowed_preflight_capabilities=preflight_capabilities,
            allowed_execution_capabilities=execution_capabilities,
            warnings=[*warnings, "Sample-owned demo runs bypass user-owned consent gates."],
            checked_at_utc=checked_at,
        )
    if ownership.ownership != "user_owned":
        blockers.append("The run ownership could not be classified as user-owned.")
    if run_state in TERMINAL_OR_CANCELLED_STATES:
        blockers.append("Terminal, cancelled, or reset runs cannot start user-owned guided workflow gates.")
    if not isinstance(entry.plan_snapshot, dict) or not entry.plan_snapshot.get("plan_id"):
        blockers.append("A valid ledgered plan snapshot is required.")
    if not ownership.lifecycle_id:
        blockers.append("A safe lifecycle_id marker is required.")
    if not preflight_capabilities and not execution_capabilities:
        warnings.append("No user-owned preflight or draft execution capabilities are present in this run.")

    return UserWorkflowReadinessSummary(
        status="blocked" if blockers else "ready",
        readiness_intent=readiness_intent,
        consent_required=not blockers,
        allowed_preflight_capabilities=preflight_capabilities if not blockers else [],
        allowed_execution_capabilities=execution_capabilities if not blockers else [],
        blockers=blockers,
        warnings=warnings,
        checked_at_utc=checked_at,
    )


def _preflight_capabilities(entry: LedgerEntry) -> list[str]:
    capabilities: list[str] = []
    snapshot = entry.plan_snapshot if isinstance(entry.plan_snapshot, dict) else {}
    steps = snapshot.get("proposed_steps") if isinstance(snapshot, dict) else None
    if not isinstance(steps, list):
        return capabilities
    for step in steps:
        if not isinstance(step, dict) or step.get("preflight_required") is not True:
            continue
        capability_id = _safe_string(step.get("capability_id"))
        if capability_id and capability_id not in capabilities:
            capabilities.append(capability_id)
    return capabilities


def _execution_capabilities(entry: LedgerEntry) -> list[str]:
    capabilities: list[str] = []
    snapshot = entry.plan_snapshot if isinstance(entry.plan_snapshot, dict) else {}
    steps = snapshot.get("proposed_steps") if isinstance(snapshot, dict) else None
    if not isinstance(steps, list):
        return capabilities
    for step in steps:
        if not isinstance(step, dict):
            continue
        capability_id = _safe_string(step.get("capability_id"))
        if capability_id in SUPPORTED_EXECUTION_CAPABILITIES and capability_id not in capabilities:
            capabilities.append(capability_id)
    return capabilities


def _lifecycle_summary(entry: LedgerEntry) -> dict[str, Any] | None:
    if entry.context_preview is None or not isinstance(entry.context_preview.context, dict):
        return None
    lifecycle = entry.context_preview.context.get("lifecycle_summary")
    return lifecycle if isinstance(lifecycle, dict) else None


def _sanitized_current_context(context: dict[str, Any]) -> dict[str, Any]:
    sanitized, redaction = sanitize_value(context, path="user_workflow_context")
    if not isinstance(sanitized, dict):
        sanitized = {}
    if redaction.redacted:
        raise _rejected(
            "unsafe_user_workflow_context",
            "The user-owned workflow context included unsafe fields or values.",
        )
    return sanitized


def _safe_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _safe_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _reject_unsafe_event(record: dict[str, Any], *, code: str) -> None:
    unsafe_issues = find_unsafe_payload_issues(record, root="user_workflow")
    if unsafe_issues:
        raise RuntimeValidationError(
            PlanValidationResult(
                status="rejected",
                errors=[issue.model_copy(update={"code": code}) for issue in unsafe_issues],
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
