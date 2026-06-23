from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from quant_agent_runtime.capability_discovery import SUPPORTED_EXECUTION_CAPABILITIES
from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.models import (
    LedgerEntry,
    PlanValidationResult,
    UserPlanApprovalRequest,
    UserPlanApprovalResult,
    UserPlanApprovalSummary,
    UserPlanReviewRequest,
    UserPlanReviewResult,
    UserPlanReviewSummary,
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
PLAN_REVIEW_EVENT_TYPE = "user_plan_review"
PLAN_APPROVAL_EVENT_TYPE = "user_plan_approval"
READINESS_INTENT = "check_user_owned_readiness"
CONSENT_INTENT = "approve_user_owned_guided_execution"
CONSENT_SCOPE = "single_run_review_draft_actions"
PLAN_REVIEW_INTENT = "review_plan_assumptions"
PLAN_APPROVAL_INTENT = "approve_user_plan"
TERMINAL_OR_CANCELLED_STATES = {
    "cancelled",
    "sample_reset",
    "completed",
    "completed_with_warnings",
    "failed_terminal",
}
_USER_OWNED_GATED_ACTIONS = [
    "run_preflight",
    "confirm_step",
    "preview_action_request",
    "execute_step",
    "retry_failed_step",
]


class UserWorkflowService:
    def __init__(self, *, ledger: InMemoryLedger) -> None:
        self._ledger = ledger

    def review_plan(self, request: UserPlanReviewRequest) -> UserPlanReviewResult:
        entry = self._entry(request.run_id)
        _sanitized_current_context(request.current_context_summary)
        ownership = ownership_summary_for_entry(entry)
        _ensure_user_owned_plan_gate_target(entry, ownership)

        plan_id = _active_plan_id(entry)
        assumptions = _active_plan_assumptions(entry)
        reviews = _validated_assumption_reviews(request.assumption_reviews, assumptions)
        revise_count = sum(1 for item in reviews if item["decision"] == "revise")
        accepted_count = sum(1 for item in reviews if item["decision"] == "accept")
        reviewed_at = _utc_now_label()
        review_id = f"user_plan_review_{uuid4().hex[:12]}"
        status = "revision_requested" if revise_count else "reviewed"
        event = {
            "recovery_event_id": review_id,
            "plan_review_id": review_id,
            "event_type": PLAN_REVIEW_EVENT_TYPE,
            "status": status,
            "review_intent": request.review_intent,
            "run_id": entry.run_id,
            "plan_id": plan_id,
            "total_assumption_count": len(assumptions),
            "accepted_assumption_count": accepted_count,
            "revise_assumption_count": revise_count,
            "assumption_reviews": reviews,
            "revision_notes": [
                {
                    "assumption_index": item["assumption_index"],
                    "safe_note": item["safe_note"],
                }
                for item in reviews
                if item["decision"] == "revise"
            ],
            "blockers": (
                ["Plan approval is blocked until requested assumption revisions are handled."]
                if revise_count
                else []
            ),
            "warnings": [],
            "reviewed_by": "local_user",
            "reviewed_at_utc": reviewed_at,
            "data_policy": "summaries_and_references_only",
            "execution_permitted": False,
        }
        _reject_unsafe_event(event, code="unsafe_user_plan_review_record")

        try:
            recorded_entry = self._ledger.append_recovery_event(request.run_id, event)
        except ValueError as exc:
            raise _rejected(
                "unsafe_user_plan_review_record",
                "The user plan review record could not be safely ledgered.",
            ) from exc

        return UserPlanReviewResult(
            run_id=request.run_id,
            ownership_summary=ownership_summary_for_entry(recorded_entry),
            plan_review_summary=latest_plan_review_summary(recorded_entry),
            plan_approval_summary=latest_plan_approval_summary(recorded_entry),
            readiness_summary=latest_readiness_summary(recorded_entry),
            consent_summary=latest_consent_summary(recorded_entry),
            run_state=run_state_for_entry(recorded_entry),
            orchestration=orchestration_for_entry(recorded_entry),
            validation=PlanValidationResult(status="valid" if not revise_count else "rejected"),
            ledger_recorded=True,
        )

    def approve_plan(self, request: UserPlanApprovalRequest) -> UserPlanApprovalResult:
        entry = self._entry(request.run_id)
        ownership = ownership_summary_for_entry(entry)
        _ensure_user_owned_plan_gate_target(entry, ownership)
        review_event = _plan_review_event_by_id(entry, request.plan_review_id)
        if review_event is None:
            raise _rejected(
                "user_plan_review_required",
                "A matching active-plan assumption review is required before plan approval.",
            )
        plan_id = _active_plan_id(entry)
        if _safe_string(review_event.get("plan_id")) != plan_id:
            raise _rejected(
                "stale_user_plan_review",
                "The requested plan review does not match the active ledgered plan.",
            )
        if review_event.get("status") == "revision_requested":
            raise _rejected(
                "user_plan_revision_requested",
                "The latest plan review requested assumption revisions and cannot be approved.",
            )
        if review_event.get("status") != "reviewed":
            raise _rejected(
                "user_plan_review_not_approvable",
                "The requested plan review is not in an approvable state.",
            )

        existing = latest_plan_approval_summary(entry)
        if (
            existing.status == "approved"
            and existing.plan_id == plan_id
            and existing.plan_review_id == request.plan_review_id
        ):
            return UserPlanApprovalResult(
                run_id=request.run_id,
                ownership_summary=ownership,
                plan_review_summary=latest_plan_review_summary(entry),
                plan_approval_summary=existing,
                readiness_summary=latest_readiness_summary(entry),
                consent_summary=latest_consent_summary(entry),
                run_state=run_state_for_entry(entry),
                orchestration=orchestration_for_entry(entry),
                validation=PlanValidationResult(status="valid"),
                ledger_recorded=True,
            )

        approved_at = _utc_now_label()
        approval_id = f"user_plan_approval_{uuid4().hex[:12]}"
        event = {
            "recovery_event_id": approval_id,
            "plan_approval_id": approval_id,
            "event_type": PLAN_APPROVAL_EVENT_TYPE,
            "status": "approved",
            "approval_intent": request.approval_intent,
            "run_id": entry.run_id,
            "plan_id": plan_id,
            "plan_review_id": request.plan_review_id,
            "approved_by": "local_user",
            "approved_at_utc": approved_at,
            "execution_permitted": False,
        }
        _reject_unsafe_event(event, code="unsafe_user_plan_approval_record")

        try:
            recorded_entry = self._ledger.append_recovery_event(request.run_id, event)
        except ValueError as exc:
            raise _rejected(
                "unsafe_user_plan_approval_record",
                "The user plan approval record could not be safely ledgered.",
            ) from exc

        return UserPlanApprovalResult(
            run_id=request.run_id,
            ownership_summary=ownership_summary_for_entry(recorded_entry),
            plan_review_summary=latest_plan_review_summary(recorded_entry),
            plan_approval_summary=latest_plan_approval_summary(recorded_entry),
            readiness_summary=latest_readiness_summary(recorded_entry),
            consent_summary=latest_consent_summary(recorded_entry),
            run_state=run_state_for_entry(recorded_entry),
            orchestration=orchestration_for_entry(recorded_entry),
            validation=PlanValidationResult(status="valid"),
            ledger_recorded=True,
        )

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
            plan_review_summary=latest_plan_review_summary(recorded_entry),
            plan_approval_summary=latest_plan_approval_summary(recorded_entry),
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
        approval = latest_plan_approval_summary(entry)
        if approval.status != "approved" or approval.plan_id != _active_plan_id(entry):
            raise _rejected(
                "user_plan_approval_required",
                "The active user-owned plan must be reviewed and approved before workflow consent.",
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
                plan_review_summary=latest_plan_review_summary(entry),
                plan_approval_summary=latest_plan_approval_summary(entry),
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
            plan_review_summary=latest_plan_review_summary(recorded_entry),
            plan_approval_summary=latest_plan_approval_summary(recorded_entry),
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


def latest_plan_review_summary(entry: LedgerEntry) -> UserPlanReviewSummary:
    ownership = ownership_summary_for_entry(entry)
    if ownership.ownership == "sample_owned":
        return UserPlanReviewSummary(
            status="not_required",
            warnings=["Sample-owned demo runs use sample-specific gates, not user plan approval."],
        )

    plan_id = _active_plan_id(entry)
    if not plan_id:
        return UserPlanReviewSummary(
            status="blocked",
            blockers=["A valid ledgered plan snapshot is required before plan review."],
        )

    for event in reversed(entry.recovery_events):
        if not isinstance(event, dict) or event.get("event_type") != PLAN_REVIEW_EVENT_TYPE:
            continue
        event_plan_id = _safe_string(event.get("plan_id"))
        if event_plan_id != plan_id:
            return UserPlanReviewSummary(
                status="not_reviewed",
                plan_id=plan_id,
                total_assumption_count=len(_active_plan_assumptions(entry)),
                blockers=["Review the current active plan assumptions before approval."],
                warnings=["The latest plan review belongs to a previous plan."],
            )
        status = str(event.get("status") or "not_reviewed")
        return UserPlanReviewSummary(
            status=status,  # type: ignore[arg-type]
            plan_review_id=_safe_string(event.get("plan_review_id")) or _safe_string(event.get("recovery_event_id")),
            plan_id=event_plan_id,
            review_intent=_safe_string(event.get("review_intent")),
            total_assumption_count=_safe_int(event.get("total_assumption_count")),
            accepted_assumption_count=_safe_int(event.get("accepted_assumption_count")),
            revise_assumption_count=_safe_int(event.get("revise_assumption_count")),
            revision_notes=_safe_dict_list(event.get("revision_notes")),
            blockers=_safe_string_list(event.get("blockers")),
            warnings=_safe_string_list(event.get("warnings")),
            reviewed_by=_safe_string(event.get("reviewed_by")),
            reviewed_at_utc=_safe_string(event.get("reviewed_at_utc")),
        )

    return UserPlanReviewSummary(
        status="not_reviewed",
        plan_id=plan_id,
        total_assumption_count=len(_active_plan_assumptions(entry)),
        blockers=["Review the active plan assumptions before approval."],
    )


def latest_plan_approval_summary(entry: LedgerEntry) -> UserPlanApprovalSummary:
    ownership = ownership_summary_for_entry(entry)
    if ownership.ownership == "sample_owned":
        return UserPlanApprovalSummary(
            status="not_required",
            warnings=["Sample-owned demo runs use sample-specific gates, not user plan approval."],
        )

    plan_id = _active_plan_id(entry)
    if not plan_id:
        return UserPlanApprovalSummary(
            status="blocked",
            blockers=["A valid ledgered plan snapshot is required before plan approval."],
        )

    review = latest_plan_review_summary(entry)
    for event in reversed(entry.recovery_events):
        if not isinstance(event, dict) or event.get("event_type") != PLAN_APPROVAL_EVENT_TYPE:
            continue
        event_plan_id = _safe_string(event.get("plan_id"))
        if event_plan_id != plan_id:
            return UserPlanApprovalSummary(
                status="not_approved",
                plan_id=plan_id,
                blockers=["Approve the current active plan before user-owned workflow consent."],
                warnings=["The latest plan approval belongs to a previous plan."],
            )
        return UserPlanApprovalSummary(
            status="approved",
            plan_approval_id=_safe_string(event.get("plan_approval_id")) or _safe_string(event.get("recovery_event_id")),
            plan_review_id=_safe_string(event.get("plan_review_id")),
            plan_id=event_plan_id,
            approval_intent=_safe_string(event.get("approval_intent")),
            approved_by=_safe_string(event.get("approved_by")),
            approved_at_utc=_safe_string(event.get("approved_at_utc")),
            execution_permitted=False,
        )

    blockers = ["Approve the active plan before user-owned workflow consent."]
    if review.status == "not_reviewed":
        blockers = ["Review the active plan assumptions before approval."]
    elif review.status == "revision_requested":
        blockers = ["Resolve requested assumption revisions before approving the plan."]
    elif review.status == "blocked":
        blockers = list(review.blockers)
    return UserPlanApprovalSummary(
        status="not_approved",
        plan_review_id=review.plan_review_id,
        plan_id=plan_id,
        blockers=blockers,
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


def user_workflow_summaries_for_entry(
    entry: LedgerEntry,
    *,
    run_state: str | None = None,
) -> dict[str, Any]:
    ownership = ownership_summary_for_entry(entry)
    readiness = latest_readiness_summary(entry)
    consent = latest_consent_summary(entry)
    return {
        "ownership_summary": ownership,
        "plan_review_summary": latest_plan_review_summary(entry),
        "plan_approval_summary": latest_plan_approval_summary(entry),
        "readiness_summary": readiness,
        "consent_summary": consent,
        "allowed_user_owned_actions": allowed_user_owned_actions_for_entry(
            entry,
            ownership=ownership,
            readiness=readiness,
            consent=consent,
            run_state=run_state,
        ),
    }


def allowed_user_owned_actions_for_entry(
    entry: LedgerEntry,
    *,
    ownership: UserWorkflowOwnershipSummary | None = None,
    readiness: UserWorkflowReadinessSummary | None = None,
    consent: UserWorkflowConsentSummary | None = None,
    run_state: str | None = None,
) -> list[str]:
    ownership = ownership or ownership_summary_for_entry(entry)
    if ownership.ownership != "user_owned":
        return []

    effective_state = run_state or run_state_for_entry(entry)
    if effective_state in TERMINAL_OR_CANCELLED_STATES or effective_state == "paused":
        return []

    readiness = readiness or latest_readiness_summary(entry)
    if readiness.status != "ready":
        return ["check_user_owned_readiness"]

    plan_review = latest_plan_review_summary(entry)
    plan_approval = latest_plan_approval_summary(entry)
    if plan_approval.status != "approved":
        if plan_review.status == "reviewed" and plan_review.revise_assumption_count == 0:
            return ["approve_user_plan"]
        if plan_review.status == "revision_requested":
            return ["revise_plan"]
        return ["review_plan_assumptions"]

    consent = consent or latest_consent_summary(entry)
    if consent.status != "consented" or consent.consent_scope != CONSENT_SCOPE:
        return ["approve_user_owned_guided_execution"]

    actions: list[str] = []
    if readiness.allowed_preflight_capabilities:
        actions.append("run_preflight")
    if readiness.allowed_execution_capabilities:
        actions.extend(["confirm_step", "preview_action_request", "execute_step", "retry_failed_step"])
    return list(dict.fromkeys(actions))


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
    ensure_user_plan_approval(entry, step_id=step_id, capability_id=capability_id)


def ensure_user_plan_approval(
    entry: LedgerEntry,
    *,
    step_id: str | None = None,
    capability_id: str | None = None,
) -> None:
    ownership = ownership_summary_for_entry(entry)
    if ownership.ownership != "user_owned":
        return
    approval = latest_plan_approval_summary(entry)
    if approval.status != "approved" or approval.plan_id != _active_plan_id(entry):
        raise _rejected(
            "user_plan_approval_required",
            "User-owned workflow actions require an approved active plan.",
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


def _ensure_user_owned_plan_gate_target(entry: LedgerEntry, ownership: UserWorkflowOwnershipSummary) -> None:
    if ownership.ownership == "sample_owned":
        raise _rejected(
            "sample_plan_approval_not_required",
            "Sample-owned demo runs do not use the user-owned plan approval gate.",
        )
    if ownership.ownership != "user_owned":
        raise _rejected(
            "unknown_run_ownership",
            "The recorded run does not include explicit user-owned lifecycle ownership markers.",
        )
    run_state = run_state_for_entry(entry)
    if run_state in TERMINAL_OR_CANCELLED_STATES or run_state == "paused":
        raise _rejected(
            "inactive_run_user_plan_approval",
            "Paused, terminal, cancelled, or reset runs cannot record user plan review or approval.",
        )
    if not _active_plan_id(entry):
        raise _rejected(
            "missing_active_plan",
            "A valid ledgered plan snapshot is required for user plan review and approval.",
        )


def _active_plan_id(entry: LedgerEntry) -> str | None:
    snapshot = entry.plan_snapshot if isinstance(entry.plan_snapshot, dict) else {}
    return _safe_string(snapshot.get("plan_id")) if isinstance(snapshot, dict) else None


def _active_plan_assumptions(entry: LedgerEntry) -> list[str]:
    snapshot = entry.plan_snapshot if isinstance(entry.plan_snapshot, dict) else {}
    assumptions = snapshot.get("assumptions") if isinstance(snapshot, dict) else None
    if not isinstance(assumptions, list):
        return []
    return [assumption for assumption in assumptions if isinstance(assumption, str)]


def _validated_assumption_reviews(reviews: list[Any], assumptions: list[str]) -> list[dict[str, Any]]:
    expected_count = len(assumptions)
    if len(reviews) != expected_count:
        raise _rejected(
            "user_plan_assumption_review_count_mismatch",
            "Every active plan assumption must be reviewed exactly once.",
        )
    seen: set[int] = set()
    normalized: list[dict[str, Any]] = []
    for item in reviews:
        index = item.assumption_index
        if index in seen:
            raise _rejected(
                "duplicate_user_plan_assumption_review",
                "Each active plan assumption can be reviewed only once.",
            )
        if index < 0 or index >= expected_count:
            raise _rejected(
                "invalid_user_plan_assumption_review_index",
                "The assumption review index does not match the active plan assumptions.",
            )
        seen.add(index)
        safe_note = item.safe_note.strip() if isinstance(item.safe_note, str) else None
        if item.decision == "revise" and not safe_note:
            raise _rejected(
                "missing_user_plan_revision_note",
                "Assumption reviews marked for revision require a safe note.",
            )
        review = {
            "assumption_index": index,
            "decision": item.decision,
            "assumption_label": f"assumption_{index + 1}",
            "safe_note": safe_note,
        }
        _reject_unsafe_event(review, code="unsafe_user_plan_review_record")
        normalized.append(review)
    if seen != set(range(expected_count)):
        raise _rejected(
            "missing_user_plan_assumption_review",
            "Every active plan assumption must have a matching review.",
        )
    return sorted(normalized, key=lambda item: item["assumption_index"])


def _plan_review_event_by_id(entry: LedgerEntry, plan_review_id: str) -> dict[str, Any] | None:
    for event in reversed(entry.recovery_events):
        if not isinstance(event, dict) or event.get("event_type") != PLAN_REVIEW_EVENT_TYPE:
            continue
        if event.get("plan_review_id") == plan_review_id or event.get("recovery_event_id") == plan_review_id:
            return event
    return None


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


def _safe_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _safe_int(value: Any) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


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
