from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from quant_agent_runtime.capabilities import CapabilityRegistry
from quant_agent_runtime.context_preview import build_context_preview
from quant_agent_runtime.contracts import QuantSuiteContractLoader
from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.model_gateway import ModelProvider, ProviderPlanRequest
from quant_agent_runtime.models import (
    CapabilityDefinition,
    ConfirmationRequirement,
    ContextPreview,
    LedgerEntry,
    PlanRevisionReason,
    PlanRevisionRequest,
    PlanRevisionResult,
    PlanValidationResult,
    PolicySettings,
    ProviderPlanOutput,
    RiskTier,
    StructuredPlan,
    ValidationIssue,
)
from quant_agent_runtime.orchestration import orchestration_for_entry
from quant_agent_runtime.redaction import (
    find_unsafe_payload_issues,
    sanitize_value,
)
from quant_agent_runtime.run_state import run_state_for_entry
from quant_agent_runtime.validation import PlanValidator
from quant_agent_runtime.validation.errors import RuntimeValidationError


USER_PLAN_REVIEW_EVENT_TYPE = "user_plan_review"


class PlanRevisionService:
    def __init__(
        self,
        *,
        provider: ModelProvider,
        ledger: InMemoryLedger,
        contract_loader: QuantSuiteContractLoader,
        validator: PlanValidator | None = None,
        default_capabilities: list[CapabilityDefinition] | None = None,
    ) -> None:
        self._provider = provider
        self._ledger = ledger
        self._contract_loader = contract_loader
        self._validator = validator or PlanValidator()
        self._default_capabilities = default_capabilities

    def preview_revision(self, request: PlanRevisionRequest) -> PlanRevisionResult:
        entry = self._ledger.get(request.run_id)
        if entry is None:
            raise _rejected("unknown_run", "No recorded run was found for the requested run_id.")

        snapshot = entry.plan_snapshot if isinstance(entry.plan_snapshot, dict) else {}
        parent_plan_id = snapshot.get("plan_id")
        if not isinstance(parent_plan_id, str) or not parent_plan_id:
            raise _rejected(
                "missing_plan_revision_source",
                "The recorded run does not have a valid parent plan snapshot.",
            )
        steps = snapshot.get("proposed_steps")
        if not isinstance(steps, list):
            raise _rejected(
                "malformed_plan_revision_source",
                "The recorded parent plan snapshot is malformed.",
            )

        run_state = run_state_for_entry(entry)
        if run_state == "paused":
            raise _rejected(
                "paused_run_plan_revision",
                "The recorded run is paused and must be resumed before plan revision preview.",
            )
        if run_state in {"cancelled", "completed", "completed_with_warnings", "failed_terminal", "sample_reset"}:
            raise _rejected(
                "terminal_run_plan_revision",
                "The recorded run is terminal or cancelled and cannot be revised.",
            )

        sanitized_context, context_redaction = sanitize_value(
            request.current_context_summary,
            path="current_context_summary",
        )
        if not isinstance(sanitized_context, dict):
            sanitized_context = {}
        if context_redaction.redacted:
            raise RuntimeValidationError(
                PlanValidationResult(
                    status="rejected",
                    errors=[
                        ValidationIssue(
                            code="unsafe_revision_context",
                            message="The revision context included unsafe fields or values.",
                        )
                    ],
                )
            )

        original_context = _original_context(entry)
        stale_state_summary = _stale_state_summary(original_context, sanitized_context)
        orchestration = orchestration_for_entry(entry)
        review_revision_summary = _review_revision_summary(entry, parent_plan_id)
        blocker_summary = _blocker_summary(entry, orchestration, review_revision_summary=review_revision_summary)
        _ensure_revision_allowed(
            reason=request.reason,
            run_state=run_state,
            stale_state_summary=stale_state_summary,
            blocker_summary=blocker_summary,
            review_revision_summary=review_revision_summary,
        )

        context_preview = build_context_preview(
            sanitized_context or original_context,
            redaction_summary=context_redaction,
            context_sources=["current_context_summary"] if sanitized_context else ["ledger.context_preview"],
            warnings=_context_warnings(request.reason, stale_state_summary, blocker_summary),
        )
        fingerprint = _revision_fingerprint(
            parent_plan_id=parent_plan_id,
            reason=request.reason,
            sanitized_context=sanitized_context,
            blocker_summary=blocker_summary,
            review_revision_summary=review_revision_summary,
        )
        existing = _matching_revision_event(entry, fingerprint)
        if existing is not None:
            revised_plan = existing.get("revised_plan_snapshot")
            if not isinstance(revised_plan, dict):
                raise _rejected(
                    "malformed_existing_plan_revision",
                    "The existing plan revision preview record is malformed.",
                )
            return PlanRevisionResult(
                run_id=request.run_id,
                parent_plan_id=parent_plan_id,
                revision_id=str(existing.get("revision_id") or existing.get("recovery_event_id")),
                revised_plan=revised_plan,
                revision_event=existing,
                run_state=run_state,
                orchestration=orchestration,
                context_preview=ContextPreview.model_validate(existing.get("context_preview", context_preview.model_dump(mode="json"))),
                stale_state_summary=stale_state_summary,
                validation=PlanValidationResult.model_validate(existing.get("validation", {"status": "valid", "errors": [], "warnings": []})),
                ledger_recorded=True,
            )

        registry = CapabilityRegistry.from_request(None, default_registry=self._default_capabilities)
        revision_context = _revision_context(
            original_context=original_context,
            sanitized_context=sanitized_context,
            blocker_summary=blocker_summary,
            stale_state_summary=stale_state_summary,
            parent_plan=snapshot,
            review_revision_summary=review_revision_summary,
        )
        provider_result = self._provider.generate_plan(
            ProviderPlanRequest(
                user_goal=_revision_goal(entry, request.reason),
                context_summary=revision_context,
                capabilities=registry.enabled(),
                policy=PolicySettings(),
            )
        )
        try:
            provider_plan = ProviderPlanOutput.model_validate(provider_result.raw_output)
        except ValidationError as exc:
            raise _rejected(
                "malformed_revision_provider_output",
                _summarize_validation_error(exc),
            ) from exc

        validation = self._validator.validate(provider_plan, registry, PolicySettings())
        if validation.status == "rejected":
            raise RuntimeValidationError(validation)

        revision_id = f"revision_{uuid4().hex[:12]}"
        revised_plan = _structured_plan(revision_id, provider_plan).model_dump(mode="json")
        revised_plan.update(
            {
                "schema_version": "1.0",
                "data_policy": "summaries_and_references_only",
                "parent_plan_id": parent_plan_id,
                "revision_id": revision_id,
                "revision_source_run_id": request.run_id,
                "revision_reason": request.reason,
                "execution_permitted": False,
            }
        )
        try:
            self._contract_loader.validate_agent_contract_payload(
                revised_plan,
                "agent_plan.v1.schema.json",
            )
        except Exception as exc:
            raise _rejected(
                "malformed_revised_plan",
                "The generated revised plan failed contract validation.",
            ) from exc

        revision_event = {
            "recovery_event_id": revision_id,
            "revision_id": revision_id,
            "event_type": "plan_revision_preview",
            "status": "previewed",
            "revision_intent": request.revision_intent,
            "reason": request.reason,
            "parent_plan_id": parent_plan_id,
            "revised_plan_id": revised_plan["plan_id"],
            "created_at_utc": _utc_now_label(),
            "context_fingerprint": _stable_hash(sanitized_context),
            "blocker_fingerprint": _stable_hash(blocker_summary),
            "idempotency_fingerprint": fingerprint,
            "blocker_summary": blocker_summary,
            "stale_state_summary": stale_state_summary,
            "context_preview": context_preview.model_dump(mode="json"),
            "revised_plan_snapshot": revised_plan,
            "validation": validation.model_dump(mode="json"),
            "execution_permitted": False,
        }
        unsafe_issues = find_unsafe_payload_issues(revision_event, root="plan_revision")
        if unsafe_issues:
            raise RuntimeValidationError(
                PlanValidationResult(
                    status="rejected",
                    errors=[
                        issue.model_copy(update={"code": "unsafe_plan_revision_record"})
                        for issue in unsafe_issues
                    ],
                )
            )

        try:
            recorded_entry = self._ledger.append_recovery_event(request.run_id, revision_event)
        except ValueError as exc:
            raise _rejected(
                "unsafe_plan_revision_record",
                "The plan revision preview could not be safely ledgered.",
            ) from exc

        recorded_orchestration = orchestration_for_entry(recorded_entry)
        return PlanRevisionResult(
            run_id=request.run_id,
            parent_plan_id=parent_plan_id,
            revision_id=revision_id,
            revised_plan=revised_plan,
            revision_event=revision_event,
            run_state=run_state_for_entry(recorded_entry),
            orchestration=recorded_orchestration,
            context_preview=context_preview,
            stale_state_summary=stale_state_summary,
            validation=validation,
            ledger_recorded=True,
        )


def _structured_plan(plan_id: str, provider_plan: ProviderPlanOutput) -> StructuredPlan:
    risk_tiers = sorted({step.risk_tier for step in provider_plan.steps}, key=lambda item: item.value)
    required_confirmations = [
        ConfirmationRequirement(
            step_id=step.step_id,
            capability_id=step.capability_id,
            risk_tier=step.risk_tier,
            reason="Policy requires explicit confirmation before this step can execute.",
        )
        for step in provider_plan.steps
        if step.requires_confirmation
    ]
    return StructuredPlan(
        plan_id=f"plan_{plan_id}",
        user_goal_summary=provider_plan.user_goal_summary,
        assumptions=provider_plan.assumptions,
        missing_inputs=provider_plan.missing_inputs,
        proposed_steps=provider_plan.steps,
        risk_tiers=risk_tiers,
        required_confirmations=required_confirmations,
        status="blocked" if provider_plan.missing_inputs else "valid",
        execution_permitted=False,
    )


def _original_context(entry: LedgerEntry) -> dict[str, Any]:
    preview = entry.context_preview
    if preview is None or not isinstance(preview.context, dict):
        return {}
    return preview.context


def _revision_context(
    *,
    original_context: dict[str, Any],
    sanitized_context: dict[str, Any],
    blocker_summary: dict[str, Any],
    stale_state_summary: dict[str, Any],
    parent_plan: dict[str, Any],
    review_revision_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    context = dict(original_context)
    context.update(sanitized_context)
    context["revision_summary"] = {
        "parent_plan_id": parent_plan.get("plan_id"),
        "parent_plan_status": parent_plan.get("status"),
        "parent_missing_inputs": parent_plan.get("missing_inputs", []),
        "blocker_summary": blocker_summary,
        "stale_state_summary": stale_state_summary,
        "requested_assumption_revisions": review_revision_summary or {},
    }
    return context


def _blocker_summary(
    entry: LedgerEntry,
    orchestration: Any,
    *,
    review_revision_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot = entry.plan_snapshot if isinstance(entry.plan_snapshot, dict) else {}
    current = next((step for step in orchestration.steps if step.is_current), None)
    latest_preflight = entry.preflight_records[-1] if entry.preflight_records else None
    latest_action_result = entry.action_results[-1] if entry.action_results else None
    summary = {
        "run_state": orchestration.run_state,
        "final_status": entry.final_status,
        "plan_status": snapshot.get("status"),
        "missing_inputs": snapshot.get("missing_inputs", []),
        "current_step_id": current.step_id if current else None,
        "current_capability_id": current.capability_id if current else None,
        "current_step_status": current.status if current else None,
        "required_gate": current.required_gate if current else None,
        "blocker_reason": current.blocker_reason if current else None,
        "latest_preflight_status": latest_preflight.get("status") if isinstance(latest_preflight, dict) else None,
        "latest_preflight_blocker_count": len(latest_preflight.get("blockers", [])) if isinstance(latest_preflight, dict) and isinstance(latest_preflight.get("blockers"), list) else 0,
        "latest_action_result_status": latest_action_result.get("execution_status") if isinstance(latest_action_result, dict) else None,
    }
    if review_revision_summary:
        summary["requested_assumption_revisions"] = review_revision_summary
    return summary


def _review_revision_summary(entry: LedgerEntry, parent_plan_id: str) -> dict[str, Any] | None:
    latest_review: dict[str, Any] | None = None
    for record in reversed(entry.recovery_events):
        if isinstance(record, dict) and record.get("event_type") == USER_PLAN_REVIEW_EVENT_TYPE:
            latest_review = record
            break
    if latest_review is None:
        return None

    plan_id = latest_review.get("plan_id")
    if plan_id != parent_plan_id:
        return {
            "status": "stale_review",
            "plan_review_id": _safe_string(latest_review.get("plan_review_id") or latest_review.get("recovery_event_id")),
            "review_plan_id": _safe_string(plan_id),
            "active_plan_id": parent_plan_id,
            "revision_requested": False,
            "blocker_reason": "The latest user plan review does not match the active parent plan.",
        }

    status = _safe_string(latest_review.get("status")) or "not_reviewed"
    revision_notes = [
        {
            "assumption_index": item.get("assumption_index"),
            "safe_note": item.get("safe_note"),
        }
        for item in latest_review.get("revision_notes", [])
        if isinstance(item, dict)
    ]
    return {
        "status": status,
        "plan_review_id": _safe_string(latest_review.get("plan_review_id") or latest_review.get("recovery_event_id")),
        "plan_id": parent_plan_id,
        "revision_requested": status == "revision_requested",
        "total_assumption_count": _safe_int(latest_review.get("total_assumption_count")),
        "accepted_assumption_count": _safe_int(latest_review.get("accepted_assumption_count")),
        "revise_assumption_count": _safe_int(latest_review.get("revise_assumption_count")),
        "revision_notes": revision_notes,
        "blockers": _safe_string_list(latest_review.get("blockers")),
    }


def _stale_state_summary(
    original_context: dict[str, Any],
    current_context: dict[str, Any],
) -> dict[str, Any]:
    if not current_context:
        return {
            "current_context_provided": False,
            "state_changed_since_planning": False,
            "changed_top_level_keys": [],
            "added_top_level_keys": [],
            "removed_top_level_keys": [],
        }
    original_keys = set(original_context)
    current_keys = set(current_context)
    shared = sorted(original_keys & current_keys)
    changed = [
        key
        for key in shared
        if _stable_json(original_context.get(key)) != _stable_json(current_context.get(key))
    ]
    return {
        "current_context_provided": True,
        "state_changed_since_planning": bool(changed or (current_keys - original_keys) or (original_keys - current_keys)),
        "changed_top_level_keys": changed,
        "added_top_level_keys": sorted(current_keys - original_keys),
        "removed_top_level_keys": sorted(original_keys - current_keys),
    }


def _ensure_revision_allowed(
    *,
    reason: PlanRevisionReason,
    run_state: str,
    stale_state_summary: dict[str, Any],
    blocker_summary: dict[str, Any],
    review_revision_summary: dict[str, Any] | None,
) -> None:
    stale = stale_state_summary.get("state_changed_since_planning") is True
    missing = bool(blocker_summary.get("missing_inputs"))
    preflight_blocked = run_state == "preflight_blocked"
    failed_recoverable = run_state == "failed_recoverable"
    if reason == "missing_inputs" and not (run_state == "waiting_for_input" or missing):
        raise _rejected("no_plan_revision_needed", "The recorded run is not waiting for missing inputs.")
    if reason == "preflight_blocked" and not preflight_blocked:
        raise _rejected("no_plan_revision_needed", "The recorded run is not blocked by preflight.")
    if reason == "failed_recoverable" and not failed_recoverable:
        raise _rejected("no_plan_revision_needed", "The recorded run is not in a recoverable failure state.")
    if reason == "stale_state" and not stale:
        raise _rejected("no_plan_revision_needed", "The provided current context does not differ from the planned context.")
    revision_requested = (
        isinstance(review_revision_summary, dict)
        and review_revision_summary.get("revision_requested") is True
    )
    if reason == "user_requested" and not (
        run_state in {"waiting_for_input", "preflight_blocked", "failed_recoverable"} or stale or revision_requested
    ):
        raise _rejected(
            "no_plan_revision_needed",
            "The recorded run has no blocker, recoverable failure, stale context evidence, or revision-requested plan review.",
        )


def _context_warnings(
    reason: str,
    stale_state_summary: dict[str, Any],
    blocker_summary: dict[str, Any],
) -> list[str]:
    warnings = [f"Plan revision preview reason: {reason}."]
    if stale_state_summary.get("state_changed_since_planning") is True:
        warnings.append("Current context differs from the original planning context.")
    if blocker_summary.get("blocker_reason"):
        warnings.append("The active run has a blocker recorded in orchestration.")
    return warnings


def _revision_goal(entry: LedgerEntry, reason: str) -> str:
    return f"Revise the existing governed agent plan for reason {reason}: {entry.user_goal_summary}"


def _matching_revision_event(entry: LedgerEntry, fingerprint: str) -> dict[str, Any] | None:
    for record in reversed(entry.recovery_events):
        if not isinstance(record, dict):
            continue
        if record.get("event_type") == "plan_revision_preview" and record.get("idempotency_fingerprint") == fingerprint:
            return record
    return None


def _revision_fingerprint(
    *,
    parent_plan_id: str,
    reason: str,
    sanitized_context: dict[str, Any],
    blocker_summary: dict[str, Any],
    review_revision_summary: dict[str, Any] | None,
) -> str:
    return _stable_hash(
        {
            "parent_plan_id": parent_plan_id,
            "reason": reason,
            "current_context": sanitized_context,
            "blocker_summary": blocker_summary,
            "review_revision_summary": review_revision_summary or {},
        }
    )


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()[:24]


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)


def _safe_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _safe_int(value: Any) -> int:
    return value if isinstance(value, int) else 0


def _safe_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _utc_now_label() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _summarize_validation_error(exc: ValidationError) -> str:
    first_error = exc.errors()[0] if exc.errors() else {}
    location = ".".join(str(part) for part in first_error.get("loc", []))
    error_type = first_error.get("type", "validation_error")
    if location:
        return f"Revision provider output failed schema validation at {location}: {error_type}."
    return f"Revision provider output failed schema validation: {error_type}."


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
