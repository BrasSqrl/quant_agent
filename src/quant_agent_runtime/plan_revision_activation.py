from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from quant_agent_runtime.contracts import QuantSuiteContractLoader
from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.models import (
    ContextPreview,
    LedgerEntry,
    PlanRevisionActivationRequest,
    PlanRevisionActivationResult,
    PlanValidationResult,
    ValidationIssue,
)
from quant_agent_runtime.orchestration import orchestration_for_entry
from quant_agent_runtime.redaction import find_unsafe_payload_issues
from quant_agent_runtime.run_state import run_state_for_entry
from quant_agent_runtime.validation.errors import RuntimeValidationError


class PlanRevisionActivationService:
    def __init__(
        self,
        *,
        ledger: InMemoryLedger,
        contract_loader: QuantSuiteContractLoader,
    ) -> None:
        self._ledger = ledger
        self._contract_loader = contract_loader

    def activate_revision(
        self,
        request: PlanRevisionActivationRequest,
    ) -> PlanRevisionActivationResult:
        parent = self._ledger.get(request.run_id)
        if parent is None:
            raise _rejected("unknown_run", "No recorded run was found for the requested run_id.")

        parent_state = run_state_for_entry(parent)
        if parent_state == "paused":
            raise _rejected(
                "paused_run_revision_activation",
                "The recorded run is paused and must be resumed before activating a revised plan.",
            )
        if parent_state in {"cancelled", "completed", "completed_with_warnings", "failed_terminal"}:
            raise _rejected(
                "terminal_run_revision_activation",
                "The recorded run is terminal or cancelled and cannot activate a revised plan.",
            )

        preview = _revision_preview(parent, request.revision_id)
        if preview is None:
            raise _rejected(
                "unknown_plan_revision",
                "No ledgered plan revision preview was found for the requested revision_id.",
            )
        if preview.get("status") != "previewed":
            raise _rejected(
                "inactive_plan_revision",
                "Only previewed plan revisions can be activated.",
            )

        existing = _existing_activation(parent, request.revision_id)
        if existing is not None:
            return self._existing_activation_result(parent, existing, request.revision_id)

        revised_plan = preview.get("revised_plan_snapshot")
        if not isinstance(revised_plan, dict):
            raise _rejected(
                "malformed_plan_revision_event",
                "The ledgered plan revision does not contain a revised plan snapshot.",
            )
        if revised_plan.get("execution_permitted") is True:
            raise _rejected(
                "unsafe_plan_revision_activation",
                "Revised plans cannot grant execution permission during activation.",
            )
        _validate_revised_plan_contract(self._contract_loader, revised_plan)
        _validate_capability_snapshot(parent, revised_plan)

        child_run_id = f"run_revision_{uuid4().hex[:12]}"
        parent_plan_id = str(preview.get("parent_plan_id") or revised_plan.get("parent_plan_id") or "")
        if not parent_plan_id:
            raise _rejected(
                "malformed_plan_revision_event",
                "The ledgered plan revision does not contain a parent plan id.",
            )
        activation_id = f"activation_{uuid4().hex[:12]}"
        activated_at = _utc_now_label()
        activation_event = {
            "recovery_event_id": activation_id,
            "activation_id": activation_id,
            "event_type": "plan_revision_activation",
            "status": "activated",
            "activation_intent": request.activation_intent,
            "revision_id": request.revision_id,
            "parent_run_id": parent.run_id,
            "parent_plan_id": parent_plan_id,
            "revised_plan_id": revised_plan.get("plan_id"),
            "child_run_id": child_run_id,
            "active_plan_replaced": False,
            "activated_by": "local_user",
            "activated_at_utc": activated_at,
            "idempotency_key": f"activate_{parent.run_id}_{request.revision_id}",
            "execution_permitted": False,
        }
        _reject_unsafe_activation_event(activation_event)

        validation = _preview_validation(preview)
        child_entry = LedgerEntry(
            run_id=child_run_id,
            parent_run_id=parent.run_id,
            parent_plan_id=parent_plan_id,
            activated_revision_id=request.revision_id,
            user_goal_summary=str(revised_plan.get("user_goal_summary") or parent.user_goal_summary)[:240],
            provider_mode=parent.provider_mode,
            provider_metadata=parent.provider_metadata,
            redaction_summary=parent.redaction_summary,
            context_preview=_activation_context_preview(parent, preview),
            plan_snapshot=revised_plan,
            capability_snapshot=[dict(item) for item in parent.capability_snapshot],
            validation_results=validation,
            policy_rejections=[],
            final_status="planned",
        )
        try:
            recorded_child = self._ledger.append(child_entry)
            recorded_parent = self._ledger.append_plan_revision_activation(
                parent.run_id,
                activation_event,
                child_run_id=child_run_id,
            )
        except ValueError as exc:
            raise _rejected(
                "unsafe_plan_revision_activation_record",
                "The plan revision activation could not be safely ledgered.",
            ) from exc

        return _activation_result(
            parent=recorded_parent,
            child=recorded_child,
            activation_event=activation_event,
            revision_id=request.revision_id,
            validation=validation,
        )

    def _existing_activation_result(
        self,
        parent: LedgerEntry,
        event: dict[str, Any],
        revision_id: str,
    ) -> PlanRevisionActivationResult:
        child_run_id = event.get("child_run_id")
        if not isinstance(child_run_id, str) or not child_run_id:
            raise _rejected(
                "malformed_plan_revision_activation",
                "The existing activation record does not contain a child run id.",
            )
        child = self._ledger.get(child_run_id)
        if child is None:
            raise _rejected(
                "missing_plan_revision_child_run",
                "The existing activation record references a missing child run.",
            )
        if child.parent_run_id != parent.run_id or child.activated_revision_id != revision_id:
            raise _rejected(
                "conflicting_plan_revision_activation",
                "The existing activation child run does not match the requested revision.",
            )
        _validate_revised_plan_contract(self._contract_loader, child.plan_snapshot or {})
        return _activation_result(
            parent=parent,
            child=child,
            activation_event=event,
            revision_id=revision_id,
            validation=child.validation_results,
        )


def _activation_result(
    *,
    parent: LedgerEntry,
    child: LedgerEntry,
    activation_event: dict[str, Any],
    revision_id: str,
    validation: PlanValidationResult,
) -> PlanRevisionActivationResult:
    return PlanRevisionActivationResult(
        parent_run_id=parent.run_id,
        child_run_id=child.run_id,
        revision_id=revision_id,
        activated_plan=child.plan_snapshot or {},
        activation_event=activation_event,
        child_run_state=run_state_for_entry(child),
        child_orchestration=orchestration_for_entry(child),
        parent_run_state=run_state_for_entry(parent),
        validation=validation,
        ledger_recorded=True,
    )


def _revision_preview(entry: LedgerEntry, revision_id: str) -> dict[str, Any] | None:
    for record in reversed(entry.recovery_events):
        if not isinstance(record, dict):
            continue
        if record.get("event_type") == "plan_revision_preview" and record.get("revision_id") == revision_id:
            return record
    return None


def _existing_activation(entry: LedgerEntry, revision_id: str) -> dict[str, Any] | None:
    matches = [
        record
        for record in entry.recovery_events
        if isinstance(record, dict)
        and record.get("event_type") == "plan_revision_activation"
        and record.get("revision_id") == revision_id
    ]
    child_ids = {record.get("child_run_id") for record in matches if isinstance(record.get("child_run_id"), str)}
    if len(child_ids) > 1:
        raise _rejected(
            "conflicting_plan_revision_activation",
            "The requested revision has conflicting activation records.",
        )
    return matches[-1] if matches else None


def _validate_revised_plan_contract(
    contract_loader: QuantSuiteContractLoader,
    revised_plan: dict[str, Any],
) -> None:
    try:
        contract_loader.validate_agent_contract_payload(
            revised_plan,
            "agent_plan.v1.schema.json",
        )
    except Exception as exc:
        raise _rejected(
            "malformed_revised_plan",
            "The revised plan snapshot failed contract validation.",
        ) from exc


def _validate_capability_snapshot(parent: LedgerEntry, revised_plan: dict[str, Any]) -> None:
    snapshot = parent.capability_snapshot
    if not snapshot:
        raise _rejected(
            "stale_capability_snapshot",
            "The parent run does not have a capability snapshot to activate the revised plan.",
        )
    raw_steps = revised_plan.get("proposed_steps")
    if not isinstance(raw_steps, list):
        raise _rejected(
            "malformed_revised_plan",
            "The revised plan does not contain proposed steps.",
        )
    for step in raw_steps:
        if not isinstance(step, dict):
            raise _rejected(
                "malformed_revised_plan",
                "The revised plan contains a malformed step.",
            )
        capability_id = step.get("capability_id")
        app_id = step.get("app_id")
        risk_tier = step.get("risk_tier")
        if not isinstance(capability_id, str) or not isinstance(app_id, str):
            raise _rejected(
                "malformed_revised_plan",
                "The revised plan contains a step without a capability or app id.",
            )
        matching = [
            capability
            for capability in snapshot
            if isinstance(capability, dict) and capability.get("capability_id") == capability_id
        ]
        if not matching:
            raise _rejected(
                "unsupported_revision_capability",
                "The revised plan references a capability that was not in the recorded snapshot.",
                capability_id=capability_id,
            )
        capability = matching[0]
        if capability.get("enabled") is False:
            raise _rejected(
                "stale_capability_snapshot",
                "The recorded capability snapshot is no longer enabled for this revision.",
                capability_id=capability_id,
            )
        if capability.get("app_id") != app_id or str(capability.get("risk_tier")) != str(risk_tier):
            raise _rejected(
                "stale_capability_snapshot",
                "The revised plan no longer matches the recorded capability snapshot.",
                capability_id=capability_id,
            )


def _preview_validation(preview: dict[str, Any]) -> PlanValidationResult:
    validation = preview.get("validation")
    if isinstance(validation, dict):
        try:
            return PlanValidationResult.model_validate(validation)
        except ValidationError as exc:
            raise _rejected(
                "malformed_plan_revision_event",
                "The plan revision preview validation record is malformed.",
            ) from exc
    return PlanValidationResult(status="valid")


def _activation_context_preview(parent: LedgerEntry, preview: dict[str, Any]) -> ContextPreview | None:
    context_preview = preview.get("context_preview")
    if isinstance(context_preview, dict):
        try:
            return ContextPreview.model_validate(context_preview)
        except ValidationError as exc:
            raise _rejected(
                "malformed_plan_revision_event",
                "The plan revision preview context record is malformed.",
            ) from exc
    return parent.context_preview


def _reject_unsafe_activation_event(event: dict[str, Any]) -> None:
    unsafe_issues = find_unsafe_payload_issues(event, root="plan_revision_activation")
    if unsafe_issues:
        raise RuntimeValidationError(
            PlanValidationResult(
                status="rejected",
                errors=[
                    issue.model_copy(update={"code": "unsafe_plan_revision_activation_record"})
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
    capability_id: str | None = None,
) -> RuntimeValidationError:
    return RuntimeValidationError(
        PlanValidationResult(
            status="rejected",
            errors=[
                ValidationIssue(
                    code=code,
                    message=message,
                    capability_id=capability_id,
                )
            ],
        )
    )
