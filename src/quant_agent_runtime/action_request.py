from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from quant_agent_runtime.contracts import QuantSuiteContractLoader
from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.models import (
    ActionRequestPreviewRequest,
    ActionRequestPreviewResult,
    LedgerEntry,
    PlanValidationResult,
    ValidationIssue,
)
from quant_agent_runtime.orchestration import ensure_step_action_allowed
from quant_agent_runtime.redaction import find_unsafe_payload_issues
from quant_agent_runtime.run_state import run_state_for_entry
from quant_agent_runtime.user_workflow import ensure_user_workflow_consent
from quant_agent_runtime.validation.errors import RuntimeValidationError
from quant_agent_runtime.workflow_handoffs import action_input_with_workflow_handoffs


ACTION_REQUEST_CONTRACT_SCHEMA = "agent_action_request.v1.schema.json"


class ActionRequestPreviewService:
    def __init__(self, *, ledger: InMemoryLedger, contract_loader: QuantSuiteContractLoader) -> None:
        self._ledger = ledger
        self._contract_loader = contract_loader

    def create_action_request(self, request: ActionRequestPreviewRequest) -> ActionRequestPreviewResult:
        entry = self._ledger.get(request.run_id)
        if entry is None:
            raise _rejected("unknown_run", "No recorded plan was found for the requested run_id.")

        step = _plan_step(entry, request.step_id)
        if step is None:
            raise _rejected(
                "unknown_step",
                "No recorded plan step was found for the requested step_id.",
                step_id=request.step_id,
            )

        plan_state = run_state_for_entry(entry)
        capability_id = str(step.get("capability_id") or "")
        app_id = str(step.get("app_id") or "")
        if plan_state == "waiting_for_input":
            raise _rejected(
                "blocked_plan_action_request",
                "The recorded plan is blocked by missing inputs and cannot produce an action request.",
                step_id=request.step_id,
                capability_id=capability_id or None,
            )
        if plan_state == "preflight_blocked":
            raise _rejected(
                "preflight_blocked_action_request",
                "The recorded run has a blocked preflight and cannot produce an action request.",
                step_id=request.step_id,
                capability_id=capability_id or None,
            )
        if plan_state == "cancelled":
            raise _rejected(
                "cancelled_run_action_request",
                "The recorded run has been cancelled and cannot produce an action request.",
                step_id=request.step_id,
                capability_id=capability_id or None,
            )
        if plan_state == "sample_reset":
            raise _rejected(
                "terminal_run_action_request",
                "The recorded run is terminal and cannot produce an action request.",
                step_id=request.step_id,
                capability_id=capability_id or None,
            )
        if plan_state == "paused":
            raise _rejected(
                "paused_run_action_request",
                "The recorded run is paused and must be resumed before action request preview.",
                step_id=request.step_id,
                capability_id=capability_id or None,
            )

        ensure_user_workflow_consent(
            entry,
            step_id=request.step_id,
            capability_id=capability_id or None,
        )
        existing = _existing_action_request(entry, request.step_id, capability_id)
        if existing is not None:
            self._validate_action_request(existing, request.step_id, capability_id)
            return ActionRequestPreviewResult(
                run_id=request.run_id,
                step_id=request.step_id,
                capability_id=capability_id,
                action_request=existing,
                run_state=run_state_for_entry(entry),
                validation=PlanValidationResult(status="valid"),
                ledger_recorded=True,
            )

        ensure_step_action_allowed(entry, request.step_id, "preview_action_request")

        capability = _capability_snapshot(entry, capability_id, app_id)
        if capability is None:
            raise _rejected(
                "stale_action_request_capability_snapshot",
                "The recorded plan step capability was not found in the ledger capability snapshot.",
                step_id=request.step_id,
                capability_id=capability_id or None,
            )
        if str(capability.get("risk_tier") or "") != str(step.get("risk_tier") or ""):
            raise _rejected(
                "stale_action_request_capability_snapshot",
                "The recorded plan step risk tier no longer matches the ledger capability snapshot.",
                step_id=request.step_id,
                capability_id=capability_id or None,
            )
        if capability.get("enabled", True) is not True:
            raise _rejected(
                "stale_action_request_capability_snapshot",
                "The recorded plan step capability is disabled in the ledger capability snapshot.",
                step_id=request.step_id,
                capability_id=capability_id or None,
            )

        action_input = action_input_with_workflow_handoffs(entry, step, fail_on_missing=True)
        action_input_issues = find_unsafe_payload_issues(action_input, root="action_input")
        if action_input_issues:
            raise RuntimeValidationError(
                PlanValidationResult(
                    status="rejected",
                    errors=[
                        issue.model_copy(
                            update={
                                "code": "unsafe_action_input",
                                "step_id": request.step_id,
                                "capability_id": capability_id or None,
                            }
                        )
                        for issue in action_input_issues
                    ],
                )
            )

        confirmation_reference = self._confirmation_reference(entry, request.step_id, capability_id, step)
        preflight_reference = self._preflight_reference(entry, request.step_id, capability_id, app_id, step, capability)
        lifecycle_reference = _lifecycle_state_reference(entry, request.step_id, capability_id)
        plan_id = _plan_id(entry, request.step_id, capability_id)
        input_schema_version = str(capability.get("version") or "1.0-draft")

        action_request = {
            "schema_version": "1.0",
            "data_policy": "summaries_and_references_only",
            "agent_run_id": request.run_id,
            "plan_id": plan_id,
            "step_id": request.step_id,
            "capability_id": capability_id,
            "app_id": app_id,
            "action_input": action_input,
            "input_schema_version": input_schema_version,
            "confirmation_reference": confirmation_reference,
            "preflight_reference": preflight_reference,
            "lifecycle_state_reference": lifecycle_reference,
            "idempotency_key": f"idem_{request.run_id}_{request.step_id}_{capability_id}",
            "requested_at_utc": _utc_now_label(),
            "execution_permitted": False,
            "redaction_summary": entry.redaction_summary.model_dump(mode="json"),
        }
        self._validate_action_request(action_request, request.step_id, capability_id)

        try:
            recorded_entry = self._ledger.append_action_request(request.run_id, action_request)
        except ValueError as exc:
            raise _rejected(
                "unsafe_action_request_record",
                "The action request could not be safely ledgered.",
                step_id=request.step_id,
                capability_id=capability_id or None,
            ) from exc

        return ActionRequestPreviewResult(
            run_id=request.run_id,
            step_id=request.step_id,
            capability_id=capability_id,
            action_request=action_request,
            run_state=run_state_for_entry(recorded_entry),
            validation=PlanValidationResult(status="valid"),
            ledger_recorded=True,
        )

    def _confirmation_reference(
        self,
        entry: LedgerEntry,
        step_id: str,
        capability_id: str,
        step: dict[str, Any],
    ) -> dict[str, Any] | None:
        required = _required_confirmation(entry, step_id, capability_id)
        if required is None and not bool(step.get("requires_confirmation")):
            return None
        confirmed = _confirmed_record(entry, step_id, capability_id)
        if confirmed is None:
            raise _rejected(
                "missing_confirmation_for_action_request",
                "The recorded plan step requires confirmation before an action request preview can be created.",
                step_id=step_id,
                capability_id=capability_id or None,
            )
        return {
            "confirmation_id": confirmed.get("confirmation_id"),
            "status": confirmed.get("status"),
            "reason": confirmed.get("reason"),
            "confirmed_by": confirmed.get("confirmed_by"),
            "confirmed_at_utc": confirmed.get("confirmed_at_utc"),
            "execution_permitted": False,
        }

    def _preflight_reference(
        self,
        entry: LedgerEntry,
        step_id: str,
        capability_id: str,
        app_id: str,
        step: dict[str, Any],
        capability: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not (bool(step.get("preflight_required")) or bool(capability.get("preflight_required"))):
            return None
        preflight = _latest_preflight(entry, capability_id, app_id)
        if preflight is None:
            raise _rejected(
                "missing_preflight_for_action_request",
                "The recorded plan step requires a ready or warning preflight before action request preview.",
                step_id=step_id,
                capability_id=capability_id or None,
            )
        blockers = preflight.get("blockers")
        if preflight.get("status") == "blocked" or (isinstance(blockers, list) and blockers):
            raise _rejected(
                "blocked_preflight_for_action_request",
                "The latest matching preflight has blockers and cannot produce an action request preview.",
                step_id=step_id,
                capability_id=capability_id or None,
            )
        if preflight.get("status") not in {"ready", "warning"}:
            raise _rejected(
                "invalid_preflight_for_action_request",
                "The latest matching preflight must have ready or warning status.",
                step_id=step_id,
                capability_id=capability_id or None,
            )
        return {
            "preflight_id": preflight.get("preflight_id"),
            "status": preflight.get("status"),
            "capability_id": preflight.get("capability_id"),
            "app_id": preflight.get("app_id"),
            "warnings": preflight.get("warnings") if isinstance(preflight.get("warnings"), list) else [],
            "safe_artifact_references": (
                preflight.get("safe_artifact_references")
                if isinstance(preflight.get("safe_artifact_references"), list)
                else []
            ),
            "revalidation_required": bool(preflight.get("revalidation_required")),
        }

    def _validate_action_request(
        self,
        action_request: dict[str, Any],
        step_id: str,
        capability_id: str,
    ) -> None:
        unsafe_issues = find_unsafe_payload_issues(action_request, root="action_request")
        if unsafe_issues:
            raise RuntimeValidationError(
                PlanValidationResult(
                    status="rejected",
                    errors=[
                        issue.model_copy(
                            update={
                                "code": "unsafe_action_request_record",
                                "step_id": step_id,
                                "capability_id": capability_id or None,
                            }
                        )
                        for issue in unsafe_issues
                    ],
                )
            )
        try:
            self._contract_loader.validate_agent_contract_payload(
                action_request,
                ACTION_REQUEST_CONTRACT_SCHEMA,
            )
        except Exception as exc:
            raise _rejected(
                "malformed_generated_action_request",
                f"The generated action request failed contract validation: {exc}",
                step_id=step_id,
                capability_id=capability_id or None,
            ) from exc


def _plan_step(entry: LedgerEntry, step_id: str) -> dict[str, Any] | None:
    snapshot = entry.plan_snapshot if isinstance(entry.plan_snapshot, dict) else {}
    steps = snapshot.get("proposed_steps")
    if not isinstance(steps, list):
        return None
    for step in steps:
        if isinstance(step, dict) and step.get("step_id") == step_id:
            return step
    return None


def _plan_id(entry: LedgerEntry, step_id: str, capability_id: str) -> str:
    snapshot = entry.plan_snapshot if isinstance(entry.plan_snapshot, dict) else {}
    plan_id = snapshot.get("plan_id")
    if isinstance(plan_id, str) and plan_id:
        return plan_id
    raise _rejected(
        "missing_plan_id_for_action_request",
        "The recorded plan snapshot is missing plan_id.",
        step_id=step_id,
        capability_id=capability_id or None,
    )


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


def _confirmed_record(entry: LedgerEntry, step_id: str, capability_id: str) -> dict[str, Any] | None:
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


def _latest_preflight(entry: LedgerEntry, capability_id: str, app_id: str) -> dict[str, Any] | None:
    for record in reversed(entry.preflight_records):
        if not isinstance(record, dict):
            continue
        if record.get("capability_id") == capability_id and record.get("app_id") == app_id:
            return record
    return None


def _existing_action_request(
    entry: LedgerEntry,
    step_id: str,
    capability_id: str,
) -> dict[str, Any] | None:
    for record in reversed(entry.action_requests):
        if not isinstance(record, dict):
            continue
        if (
            record.get("agent_run_id") == entry.run_id
            and record.get("step_id") == step_id
            and record.get("capability_id") == capability_id
        ):
            return record
    return None


def _lifecycle_state_reference(entry: LedgerEntry, step_id: str, capability_id: str) -> dict[str, str]:
    context = entry.context_preview.context if entry.context_preview is not None else {}
    lifecycle_summary = context.get("lifecycle_summary") if isinstance(context, dict) else None
    if not isinstance(lifecycle_summary, dict):
        raise _rejected(
            "missing_lifecycle_state_reference",
            "The recorded context preview is missing a structured lifecycle summary.",
            step_id=step_id,
            capability_id=capability_id or None,
        )
    lifecycle_id = lifecycle_summary.get("lifecycle_id")
    state = lifecycle_summary.get("state")
    summary = lifecycle_summary.get("summary")
    if not all(isinstance(value, str) and value for value in [lifecycle_id, state, summary]):
        raise _rejected(
            "missing_lifecycle_state_reference",
            "The recorded lifecycle summary must include lifecycle_id, state, and summary.",
            step_id=step_id,
            capability_id=capability_id or None,
        )
    return {
        "lifecycle_id": lifecycle_id,
        "state": state,
        "summary": summary,
    }


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
