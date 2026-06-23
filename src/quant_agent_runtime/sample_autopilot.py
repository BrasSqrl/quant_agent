from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from quant_agent_runtime.action_request import ActionRequestPreviewService
from quant_agent_runtime.app_clients import AppClientError
from quant_agent_runtime.execution import ExecutionService
from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.models import (
    ActionRequestPreviewRequest,
    ExecutionRequest,
    LedgerEntry,
    PlanValidationResult,
    PreflightRequest,
    SampleAutopilotEligibility,
    SampleAutopilotPreview,
    SampleAutopilotPreviewRequest,
    SampleAutopilotPreviewResult,
    SampleAutopilotPreviewStep,
    SampleAutopilotStepRequest,
    SampleAutopilotStepResult,
    ValidationIssue,
)
from quant_agent_runtime.orchestration import orchestration_for_entry
from quant_agent_runtime.preflight import PreflightService
from quant_agent_runtime.redaction import find_unsafe_payload_issues, sanitize_value
from quant_agent_runtime.run_state import run_state_for_entry
from quant_agent_runtime.validation.errors import RuntimeValidationError


DEFAULT_SAMPLE_AUTOPILOT_ALLOWLIST = {"credit_pd_scorecard_panel"}
PHASE7_SAMPLE_DEMO_CAPABILITY_SEQUENCE = [
    "quant_data.run_source_preflight",
    "quant_studio.prepare_model_config_draft",
    "quant_documentation.inspect_package",
    "quant_documentation.create_draft_workspace",
    "quant_monitoring.validate_bundle",
]
PHASE7_SAMPLE_DEMO_GATE_REQUIREMENTS = {
    "quant_data.run_source_preflight": {
        "preflight_required": True,
        "requires_confirmation": False,
    },
    "quant_studio.prepare_model_config_draft": {
        "preflight_required": False,
        "requires_confirmation": True,
    },
    "quant_documentation.inspect_package": {
        "preflight_required": False,
        "requires_confirmation": False,
    },
    "quant_documentation.create_draft_workspace": {
        "preflight_required": False,
        "requires_confirmation": True,
    },
    "quant_monitoring.validate_bundle": {
        "preflight_required": True,
        "requires_confirmation": False,
    },
}
_TERMINAL_OR_CANCELLED_RUN_STATES = {
    "cancelled",
    "completed",
    "completed_with_warnings",
    "failed_terminal",
    "sample_reset",
}


@dataclass(frozen=True)
class SampleWorkspaceMetadata:
    sample_workspace_id: str
    label: str
    lifecycle_id: str | None
    sample_owned: bool
    reset_boundary_available: bool
    allowed_delete_scopes: list[str]
    protected_scopes: list[str]


@dataclass(frozen=True)
class SampleAutopilotEvaluation:
    entry: LedgerEntry
    current_context: dict[str, Any]
    orchestration: Any
    eligibility: SampleAutopilotEligibility


class SampleAutopilotPreviewService:
    def __init__(
        self,
        *,
        ledger: InMemoryLedger,
        sample_workspace_root: Path | None = None,
        allowlist: set[str] | None = None,
    ) -> None:
        self._ledger = ledger
        self._sample_workspace_root = sample_workspace_root or _default_sample_workspace_root()
        self._allowlist = allowlist or DEFAULT_SAMPLE_AUTOPILOT_ALLOWLIST

    def preview_autopilot(
        self,
        request: SampleAutopilotPreviewRequest,
    ) -> SampleAutopilotPreviewResult:
        evaluation = self.evaluate_eligibility(
            run_id=request.run_id,
            current_context_summary=request.current_context_summary,
        )
        entry = evaluation.entry
        orchestration = evaluation.orchestration
        eligibility = evaluation.eligibility
        preview = _preview_from_orchestration(orchestration, eligibility)
        event = {
            "recovery_event_id": f"sample_autopilot_preview_{uuid4().hex[:12]}",
            "event_type": "sample_autopilot_preview",
            "status": "eligible_previewed" if eligibility.eligible else "blocked",
            "autopilot_intent": request.autopilot_intent,
            "run_id": entry.run_id,
            "plan_id": orchestration.plan_id,
            "created_by": "local_user",
            "created_at_utc": _utc_now_label(),
            "sample_ownership_summary": {
                "sample_workspace_id": eligibility.sample_workspace_id,
                "sample_label": eligibility.sample_label,
                "lifecycle_id": eligibility.lifecycle_id,
                "sample_owned": eligibility.sample_owned,
                "allowlisted": eligibility.allowlisted,
                "reset_boundary_available": eligibility.reset_boundary_available,
            },
            "dry_run_step_count": preview.step_count,
            "dry_run_blocked_step_count": preview.blocked_step_count,
            "blockers": [*eligibility.blockers, *preview.blockers],
            "warnings": [*eligibility.warnings, *preview.warnings],
            "dry_run_only": True,
            "execution_permitted": False,
        }
        _reject_unsafe_autopilot_event(event)

        try:
            recorded_entry = self._ledger.append_recovery_event(request.run_id, event)
        except ValueError as exc:
            raise _rejected(
                "unsafe_autopilot_preview_record",
                "The sample autopilot preview could not be safely ledgered.",
            ) from exc

        recorded_orchestration = orchestration_for_entry(recorded_entry)
        return SampleAutopilotPreviewResult(
            run_id=request.run_id,
            sample_eligibility=eligibility,
            autopilot_preview=preview,
            run_progress_summary=recorded_orchestration.run_progress_summary,
            orchestration=recorded_orchestration,
            validation=PlanValidationResult(status="valid"),
            ledger_recorded=True,
        )

    def evaluate_eligibility(
        self,
        *,
        run_id: str,
        current_context_summary: dict[str, Any],
        include_run_state_checks: bool = True,
    ) -> SampleAutopilotEvaluation:
        entry = self._ledger.get(run_id)
        if entry is None:
            raise _rejected("unknown_run", "No recorded run was found for the requested run_id.")

        sanitized_context, redaction = sanitize_value(
            current_context_summary,
            path="current_context_summary",
        )
        if not isinstance(sanitized_context, dict):
            sanitized_context = {}

        orchestration = orchestration_for_entry(entry)
        metadata_by_id = self._load_sample_workspace_metadata()
        eligibility = self._eligibility(
            entry=entry,
            current_context=sanitized_context,
            metadata_by_id=metadata_by_id,
            unsafe_context=redaction.redacted,
            include_run_state_checks=include_run_state_checks,
        )
        return SampleAutopilotEvaluation(
            entry=entry,
            current_context=sanitized_context,
            orchestration=orchestration,
            eligibility=eligibility,
        )

    def _eligibility(
        self,
        *,
        entry: LedgerEntry,
        current_context: dict[str, Any],
        metadata_by_id: dict[str, SampleWorkspaceMetadata],
        unsafe_context: bool,
        include_run_state_checks: bool,
    ) -> SampleAutopilotEligibility:
        original_context = _original_context(entry)
        original_marker = _sample_marker(original_context)
        current_marker = _sample_marker(current_context)
        marker = current_marker or original_marker
        sample_workspace_id = _sample_workspace_id(marker)
        metadata = metadata_by_id.get(sample_workspace_id or "")
        lifecycle_id = _safe_label(
            _nested_value(current_context, ["lifecycle_summary", "lifecycle_id"])
            or _nested_value(original_context, ["lifecycle_summary", "lifecycle_id"])
            or (metadata.lifecycle_id if metadata else None)
        )

        blockers: list[str] = []
        warnings: list[str] = []
        if unsafe_context:
            blockers.append("Current context included unsafe fields or values and cannot be used for autopilot.")
        if not sample_workspace_id:
            blockers.append("No sample_workspace_id was found in the ledgered or current context.")
        if marker.get("sample_owned") is not True:
            blockers.append("The selected lifecycle is not marked sample_owned.")
        if original_marker and current_marker and original_marker != current_marker:
            blockers.append("Current context sample ownership does not match the ledgered plan context.")
        if sample_workspace_id and sample_workspace_id not in self._allowlist:
            blockers.append("The sample workspace is not allowlisted for Phase 7 dry-run autopilot.")
        if sample_workspace_id and metadata is None:
            blockers.append("No matching sample workspace metadata was found in quant_suite fixtures.")
        if metadata is not None and not metadata.reset_boundary_available:
            blockers.append("The sample workspace does not declare a safe reset boundary.")

        if include_run_state_checks:
            run_state = run_state_for_entry(entry)
            if run_state == "paused":
                blockers.append("Paused runs must be resumed before sample autopilot preview.")
            if run_state in _TERMINAL_OR_CANCELLED_RUN_STATES:
                blockers.append("Terminal or cancelled runs cannot use sample autopilot preview.")

        capability_blockers = _capability_snapshot_blockers(entry)
        blockers.extend(capability_blockers)
        blockers.extend(_sample_demo_plan_blockers(entry))

        if not current_context:
            warnings.append("No current sanitized lifecycle context was provided; using ledgered context only.")
        eligible = not blockers
        return SampleAutopilotEligibility(
            eligible=eligible,
            status="eligible" if eligible else "blocked",
            sample_workspace_id=sample_workspace_id,
            sample_label=metadata.label if metadata else None,
            lifecycle_id=lifecycle_id,
            sample_owned=marker.get("sample_owned") is True,
            allowlisted=bool(sample_workspace_id and sample_workspace_id in self._allowlist),
            reset_boundary_available=bool(metadata and metadata.reset_boundary_available),
            blockers=blockers,
            warnings=warnings,
            safe_labels={
                "allowed_delete_scopes": metadata.allowed_delete_scopes if metadata else [],
                "protected_scopes": metadata.protected_scopes if metadata else [],
                "sample_demo_capability_sequence": PHASE7_SAMPLE_DEMO_CAPABILITY_SEQUENCE,
            },
        )

    def _load_sample_workspace_metadata(self) -> dict[str, SampleWorkspaceMetadata]:
        if not self._sample_workspace_root.is_dir():
            return {}
        metadata_by_id: dict[str, SampleWorkspaceMetadata] = {}
        for path in sorted(self._sample_workspace_root.glob("*/sample_workspace.v1.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            sample_id = _safe_label(payload.get("sample_workspace_id"))
            if not sample_id:
                continue
            marker = payload.get("owned_marker") if isinstance(payload.get("owned_marker"), dict) else {}
            reset_scope = payload.get("reset_scope") if isinstance(payload.get("reset_scope"), dict) else {}
            allowed_delete_scopes = _safe_string_list(reset_scope.get("allowed_delete_scopes"))
            protected_scopes = _safe_string_list(reset_scope.get("protected_scopes"))
            metadata_by_id[sample_id] = SampleWorkspaceMetadata(
                sample_workspace_id=sample_id,
                label=_safe_label(payload.get("label")) or sample_id,
                lifecycle_id=_safe_label(payload.get("lifecycle_id")) or None,
                sample_owned=marker.get("sample_owned") is True,
                reset_boundary_available=(
                    marker.get("sample_owned") is True
                    and reset_scope.get("sample_owned_only") is True
                    and bool(allowed_delete_scopes)
                    and bool(protected_scopes)
                ),
                allowed_delete_scopes=allowed_delete_scopes,
                protected_scopes=protected_scopes,
            )
        return metadata_by_id


class SampleAutopilotStepService:
    def __init__(
        self,
        *,
        ledger: InMemoryLedger,
        preflight: PreflightService,
        action_request: ActionRequestPreviewService,
        execution: ExecutionService,
        sample_workspace_root: Path | None = None,
        allowlist: set[str] | None = None,
    ) -> None:
        self._ledger = ledger
        self._preflight = preflight
        self._action_request = action_request
        self._execution = execution
        self._eligibility = SampleAutopilotPreviewService(
            ledger=ledger,
            sample_workspace_root=sample_workspace_root,
            allowlist=allowlist,
        )

    def advance_one_step(self, request: SampleAutopilotStepRequest) -> SampleAutopilotStepResult:
        evaluation = self._eligibility.evaluate_eligibility(
            run_id=request.run_id,
            current_context_summary=request.current_context_summary,
        )
        current_step = next((step for step in evaluation.orchestration.steps if step.is_current), None)
        selected_action = _selected_autopilot_action(current_step)
        if not evaluation.eligibility.eligible:
            return self._record_result(
                request=request,
                evaluation=evaluation,
                current_step=current_step,
                selected_action=selected_action,
                advance_status="blocked",
                blockers=evaluation.eligibility.blockers,
                validation=PlanValidationResult(
                    status="rejected",
                    errors=[
                        ValidationIssue(
                            code="sample_autopilot_ineligible",
                            message="The recorded run is not eligible for sample autopilot one-step advance.",
                            step_id=current_step.step_id if current_step else None,
                            capability_id=current_step.capability_id if current_step else None,
                        )
                    ],
                ),
            )

        if current_step is None:
            return self._record_result(
                request=request,
                evaluation=evaluation,
                current_step=None,
                selected_action=None,
                advance_status="no_current_step",
                warnings=["No current orchestration step is available for one-step autopilot advance."],
            )
        if selected_action == "confirm_step":
            return self._record_result(
                request=request,
                evaluation=evaluation,
                current_step=current_step,
                selected_action=selected_action,
                advance_status="manual_confirmation_required",
                blockers=["Autopilot cannot create confirmation records; manual confirmation is required."],
            )
        if selected_action == "retry_failed_step":
            return self._record_result(
                request=request,
                evaluation=evaluation,
                current_step=current_step,
                selected_action=selected_action,
                advance_status="manual_retry_required",
                blockers=["Autopilot cannot retry failed steps; manual retry is required."],
            )
        if selected_action not in {"run_preflight", "preview_action_request", "execute_step"}:
            return self._record_result(
                request=request,
                evaluation=evaluation,
                current_step=current_step,
                selected_action=selected_action,
                advance_status="unsupported_action",
                blockers=["The current orchestration action is not supported by one-step sample autopilot."],
            )

        delegated_result: Any | None = None
        try:
            if selected_action == "run_preflight":
                delegated_result = self._preflight.create_preflight(
                    PreflightRequest(run_id=request.run_id, step_id=current_step.step_id)
                )
            elif selected_action == "preview_action_request":
                delegated_result = self._action_request.create_action_request(
                    ActionRequestPreviewRequest(run_id=request.run_id, step_id=current_step.step_id)
                )
            elif selected_action == "execute_step":
                delegated_result = self._execution.execute_step(
                    ExecutionRequest(run_id=request.run_id, step_id=current_step.step_id)
                )
        except AppClientError as exc:
            return self._record_result(
                request=request,
                evaluation=evaluation,
                current_step=current_step,
                selected_action=selected_action,
                advance_status="delegated_app_unavailable",
                blockers=[f"app_status_{exc.status_code}"],
                validation=PlanValidationResult(
                    status="rejected",
                    errors=[
                        ValidationIssue(
                            code="sample_autopilot_app_unavailable",
                            message="The owning app could not complete the delegated autopilot step.",
                            step_id=current_step.step_id,
                            capability_id=current_step.capability_id,
                        )
                    ],
                ),
            )
        except RuntimeValidationError as exc:
            return self._record_result(
                request=request,
                evaluation=evaluation,
                current_step=current_step,
                selected_action=selected_action,
                advance_status="delegated_rejected",
                blockers=[issue.code for issue in exc.validation.errors],
                validation=exc.validation,
            )

        return self._record_result(
            request=request,
            evaluation=evaluation,
            current_step=current_step,
            selected_action=selected_action,
            advance_status="advanced",
            delegated_result=delegated_result.model_dump(mode="json") if delegated_result else None,
        )

    def _record_result(
        self,
        *,
        request: SampleAutopilotStepRequest,
        evaluation: SampleAutopilotEvaluation,
        current_step: Any | None,
        selected_action: str | None,
        advance_status: str,
        delegated_result: dict[str, Any] | None = None,
        blockers: list[str] | None = None,
        warnings: list[str] | None = None,
        validation: PlanValidationResult | None = None,
    ) -> SampleAutopilotStepResult:
        blockers = blockers or []
        warnings = warnings or []
        event = {
            "recovery_event_id": f"sample_autopilot_step_{uuid4().hex[:12]}",
            "event_type": "sample_autopilot_step",
            "status": advance_status,
            "advance_status": advance_status,
            "autopilot_intent": request.autopilot_intent,
            "run_id": request.run_id,
            "plan_id": evaluation.orchestration.plan_id,
            "step_id": current_step.step_id if current_step else None,
            "capability_id": current_step.capability_id if current_step else None,
            "selected_action": selected_action,
            "created_by": "local_user",
            "created_at_utc": _utc_now_label(),
            "sample_ownership_summary": _sample_ownership_summary(evaluation.eligibility),
            "delegated_result_reference": _delegated_result_reference(selected_action, delegated_result),
            "blockers": blockers,
            "warnings": warnings,
            "single_step_only": True,
            "execution_permitted": False,
        }
        _reject_unsafe_autopilot_event(event)
        try:
            recorded_entry = self._ledger.append_recovery_event(request.run_id, event)
        except ValueError as exc:
            raise _rejected(
                "unsafe_autopilot_step_record",
                "The sample autopilot step event could not be safely ledgered.",
                step_id=current_step.step_id if current_step else None,
                capability_id=current_step.capability_id if current_step else None,
            ) from exc

        recorded_orchestration = orchestration_for_entry(recorded_entry)
        return SampleAutopilotStepResult(
            run_id=request.run_id,
            step_id=current_step.step_id if current_step else None,
            capability_id=current_step.capability_id if current_step else None,
            selected_action=selected_action,
            advance_status=advance_status,
            autopilot_event=event,
            delegated_result=delegated_result,
            sample_eligibility=evaluation.eligibility,
            run_progress_summary=recorded_orchestration.run_progress_summary,
            orchestration=recorded_orchestration,
            validation=validation or PlanValidationResult(status="valid"),
            ledger_recorded=True,
        )


def _preview_from_orchestration(
    orchestration: Any,
    eligibility: SampleAutopilotEligibility,
) -> SampleAutopilotPreview:
    steps = [
        SampleAutopilotPreviewStep(
            step_id=step.step_id,
            capability_id=step.capability_id,
            app_id=step.app_id,
            title=step.title,
            status=step.status,
            dry_run_action=_dry_run_action(step.status),
            allowed_manual_actions=step.allowed_actions,
            blocker_reason=step.blocker_reason,
            preflight_required=step.preflight_required,
            confirmation_required=step.confirmation_required,
            execution_supported=step.execution_supported,
        )
        for step in orchestration.steps
    ]
    preview_blockers = [
        step.blocker_reason
        for step in steps
        if step.blocker_reason and step.status in {"not_ready", "preflight_blocked", "failed_recoverable", "failed_terminal"}
    ]
    return SampleAutopilotPreview(
        sample_workspace_id=eligibility.sample_workspace_id,
        current_step_id=orchestration.current_step_id,
        step_count=len(steps),
        blocked_step_count=len(
            [
                step
                for step in steps
                if step.status in {"not_ready", "needs_preflight", "preflight_blocked", "needs_confirmation"}
            ]
        ),
        next_manual_actions=orchestration.allowed_next_actions if eligibility.eligible else [],
        steps=steps,
        blockers=preview_blockers,
        warnings=["Autopilot is dry-run only in this slice; every action remains manually gated."],
    )


def _dry_run_action(status: str) -> str:
    return {
        "not_ready": "wait_for_prior_gate",
        "needs_preflight": "request_manual_preflight",
        "preflight_blocked": "surface_preflight_blockers",
        "needs_confirmation": "request_manual_confirmation",
        "ready_for_action_request": "build_manual_action_request_preview",
        "ready_for_execution": "request_manual_guarded_execution",
        "completed": "skip_completed_step",
        "completed_with_warnings": "skip_completed_step_with_warnings",
        "failed_recoverable": "offer_manual_retry_or_revision",
        "failed_terminal": "stop_for_terminal_failure",
        "cancelled": "stop_for_cancelled_run",
        "informational": "skip_informational_step",
        "unsupported": "skip_unsupported_step",
    }.get(status, "inspect_step_state")


def _selected_autopilot_action(step: Any | None) -> str | None:
    if step is None:
        return None
    allowed = step.allowed_actions
    if "run_preflight" in allowed:
        return "run_preflight"
    if step.status == "ready_for_action_request" and "preview_action_request" in allowed:
        return "preview_action_request"
    if step.status == "ready_for_execution" and "execute_step" in allowed:
        return "execute_step"
    if "confirm_step" in allowed:
        return "confirm_step"
    if "retry_failed_step" in allowed:
        return "retry_failed_step"
    if allowed:
        return str(allowed[0])
    return None


def _sample_ownership_summary(eligibility: SampleAutopilotEligibility) -> dict[str, Any]:
    return {
        "sample_workspace_id": eligibility.sample_workspace_id,
        "sample_label": eligibility.sample_label,
        "lifecycle_id": eligibility.lifecycle_id,
        "sample_owned": eligibility.sample_owned,
        "allowlisted": eligibility.allowlisted,
        "reset_boundary_available": eligibility.reset_boundary_available,
    }


def _delegated_result_reference(
    selected_action: str | None,
    delegated_result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not delegated_result:
        return None
    if selected_action == "run_preflight":
        preflight = delegated_result.get("preflight")
        if not isinstance(preflight, dict):
            return {"result_type": "preflight"}
        return {
            "result_type": "preflight",
            "preflight_id": preflight.get("preflight_id"),
            "status": preflight.get("status"),
            "capability_id": preflight.get("capability_id"),
            "app_id": preflight.get("app_id"),
        }
    if selected_action == "preview_action_request":
        action_request = delegated_result.get("action_request")
        if not isinstance(action_request, dict):
            return {"result_type": "action_request"}
        return {
            "result_type": "action_request",
            "step_id": action_request.get("step_id"),
            "capability_id": action_request.get("capability_id"),
            "app_id": action_request.get("app_id"),
            "idempotency_key": action_request.get("idempotency_key"),
            "execution_permitted": action_request.get("execution_permitted"),
        }
    if selected_action == "execute_step":
        action_result = delegated_result.get("action_result")
        if not isinstance(action_result, dict):
            return {"result_type": "execution"}
        return {
            "result_type": "execution",
            "action_run_id": action_result.get("action_run_id"),
            "step_id": action_result.get("step_id"),
            "capability_id": action_result.get("capability_id"),
            "app_id": action_result.get("app_id"),
            "execution_status": action_result.get("execution_status"),
            "retry_allowed": action_result.get("retry_allowed"),
        }
    return {"result_type": selected_action}


def _capability_snapshot_blockers(entry: LedgerEntry) -> list[str]:
    snapshot = entry.plan_snapshot if isinstance(entry.plan_snapshot, dict) else {}
    raw_steps = snapshot.get("proposed_steps")
    steps = raw_steps if isinstance(raw_steps, list) else []
    blockers: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        capability_id = str(step.get("capability_id") or "")
        app_id = str(step.get("app_id") or "")
        risk_tier = str(step.get("risk_tier") or "")
        capability = _capability_snapshot(entry, capability_id, app_id)
        if capability is None:
            blockers.append(f"Capability snapshot is missing for {capability_id}.")
            continue
        if capability.get("enabled", True) is not True:
            blockers.append(f"Capability snapshot is disabled for {capability_id}.")
        if str(capability.get("risk_tier") or "") != risk_tier:
            blockers.append(f"Capability snapshot risk tier is stale for {capability_id}.")
    return blockers


def _sample_demo_plan_blockers(entry: LedgerEntry) -> list[str]:
    snapshot = entry.plan_snapshot if isinstance(entry.plan_snapshot, dict) else {}
    raw_steps = snapshot.get("proposed_steps")
    steps = raw_steps if isinstance(raw_steps, list) else []
    capability_ids = [
        str(step.get("capability_id") or "")
        for step in steps
        if isinstance(step, dict)
    ]
    blockers: list[str] = []
    if capability_ids != PHASE7_SAMPLE_DEMO_CAPABILITY_SEQUENCE:
        blockers.append(
            "The active plan does not match the Phase 7 sample demo capability sequence."
        )
        return blockers

    for step in steps:
        if not isinstance(step, dict):
            blockers.append("The active sample demo plan includes a malformed step.")
            continue
        capability_id = str(step.get("capability_id") or "")
        requirements = PHASE7_SAMPLE_DEMO_GATE_REQUIREMENTS.get(capability_id, {})
        for field_name, expected_value in requirements.items():
            if step.get(field_name) is not expected_value:
                blockers.append(
                    f"The active sample demo plan gate metadata is stale for {capability_id}."
                )
                break
    return blockers


def _capability_snapshot(
    entry: LedgerEntry,
    capability_id: str,
    app_id: str,
) -> dict[str, Any] | None:
    for item in entry.capability_snapshot:
        if not isinstance(item, dict):
            continue
        if item.get("capability_id") == capability_id and item.get("app_id") == app_id:
            return item
    return None


def _sample_marker(context: dict[str, Any]) -> dict[str, Any]:
    lifecycle = context.get("lifecycle_summary")
    if isinstance(lifecycle, dict):
        nested = lifecycle.get("sample_workspace")
        if isinstance(nested, dict):
            return nested
    top_level = context.get("sample_workspace")
    if isinstance(top_level, dict):
        return top_level
    return {}


def _sample_workspace_id(marker: dict[str, Any]) -> str | None:
    sample_id = _safe_label(marker.get("sample_workspace_id"))
    return sample_id or None


def _original_context(entry: LedgerEntry) -> dict[str, Any]:
    preview = entry.context_preview
    if preview is None or not isinstance(preview.context, dict):
        return {}
    return preview.context


def _nested_value(context: dict[str, Any], keys: list[str]) -> Any:
    current: Any = context
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _safe_label(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()[:120]


def _safe_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    labels: list[str] = []
    for item in value:
        label = _safe_label(item)
        if label:
            labels.append(label)
    return labels[:12]


def _reject_unsafe_autopilot_event(record: dict[str, Any]) -> None:
    unsafe_issues = find_unsafe_payload_issues(record, root="sample_autopilot")
    if unsafe_issues:
        raise RuntimeValidationError(
            PlanValidationResult(
                status="rejected",
                errors=[
                    issue.model_copy(update={"code": "unsafe_autopilot_preview_record"})
                    for issue in unsafe_issues
                ],
            )
        )


def _default_sample_workspace_root() -> Path:
    return Path.cwd().parent / "quant_suite" / "fixtures" / "sample_workspaces"


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
