from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.models import (
    LedgerEntry,
    PlanValidationResult,
    SampleAutopilotEligibility,
    SampleAutopilotPreview,
    SampleAutopilotPreviewRequest,
    SampleAutopilotPreviewResult,
    SampleAutopilotPreviewStep,
    ValidationIssue,
)
from quant_agent_runtime.orchestration import orchestration_for_entry
from quant_agent_runtime.redaction import find_unsafe_payload_issues, sanitize_value
from quant_agent_runtime.run_state import run_state_for_entry
from quant_agent_runtime.validation.errors import RuntimeValidationError


DEFAULT_SAMPLE_AUTOPILOT_ALLOWLIST = {"credit_pd_scorecard_panel"}
_TERMINAL_OR_CANCELLED_RUN_STATES = {
    "cancelled",
    "completed",
    "completed_with_warnings",
    "failed_terminal",
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
        entry = self._ledger.get(request.run_id)
        if entry is None:
            raise _rejected("unknown_run", "No recorded run was found for the requested run_id.")

        sanitized_context, redaction = sanitize_value(
            request.current_context_summary,
            path="current_context_summary",
        )
        if not isinstance(sanitized_context, dict):
            sanitized_context = {}

        orchestration = orchestration_for_entry(entry)
        metadata_by_id = self._load_sample_workspace_metadata()
        unsafe_context = redaction.redacted
        eligibility = self._eligibility(
            entry=entry,
            current_context=sanitized_context,
            metadata_by_id=metadata_by_id,
            unsafe_context=unsafe_context,
        )
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

    def _eligibility(
        self,
        *,
        entry: LedgerEntry,
        current_context: dict[str, Any],
        metadata_by_id: dict[str, SampleWorkspaceMetadata],
        unsafe_context: bool,
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

        run_state = run_state_for_entry(entry)
        if run_state == "paused":
            blockers.append("Paused runs must be resumed before sample autopilot preview.")
        if run_state in _TERMINAL_OR_CANCELLED_RUN_STATES:
            blockers.append("Terminal or cancelled runs cannot use sample autopilot preview.")

        capability_blockers = _capability_snapshot_blockers(entry)
        blockers.extend(capability_blockers)

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


def _rejected(code: str, message: str) -> RuntimeValidationError:
    return RuntimeValidationError(
        PlanValidationResult(
            status="rejected",
            errors=[ValidationIssue(code=code, message=message)],
        )
    )
