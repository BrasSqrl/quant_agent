from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.models import (
    DemoNarrativeResult,
    DemoNarrativeSection,
    LedgerEntry,
    PlanValidationResult,
    SampleAutopilotEligibility,
    ValidationIssue,
)
from quant_agent_runtime.orchestration import ledger_summary, orchestration_for_entry
from quant_agent_runtime.redaction import find_unsafe_payload_issues
from quant_agent_runtime.sample_autopilot import SampleAutopilotPreviewService
from quant_agent_runtime.validation.errors import RuntimeValidationError


class DemoNarrativeService:
    def __init__(
        self,
        *,
        ledger: InMemoryLedger,
        sample_workspace_root: Path | None = None,
        allowlist: set[str] | None = None,
    ) -> None:
        self._ledger = ledger
        self._eligibility = SampleAutopilotPreviewService(
            ledger=ledger,
            sample_workspace_root=sample_workspace_root,
            allowlist=allowlist,
        )

    def get_demo_narrative(self, run_id: str) -> DemoNarrativeResult:
        entry = self._ledger.get(run_id)
        if entry is None:
            raise _rejected("unknown_run", "No recorded run was found for the requested run_id.")
        _reject_unsafe_ledger(entry)
        _reject_malformed_ledger(entry)

        evaluation = self._eligibility.evaluate_eligibility(
            run_id=run_id,
            current_context_summary={},
            include_run_state_checks=False,
        )
        orchestration = orchestration_for_entry(entry)
        demo_status = _demo_status(entry, evaluation.eligibility, orchestration.run_state)
        sections = _narrative_sections(entry, evaluation.eligibility, demo_status)
        safety_summary = _safety_summary(entry, evaluation.eligibility)
        validation = _validation_for_status(demo_status, evaluation.eligibility)
        result = DemoNarrativeResult(
            run_id=run_id,
            demo_status=demo_status,
            sample_eligibility=evaluation.eligibility,
            narrative_sections=sections,
            safety_summary=safety_summary,
            run_progress_summary=orchestration.run_progress_summary,
            orchestration=orchestration,
            ledger_summary=ledger_summary(entry),
            validation=validation,
        )
        unsafe = find_unsafe_payload_issues(result.model_dump(mode="json"), root="demo_narrative")
        if unsafe:
            raise RuntimeValidationError(
                PlanValidationResult(
                    status="rejected",
                    errors=[
                        issue.model_copy(update={"code": "unsafe_demo_narrative_output"})
                        for issue in unsafe
                    ],
                )
            )
        return result


def _demo_status(
    entry: LedgerEntry,
    eligibility: SampleAutopilotEligibility,
    run_state: str,
) -> str:
    blocker_text = " ".join(eligibility.blockers).lower()
    if (
        not eligibility.sample_workspace_id
        or "not marked sample_owned" in blocker_text
        or "no sample_workspace_id" in blocker_text
    ):
        return "not_sample_demo"
    if entry.final_status == "sample_reset" or run_state == "sample_reset":
        return "sample_reset"
    if not eligibility.eligible:
        return "blocked"
    if entry.final_status in {"completed", "completed_with_warnings"} or run_state in {
        "completed",
        "completed_with_warnings",
    }:
        return "completed"
    return "in_progress"


def _narrative_sections(
    entry: LedgerEntry,
    eligibility: SampleAutopilotEligibility,
    demo_status: str,
) -> list[DemoNarrativeSection]:
    return [
        _plan_section(entry, eligibility),
        _preflight_section(
            entry,
            section_id="data_preflight",
            title="Data preflight",
            capability_id="quant_data.run_source_preflight",
        ),
        _execution_section(
            entry,
            section_id="studio_draft",
            title="Studio model configuration draft",
            capability_id="quant_studio.prepare_model_config_draft",
        ),
        _informational_section(
            entry,
            section_id="documentation_inspection",
            title="Documentation package inspection",
            capability_id="quant_documentation.inspect_package",
        ),
        _execution_section(
            entry,
            section_id="documentation_draft",
            title="Documentation draft workspace",
            capability_id="quant_documentation.create_draft_workspace",
        ),
        _preflight_section(
            entry,
            section_id="monitoring_preflight",
            title="Monitoring bundle validation",
            capability_id="quant_monitoring.validate_bundle",
        ),
        _reset_section(entry, demo_status),
        _safety_section(entry, eligibility),
        _remaining_gates_section(entry),
    ]


def _plan_section(entry: LedgerEntry, eligibility: SampleAutopilotEligibility) -> DemoNarrativeSection:
    plan = _plan(entry)
    steps = _steps(entry)
    return DemoNarrativeSection(
        section_id="plan",
        title="Sample demo plan",
        status="recorded" if eligibility.eligible or eligibility.sample_workspace_id else "not_sample_demo",
        summary=(
            "The ledger contains a sample-owned Phase 7 plan."
            if eligibility.sample_workspace_id
            else "The ledger does not contain sample-owned demo markers."
        ),
        evidence_references=[
            {
                "reference_type": "plan",
                "plan_id": _safe_label(plan.get("plan_id")),
                "step_count": len(steps),
            }
        ],
        blockers=eligibility.blockers,
        warnings=eligibility.warnings,
    )


def _preflight_section(
    entry: LedgerEntry,
    *,
    section_id: str,
    title: str,
    capability_id: str,
) -> DemoNarrativeSection:
    step = _step_for_capability(entry, capability_id)
    latest = _latest_record(entry.preflight_records, capability_id)
    status = _safe_label(latest.get("status")) if latest else "pending"
    if status in {"ready", "warning"}:
        summary = "App-owned preflight evidence is ledgered."
    elif status == "blocked":
        summary = "App-owned preflight is blocked."
    else:
        summary = "App-owned preflight has not been ledgered yet."
    return DemoNarrativeSection(
        section_id=section_id,
        title=title,
        status=status,
        summary=summary,
        step_id=_safe_label(step.get("step_id")) or None,
        capability_id=capability_id,
        app_id=_safe_label(step.get("app_id")) or _safe_label(latest.get("app_id")) or None,
        evidence_references=[_preflight_reference(latest)] if latest else [],
        blockers=_record_messages(latest, "blockers"),
        warnings=_record_messages(latest, "warnings"),
    )


def _execution_section(
    entry: LedgerEntry,
    *,
    section_id: str,
    title: str,
    capability_id: str,
) -> DemoNarrativeSection:
    step = _step_for_capability(entry, capability_id)
    confirmation = _latest_confirmation(entry, _safe_label(step.get("step_id")), capability_id)
    action_request = _latest_record(entry.action_requests, capability_id)
    action_result = _latest_record(entry.action_results, capability_id)
    execution_status = _safe_label(action_result.get("execution_status")) if action_result else ""
    if execution_status:
        status = execution_status
        summary = "Guarded app-owned draft execution result is ledgered."
    elif action_request:
        status = "ready_for_execution"
        summary = "Action request preview is ledgered; execution remains manually gated."
    elif confirmation:
        status = "confirmed"
        summary = "Manual confirmation is ledgered; action request preview is still needed."
    else:
        status = "pending"
        summary = "Required confirmation and action request preview are not fully ledgered yet."
    references: list[dict[str, Any]] = []
    if confirmation:
        references.append(_confirmation_reference(confirmation))
    if action_request:
        references.append(_action_request_reference(action_request))
    if action_result:
        references.append(_action_result_reference(action_result))
    return DemoNarrativeSection(
        section_id=section_id,
        title=title,
        status=status,
        summary=summary,
        step_id=_safe_label(step.get("step_id")) or None,
        capability_id=capability_id,
        app_id=_safe_label(step.get("app_id")) or _safe_label(action_result.get("app_id")) or None,
        evidence_references=references,
        blockers=_record_messages(action_result, "recoverable_errors") + _record_messages(action_result, "terminal_errors"),
        warnings=_record_messages(action_result, "warnings"),
    )


def _informational_section(
    entry: LedgerEntry,
    *,
    section_id: str,
    title: str,
    capability_id: str,
) -> DemoNarrativeSection:
    step = _step_for_capability(entry, capability_id)
    status = "observed" if step else "pending"
    return DemoNarrativeSection(
        section_id=section_id,
        title=title,
        status=status,
        summary=(
            "Read-only documentation package inspection is represented in the plan."
            if step
            else "The read-only documentation inspection step is not present in the plan."
        ),
        step_id=_safe_label(step.get("step_id")) or None,
        capability_id=capability_id,
        app_id=_safe_label(step.get("app_id")) or None,
    )


def _reset_section(entry: LedgerEntry, demo_status: str) -> DemoNarrativeSection:
    preview = _latest_recovery(entry, "sample_reset_preview")
    reset = _latest_recovery(entry, "sample_reset")
    status = _safe_label(reset.get("status")) if reset else ("previewed" if preview else "pending")
    references: list[dict[str, Any]] = []
    if preview:
        references.append(
            {
                "reference_type": "sample_reset_preview",
                "recovery_event_id": _safe_label(preview.get("recovery_event_id")),
                "status": _safe_label(preview.get("status")),
            }
        )
    if reset:
        summary = reset.get("reset_result_summary") if isinstance(reset.get("reset_result_summary"), dict) else {}
        references.append(
            {
                "reference_type": "sample_reset",
                "recovery_event_id": _safe_label(reset.get("recovery_event_id")),
                "status": _safe_label(reset.get("status")),
                "deleted_lifecycle_count": _safe_int(summary.get("deleted_lifecycle_count")),
            }
        )
    return DemoNarrativeSection(
        section_id="sample_reset",
        title="Sample-owned reset",
        status="sample_reset" if demo_status == "sample_reset" else status,
        summary=(
            "Sample-owned demo reset is ledgered and the run is terminal."
            if demo_status == "sample_reset"
            else "Sample-owned reset has not been performed for this run."
        ),
        evidence_references=references,
        blockers=_record_messages(reset, "blockers"),
        warnings=_record_messages(reset, "warnings"),
    )


def _safety_section(entry: LedgerEntry, eligibility: SampleAutopilotEligibility) -> DemoNarrativeSection:
    return DemoNarrativeSection(
        section_id="safety_boundaries",
        title="Safety boundaries",
        status="enforced",
        summary="The demo narrative is derived from summaries-and-references-only ledger records.",
        evidence_references=[
            {
                "reference_type": "safety_summary",
                "data_policy": entry.data_policy,
                "sample_owned": eligibility.sample_owned,
                "allowlisted": eligibility.allowlisted,
                "reset_boundary_available": eligibility.reset_boundary_available,
            }
        ],
    )


def _remaining_gates_section(entry: LedgerEntry) -> DemoNarrativeSection:
    orchestration = orchestration_for_entry(entry)
    return DemoNarrativeSection(
        section_id="remaining_manual_gates",
        title="Remaining manual gates",
        status="none" if not orchestration.allowed_next_actions else "waiting",
        summary=(
            "No further manual gates are currently available."
            if not orchestration.allowed_next_actions
            else "The run is paused at the next manually governed action."
        ),
        evidence_references=[
            {
                "reference_type": "orchestration",
                "current_step_id": orchestration.current_step_id,
                "allowed_next_actions": orchestration.allowed_next_actions,
            }
        ],
    )


def _safety_summary(entry: LedgerEntry, eligibility: SampleAutopilotEligibility) -> dict[str, Any]:
    return {
        "data_policy": entry.data_policy,
        "sample_workspace_id": eligibility.sample_workspace_id,
        "sample_owned": eligibility.sample_owned,
        "allowlisted": eligibility.allowlisted,
        "reset_boundary_available": eligibility.reset_boundary_available,
        "sample_owned_only": eligibility.reset_boundary_available,
        "user_owned_state_protected": True,
        "row_level_data_included": False,
        "raw_prompts_included": False,
        "raw_provider_responses_included": False,
        "browser_direct_app_calls_permitted": False,
        "provider_execution_supported": False,
        "autonomous_loop_supported": False,
        "auto_confirmation_supported": False,
        "ledger_record_counts": {
            "preflight_records": len(entry.preflight_records),
            "confirmation_records": len(entry.confirmation_records),
            "action_requests": len(entry.action_requests),
            "action_results": len(entry.action_results),
            "recovery_events": len(entry.recovery_events),
            "cancellation_events": len(entry.cancellation_events),
        },
    }


def _validation_for_status(
    demo_status: str,
    eligibility: SampleAutopilotEligibility,
) -> PlanValidationResult:
    if demo_status in {"not_sample_demo", "blocked"}:
        return PlanValidationResult(
            status="rejected",
            errors=[
                ValidationIssue(
                    code="sample_demo_narrative_not_available"
                    if demo_status == "not_sample_demo"
                    else "sample_demo_narrative_blocked",
                    message="The recorded run is not currently eligible for the sample demo narrative.",
                )
            ],
            warnings=[
                ValidationIssue(code="sample_demo_blocker", message=blocker)
                for blocker in eligibility.blockers[:12]
            ],
        )
    return PlanValidationResult(status="valid")


def _reject_unsafe_ledger(entry: LedgerEntry) -> None:
    issues = find_unsafe_payload_issues(entry.model_dump(mode="json"), root="demo_ledger")
    if issues:
        raise RuntimeValidationError(
            PlanValidationResult(
                status="rejected",
                errors=[
                    issue.model_copy(update={"code": "unsafe_demo_ledger"})
                    for issue in issues
                ],
            )
        )


def _reject_malformed_ledger(entry: LedgerEntry) -> None:
    plan = entry.plan_snapshot
    if not isinstance(plan, dict) or not isinstance(plan.get("proposed_steps"), list):
        raise _rejected(
            "malformed_demo_ledger",
            "The recorded run does not include a valid plan snapshot for demo narrative.",
        )


def _plan(entry: LedgerEntry) -> dict[str, Any]:
    return entry.plan_snapshot if isinstance(entry.plan_snapshot, dict) else {}


def _steps(entry: LedgerEntry) -> list[dict[str, Any]]:
    steps = _plan(entry).get("proposed_steps")
    return [step for step in steps if isinstance(step, dict)] if isinstance(steps, list) else []


def _step_for_capability(entry: LedgerEntry, capability_id: str) -> dict[str, Any]:
    return next(
        (step for step in _steps(entry) if step.get("capability_id") == capability_id),
        {},
    )


def _latest_record(records: list[dict[str, Any]], capability_id: str) -> dict[str, Any]:
    for record in reversed(records):
        if isinstance(record, dict) and record.get("capability_id") == capability_id:
            return record
    return {}


def _latest_confirmation(entry: LedgerEntry, step_id: str, capability_id: str) -> dict[str, Any]:
    for record in reversed(entry.confirmation_records):
        if not isinstance(record, dict):
            continue
        if record.get("step_id") == step_id and record.get("capability_id") == capability_id:
            return record
    return {}


def _latest_recovery(entry: LedgerEntry, event_type: str) -> dict[str, Any]:
    for record in reversed(entry.recovery_events):
        if isinstance(record, dict) and record.get("event_type") == event_type:
            return record
    return {}


def _preflight_reference(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "reference_type": "preflight",
        "preflight_id": _safe_label(record.get("preflight_id")),
        "status": _safe_label(record.get("status")),
        "capability_id": _safe_label(record.get("capability_id")),
        "app_id": _safe_label(record.get("app_id")),
    }


def _confirmation_reference(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "reference_type": "confirmation",
        "confirmation_id": _safe_label(record.get("confirmation_id")),
        "status": _safe_label(record.get("status")),
        "step_id": _safe_label(record.get("step_id")),
        "capability_id": _safe_label(record.get("capability_id")),
    }


def _action_request_reference(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "reference_type": "action_request",
        "step_id": _safe_label(record.get("step_id")),
        "capability_id": _safe_label(record.get("capability_id")),
        "app_id": _safe_label(record.get("app_id")),
        "execution_permitted": record.get("execution_permitted") is True,
    }


def _action_result_reference(record: dict[str, Any]) -> dict[str, Any]:
    output_references = record.get("output_references") if isinstance(record.get("output_references"), list) else []
    return {
        "reference_type": "action_result",
        "action_run_id": _safe_label(record.get("action_run_id")),
        "step_id": _safe_label(record.get("step_id")),
        "capability_id": _safe_label(record.get("capability_id")),
        "app_id": _safe_label(record.get("app_id")),
        "execution_status": _safe_label(record.get("execution_status")),
        "output_reference_types": [
            _safe_label(item.get("reference_type"))
            for item in output_references
            if isinstance(item, dict) and _safe_label(item.get("reference_type"))
        ][:12],
    }


def _record_messages(record: dict[str, Any], key: str) -> list[str]:
    value = record.get(key)
    if not isinstance(value, list):
        return []
    messages: list[str] = []
    for item in value:
        if isinstance(item, str):
            label = _safe_label(item)
        elif isinstance(item, dict):
            label = _safe_label(item.get("message") or item.get("summary") or item.get("code") or item.get("status"))
        else:
            label = ""
        if label:
            messages.append(label)
    return messages[:12]


def _safe_label(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()[:160]


def _safe_int(value: Any) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def _rejected(code: str, message: str) -> RuntimeValidationError:
    return RuntimeValidationError(
        PlanValidationResult(
            status="rejected",
            errors=[
                ValidationIssue(
                    code=code,
                    message=message,
                )
            ],
        )
    )
