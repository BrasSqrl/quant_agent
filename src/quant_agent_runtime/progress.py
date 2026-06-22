from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import ValidationError

from quant_agent_runtime.models import (
    LedgerEntry,
    OrchestrationStepSummary,
    RunProgressSummary,
    RunState,
    StaleAssumptionSummary,
)


_BLOCKED_STEP_STATUSES = {"not_ready", "needs_preflight", "preflight_blocked"}


def build_run_progress_summary(
    entry: LedgerEntry,
    *,
    run_state: RunState,
    final_status: str,
    plan_id: str | None,
    current_step: OrchestrationStepSummary | None,
    steps: list[OrchestrationStepSummary],
    allowed_next_actions: list[str],
) -> RunProgressSummary:
    counts = Counter(step.status for step in steps)
    current_blocker = None
    if current_step is not None:
        current_blocker = current_step.blocker_reason or (
            f"Waiting for {current_step.required_gate}."
            if current_step.required_gate
            else None
        )
    return RunProgressSummary(
        run_id=entry.run_id,
        parent_run_id=entry.parent_run_id,
        parent_plan_id=entry.parent_plan_id,
        activated_revision_id=entry.activated_revision_id,
        child_run_ids=entry.child_run_ids,
        plan_id=plan_id,
        run_state=run_state,
        final_status=final_status,
        total_steps=len(steps),
        completed_steps=counts["completed"],
        completed_with_warnings_steps=counts["completed_with_warnings"],
        informational_steps=counts["informational"],
        unsupported_steps=counts["unsupported"],
        blocked_steps=sum(counts[status] for status in _BLOCKED_STEP_STATUSES),
        failed_recoverable_steps=counts["failed_recoverable"],
        failed_terminal_steps=counts["failed_terminal"],
        not_ready_steps=counts["not_ready"],
        current_step_id=current_step.step_id if current_step else None,
        current_step_title=current_step.title if current_step else None,
        current_step_status=current_step.status if current_step else None,
        current_blocker=current_blocker,
        latest_record_counts={
            "preflight_records": len(entry.preflight_records),
            "confirmation_records": len(entry.confirmation_records),
            "action_requests": len(entry.action_requests),
            "action_results": len(entry.action_results),
            "recovery_events": len(entry.recovery_events),
            "cancellation_events": len(entry.cancellation_events),
        },
        allowed_next_actions=allowed_next_actions,
    )


def latest_stale_assumption_summary(entry: LedgerEntry) -> StaleAssumptionSummary:
    for record in reversed(entry.recovery_events):
        if not isinstance(record, dict) or record.get("event_type") != "run_revalidation":
            continue
        raw_summary = record.get("stale_assumption_summary")
        if not isinstance(raw_summary, dict):
            return StaleAssumptionSummary(
                status="insufficient_context",
                warnings=["Latest run revalidation record did not include a valid stale-assumption summary."],
                revalidation_required=True,
            )
        try:
            return StaleAssumptionSummary.model_validate(raw_summary)
        except ValidationError:
            return StaleAssumptionSummary(
                status="insufficient_context",
                warnings=["Latest run revalidation summary was malformed."],
                revalidation_required=True,
            )
    return StaleAssumptionSummary()


def safe_section_labels(context: dict[str, Any]) -> list[str]:
    return sorted(str(key) for key in context if str(key).strip())
