from __future__ import annotations

from typing import Any

from quant_agent_runtime.models import LedgerEntry, RunState


def run_state_for_entry(entry: LedgerEntry) -> RunState:
    if entry.final_status == "cancelled" or entry.cancellation_events:
        return "cancelled"

    action_state = _latest_action_result_state(entry)
    if action_state is not None:
        return action_state

    snapshot = entry.plan_snapshot if isinstance(entry.plan_snapshot, dict) else {}
    missing_inputs = snapshot.get("missing_inputs")
    if snapshot.get("status") == "blocked" or (
        isinstance(missing_inputs, list) and len(missing_inputs) > 0
    ):
        return "waiting_for_input"

    if _latest_preflight_is_blocked(entry):
        return "preflight_blocked"

    required_confirmations = _required_confirmations(snapshot)
    if required_confirmations:
        confirmed = _confirmed_step_capabilities(entry)
        if all(item in confirmed for item in required_confirmations):
            return "ready_for_execution_preview"
        return "waiting_for_confirmation"

    return "planned"


def _latest_action_result_state(entry: LedgerEntry) -> RunState | None:
    for record in reversed(entry.action_results):
        if not isinstance(record, dict):
            continue
        status = record.get("execution_status")
        if status == "succeeded":
            return "completed"
        if status == "succeeded_with_warnings":
            return "completed_with_warnings"
        if status == "failed_recoverable":
            return "failed_recoverable"
        if status == "failed_terminal":
            return "failed_terminal"
        if status == "cancelled":
            return "cancelled"
    return None


def _required_confirmations(snapshot: dict[str, Any]) -> set[tuple[str, str]]:
    required = snapshot.get("required_confirmations")
    if not isinstance(required, list):
        return set()
    items: set[tuple[str, str]] = set()
    for item in required:
        if not isinstance(item, dict):
            continue
        step_id = item.get("step_id")
        capability_id = item.get("capability_id")
        if isinstance(step_id, str) and isinstance(capability_id, str):
            items.add((step_id, capability_id))
    return items


def _confirmed_step_capabilities(entry: LedgerEntry) -> set[tuple[str, str]]:
    confirmed: set[tuple[str, str]] = set()
    for item in entry.confirmation_records:
        if not isinstance(item, dict) or item.get("status") != "confirmed":
            continue
        step_id = item.get("step_id")
        capability_id = item.get("capability_id")
        if isinstance(step_id, str) and isinstance(capability_id, str):
            confirmed.add((step_id, capability_id))
    return confirmed


def _latest_preflight_is_blocked(entry: LedgerEntry) -> bool:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for record in entry.preflight_records:
        if not isinstance(record, dict):
            continue
        step_id = str(record.get("step_id") or "")
        capability_id = str(record.get("capability_id") or "")
        key = (step_id, capability_id)
        latest[key] = record

    for record in latest.values():
        blockers = record.get("blockers")
        if record.get("status") == "blocked":
            return True
        if isinstance(blockers, list) and blockers:
            return True
    return False
