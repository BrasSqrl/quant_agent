from __future__ import annotations

import copy
from datetime import UTC, datetime
from typing import Any

from quant_agent_runtime.app_clients import AgentAppClient, AppClientError
from quant_agent_runtime.capability_discovery import (
    CapabilityDiscoveryService,
    SUPPORTED_EXECUTION_CAPABILITIES,
)
from quant_agent_runtime.contracts import QuantSuiteContractLoader
from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.models import (
    ExecutionRequest,
    ExecutionResult,
    LedgerEntry,
    PlanValidationResult,
    ValidationIssue,
)
from quant_agent_runtime.redaction import find_unsafe_payload_issues
from quant_agent_runtime.run_state import run_state_for_entry
from quant_agent_runtime.validation.errors import RuntimeValidationError

ACTION_REQUEST_CONTRACT_SCHEMA = "agent_action_request.v1.schema.json"
ACTION_RESULT_CONTRACT_SCHEMA = "agent_action_result.v1.schema.json"


class ExecutionService:
    def __init__(
        self,
        *,
        ledger: InMemoryLedger,
        contract_loader: QuantSuiteContractLoader,
        app_client: AgentAppClient,
        capability_discovery: CapabilityDiscoveryService,
    ) -> None:
        self._ledger = ledger
        self._contract_loader = contract_loader
        self._app_client = app_client
        self._capability_discovery = capability_discovery

    def execute_step(self, request: ExecutionRequest) -> ExecutionResult:
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

        capability_id = str(step.get("capability_id") or "")
        app_id = str(step.get("app_id") or "")
        existing = _existing_successful_result(entry, request.step_id, capability_id, app_id)
        if existing is not None:
            action_request = _latest_execution_action_request(
                entry,
                request.step_id,
                capability_id,
                app_id,
            ) or _latest_action_request_preview(entry, request.step_id, capability_id, app_id)
            if action_request is None:
                raise _rejected(
                    "missing_action_request_for_existing_result",
                    "The ledger has an action result but no matching safe action request.",
                    step_id=request.step_id,
                    capability_id=capability_id or None,
                )
            self._validate_action_request(action_request, request.step_id, capability_id)
            self._validate_action_result(existing, step)
            return ExecutionResult(
                run_id=request.run_id,
                step_id=request.step_id,
                capability_id=capability_id,
                action_request=action_request,
                action_result=existing,
                run_state=run_state_for_entry(entry),
                validation=PlanValidationResult(status="valid"),
                ledger_recorded=True,
            )

        self._validate_execution_gate(entry, step, request.step_id, capability_id, app_id)
        preview_request = _latest_action_request_preview(entry, request.step_id, capability_id, app_id)
        if preview_request is None:
            raise _rejected(
                "missing_action_request_preview",
                "A ledgered action request preview is required before execution.",
                step_id=request.step_id,
                capability_id=capability_id or None,
            )
        self._validate_action_request(preview_request, request.step_id, capability_id)

        action_request = _execution_action_request(preview_request)
        self._validate_action_request(action_request, request.step_id, capability_id)

        try:
            action_result = self._app_client.execute_action(
                app_id=app_id,
                capability_id=capability_id,
                payload={"action_request": action_request},
            )
        except AppClientError:
            raise
        except Exception as exc:
            raise AppClientError("Execution app call failed.", status_code=502) from exc

        self._validate_action_result(action_result, step)
        try:
            recorded_entry = self._ledger.append_action_request(request.run_id, action_request)
            recorded_entry = self._ledger.append_action_result(request.run_id, action_result)
        except ValueError as exc:
            raise _rejected(
                "unsafe_execution_ledger_record",
                "The execution request or result could not be safely ledgered.",
                step_id=request.step_id,
                capability_id=capability_id or None,
            ) from exc

        return ExecutionResult(
            run_id=request.run_id,
            step_id=request.step_id,
            capability_id=capability_id,
            action_request=action_request,
            action_result=action_result,
            run_state=run_state_for_entry(recorded_entry),
            validation=PlanValidationResult(status="valid"),
            ledger_recorded=True,
        )

    def _validate_execution_gate(
        self,
        entry: LedgerEntry,
        step: dict[str, Any],
        step_id: str,
        capability_id: str,
        app_id: str,
    ) -> None:
        if capability_id not in SUPPORTED_EXECUTION_CAPABILITIES:
            raise _rejected(
                "unsupported_execution_capability",
                "Only the confirmed Studio model configuration draft action can execute in this slice.",
                step_id=step_id,
                capability_id=capability_id or None,
            )
        if app_id != "quant_studio":
            raise _rejected(
                "unsupported_execution_app",
                "The recorded plan step is owned by an app without configured execution routing.",
                step_id=step_id,
                capability_id=capability_id or None,
            )

        plan_state = run_state_for_entry(entry)
        if plan_state == "waiting_for_input":
            raise _rejected(
                "blocked_plan_execution",
                "The recorded plan is blocked by missing inputs and cannot execute.",
                step_id=step_id,
                capability_id=capability_id or None,
            )
        if plan_state == "preflight_blocked":
            raise _rejected(
                "preflight_blocked_execution",
                "The recorded run has a blocked preflight and cannot execute.",
                step_id=step_id,
                capability_id=capability_id or None,
            )
        if plan_state == "cancelled":
            raise _rejected(
                "cancelled_run_execution",
                "The recorded run is cancelled and cannot execute.",
                step_id=step_id,
                capability_id=capability_id or None,
            )

        capability = _capability_snapshot(entry, capability_id, app_id)
        if capability is None:
            raise _rejected(
                "stale_execution_capability_snapshot",
                "The recorded plan step capability was not found in the ledger capability snapshot.",
                step_id=step_id,
                capability_id=capability_id or None,
            )
        if str(capability.get("risk_tier") or "") != str(step.get("risk_tier") or ""):
            raise _rejected(
                "stale_execution_capability_snapshot",
                "The recorded plan step risk tier no longer matches the ledger capability snapshot.",
                step_id=step_id,
                capability_id=capability_id or None,
            )
        if capability.get("enabled", True) is not True:
            raise _rejected(
                "stale_execution_capability_snapshot",
                "The recorded plan step capability is disabled in the ledger capability snapshot.",
                step_id=step_id,
                capability_id=capability_id or None,
            )

        action_input = step.get("action_input") if isinstance(step.get("action_input"), dict) else {}
        action_input_issues = find_unsafe_payload_issues(action_input, root="action_input")
        if action_input_issues:
            raise RuntimeValidationError(
                PlanValidationResult(
                    status="rejected",
                    errors=[
                        issue.model_copy(
                            update={
                                "code": "unsafe_execution_action_input",
                                "step_id": step_id,
                                "capability_id": capability_id or None,
                            }
                        )
                        for issue in action_input_issues
                    ],
                )
            )

        if not _confirmed_record(entry, step_id, capability_id):
            raise _rejected(
                "missing_confirmation_for_execution",
                "The recorded plan step requires confirmation before execution.",
                step_id=step_id,
                capability_id=capability_id or None,
            )

        discovery_result = self._capability_discovery.discover()
        if not discovery_result.supports_execution(capability_id):
            if discovery_result.app_is_unavailable(app_id):
                raise _rejected(
                    "execution_app_unavailable",
                    "The owning app is not currently advertising agent capabilities.",
                    step_id=step_id,
                    capability_id=capability_id or None,
                )
            raise _rejected(
                "execution_capability_unavailable",
                "The recorded plan step capability is not currently reconciled as app-owned execution support.",
                step_id=step_id,
                capability_id=capability_id or None,
            )

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
                                "code": "unsafe_execution_action_request",
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
                "malformed_execution_action_request",
                f"The generated execution action request failed contract validation: {exc}",
                step_id=step_id,
                capability_id=capability_id or None,
            ) from exc

    def _validate_action_result(self, action_result: dict[str, Any], step: dict[str, Any]) -> None:
        step_id = str(step.get("step_id") or "") or None
        capability_id = str(step.get("capability_id") or "") or None
        unsafe_issues = find_unsafe_payload_issues(action_result, root="action_result")
        if unsafe_issues:
            raise RuntimeValidationError(
                PlanValidationResult(
                    status="rejected",
                    errors=[
                        issue.model_copy(
                            update={
                                "code": "unsafe_app_action_result",
                                "step_id": step_id,
                                "capability_id": capability_id,
                            }
                        )
                        for issue in unsafe_issues
                    ],
                )
            )
        if action_result.get("capability_id") != step.get("capability_id"):
            raise _rejected(
                "action_result_capability_mismatch",
                "The app action result capability_id does not match the recorded step.",
                step_id=step_id,
                capability_id=capability_id,
            )
        if action_result.get("app_id") != step.get("app_id"):
            raise _rejected(
                "action_result_app_mismatch",
                "The app action result app_id does not match the recorded step.",
                step_id=step_id,
                capability_id=capability_id,
            )
        try:
            self._contract_loader.validate_agent_contract_payload(
                action_result,
                ACTION_RESULT_CONTRACT_SCHEMA,
            )
        except Exception as exc:
            raise _rejected(
                "malformed_app_action_result",
                f"The app action result failed contract validation: {exc}",
                step_id=step_id,
                capability_id=capability_id,
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


def _latest_action_request_preview(
    entry: LedgerEntry,
    step_id: str,
    capability_id: str,
    app_id: str,
) -> dict[str, Any] | None:
    for record in reversed(entry.action_requests):
        if not isinstance(record, dict):
            continue
        if (
            record.get("agent_run_id") == entry.run_id
            and record.get("step_id") == step_id
            and record.get("capability_id") == capability_id
            and record.get("app_id") == app_id
            and record.get("execution_permitted") is False
        ):
            return record
    return None


def _latest_execution_action_request(
    entry: LedgerEntry,
    step_id: str,
    capability_id: str,
    app_id: str,
) -> dict[str, Any] | None:
    for record in reversed(entry.action_requests):
        if not isinstance(record, dict):
            continue
        if (
            record.get("agent_run_id") == entry.run_id
            and record.get("step_id") == step_id
            and record.get("capability_id") == capability_id
            and record.get("app_id") == app_id
            and record.get("execution_permitted") is True
        ):
            return record
    return None


def _existing_successful_result(
    entry: LedgerEntry,
    step_id: str,
    capability_id: str,
    app_id: str,
) -> dict[str, Any] | None:
    for record in reversed(entry.action_results):
        if not isinstance(record, dict):
            continue
        if (
            record.get("step_id") == step_id
            and record.get("capability_id") == capability_id
            and record.get("app_id") == app_id
            and record.get("execution_status") in {"succeeded", "succeeded_with_warnings"}
        ):
            return record
    return None


def _execution_action_request(preview_request: dict[str, Any]) -> dict[str, Any]:
    action_request = copy.deepcopy(preview_request)
    preview_key = str(action_request.get("idempotency_key") or "")
    action_request["execution_permitted"] = True
    action_request["requested_at_utc"] = _utc_now_label()
    action_request["idempotency_key"] = f"exec_{preview_key}" if preview_key else None
    action_request["execution_request"] = True
    action_request["preview_idempotency_key"] = preview_key
    return action_request


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
