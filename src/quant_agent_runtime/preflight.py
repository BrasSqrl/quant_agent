from __future__ import annotations

from typing import Any

from quant_agent_runtime.app_clients import AgentAppClient, AppClientError
from quant_agent_runtime.contracts import QuantSuiteContractLoader
from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.models import (
    LedgerEntry,
    PlanValidationResult,
    PreflightRequest,
    PreflightResult,
    ValidationIssue,
)
from quant_agent_runtime.redaction import find_unsafe_payload_issues
from quant_agent_runtime.validation.errors import RuntimeValidationError

SOURCE_PREFLIGHT_CAPABILITY_ID = "quant_data.run_source_preflight"
PREFLIGHT_CONTRACT_SCHEMA = "agent_action_preflight.v1.schema.json"


class PreflightService:
    def __init__(
        self,
        *,
        ledger: InMemoryLedger,
        contract_loader: QuantSuiteContractLoader,
        app_client: AgentAppClient,
    ) -> None:
        self._ledger = ledger
        self._contract_loader = contract_loader
        self._app_client = app_client

    def create_preflight(self, request: PreflightRequest) -> PreflightResult:
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
        if capability_id != SOURCE_PREFLIGHT_CAPABILITY_ID or app_id != "quant_data":
            raise _rejected(
                "unsupported_preflight_step",
                "The recorded plan step does not support app-owned preflight in this runtime slice.",
                step_id=request.step_id,
                capability_id=capability_id or None,
            )

        app_request = {
            "run_id": request.run_id,
            "step_id": request.step_id,
            "capability_id": capability_id,
            "context_summary": _safe_context(entry),
            "action_input": step.get("action_input") if isinstance(step.get("action_input"), dict) else {},
        }
        try:
            preflight = self._app_client.create_preflight(
                app_id=app_id,
                capability_id=capability_id,
                payload=app_request,
            )
        except AppClientError:
            raise
        except Exception as exc:
            raise AppClientError("Preflight app call failed.", status_code=502) from exc

        self._validate_preflight_response(preflight, step)
        self._ledger.append_preflight_record(request.run_id, preflight)
        return PreflightResult(
            run_id=request.run_id,
            step_id=request.step_id,
            capability_id=capability_id,
            preflight=preflight,
            validation=PlanValidationResult(status="valid"),
            ledger_recorded=True,
        )

    def _validate_preflight_response(self, preflight: dict[str, Any], step: dict[str, Any]) -> None:
        unsafe_issues = find_unsafe_payload_issues(preflight, root="preflight")
        if unsafe_issues:
            raise RuntimeValidationError(
                PlanValidationResult(
                    status="rejected",
                    errors=[
                        issue.model_copy(
                            update={
                                "code": "unsafe_app_preflight_response",
                                "step_id": str(step.get("step_id") or "") or None,
                                "capability_id": str(step.get("capability_id") or "") or None,
                            }
                        )
                        for issue in unsafe_issues
                    ],
                )
            )

        if preflight.get("capability_id") != step.get("capability_id"):
            raise _rejected(
                "preflight_capability_mismatch",
                "The app preflight response capability_id does not match the recorded step.",
                step_id=str(step.get("step_id") or "") or None,
                capability_id=str(step.get("capability_id") or "") or None,
            )
        if preflight.get("app_id") != step.get("app_id"):
            raise _rejected(
                "preflight_app_mismatch",
                "The app preflight response app_id does not match the recorded step.",
                step_id=str(step.get("step_id") or "") or None,
                capability_id=str(step.get("capability_id") or "") or None,
            )

        try:
            self._contract_loader.validate_agent_contract_payload(preflight, PREFLIGHT_CONTRACT_SCHEMA)
        except Exception as exc:
            raise _rejected(
                "malformed_app_preflight_response",
                f"The app preflight response failed contract validation: {exc}",
                step_id=str(step.get("step_id") or "") or None,
                capability_id=str(step.get("capability_id") or "") or None,
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


def _safe_context(entry: LedgerEntry) -> dict[str, Any]:
    if entry.context_preview is None:
        return {}
    return entry.context_preview.context


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
