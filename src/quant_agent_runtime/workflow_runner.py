from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from quant_agent_runtime.action_request import ActionRequestPreviewService
from quant_agent_runtime.capabilities import default_capabilities
from quant_agent_runtime.confirmation import ConfirmationService
from quant_agent_runtime.contracts import QuantSuiteContractLoader
from quant_agent_runtime.execution import ExecutionService
from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.models import (
    ActionRequestPreviewRequest,
    ConfirmationRequest,
    ExecutionRequest,
    PlanRequest,
    PlanValidationResult,
    PreflightRequest,
    RetryRequest,
    ValidationIssue,
    WorkflowAdvanceRequest,
    WorkflowAdvanceResult,
    WorkflowAdvanceUntilBlockedRequest,
    WorkflowAdvanceUntilBlockedResult,
    WorkflowRunRequest,
    WorkflowRunResult,
    WorkflowRunScopeSummary,
    WorkflowRunStatusResult,
)
from quant_agent_runtime.orchestration import OrchestrationService
from quant_agent_runtime.planner import PlannerService
from quant_agent_runtime.preflight import PreflightService
from quant_agent_runtime.retry import RetryService
from quant_agent_runtime.run_status import RunStatusService
from quant_agent_runtime.summary_text import compact_safe_summary_text, has_meaningful_summary
from quant_agent_runtime.validation.errors import RuntimeValidationError


WORKFLOW_RUN_SUPPORT_LEVEL = "scoped_template_guided_manual_and_until_blocked"
WORKFLOW_TEMPLATE_SUPPORT_LEVEL = "canonical_suite_workflow_templates_v1"
WORKFLOW_CREATED_EVENT_TYPE = "workflow_run_created"
WORKFLOW_ADVANCE_EVENT_TYPE = "workflow_run_advance"


class WorkflowRunService:
    def __init__(
        self,
        *,
        planner: PlannerService,
        ledger: InMemoryLedger,
        contract_loader: QuantSuiteContractLoader,
        run_status: RunStatusService,
        orchestration: OrchestrationService,
        preflight: PreflightService,
        confirmation: ConfirmationService,
        action_request: ActionRequestPreviewService,
        execution: ExecutionService,
        retry: RetryService,
        governance: Any | None = None,
    ) -> None:
        self._planner = planner
        self._ledger = ledger
        self._contract_loader = contract_loader
        self._run_status = run_status
        self._orchestration = orchestration
        self._preflight = preflight
        self._confirmation = confirmation
        self._action_request = action_request
        self._execution = execution
        self._retry = retry
        self._governance = governance

    def create_workflow_run(self, request: WorkflowRunRequest) -> WorkflowRunResult:
        scope = self._resolve_scope(request)
        if not scope.selected_capability_ids:
            raise _rejected(
                "workflow_scope_has_no_enabled_capabilities",
                "The selected workflow scope does not currently have any enabled app-owned agent capabilities.",
            )

        capabilities = self._contract_loader.load_agent_capabilities() or default_capabilities()
        by_capability_id = {capability.capability_id: capability for capability in capabilities}
        selected = [
            by_capability_id[capability_id]
            for capability_id in scope.selected_capability_ids
            if capability_id in by_capability_id
        ]
        context_summary = _context_summary_for_scope(request.context_summary, scope)
        plan_result = self._planner.create_plan(
            PlanRequest(
                user_goal=request.goal,
                context_summary={
                    **context_summary,
                    "workflow_scope": scope.model_dump(mode="json"),
                },
                capabilities=selected,
                policy=request.policy,
            )
        )
        self._ledger.append_recovery_event(
            plan_result.run_id,
            _workflow_created_event(scope),
        )
        orchestration = self._orchestration.get_run_orchestration(plan_result.run_id)
        return WorkflowRunResult(
            run_id=plan_result.run_id,
            workflow_scope=scope,
            plan=plan_result.plan,
            run_state=orchestration.run_state,
            orchestration=orchestration,
            provider_metadata=plan_result.provider_metadata,
            redaction_summary=plan_result.redaction_summary,
            context_preview=plan_result.context_preview,
            validation=plan_result.validation,
            ledger_recorded=True,
        )

    def get_workflow_run(self, run_id: str) -> WorkflowRunStatusResult:
        return WorkflowRunStatusResult(
            run_id=run_id,
            workflow_scope=self._workflow_scope_for_run(run_id),
            run_status=self._run_status.get_run_status(run_id),
            orchestration=self._orchestration.get_run_orchestration(run_id),
            validation=PlanValidationResult(status="valid"),
        )

    def resolve_scope_summary(self, request: WorkflowRunRequest) -> WorkflowRunScopeSummary:
        return self._resolve_scope(request)

    def advance_one(
        self,
        run_id: str,
        request: WorkflowAdvanceRequest | None = None,
    ) -> WorkflowAdvanceResult:
        _ = request or WorkflowAdvanceRequest()
        scope = self._workflow_scope_for_run(run_id)
        orchestration = self._orchestration.get_run_orchestration(run_id)
        current_step = next((step for step in orchestration.steps if step.is_current), None)
        if current_step is None:
            return self._record_advance_result(
                run_id=run_id,
                scope=scope,
                advance_status="completed",
                selected_action=None,
                delegated_result=None,
            )

        action = _select_advance_action(current_step.allowed_actions)
        if action is None:
            return self._record_advance_result(
                run_id=run_id,
                scope=scope,
                step_id=current_step.step_id,
                capability_id=current_step.capability_id,
                advance_status="blocked",
                selected_action=None,
                delegated_result={
                    "blocker_reason": current_step.blocker_reason,
                    "allowed_actions": current_step.allowed_actions,
                    "step_status": current_step.status,
                },
            )
        if action == "confirm_step":
            return self._record_advance_result(
                run_id=run_id,
                scope=scope,
                step_id=current_step.step_id,
                capability_id=current_step.capability_id,
                advance_status="manual_confirmation_required",
                selected_action=action,
                delegated_result={
                    "message": "Manual confirmation is required before the workflow runner can continue.",
                },
            )
        if action == "retry_failed_step":
            return self._record_advance_result(
                run_id=run_id,
                scope=scope,
                step_id=current_step.step_id,
                capability_id=current_step.capability_id,
                advance_status="manual_retry_required",
                selected_action=action,
                delegated_result={
                    "message": "Manual retry is required before the workflow runner can continue.",
                },
            )

        delegated = self._delegate_action(
            action=action,
            run_id=run_id,
            step_id=current_step.step_id,
            capability_id=current_step.capability_id,
        )
        return self._record_advance_result(
            run_id=run_id,
            scope=scope,
            step_id=current_step.step_id,
            capability_id=current_step.capability_id,
            advance_status="advanced",
            selected_action=action,
            delegated_result=delegated,
        )

    def advance_until_blocked(
        self,
        run_id: str,
        request: WorkflowAdvanceUntilBlockedRequest | None = None,
    ) -> WorkflowAdvanceUntilBlockedResult:
        request = request or WorkflowAdvanceUntilBlockedRequest()
        completed = 0
        last_result: WorkflowAdvanceResult | None = None
        for _index in range(request.max_steps):
            last_result = self.advance_one(run_id)
            if last_result.advance_status != "advanced":
                break
            completed += 1
        orchestration = self._orchestration.get_run_orchestration(run_id)
        scope = self._workflow_scope_for_run(run_id)
        status = last_result.advance_status if last_result is not None else "blocked"
        return WorkflowAdvanceUntilBlockedResult(
            run_id=run_id,
            workflow_scope=scope,
            advance_status=status,
            completed_action_count=completed,
            last_result=last_result,
            run_state=orchestration.run_state,
            orchestration=orchestration,
            validation=PlanValidationResult(status="valid"),
            ledger_recorded=True,
        )

    def _delegate_action(
        self,
        *,
        action: str,
        run_id: str,
        step_id: str,
        capability_id: str,
    ) -> dict[str, Any]:
        if action == "run_preflight":
            self._require_governance("POST /preflights", run_id, step_id, capability_id)
            result = self._preflight.create_preflight(PreflightRequest(run_id=run_id, step_id=step_id))
            return {"result_type": "preflight", "result": result.model_dump(mode="json")}
        if action == "preview_action_request":
            self._require_governance("POST /action-requests", run_id, step_id, capability_id)
            result = self._action_request.create_action_request(
                ActionRequestPreviewRequest(run_id=run_id, step_id=step_id)
            )
            return {"result_type": "action_request_preview", "result": result.model_dump(mode="json")}
        if action == "execute_step":
            self._require_governance("POST /executions", run_id, step_id, capability_id)
            result = self._execution.execute_step(ExecutionRequest(run_id=run_id, step_id=step_id))
            return {"result_type": "execution", "result": result.model_dump(mode="json")}
        if action == "retry_failed_step":
            self._require_governance("POST /retries", run_id, step_id, capability_id)
            result = self._retry.retry_step(
                RetryRequest(run_id=run_id, step_id=step_id, retry_intent="retry_failed_step")
            )
            return {"result_type": "retry", "result": result.model_dump(mode="json")}
        if action == "confirm_step":
            self._require_governance("POST /confirmations", run_id, step_id, capability_id)
            result = self._confirmation.create_confirmation(
                ConfirmationRequest(
                    run_id=run_id,
                    step_id=step_id,
                    confirmation_intent="approve_plan_step",
                )
            )
            return {"result_type": "confirmation", "result": result.model_dump(mode="json")}
        raise _rejected(
            "unsupported_workflow_advance_action",
            "The current orchestration action cannot be advanced by the workflow runner.",
            step_id=step_id,
            capability_id=capability_id,
        )

    def _require_governance(
        self,
        route: str,
        run_id: str,
        step_id: str,
        capability_id: str,
    ) -> None:
        if self._governance is None:
            return
        self._governance.require_allowed(
            route=route,
            run_id=run_id,
            step_id=step_id,
            capability_id=capability_id,
        )

    def _record_advance_result(
        self,
        *,
        run_id: str,
        scope: WorkflowRunScopeSummary | None,
        advance_status: str,
        selected_action: str | None,
        delegated_result: dict[str, Any] | None,
        step_id: str | None = None,
        capability_id: str | None = None,
    ) -> WorkflowAdvanceResult:
        event = {
            "event_type": WORKFLOW_ADVANCE_EVENT_TYPE,
            "workflow_advance_id": f"workflow_advance_{uuid4().hex[:12]}",
            "status": advance_status,
            "selected_action": selected_action,
            "step_id": step_id,
            "capability_id": capability_id,
            "workflow_scope": scope.model_dump(mode="json") if scope is not None else None,
            "delegated_result_type": delegated_result.get("result_type") if isinstance(delegated_result, dict) else None,
            "recorded_at_utc": _utc_now_label(),
            "single_step_only": True,
            "execution_permitted": False,
        }
        try:
            self._ledger.append_recovery_event(run_id, event)
            ledger_recorded = True
        except Exception:
            ledger_recorded = False
        orchestration = self._orchestration.get_run_orchestration(run_id)
        return WorkflowAdvanceResult(
            run_id=run_id,
            workflow_scope=scope,
            step_id=step_id,
            capability_id=capability_id,
            selected_action=selected_action,
            advance_status=advance_status,
            delegated_result=delegated_result,
            run_state=orchestration.run_state,
            orchestration=orchestration,
            validation=PlanValidationResult(status="valid"),
            ledger_recorded=ledger_recorded,
        )

    def _workflow_scope_for_run(self, run_id: str) -> WorkflowRunScopeSummary | None:
        entry = self._ledger.get(run_id)
        if entry is None:
            raise _rejected("unknown_run", "No recorded run was found for the requested run_id.")
        for record in reversed(entry.recovery_events):
            if not isinstance(record, dict):
                continue
            if record.get("event_type") != WORKFLOW_CREATED_EVENT_TYPE:
                continue
            raw_scope = record.get("workflow_scope")
            if isinstance(raw_scope, dict):
                try:
                    return WorkflowRunScopeSummary.model_validate(raw_scope)
                except Exception:
                    return None
        return None

    def _resolve_scope(self, request: WorkflowRunRequest) -> WorkflowRunScopeSummary:
        capability_ids, template_ids, gaps = self._capability_ids_for_request(request)
        capabilities = self._contract_loader.load_agent_capabilities() or default_capabilities()
        by_id = {capability.capability_id: capability for capability in capabilities}
        selected: list[str] = []
        omitted: list[str] = []
        final_gaps = list(gaps)
        for capability_id in _unique(capability_ids):
            capability = by_id.get(capability_id)
            if capability is None:
                final_gaps.append(
                    {
                        "capability_id": capability_id,
                        "status": "missing_canonical_capability",
                        "message": "The workflow template references a capability that is not in the canonical capability registry.",
                    }
                )
                omitted.append(capability_id)
                continue
            if not capability.enabled:
                final_gaps.append(
                    {
                        "capability_id": capability_id,
                        "app_id": capability.app_id,
                        "status": "planned_or_disabled",
                        "message": "The capability is part of the workflow template but is not yet enabled by the canonical registry.",
                    }
                )
                omitted.append(capability_id)
                continue
            selected.append(capability_id)
        return WorkflowRunScopeSummary(
            workflow_scope=request.workflow_scope,
            source_app=request.source_app,
            start_stage=request.start_stage,
            end_stage=request.end_stage,
            requested_capability_ids=request.requested_capability_ids,
            selected_template_ids=template_ids,
            selected_capability_ids=selected,
            omitted_capability_ids=_unique(omitted),
            workflow_gaps=final_gaps,
        )

    def _capability_ids_for_request(
        self,
        request: WorkflowRunRequest,
    ) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        if request.workflow_scope == "capability_set":
            if not request.requested_capability_ids:
                raise _rejected(
                    "workflow_capability_set_required",
                    "A capability-set workflow requires requested_capability_ids.",
                )
            canonical = {
                capability.capability_id
                for capability in (self._contract_loader.load_agent_capabilities() or default_capabilities())
            }
            unknown = [item for item in request.requested_capability_ids if item not in canonical]
            if unknown:
                raise _rejected(
                    "unknown_workflow_capability",
                    "The requested workflow capability is not in the canonical capability registry.",
                    capability_id=unknown[0],
                )
            return list(request.requested_capability_ids), [], []

        templates = _workflow_templates(self._contract_loader.load_agent_workflow_templates())
        selected_templates = _select_templates(templates, request)
        if not selected_templates:
            raise _rejected(
                "workflow_template_not_found",
                "No canonical workflow template matched the requested workflow scope.",
            )
        capability_ids: list[str] = []
        gaps: list[dict[str, Any]] = []
        template_ids: list[str] = []
        for template in selected_templates:
            template_id = _safe_str(template.get("workflow_id"))
            if template_id:
                template_ids.append(template_id)
            steps = _steps_in_range(template.get("steps"), request)
            for step in steps:
                capability_id = _safe_str(step.get("capability_id"))
                if capability_id:
                    capability_ids.append(capability_id)
                if step.get("implementation_status") not in {None, "available"}:
                    gaps.append(
                        {
                            "template_id": template_id,
                            "step_key": _safe_str(step.get("step_key")),
                            "capability_id": capability_id,
                            "app_id": _safe_str(step.get("app_id")),
                            "stage_number": step.get("stage_number"),
                            "stage_label": _safe_str(step.get("stage_label")),
                            "status": _safe_str(step.get("implementation_status")),
                            "message": "This workflow step is defined but still needs app-owned agent endpoint support.",
                        }
                    )
        return capability_ids, _unique(template_ids), gaps


def _workflow_templates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_templates = payload.get("workflow_templates") if isinstance(payload, dict) else None
    if isinstance(raw_templates, list) and raw_templates:
        return [item for item in raw_templates if isinstance(item, dict)]
    return _fallback_templates()


def _select_templates(templates: list[dict[str, Any]], request: WorkflowRunRequest) -> list[dict[str, Any]]:
    if request.workflow_scope == "full_lifecycle":
        return [item for item in templates if item.get("workflow_scope") == "full_lifecycle"]
    if request.workflow_scope in {"app_workflow", "stage_range"}:
        if not request.source_app:
            raise _rejected(
                "workflow_source_app_required",
                "An app-scoped or stage-range workflow requires source_app.",
            )
        return [
            item
            for item in templates
            if item.get("workflow_scope") == "app_workflow" and item.get("app_id") == request.source_app
        ]
    return []


def _steps_in_range(raw_steps: Any, request: WorkflowRunRequest) -> list[dict[str, Any]]:
    steps = [item for item in raw_steps if isinstance(item, dict)] if isinstance(raw_steps, list) else []
    if request.workflow_scope != "stage_range":
        return steps
    start = _stage_position(request.start_stage, steps, default=1, field_name="start_stage")
    end = _stage_position(
        request.end_stage,
        steps,
        default=max([_stage_number(step) for step in steps] or [1]),
        field_name="end_stage",
    )
    low, high = sorted((start, end))
    return [step for step in steps if low <= _stage_number(step) <= high]


def _stage_position(
    value: str | None,
    steps: list[dict[str, Any]],
    *,
    default: int,
    field_name: str,
) -> int:
    if value is None or value == "":
        return default
    if value.isdigit():
        position = int(value)
        if position in {_stage_number(step) for step in steps}:
            return position
        raise _rejected(
            "workflow_stage_not_found",
            f"The requested {field_name} does not match a stage in the selected app workflow.",
        )
    for step in steps:
        if step.get("stage_id") == value or step.get("step_key") == value:
            return _stage_number(step)
    raise _rejected(
        "workflow_stage_not_found",
        f"The requested {field_name} does not match a stage in the selected app workflow.",
    )


def _stage_number(step: dict[str, Any]) -> int:
    value = step.get("stage_number")
    return value if isinstance(value, int) else 0


def _select_advance_action(actions: list[str]) -> str | None:
    for action in ("run_preflight", "execute_step", "preview_action_request", "confirm_step", "retry_failed_step"):
        if action in actions:
            return action
    return None


def _workflow_created_event(scope: WorkflowRunScopeSummary) -> dict[str, Any]:
    return {
        "event_type": WORKFLOW_CREATED_EVENT_TYPE,
        "workflow_run_id": f"workflow_{uuid4().hex[:12]}",
        "status": "planned",
        "workflow_scope": scope.model_dump(mode="json"),
        "workflow_gap_count": len(scope.workflow_gaps),
        "recorded_at_utc": _utc_now_label(),
        "execution_permitted": False,
    }


def _fallback_templates() -> list[dict[str, Any]]:
    return [
        {
            "workflow_id": "full_lifecycle_fallback",
            "workflow_scope": "full_lifecycle",
            "steps": [
                {"stage_number": 1, "capability_id": "quant_data.run_source_preflight"},
                {"stage_number": 2, "capability_id": "quant_studio.prepare_model_config_draft"},
                {"stage_number": 3, "capability_id": "quant_documentation.inspect_package"},
                {"stage_number": 4, "capability_id": "quant_documentation.create_draft_workspace"},
                {"stage_number": 5, "capability_id": "quant_monitoring.validate_bundle"},
            ],
        }
    ]


def _unique(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result


def _context_summary_for_scope(
    context_summary: dict[str, Any],
    scope: WorkflowRunScopeSummary,
) -> dict[str, Any]:
    result = dict(context_summary)
    if not _scope_selects_studio(scope):
        return result

    target_text = compact_safe_summary_text(
        result.get("target_summary"),
        label="Studio target summary",
    )
    if target_text is not None:
        result["target_summary"] = target_text
        return result

    source_summary = result.get("source_summary")
    if has_meaningful_summary(source_summary):
        source_text = compact_safe_summary_text(
            source_summary,
            label="Direct Studio source summary",
        )
        if source_text is not None:
            result["target_summary"] = source_text
    return result


def _scope_selects_studio(scope: WorkflowRunScopeSummary) -> bool:
    return any(capability_id.startswith("quant_studio.") for capability_id in scope.selected_capability_ids)


def _safe_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


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
