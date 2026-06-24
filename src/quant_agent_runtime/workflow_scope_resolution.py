from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from quant_agent_runtime.models import PlanValidationResult, ValidationIssue, WorkflowRunRequest
from quant_agent_runtime.redaction import find_unsafe_payload_issues
from quant_agent_runtime.validation.errors import RuntimeValidationError


WORKFLOW_SCOPE_RESOLUTION_SUPPORT_LEVEL = "deterministic_prompt_to_workflow_scope_v1"

_KNOWN_APPS = {
    "quant_data": ("quant data", "data"),
    "quant_studio": ("quant studio", "studio"),
    "quant_documentation": ("quant documentation", "documentation", "docs", "doc"),
    "quant_monitoring": ("quant monitoring", "monitoring"),
}

_CAPABILITY_ALIASES = {
    "quant_data.run_source_preflight": (
        "source preflight",
        "source readiness",
        "data readiness",
    ),
    "quant_studio.prepare_model_config_draft": (
        "model config draft",
        "model configuration draft",
    ),
    "quant_studio.fit_candidate_model": (
        "fit candidate model",
        "candidate model fit",
        "fit a model",
        "model fit",
    ),
    "quant_documentation.inspect_package": (
        "inspect documentation package",
        "documentation package inspection",
    ),
    "quant_documentation.create_draft_workspace": (
        "documentation draft workspace",
        "draft workspace",
    ),
    "quant_monitoring.validate_bundle": (
        "validate monitoring bundle",
        "monitoring bundle validation",
        "validate the monitoring bundle",
        "bundle validation",
    ),
}


@dataclass(frozen=True)
class WorkflowScopeResolution:
    request: WorkflowRunRequest
    summary: dict[str, Any]


class WorkflowScopeResolutionService:
    def resolve(self, request_goal: str, *, source_app: str | None, context_summary: dict[str, Any]) -> WorkflowScopeResolution:
        goal = " ".join(request_goal.strip().split())
        _reject_unsafe(goal, context_summary)
        normalized = _normalize(goal)
        detected_app = _detect_app(normalized, source_app, context_summary)
        detected_capabilities = _detect_capabilities(normalized)
        stage_range = _detect_stage_range(normalized)

        if _looks_like_unsupported_quant_app(normalized):
            raise _rejected(
                "workflow_scope_unknown_app",
                "The prompt references an app that is not a known Quant Suite workflow app.",
            )

        if _is_full_lifecycle_request(normalized):
            workflow_request = WorkflowRunRequest(
                goal=goal,
                workflow_scope="full_lifecycle",
                source_app="quant_suite",
                context_summary=context_summary,
            )
            return WorkflowScopeResolution(
                request=workflow_request,
                summary=_summary("full_lifecycle_keywords", detected_app, [], None, 0.93),
            )

        if _studio_fit_range_requested(normalized, detected_app, detected_capabilities, stage_range):
            workflow_request = WorkflowRunRequest(
                goal=goal,
                workflow_scope="stage_range",
                source_app="quant_studio",
                start_stage="lifecycle_handoff_intake",
                end_stage="candidate_model_fit",
                context_summary=context_summary,
            )
            return WorkflowScopeResolution(
                request=workflow_request,
                summary=_summary(
                    "studio_fit_stage_range",
                    "quant_studio",
                    [],
                    {"start_stage": "lifecycle_handoff_intake", "end_stage": "candidate_model_fit"},
                    0.86,
                ),
            )

        if detected_capabilities and stage_range is None:
            workflow_request = WorkflowRunRequest(
                goal=goal,
                workflow_scope="capability_set",
                source_app=detected_app,
                requested_capability_ids=detected_capabilities,
                context_summary=context_summary,
            )
            return WorkflowScopeResolution(
                request=workflow_request,
                summary=_summary("capability_alias_match", detected_app, detected_capabilities, None, 0.88),
            )

        if detected_app is None:
            raise _rejected(
                "workflow_scope_unresolved",
                "The prompt did not identify a full lifecycle, known app workflow, stage range, or supported capability set.",
            )

        if stage_range is not None:
            start_stage, end_stage = stage_range
            if start_stage == "1" and end_stage == "5":
                workflow_request = WorkflowRunRequest(
                    goal=goal,
                    workflow_scope="app_workflow",
                    source_app=detected_app,
                    context_summary=context_summary,
                )
                return WorkflowScopeResolution(
                    request=workflow_request,
                    summary=_summary("app_steps_1_to_5", detected_app, [], {"start_stage": "1", "end_stage": "5"}, 0.91),
                )
            workflow_request = WorkflowRunRequest(
                goal=goal,
                workflow_scope="stage_range",
                source_app=detected_app,
                start_stage=start_stage,
                end_stage=end_stage,
                context_summary=context_summary,
            )
            return WorkflowScopeResolution(
                request=workflow_request,
                summary=_summary(
                    "app_stage_range",
                    detected_app,
                    [],
                    {"start_stage": start_stage, "end_stage": end_stage},
                    0.89,
                ),
            )

        workflow_request = WorkflowRunRequest(
            goal=goal,
            workflow_scope="app_workflow",
            source_app=detected_app,
            context_summary=context_summary,
        )
        return WorkflowScopeResolution(
            request=workflow_request,
            summary=_summary("app_workflow_alias", detected_app, [], None, 0.82),
        )


def _normalize(value: str) -> str:
    normalized = value.lower()
    normalized = normalized.replace("->", " to ")
    normalized = normalized.replace("through", " through ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _reject_unsafe(goal: str, context_summary: dict[str, Any]) -> None:
    issues = find_unsafe_payload_issues({"goal": goal, "context_summary": context_summary}, root="workflow_scope_resolution")
    if not issues:
        return
    raise RuntimeValidationError(
        PlanValidationResult(
            status="rejected",
            errors=[
                issue.model_copy(update={"code": "unsafe_workflow_scope_resolution_request"})
                for issue in issues
            ],
        )
    )


def _detect_app(goal: str, source_app: str | None, context_summary: dict[str, Any]) -> str | None:
    for app_id, aliases in _KNOWN_APPS.items():
        if any(_contains_phrase(goal, alias) for alias in aliases):
            return app_id
    launch_source = context_summary.get("launch_source")
    launch_app = launch_source.get("source_app") if isinstance(launch_source, dict) else None
    if source_app in _KNOWN_APPS:
        return source_app
    if launch_app in _KNOWN_APPS:
        return str(launch_app)
    return None


def _detect_capabilities(goal: str) -> list[str]:
    exact_matches = [capability_id for capability_id in _CAPABILITY_ALIASES if capability_id in goal]
    alias_matches = [
        capability_id
        for capability_id, aliases in _CAPABILITY_ALIASES.items()
        if any(_contains_phrase(goal, alias) for alias in aliases)
    ]
    result: list[str] = []
    for capability_id in [*exact_matches, *alias_matches]:
        if capability_id not in result:
            result.append(capability_id)
    return result


def _detect_stage_range(goal: str) -> tuple[str, str] | None:
    range_match = re.search(r"\b(?:steps?|stages?)\s+(\d+)\s*(?:-|to|through|thru)\s*(\d+)\b", goal)
    if range_match:
        return range_match.group(1), range_match.group(2)
    single_match = re.search(r"\b(?:step|stage)\s+(\d+)\b", goal)
    if single_match:
        return single_match.group(1), single_match.group(1)
    return None


def _is_full_lifecycle_request(goal: str) -> bool:
    return (
        ("full" in goal or "entire" in goal or "whole" in goal or "end to end" in goal)
        and ("workflow" in goal or "lifecycle" in goal or "suite" in goal)
    ) or "data to studio" in goal or "data studio documentation monitoring" in goal


def _studio_fit_range_requested(
    goal: str,
    detected_app: str | None,
    detected_capabilities: list[str],
    stage_range: tuple[str, str] | None,
) -> bool:
    return (
        stage_range is None
        and detected_app == "quant_studio"
        and "quant_studio.fit_candidate_model" in detected_capabilities
        and ("handoff" in goal or "from this" in goal)
    )


def _looks_like_unsupported_quant_app(goal: str) -> bool:
    if "quant " not in goal:
        return False
    return not any(alias in goal for aliases in _KNOWN_APPS.values() for alias in aliases) and "quant suite" not in goal


def _contains_phrase(value: str, phrase: str) -> bool:
    escaped = re.escape(phrase.lower())
    return re.search(rf"(?<![a-z0-9_]){escaped}(?![a-z0-9_])", value) is not None


def _summary(
    strategy: str,
    app_id: str | None,
    capability_ids: list[str],
    stage_range: dict[str, str] | None,
    confidence: float,
) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "detected_app": app_id,
        "detected_capability_ids": capability_ids,
        "detected_stage_range": stage_range,
        "confidence": confidence,
        "llm_used": False,
        "data_policy": "summaries_and_references_only",
    }


def _rejected(code: str, message: str) -> RuntimeValidationError:
    return RuntimeValidationError(
        PlanValidationResult(
            status="rejected",
            errors=[ValidationIssue(code=code, message=message)],
        )
    )
