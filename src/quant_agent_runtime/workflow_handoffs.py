from __future__ import annotations

from typing import Any

from quant_agent_runtime.models import LedgerEntry, PlanValidationResult, ValidationIssue
from quant_agent_runtime.validation.errors import RuntimeValidationError


STUDIO_READINESS_CAPABILITY_ID = "quant_studio.run_model_readiness_check"
STUDIO_DRAFT_CAPABILITY_ID = "quant_studio.prepare_model_config_draft"
STUDIO_FIT_CAPABILITY_ID = "quant_studio.fit_candidate_model"
STUDIO_COMPARE_CAPABILITY_ID = "quant_studio.compare_candidate_runs"
STUDIO_PACKAGE_CAPABILITY_ID = "quant_studio.create_documentation_package"
DATA_REGISTER_SOURCE_CAPABILITY_ID = "quant_data.register_source_reference"
DATA_SOURCE_PREFLIGHT_CAPABILITY_ID = "quant_data.run_source_preflight"
DATA_CREATE_EDA_PLAN_CAPABILITY_ID = "quant_data.create_eda_plan"
DATA_RUN_EDA_REVIEW_CAPABILITY_ID = "quant_data.run_eda_review"
DATA_EXPORT_EDA_HANDOFF_CAPABILITY_ID = "quant_data.export_eda_handoff"
DOCUMENTATION_DRAFT_WORKSPACE_CAPABILITY_ID = "quant_documentation.create_draft_workspace"
DOCUMENTATION_DRAFT_SECTION_CAPABILITY_ID = "quant_documentation.draft_section"
DOCUMENTATION_CLAIM_REVIEW_CAPABILITY_ID = "quant_documentation.find_unsupported_claims"
DOCUMENTATION_REVIEW_EXPORT_CAPABILITY_ID = "quant_documentation.export_markdown_review_package"

_REQUIRED_STUDIO_HANDOFFS = {
    STUDIO_FIT_CAPABILITY_ID: ("model_config_draft", "model configuration draft"),
    STUDIO_COMPARE_CAPABILITY_ID: ("studio_run", "Studio candidate run"),
    STUDIO_PACKAGE_CAPABILITY_ID: ("champion_recommendation", "champion recommendation"),
}
_REQUIRED_DATA_HANDOFFS = {
    DATA_CREATE_EDA_PLAN_CAPABILITY_ID: ("preflight_summary", "source preflight summary"),
    DATA_RUN_EDA_REVIEW_CAPABILITY_ID: ("eda_plan", "EDA plan"),
    DATA_EXPORT_EDA_HANDOFF_CAPABILITY_ID: ("eda_package", "EDA package"),
}
_REQUIRED_DOCUMENTATION_HANDOFFS = {
    DOCUMENTATION_DRAFT_SECTION_CAPABILITY_ID: ("documentation_draft", "Documentation draft workspace"),
    DOCUMENTATION_CLAIM_REVIEW_CAPABILITY_ID: ("draft_section", "draft section"),
    DOCUMENTATION_REVIEW_EXPORT_CAPABILITY_ID: ("claim_review_summary", "claim review summary"),
}


def action_input_with_workflow_handoffs(
    entry: LedgerEntry,
    step: dict[str, Any],
    *,
    fail_on_missing: bool,
) -> dict[str, Any]:
    """Adds safe prior-step references required by known workflow handoffs."""

    capability_id = str(step.get("capability_id") or "")
    step_id = str(step.get("step_id") or "")
    action_input = dict(step.get("action_input") if isinstance(step.get("action_input"), dict) else {})

    if capability_id == STUDIO_DRAFT_CAPABILITY_ID and "model_readiness_summary" not in action_input:
        readiness = _latest_preflight_reference(entry, "model_readiness_summary")
        if readiness is not None:
            action_input["model_readiness_summary"] = readiness

    if capability_id == DATA_SOURCE_PREFLIGHT_CAPABILITY_ID and "source_reference" not in action_input:
        source_reference = _latest_reference(entry, "source_reference")
        if source_reference is not None:
            action_input["source_reference"] = source_reference

    if capability_id == DOCUMENTATION_DRAFT_WORKSPACE_CAPABILITY_ID and "documentation_package_summary" not in action_input:
        package_summary = _latest_reference(entry, "documentation_package_summary")
        if package_summary is not None:
            action_input["documentation_package_summary"] = package_summary

    required = _REQUIRED_DATA_HANDOFFS.get(capability_id)
    if required is not None:
        reference_type, label = required
        if isinstance(action_input.get(reference_type), dict):
            return action_input
        reference = _latest_reference(entry, reference_type)
        if reference is not None:
            action_input[reference_type] = reference
            return action_input
        if fail_on_missing:
            raise RuntimeValidationError(
                PlanValidationResult(
                    status="rejected",
                    errors=[
                        ValidationIssue(
                            code="missing_workflow_handoff_reference",
                            message=f"The Quant Data workflow step requires a prior {label} reference.",
                            step_id=step_id or None,
                            capability_id=capability_id or None,
                        )
                    ],
                )
            )
        return action_input

    required = _REQUIRED_DOCUMENTATION_HANDOFFS.get(capability_id)
    if required is not None:
        reference_type, label = required
        if isinstance(action_input.get(reference_type), dict):
            return action_input
        reference = _latest_reference(entry, reference_type)
        if reference is not None:
            action_input[reference_type] = reference
            return action_input
        if fail_on_missing:
            raise RuntimeValidationError(
                PlanValidationResult(
                    status="rejected",
                    errors=[
                        ValidationIssue(
                            code="missing_workflow_handoff_reference",
                            message=f"The Quant Documentation workflow step requires a prior {label} reference.",
                            step_id=step_id or None,
                            capability_id=capability_id or None,
                        )
                    ],
                )
            )
        return action_input

    required = _REQUIRED_STUDIO_HANDOFFS.get(capability_id)
    if required is None:
        return action_input

    reference_type, label = required
    if isinstance(action_input.get(reference_type), dict):
        return action_input
    reference = _latest_reference(entry, reference_type)
    if reference is not None:
        action_input[reference_type] = reference
        return action_input
    if fail_on_missing:
        raise RuntimeValidationError(
            PlanValidationResult(
                status="rejected",
                errors=[
                    ValidationIssue(
                        code="missing_workflow_handoff_reference",
                        message=f"The Studio workflow step requires a prior {label} reference.",
                        step_id=step_id or None,
                        capability_id=capability_id or None,
                    )
                ],
            )
        )
    return action_input


def _latest_action_result_reference(entry: LedgerEntry, reference_type: str) -> dict[str, Any] | None:
    for result in reversed(entry.action_results):
        if not isinstance(result, dict):
            continue
        references = result.get("output_references")
        if not isinstance(references, list):
            continue
        for reference in references:
            normalized = _safe_reference(reference, reference_type)
            if normalized is not None:
                return normalized
    return None


def _latest_preflight_reference(entry: LedgerEntry, reference_type: str) -> dict[str, Any] | None:
    for result in reversed(entry.preflight_records):
        if not isinstance(result, dict):
            continue
        references = result.get("safe_artifact_references")
        if not isinstance(references, list):
            continue
        for reference in references:
            normalized = _safe_reference(reference, reference_type)
            if normalized is not None:
                return normalized
    return None


def _latest_reference(entry: LedgerEntry, reference_type: str) -> dict[str, Any] | None:
    action_result = _latest_action_result_reference(entry, reference_type)
    if action_result is not None:
        return action_result
    return _latest_preflight_reference(entry, reference_type)


def _safe_reference(value: Any, reference_type: str) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if value.get("reference_type") != reference_type:
        return None
    reference_id = value.get("reference_id")
    if not isinstance(reference_id, str) or not reference_id:
        return None
    label = value.get("label")
    stored_in = value.get("stored_in")
    return {
        "reference_type": reference_type,
        "reference_id": reference_id,
        "label": label if isinstance(label, str) and label else reference_type.replace("_", " ").title(),
        "stored_in": stored_in if isinstance(stored_in, str) and stored_in else "quant_agent_ledger",
    }
