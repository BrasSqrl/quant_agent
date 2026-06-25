from __future__ import annotations

from typing import Any

from quant_agent_runtime.models import LedgerEntry, PlanValidationResult, ValidationIssue
from quant_agent_runtime.summary_text import compact_safe_summary_text
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
DOCUMENTATION_INSPECT_PACKAGE_CAPABILITY_ID = "quant_documentation.inspect_package"
DOCUMENTATION_DRAFT_WORKSPACE_CAPABILITY_ID = "quant_documentation.create_draft_workspace"
DOCUMENTATION_DRAFT_SECTION_CAPABILITY_ID = "quant_documentation.draft_section"
DOCUMENTATION_CLAIM_REVIEW_CAPABILITY_ID = "quant_documentation.find_unsupported_claims"
DOCUMENTATION_REVIEW_EXPORT_CAPABILITY_ID = "quant_documentation.export_markdown_review_package"
MONITORING_INSPECT_BUNDLE_CAPABILITY_ID = "quant_monitoring.inspect_bundle"
MONITORING_PROFILE_DRAFT_CAPABILITY_ID = "quant_monitoring.prepare_profile_draft"
MONITORING_BUNDLE_PREFLIGHT_CAPABILITY_ID = "quant_monitoring.validate_bundle"
MONITORING_RUN_REVIEW_CAPABILITY_ID = "quant_monitoring.run_monitoring_review"
MONITORING_FEEDBACK_SIGNAL_CAPABILITY_ID = "quant_monitoring.create_feedback_signal"

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
_REQUIRED_MONITORING_HANDOFFS = {
    MONITORING_RUN_REVIEW_CAPABILITY_ID: ("bundle_validation_summary", "bundle validation summary"),
    MONITORING_FEEDBACK_SIGNAL_CAPABILITY_ID: ("monitoring_run", "monitoring run"),
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

    if capability_id.startswith("quant_studio."):
        target_text = compact_safe_summary_text(
            action_input.get("target_summary"),
            label="Studio target summary",
        )
        if target_text is not None:
            action_input["target_summary"] = target_text

    if capability_id in {STUDIO_READINESS_CAPABILITY_ID, STUDIO_DRAFT_CAPABILITY_ID}:
        eda_handoff = _latest_reference(entry, "eda_handoff")
        if eda_handoff is not None:
            action_input["eda_handoff"] = eda_handoff
            if not _usable_text(action_input.get("target_summary")):
                action_input["target_summary"] = f"Target summary from {eda_handoff['label']}."

    if capability_id == STUDIO_DRAFT_CAPABILITY_ID and "model_readiness_summary" not in action_input:
        readiness = _latest_preflight_reference(entry, "model_readiness_summary")
        if readiness is not None:
            action_input["model_readiness_summary"] = readiness

    if capability_id == DATA_SOURCE_PREFLIGHT_CAPABILITY_ID and "source_reference" not in action_input:
        source_reference = _latest_reference(entry, "source_reference")
        if source_reference is not None:
            action_input["source_reference"] = source_reference

    if capability_id in {
        DOCUMENTATION_INSPECT_PACKAGE_CAPABILITY_ID,
        DOCUMENTATION_DRAFT_WORKSPACE_CAPABILITY_ID,
        DOCUMENTATION_DRAFT_SECTION_CAPABILITY_ID,
        DOCUMENTATION_CLAIM_REVIEW_CAPABILITY_ID,
        DOCUMENTATION_REVIEW_EXPORT_CAPABILITY_ID,
    }:
        package_summary = _documentation_package_summary(entry)
        if package_summary is not None and not _usable_mapping(action_input.get("package_summary")):
            action_input["package_summary"] = package_summary
        documentation_package = _latest_reference(entry, "documentation_package")
        if documentation_package is not None and "documentation_package" not in action_input:
            action_input["documentation_package"] = documentation_package

    if capability_id == DOCUMENTATION_DRAFT_WORKSPACE_CAPABILITY_ID and "documentation_package_summary" not in action_input:
        package_summary = _latest_reference(entry, "documentation_package_summary")
        if package_summary is not None:
            action_input["documentation_package_summary"] = package_summary

    if capability_id in {
        MONITORING_INSPECT_BUNDLE_CAPABILITY_ID,
        MONITORING_PROFILE_DRAFT_CAPABILITY_ID,
        MONITORING_BUNDLE_PREFLIGHT_CAPABILITY_ID,
        MONITORING_RUN_REVIEW_CAPABILITY_ID,
        MONITORING_FEEDBACK_SIGNAL_CAPABILITY_ID,
    }:
        monitoring_bundle = _latest_reference(entry, "monitoring_bundle")
        if monitoring_bundle is not None:
            action_input["monitoring_bundle"] = monitoring_bundle
            action_input.setdefault("bundle_id", monitoring_bundle["reference_id"])
            action_input.setdefault("bundle_reference_id", monitoring_bundle["reference_id"])
        bundle_summary = _monitoring_bundle_summary(entry)
        if bundle_summary is not None and not _usable_mapping(action_input.get("bundle_summary")):
            action_input["bundle_summary"] = bundle_summary

    if capability_id == MONITORING_PROFILE_DRAFT_CAPABILITY_ID and not isinstance(action_input.get("bundle_summary"), dict):
        bundle_summary = _latest_reference(entry, "bundle_summary")
        if bundle_summary is not None:
            action_input["bundle_summary"] = bundle_summary

    if capability_id == MONITORING_BUNDLE_PREFLIGHT_CAPABILITY_ID and "monitoring_profile_draft" not in action_input:
        profile_draft = _latest_reference(entry, "monitoring_profile_draft")
        if profile_draft is not None:
            action_input["monitoring_profile_draft"] = profile_draft

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

    required = _REQUIRED_MONITORING_HANDOFFS.get(capability_id)
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
                            message=f"The Quant Monitoring workflow step requires a prior {label} reference.",
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


def _documentation_package_summary(entry: LedgerEntry) -> dict[str, Any] | None:
    reference = _latest_reference(entry, "documentation_package") or _latest_reference(
        entry, "documentation_package_summary"
    )
    if reference is None:
        return None
    return {
        "documentation_package_id": reference["reference_id"],
        "summary": reference["label"],
        "label": reference["label"],
        "documentation_packages": [reference],
        "section_evidence_map": [
            {
                "section_id": "model_overview",
                "document_section": "Model overview",
                "display_order": 1,
                "required_evidence": ["ledgered_model_evidence"],
            }
        ],
        "known_gaps": [],
    }


def _monitoring_bundle_summary(entry: LedgerEntry) -> dict[str, Any] | None:
    reference = _latest_reference(entry, "bundle_summary")
    if reference is None:
        reference = _latest_reference(entry, "monitoring_bundle")
    if reference is None:
        return None
    return {
        "reference_type": "bundle_summary",
        "reference_id": reference["reference_id"],
        "bundle_id": reference["reference_id"],
        "summary": reference["label"],
        "label": reference["label"],
        "stored_in": reference["stored_in"],
    }


def _usable_mapping(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    reference_id = value.get("reference_id") or value.get("documentation_package_id") or value.get("bundle_id")
    package_metadata = value.get("package_metadata")
    if not reference_id and isinstance(package_metadata, dict):
        reference_id = package_metadata.get("documentation_package_id")
    return isinstance(reference_id, str) and bool(reference_id.strip())


def _usable_text(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and value.strip() not in {"[missing]", "Pending upstream workflow handoff."}
    )


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
