from __future__ import annotations

import pytest

from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.model_gateway import ModelProvider, ProviderPlanRequest, ProviderResult
from quant_agent_runtime.models import (
    CapabilityDefinition,
    PlanRequest,
    ProviderMetadata,
    ProviderMode,
    RiskTier,
)
from quant_agent_runtime.planner import PlannerService
from quant_agent_runtime.validation.errors import RuntimeValidationError


class StaticProvider(ModelProvider):
    def __init__(self, raw_output: dict[str, object]) -> None:
        self.raw_output = raw_output
        self.seen_request: ProviderPlanRequest | None = None

    def generate_plan(self, request: ProviderPlanRequest) -> ProviderResult:
        self.seen_request = request
        return ProviderResult(
            raw_output=self.raw_output,
            metadata=ProviderMetadata(
                provider="test",
                model="static",
                provider_mode=request.policy.provider_mode,
                supports_execution=False,
            ),
        )


def valid_step(**overrides: object) -> dict[str, object]:
    step = {
        "step_id": "step_1",
        "title": "Inspect lifecycle",
        "capability_id": "quant_suite.inspect_lifecycle_context",
        "app_id": "quant_suite",
        "risk_tier": "read_only",
        "operation": "plan",
        "requires_confirmation": False,
        "action_input": {"lifecycle_summary": "safe summary"},
        "expected_artifacts": [],
        "validation_checks": ["capability_known"],
    }
    step.update(overrides)
    return step


def output_with_step(step: dict[str, object]) -> dict[str, object]:
    return {
        "user_goal_summary": "safe goal",
        "assumptions": [],
        "missing_inputs": [],
        "steps": [step],
    }


def runtime_for(raw_output: dict[str, object]) -> tuple[PlannerService, StaticProvider]:
    provider = StaticProvider(raw_output)
    return PlannerService(provider=provider, ledger=InMemoryLedger()), provider


def request_with_default_capability() -> PlanRequest:
    return PlanRequest(
        user_goal="Plan from safe summaries.",
        context_summary={"lifecycle_summary": "Lifecycle exists."},
    )


def test_malformed_provider_output_is_rejected() -> None:
    planner, _ = runtime_for({"not": "a valid plan"})

    with pytest.raises(RuntimeValidationError) as exc_info:
        planner.create_plan(request_with_default_capability())

    assert exc_info.value.validation.errors[0].code == "malformed_provider_output"
    assert planner.ledger.list_entries()[0].validation_results.status == "rejected"


def test_unknown_capability_is_rejected() -> None:
    step = valid_step(capability_id="quant_unknown.do_thing")
    planner, _ = runtime_for(output_with_step(step))

    with pytest.raises(RuntimeValidationError) as exc_info:
        planner.create_plan(request_with_default_capability())

    assert {issue.code for issue in exc_info.value.validation.errors} == {
        "unknown_capability"
    }


def test_forbidden_action_is_rejected() -> None:
    planner, _ = runtime_for(output_with_step(valid_step()))
    request = PlanRequest(
        user_goal="Plan from safe summaries.",
        context_summary={"lifecycle_summary": "Lifecycle exists."},
        policy={"forbidden_action_ids": ["quant_suite.inspect_lifecycle_context"]},
    )

    with pytest.raises(RuntimeValidationError) as exc_info:
        planner.create_plan(request)

    assert "forbidden_action" in {issue.code for issue in exc_info.value.validation.errors}
    assert planner.ledger.list_entries()[0].policy_rejections


def test_action_step_missing_required_fields_is_rejected() -> None:
    step = valid_step(action_input={})
    planner, _ = runtime_for(output_with_step(step))

    with pytest.raises(RuntimeValidationError) as exc_info:
        planner.create_plan(request_with_default_capability())

    assert "missing_required_action_field" in {
        issue.code for issue in exc_info.value.validation.errors
    }


def test_plan_that_tries_to_execute_is_rejected() -> None:
    step = valid_step(operation="execute")
    planner, _ = runtime_for(output_with_step(step))

    with pytest.raises(RuntimeValidationError) as exc_info:
        planner.create_plan(request_with_default_capability())

    assert "execution_not_allowed" in {issue.code for issue in exc_info.value.validation.errors}


def test_missing_confirmation_requirement_is_detected() -> None:
    capability = CapabilityDefinition(
        capability_id="quant_studio.fit_candidate_model",
        app_id="quant_studio",
        display_name="Fit candidate model",
        risk_tier=RiskTier.expensive_compute,
        required_fields=["target_summary"],
    )
    step = valid_step(
        capability_id="quant_studio.fit_candidate_model",
        app_id="quant_studio",
        risk_tier="expensive_compute",
        requires_confirmation=False,
        action_input={"target_summary": "safe target summary"},
    )
    planner, _ = runtime_for(output_with_step(step))

    with pytest.raises(RuntimeValidationError) as exc_info:
        planner.create_plan(
            PlanRequest(
                user_goal="Fit the model after review.",
                context_summary={"target_summary": "Default flag."},
                capabilities=[capability],
            )
        )

    assert "missing_confirmation_requirement" in {
        issue.code for issue in exc_info.value.validation.errors
    }


def test_unsafe_context_is_redacted_before_planning() -> None:
    planner, provider = runtime_for(output_with_step(valid_step()))

    result = planner.create_plan(
        PlanRequest(
            user_goal="Plan with C:\\Users\\me\\secret.csv omitted.",
            context_summary={
                "lifecycle_summary": "Lifecycle exists.",
                "secrets": {"api_key": "do-not-store"},
                "records": [{"name": "raw row"}],
                "safe_note": "Input came from s3://private-bucket/raw.csv",
            },
        )
    )

    assert provider.seen_request is not None
    assert "secrets" not in provider.seen_request.context_summary
    assert "records" not in provider.seen_request.context_summary
    assert provider.seen_request.context_summary["safe_note"] == "Input came from [redacted]"
    assert result.redaction_summary.redacted is True
    assert result.validation.status == "valid"


def test_unsafe_plan_payload_is_rejected() -> None:
    step = valid_step(action_input={"lifecycle_summary": "C:\\Users\\me\\raw.csv"})
    planner, _ = runtime_for(output_with_step(step))

    with pytest.raises(RuntimeValidationError) as exc_info:
        planner.create_plan(request_with_default_capability())

    assert "unsafe_raw_value" in {issue.code for issue in exc_info.value.validation.errors}
