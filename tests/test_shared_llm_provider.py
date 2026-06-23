from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from quant_agent_runtime.api import create_app
from quant_agent_runtime.capabilities import default_capabilities
from quant_agent_runtime.context_builder import LifecycleContextBuilder
from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.model_gateway.provider import ProviderPlanRequest
from quant_agent_runtime.model_gateway.shared_llm import SharedLlmPlanProvider
from quant_agent_runtime.models import PlanRequest, PolicySettings, ProviderMode
from quant_agent_runtime.planner import PlannerService
from quant_agent_runtime.provider_config import runtime_provider_status
from quant_agent_runtime.runtime import build_runtime
from quant_agent_runtime.validation.errors import RuntimeValidationError


AGENT_ROOT = Path(__file__).resolve().parents[1]
QUANT_SUITE_ROOT = AGENT_ROOT.parent / "quant_suite"
CREDIT_PD_LIFECYCLE_FIXTURE = (
    QUANT_SUITE_ROOT
    / "fixtures"
    / "sample_workspaces"
    / "credit_pd_scorecard_panel"
    / "quant_lifecycle_manifest.v1.json"
)
SECRET_KEY = "sk-test-secret-agent-key"


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_agent_specific_llm_env_overrides_shared_env_without_exposing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUANT_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("QUANT_LLM_MODEL", "shared-model")
    monkeypatch.setenv("QUANT_AGENT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("QUANT_AGENT_LLM_MODEL", "gpt-5.4-nano")
    monkeypatch.setenv("OPENAI_API_KEY", SECRET_KEY)

    status = runtime_provider_status()

    assert status.config_source == "QUANT_AGENT_LLM_PROVIDER"
    assert status.configured_provider_mode == "openai"
    assert status.effective_provider_mode == ProviderMode.openai
    assert status.provider_identifier == "openai"
    assert status.model_profile == "gpt-5.4-nano"
    assert status.secret_reference_present is True
    assert status.hosted_provider_enabled is True
    assert SECRET_KEY not in json.dumps(status.model_dump(mode="json"))


def test_openai_plan_provider_uses_server_side_authorization_and_parses_structured_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QUANT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("QUANT_LLM_MODEL", "gpt-5.4-nano")
    monkeypatch.setenv("OPENAI_API_KEY", SECRET_KEY)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.test/v1")
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: int) -> _Response:
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _Response({"output_text": json.dumps(_valid_provider_output())})

    monkeypatch.setattr(
        "quant_agent_runtime.model_gateway.shared_llm.urllib.request.urlopen",
        fake_urlopen,
    )

    result = SharedLlmPlanProvider(runtime_provider_status()).generate_plan(_provider_request())

    assert captured["url"] == "https://api.openai.test/v1/responses"
    assert captured["authorization"] == f"Bearer {SECRET_KEY}"
    assert captured["body"]["model"] == "gpt-5.4-nano"
    assert result.raw_output["user_goal_summary"] == "Safe model-backed plan"
    assert result.metadata.provider == "openai"
    assert result.metadata.model == "gpt-5.4-nano"
    public_result = json.dumps(
        {
            "raw_output": result.raw_output,
            "metadata": result.metadata.model_dump(mode="json"),
        }
    )
    assert SECRET_KEY not in public_result


def test_ollama_plan_provider_uses_configured_local_endpoint_without_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QUANT_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("QUANT_LLM_MODEL", "llama-test")
    monkeypatch.setenv("QUANT_OLLAMA_BASE_URL", "http://ollama.local:11434")
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: int) -> _Response:
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _Response({"response": json.dumps(_valid_provider_output())})

    monkeypatch.setattr(
        "quant_agent_runtime.model_gateway.shared_llm.urllib.request.urlopen",
        fake_urlopen,
    )

    result = SharedLlmPlanProvider(runtime_provider_status()).generate_plan(_provider_request())

    assert captured["url"] == "http://ollama.local:11434/api/generate"
    assert captured["authorization"] is None
    assert captured["body"]["model"] == "llama-test"
    assert result.raw_output["steps"][0]["capability_id"] == "quant_suite.inspect_lifecycle_context"
    assert result.metadata.provider == "ollama"


def test_missing_openai_key_falls_back_to_fake_provider_without_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QUANT_LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def fail_urlopen(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("OpenAI should not be called without a server-side key.")

    monkeypatch.setattr(
        "quant_agent_runtime.model_gateway.shared_llm.urllib.request.urlopen",
        fail_urlopen,
    )

    result = SharedLlmPlanProvider(runtime_provider_status()).generate_plan(_provider_request())

    assert result.metadata.provider == "disabled"
    assert result.metadata.provider_mode == ProviderMode.disabled_or_local_fallback
    assert result.metadata.fallback_reason
    assert result.raw_output["steps"]


def test_unsupported_shared_provider_is_reported_as_disabled_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QUANT_LLM_PROVIDER", "bedrock")

    status = runtime_provider_status()

    assert status.configured is False
    assert status.configured_provider_mode == "unsupported"
    assert status.effective_provider_mode == ProviderMode.disabled_or_local_fallback
    assert status.provider_identifier == "unsupported"
    assert status.configuration_errors == ["Unsupported LLM provider for agent planning."]


def test_unreachable_provider_falls_back_without_leaking_url_or_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QUANT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", SECRET_KEY)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.test/v1")

    def fake_urlopen(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.URLError("connection failed for secret endpoint")

    monkeypatch.setattr(
        "quant_agent_runtime.model_gateway.shared_llm.urllib.request.urlopen",
        fake_urlopen,
    )

    result = SharedLlmPlanProvider(runtime_provider_status()).generate_plan(_provider_request())
    serialized = json.dumps(
        {
            "raw_output": result.raw_output,
            "metadata": result.metadata.model_dump(mode="json"),
        }
    )

    assert result.metadata.provider == "disabled"
    assert "URLError" in (result.metadata.fallback_reason or "")
    assert SECRET_KEY not in serialized
    assert "api.openai.test" not in serialized
    assert "secret endpoint" not in serialized


def test_malformed_openai_json_is_rejected_by_existing_planner_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QUANT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", SECRET_KEY)

    def fake_urlopen(*_args: object, **_kwargs: object) -> _Response:
        return _Response({"output_text": "not json"})

    monkeypatch.setattr(
        "quant_agent_runtime.model_gateway.shared_llm.urllib.request.urlopen",
        fake_urlopen,
    )
    ledger = InMemoryLedger()
    planner = PlannerService(
        provider=SharedLlmPlanProvider(runtime_provider_status()),
        ledger=ledger,
        default_capabilities=default_capabilities(),
    )

    with pytest.raises(RuntimeValidationError) as exc_info:
        planner.create_plan(
            PlanRequest(
                user_goal="Plan with malformed output.",
                context_summary={"lifecycle_summary": "Safe lifecycle summary."},
            )
        )

    assert exc_info.value.validation.errors[0].code == "malformed_provider_output"
    assert SECRET_KEY not in json.dumps(ledger.list_entries()[0].model_dump(mode="json"))


def test_runtime_manifest_reports_shared_provider_without_key_leak(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("QUANT_SUITE_ROOT", str(QUANT_SUITE_ROOT))
    monkeypatch.setenv("QUANT_AGENT_LEDGER_DIR", str(tmp_path / "ledgers"))
    monkeypatch.setenv("QUANT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("QUANT_LLM_MODEL", "gpt-5.4-nano")
    monkeypatch.setenv("OPENAI_API_KEY", SECRET_KEY)

    client = TestClient(create_app(build_runtime()))
    response = client.get("/runtime/manifest")

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider_status"]["effective_provider_mode"] == "openai"
    assert payload["provider_status"]["model_profile"] == "gpt-5.4-nano"
    assert payload["provider_status"]["supports_execution"] is False
    assert "openai" in payload["supported_provider_modes"]
    assert SECRET_KEY not in response.text
    assert "Authorization" not in response.text


def test_openai_plan_response_is_contract_ledgered_without_key_leak(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("QUANT_SUITE_ROOT", str(QUANT_SUITE_ROOT))
    monkeypatch.setenv("QUANT_AGENT_LEDGER_DIR", str(tmp_path / "ledgers"))
    monkeypatch.setenv("QUANT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("QUANT_LLM_MODEL", "gpt-5.4-nano")
    monkeypatch.setenv("OPENAI_API_KEY", SECRET_KEY)

    def fake_urlopen(*_args: object, **_kwargs: object) -> _Response:
        return _Response({"output_text": json.dumps(_valid_canonical_provider_output())})

    monkeypatch.setattr(
        "quant_agent_runtime.model_gateway.shared_llm.urllib.request.urlopen",
        fake_urlopen,
    )

    client = TestClient(create_app(build_runtime()))
    response = client.post(
        "/plans",
        json={
            "user_goal": "Plan source readiness.",
            "context_summary": {
                "source_summary": "Safe registered source summary.",
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider_metadata"]["provider"] == "openai"
    assert payload["provider_metadata"]["model"] == "gpt-5.4-nano"
    assert SECRET_KEY not in response.text
    ledger_file = next((tmp_path / "ledgers").glob("*.json"))
    ledger_text = ledger_file.read_text(encoding="utf-8")
    assert SECRET_KEY not in ledger_text
    assert "raw_provider" not in ledger_text


@pytest.mark.parametrize("provider", ["openai", "ollama"])
def test_provider_backed_credit_pd_sample_plan_validates_and_enables_autopilot_preview(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider: str,
) -> None:
    monkeypatch.setenv("QUANT_SUITE_ROOT", str(QUANT_SUITE_ROOT))
    monkeypatch.setenv("QUANT_AGENT_LEDGER_DIR", str(tmp_path / provider / "ledgers"))
    monkeypatch.setenv("QUANT_LLM_PROVIDER", provider)
    monkeypatch.setenv("QUANT_LLM_MODEL", f"{provider}-sample-model")
    if provider == "openai":
        monkeypatch.setenv("OPENAI_API_KEY", SECRET_KEY)
        response_payload = {"output_text": json.dumps(_valid_sample_demo_provider_output())}
    else:
        response_payload = {"response": json.dumps(_valid_sample_demo_provider_output())}

    def fake_urlopen(*_args: object, **_kwargs: object) -> _Response:
        return _Response(response_payload)

    monkeypatch.setattr(
        "quant_agent_runtime.model_gateway.shared_llm.urllib.request.urlopen",
        fake_urlopen,
    )

    runtime = build_runtime()
    client = TestClient(create_app(runtime))
    plan_response = client.post(
        "/plans",
        json={
            "user_goal": "Create the Phase 7 Credit PD sample demo plan.",
            "context_summary": _credit_pd_sample_context(),
        },
    )

    assert plan_response.status_code == 200
    plan_payload = plan_response.json()
    assert plan_payload["provider_metadata"]["provider"] == provider
    assert plan_payload["plan"]["status"] == "valid"
    assert [
        step["capability_id"]
        for step in plan_payload["plan"]["proposed_steps"]
    ] == [
        "quant_data.run_source_preflight",
        "quant_studio.prepare_model_config_draft",
        "quant_documentation.inspect_package",
        "quant_documentation.create_draft_workspace",
        "quant_monitoring.validate_bundle",
    ]
    assert [
        (
            step["capability_id"],
            step["preflight_required"],
            step["requires_confirmation"],
        )
        for step in plan_payload["plan"]["proposed_steps"]
    ] == [
        ("quant_data.run_source_preflight", True, False),
        ("quant_studio.prepare_model_config_draft", False, True),
        ("quant_documentation.inspect_package", False, False),
        ("quant_documentation.create_draft_workspace", False, True),
        ("quant_monitoring.validate_bundle", True, False),
    ]
    runtime.contract_loader.validate_agent_contract_payload(
        plan_payload["plan"],
        "agent_plan.v1.schema.json",
    )

    autopilot_response = client.post(
        "/autopilot-previews",
        json={
            "run_id": plan_payload["run_id"],
            "autopilot_intent": "preview_sample_autopilot",
            "current_context_summary": _credit_pd_sample_context(),
        },
    )

    assert autopilot_response.status_code == 200
    autopilot_payload = autopilot_response.json()
    assert autopilot_payload["sample_eligibility"]["eligible"] is True
    assert autopilot_payload["sample_eligibility"]["sample_workspace_id"] == "credit_pd_scorecard_panel"
    assert autopilot_payload["autopilot_preview"]["dry_run_only"] is True
    ledger_entry = runtime.planner.ledger.get(plan_payload["run_id"])
    assert ledger_entry is not None
    ledger_dump = json.dumps(ledger_entry.model_dump(mode="json"))
    assert SECRET_KEY not in ledger_dump
    assert "raw_provider" not in ledger_dump


def test_provider_timeout_falls_back_to_fake_plan_without_secret_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QUANT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", SECRET_KEY)

    def fake_urlopen(*_args: object, **_kwargs: object) -> None:
        raise TimeoutError("provider timeout with secret details")

    monkeypatch.setattr(
        "quant_agent_runtime.model_gateway.shared_llm.urllib.request.urlopen",
        fake_urlopen,
    )

    result = SharedLlmPlanProvider(runtime_provider_status()).generate_plan(_provider_request())
    serialized = json.dumps(
        {
            "raw_output": result.raw_output,
            "metadata": result.metadata.model_dump(mode="json"),
        }
    )

    assert result.metadata.provider == "disabled"
    assert result.metadata.provider_mode == ProviderMode.disabled_or_local_fallback
    assert "TimeoutError" in (result.metadata.fallback_reason or "")
    assert SECRET_KEY not in serialized
    assert "secret details" not in serialized


def _provider_request() -> ProviderPlanRequest:
    return ProviderPlanRequest(
        user_goal="Build a safe lifecycle plan.",
        context_summary={"lifecycle_summary": "Safe lifecycle summary."},
        capabilities=default_capabilities(),
        policy=PolicySettings(),
    )


def _valid_provider_output() -> dict[str, Any]:
    return {
        "user_goal_summary": "Safe model-backed plan",
        "assumptions": ["Only sanitized summaries are available."],
        "missing_inputs": [],
        "steps": [
            {
                "step_id": "step_1",
                "title": "Inspect lifecycle context",
                "capability_id": "quant_suite.inspect_lifecycle_context",
                "app_id": "quant_suite",
                "risk_tier": "read_only",
                "operation": "plan",
                "preflight_required": False,
                "requires_confirmation": False,
                "action_input": {"lifecycle_summary": "Safe lifecycle summary."},
                "expected_artifacts": [],
                "validation_checks": ["capability_known", "policy_allowed", "plan_only"],
            }
        ],
    }


def _valid_canonical_provider_output() -> dict[str, Any]:
    return {
        "user_goal_summary": "Safe source readiness plan",
        "assumptions": ["Only sanitized source summaries are available."],
        "missing_inputs": [],
        "steps": [
            {
                "step_id": "step_1",
                "title": "Run source preflight",
                "capability_id": "quant_data.run_source_preflight",
                "app_id": "quant_data",
                "risk_tier": "workflow_preflight",
                "operation": "plan",
                "preflight_required": True,
                "requires_confirmation": False,
                "action_input": {"source_summary": "Safe registered source summary."},
                "expected_artifacts": [],
                "validation_checks": ["capability_known", "policy_allowed", "plan_only"],
            }
        ],
    }


def _credit_pd_sample_context() -> dict[str, Any]:
    context = LifecycleContextBuilder().build_from_path(CREDIT_PD_LIFECYCLE_FIXTURE)
    lifecycle_summary = context.get("lifecycle_summary")
    context["lifecycle_summary"] = {
        "lifecycle_id": "sample_credit_pd_scorecard_panel",
        "summary": str(lifecycle_summary or "Credit PD sample lifecycle has safe seeded evidence."),
        "sample_workspace": {
            "sample_workspace": True,
            "sample_workspace_id": "credit_pd_scorecard_panel",
            "sample_owned": True,
        },
    }
    return context


def _valid_sample_demo_provider_output() -> dict[str, Any]:
    return {
        "user_goal_summary": "Safe Credit PD sample demo plan",
        "assumptions": ["Only sanitized sample summaries and references are available."],
        "missing_inputs": [],
        "steps": [
            {
                "step_id": "step_1",
                "title": "Run source preflight",
                "capability_id": "quant_data.run_source_preflight",
                "app_id": "quant_data",
                "risk_tier": "workflow_preflight",
                "operation": "plan",
                "preflight_required": True,
                "requires_confirmation": False,
                "action_input": {"source_summary": "Safe Credit PD source summary."},
                "expected_artifacts": [],
                "validation_checks": ["capability_known", "policy_allowed", "plan_only"],
            },
            {
                "step_id": "step_2",
                "title": "Prepare model configuration draft",
                "capability_id": "quant_studio.prepare_model_config_draft",
                "app_id": "quant_studio",
                "risk_tier": "draft_only",
                "operation": "plan",
                "preflight_required": False,
                "requires_confirmation": True,
                "action_input": {"target_summary": "Safe Credit PD target summary."},
                "expected_artifacts": [],
                "validation_checks": ["capability_known", "policy_allowed", "confirmation_required"],
            },
            {
                "step_id": "step_3",
                "title": "Inspect documentation package",
                "capability_id": "quant_documentation.inspect_package",
                "app_id": "quant_documentation",
                "risk_tier": "read_only",
                "operation": "plan",
                "preflight_required": False,
                "requires_confirmation": False,
                "action_input": {"package_summary": "Safe Credit PD package summary."},
                "expected_artifacts": [],
                "validation_checks": ["capability_known", "policy_allowed", "plan_only"],
            },
            {
                "step_id": "step_4",
                "title": "Create documentation draft workspace",
                "capability_id": "quant_documentation.create_draft_workspace",
                "app_id": "quant_documentation",
                "risk_tier": "draft_only",
                "operation": "plan",
                "preflight_required": False,
                "requires_confirmation": True,
                "action_input": {"package_summary": "Safe Credit PD package summary."},
                "expected_artifacts": [],
                "validation_checks": ["capability_known", "policy_allowed", "confirmation_required"],
            },
            {
                "step_id": "step_5",
                "title": "Validate monitoring bundle",
                "capability_id": "quant_monitoring.validate_bundle",
                "app_id": "quant_monitoring",
                "risk_tier": "workflow_preflight",
                "operation": "plan",
                "preflight_required": True,
                "requires_confirmation": False,
                "action_input": {"bundle_summary": "Safe Credit PD monitoring bundle summary."},
                "expected_artifacts": [],
                "validation_checks": ["capability_known", "policy_allowed", "plan_only"],
            },
        ],
    }
