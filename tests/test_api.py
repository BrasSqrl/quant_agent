from fastapi.testclient import TestClient

from quant_agent_runtime.api import create_app


def test_health_endpoint_works() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["plan_only_mode"] is True
    assert response.json()["execution_supported"] is False


def test_runtime_manifest_returns_supported_modes() -> None:
    client = TestClient(create_app())

    response = client.get("/runtime/manifest")

    assert response.status_code == 200
    manifest = response.json()
    assert manifest["plan_only_mode"] is True
    assert manifest["execution_supported"] is False
    assert "fake_provider" in manifest["supported_provider_modes"]
    assert "disabled_or_local_fallback" in manifest["supported_provider_modes"]
    assert "POST /plans" in manifest["supported_routes"]
    assert manifest["runtime_health_endpoint"] == "/health"
    assert manifest["execution_support_level"] == "not_supported"


def test_fake_provider_can_produce_valid_plan() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/plans",
        json={
            "user_goal": "Build a conservative PD scorecard plan.",
            "context_summary": {
                "lifecycle_summary": "Lifecycle exists.",
                "source_summary": "Development sample is registered.",
                "target_summary": "Default flag is the candidate target.",
                "package_summary": "No documentation package exists yet.",
                "bundle_summary": "Monitoring bundle is not available.",
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["validation"]["status"] == "valid"
    assert payload["plan"]["execution_permitted"] is False
    assert payload["ledger_recorded"] is True
    assert payload["plan"]["proposed_steps"]


def test_no_execution_endpoint_exists() -> None:
    client = TestClient(create_app())

    assert client.post("/runs", json={}).status_code == 404
    assert client.post("/execute", json={}).status_code == 404
