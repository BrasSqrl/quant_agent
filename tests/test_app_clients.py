from __future__ import annotations

from io import BytesIO
from urllib.error import HTTPError

import quant_agent_runtime.app_clients as app_clients


class _Body(BytesIO):
    def close(self) -> None:
        pass


def _client() -> app_clients.LocalAgentAppClient:
    return app_clients.LocalAgentAppClient(
        quant_data_base_url="http://127.0.0.1:8830",
        quant_studio_base_url="http://127.0.0.1:8810",
        quant_documentation_base_url="http://127.0.0.1:8840",
        quant_monitoring_base_url="http://127.0.0.1:8820",
    )


def _raw_http_error() -> HTTPError:
    return HTTPError(
        url="http://127.0.0.1:8810/api/agent/actions/test/preflight",
        code=500,
        msg="boom",
        hdrs=None,
        fp=_Body(
            b'{"detail":"failed at C:\\\\Users\\\\matth\\\\secret.csv with https://example.com/token",'
            b'"api_key":"sk-test-secret"}'
        ),
    )


def test_preflight_http_error_detail_is_redacted(monkeypatch) -> None:
    def fake_urlopen(*_args, **_kwargs):
        raise _raw_http_error()

    monkeypatch.setattr(app_clients, "urlopen", fake_urlopen)

    try:
        _client().create_preflight(app_id="quant_studio", capability_id="test", payload={})
    except app_clients.AppClientError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected AppClientError")

    assert "HTTP 500" in message
    assert "[redacted]" in message
    assert "C:\\Users" not in message
    assert "https://example.com" not in message
    assert "sk-test-secret" not in message
    assert "api_key" not in message


def test_execution_http_error_detail_is_redacted(monkeypatch) -> None:
    def fake_urlopen(*_args, **_kwargs):
        raise _raw_http_error()

    monkeypatch.setattr(app_clients, "urlopen", fake_urlopen)

    try:
        _client().execute_action(app_id="quant_studio", capability_id="test", payload={})
    except app_clients.AppClientError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected AppClientError")

    assert "HTTP 500" in message
    assert "[redacted]" in message
    assert "C:\\Users" not in message
    assert "https://example.com" not in message
    assert "sk-test-secret" not in message
    assert "api_key" not in message


def test_app_client_error_sanitizes_direct_secret_assignments() -> None:
    error = app_clients.AppClientError("provider failed with token=abc123")

    assert "abc123" not in str(error)
    assert "[redacted]" in str(error)

    bearer_error = app_clients.AppClientError("provider failed with Authorization: Bearer sk-test")

    assert "sk-test" not in str(bearer_error)
    assert "[redacted]" in str(bearer_error)
