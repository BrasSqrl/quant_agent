from __future__ import annotations

from quant_agent_runtime.redaction import (
    find_unsafe_payload_issues,
    redact_text,
    sanitize_value,
)


def test_redact_text_removes_embedded_secret_assignments() -> None:
    redacted, changed = redact_text(
        "provider failed with api_key=sk-test-secret, access_token=access-secret, "
        "client_secret=client-secret, openai_api_key=openai-secret, "
        "and Authorization: Bearer bearer-secret"
    )

    assert changed is True
    assert "sk-test-secret" not in redacted
    assert "access-secret" not in redacted
    assert "client-secret" not in redacted
    assert "openai-secret" not in redacted
    assert "bearer-secret" not in redacted
    assert redacted.count("[redacted]") == 5


def test_find_unsafe_payload_issues_flags_embedded_secret_assignments() -> None:
    issues = find_unsafe_payload_issues(
        {"message": "planning note included access_token=abc123"},
        root="provider_output",
    )

    assert [issue.code for issue in issues] == ["unsafe_raw_value"]
    assert "provider_output.message" in issues[0].message


def test_sanitize_value_redacts_embedded_secret_assignments() -> None:
    sanitized, summary = sanitize_value(
        {"message": "planning note included password=hunter2"},
        path="context",
    )

    assert sanitized == {"message": "planning note included [redacted]"}
    assert summary.redacted is True
    assert summary.redacted_fields == ["context.message"]


def test_camel_case_secret_keys_are_omitted_without_dropping_count_fields() -> None:
    sanitized, summary = sanitize_value(
        {
            "apiKey": "sk-test-secret",
            "accessToken": "token-secret",
            "clientSecret": "client-secret",
            "authorizationHeader": "Bearer bearer-secret",
            "token_count": 42,
        },
        path="context",
    )

    assert sanitized == {"token_count": 42}
    assert summary.redacted is True
    assert summary.omitted_fields == [
        "context.accessToken",
        "context.apiKey",
        "context.authorizationHeader",
        "context.clientSecret",
    ]


def test_find_unsafe_payload_issues_flags_camel_case_secret_keys() -> None:
    issues = find_unsafe_payload_issues(
        {
            "apiKey": "sk-test-secret",
            "clientSecret": "client-secret",
            "token_count": 42,
        },
        root="payload",
    )

    assert [issue.code for issue in issues] == [
        "unsafe_raw_field",
        "unsafe_raw_field",
    ]
    assert "payload.apiKey" in issues[0].message
    assert "payload.clientSecret" in issues[1].message


def test_plural_secret_keys_are_omitted_without_dropping_token_usage_fields() -> None:
    sanitized, summary = sanitize_value(
        {
            "apiKeys": ["sk-test-secret"],
            "accessTokens": ["access-secret"],
            "clientSecrets": ["client-secret"],
            "secretKeys": ["secret-key"],
            "prompt_tokens": 128,
        },
        path="payload",
    )

    assert sanitized == {"prompt_tokens": 128}
    assert summary.redacted is True
    assert summary.omitted_fields == [
        "payload.accessTokens",
        "payload.apiKeys",
        "payload.clientSecrets",
        "payload.secretKeys",
    ]

    issues = find_unsafe_payload_issues(
        {
            "apiKeys": ["sk-test-secret"],
            "accessTokens": ["access-secret"],
            "prompt_tokens": 128,
        },
        root="payload",
    )
    assert [issue.code for issue in issues] == ["unsafe_raw_field", "unsafe_raw_field"]
