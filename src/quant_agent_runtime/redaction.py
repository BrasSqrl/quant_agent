from __future__ import annotations

import re
from typing import Any

from quant_agent_runtime.models import RedactionSummary, ValidationIssue


_CAMEL_CASE_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

UNSAFE_KEYS = {
    "secret",
    "secrets",
    "credential",
    "credentials",
    "password",
    "token",
    "access_token",
    "access_tokens",
    "api_key",
    "api_keys",
    "auth_token",
    "auth_tokens",
    "authorization",
    "authorization_header",
    "bearer_token",
    "bearer_tokens",
    "client_secret",
    "client_secrets",
    "id_token",
    "id_tokens",
    "refresh_token",
    "refresh_tokens",
    "secret_key",
    "secret_keys",
    "passwords",
    "records",
    "rows",
    "table_records",
    "row_level_data",
    "raw_local_path",
    "raw_local_paths",
    "local_path",
    "local_paths",
    "raw_path",
    "raw_paths",
    "s3_uri",
    "s3_uris",
    "bucket_name",
    "bucket_names",
    "hidden_command",
    "hidden_commands",
    "command",
    "shell_command",
    "provider_prompt",
    "provider_response",
    "link",
    "links",
    "query",
    "queries",
    "frontend_url",
    "frontend_urls",
    "url",
    "urls",
}
UNSAFE_KEY_SUFFIXES = (
    "_api_key",
    "_api_keys",
    "_authorization",
    "_authorization_header",
    "_bearer_token",
    "_bearer_tokens",
    "_client_secret",
    "_client_secrets",
    "_credential",
    "_credentials",
    "_id_token",
    "_id_tokens",
    "_password",
    "_password_value",
    "_passwords",
    "_refresh_token",
    "_refresh_tokens",
    "_secret",
    "_secret_key",
    "_secret_keys",
    "_token",
)

UNSAFE_VALUE_PATTERNS = [
    re.compile(r"\b[A-Za-z]:[\\/][^\s]+"),
    re.compile(r"\bs3://[^\s]+", re.IGNORECASE),
    re.compile(r"\bhttps?://[^\s]+", re.IGNORECASE),
    re.compile(
        r"\b[A-Za-z0-9_-]*(?:api[_-]?key|access[_-]?token|auth[_-]?token|authorization|bearer[_-]?token|client[_-]?secret|credential|id[_-]?token|password|refresh[_-]?token|secret[_-]?key|secret|token)\s*[:=]\s*(?:Bearer\s+)?[^\s\"',}]+",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:rm\s+-rf|del\s+/[sq]|powershell\s+-|cmd\.exe|bash\s+-c|sh\s+-c|curl\s+|Invoke-WebRequest|Start-Process)\b",
        re.IGNORECASE,
    ),
]


def normalize_key(key: str) -> str:
    normalized = _CAMEL_CASE_BOUNDARY_RE.sub("_", key.strip())
    return normalized.lower().replace("-", "_").replace(" ", "_")


def is_unsafe_key(key: str) -> bool:
    normalized = normalize_key(key)
    return normalized in UNSAFE_KEYS or any(
        normalized.endswith(suffix) for suffix in UNSAFE_KEY_SUFFIXES
    )


def redact_text(text: str) -> tuple[str, bool]:
    redacted = text
    changed = False
    for pattern in UNSAFE_VALUE_PATTERNS:
        if pattern.search(redacted):
            redacted = pattern.sub("[redacted]", redacted)
            changed = True
    return redacted, changed


def sanitize_value(value: Any, path: str = "context") -> tuple[Any, RedactionSummary]:
    omitted_fields: list[str] = []
    redacted_fields: list[str] = []

    def visit(current: Any, current_path: str) -> Any:
        if isinstance(current, dict):
            sanitized: dict[str, Any] = {}
            for key, item in current.items():
                child_path = f"{current_path}.{key}"
                if is_unsafe_key(str(key)):
                    omitted_fields.append(child_path)
                    continue
                sanitized[key] = visit(item, child_path)
            return sanitized
        if isinstance(current, list):
            return [visit(item, f"{current_path}[]") for item in current]
        if isinstance(current, str):
            redacted, changed = redact_text(current)
            if changed:
                redacted_fields.append(current_path)
            return redacted
        return current

    sanitized_value = visit(value, path)
    summary = RedactionSummary(
        redacted=bool(omitted_fields or redacted_fields),
        omitted_fields=sorted(set(omitted_fields)),
        redacted_fields=sorted(set(redacted_fields)),
    )
    return sanitized_value, summary


def merge_redaction_summaries(*summaries: RedactionSummary) -> RedactionSummary:
    omitted: list[str] = []
    redacted: list[str] = []
    for summary in summaries:
        omitted.extend(summary.omitted_fields)
        redacted.extend(summary.redacted_fields)
    return RedactionSummary(
        redacted=bool(omitted or redacted),
        omitted_fields=sorted(set(omitted)),
        redacted_fields=sorted(set(redacted)),
    )


def find_unsafe_payload_issues(payload: Any, root: str = "payload") -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    def visit(current: Any, current_path: str) -> None:
        if isinstance(current, dict):
            for key, value in current.items():
                child_path = f"{current_path}.{key}"
                if is_unsafe_key(str(key)):
                    issues.append(
                        ValidationIssue(
                            code="unsafe_raw_field",
                            message=f"Unsafe field is not allowed at {child_path}.",
                        )
                    )
                visit(value, child_path)
        elif isinstance(current, list):
            for item in current:
                visit(item, f"{current_path}[]")
        elif isinstance(current, str):
            _, changed = redact_text(current)
            if changed:
                issues.append(
                    ValidationIssue(
                        code="unsafe_raw_value",
                        message=f"Unsafe raw value is not allowed at {current_path}.",
                    )
                )

    visit(payload, root)
    return issues
