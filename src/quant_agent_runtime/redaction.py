from __future__ import annotations

import re
from typing import Any

from quant_agent_runtime.models import RedactionSummary, ValidationIssue


UNSAFE_KEYS = {
    "secret",
    "secrets",
    "credential",
    "credentials",
    "password",
    "token",
    "api_key",
    "authorization",
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

UNSAFE_VALUE_PATTERNS = [
    re.compile(r"\b[A-Za-z]:[\\/][^\s]+"),
    re.compile(r"\bs3://[^\s]+", re.IGNORECASE),
    re.compile(r"\bhttps?://[^\s]+", re.IGNORECASE),
]


def normalize_key(key: str) -> str:
    return key.strip().lower().replace("-", "_")


def is_unsafe_key(key: str) -> bool:
    return normalize_key(key) in UNSAFE_KEYS


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
