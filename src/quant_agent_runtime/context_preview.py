from __future__ import annotations

import json
from typing import Any

from quant_agent_runtime.models import ContextPreview, RedactionSummary
from quant_agent_runtime.redaction import is_unsafe_key, normalize_key, redact_text


ROW_LEVEL_KEYS = {
    "records",
    "rows",
    "table_records",
    "row_level_data",
}


def build_context_preview(
    context: dict[str, Any],
    redaction_summary: RedactionSummary | None = None,
    context_sources: list[str] | None = None,
    warnings: list[str] | None = None,
) -> ContextPreview:
    summary = redaction_summary or RedactionSummary()
    omitted_sensitive_fields, omitted_row_level_fields = _partition_omissions(summary)
    preview_warnings = list(warnings or [])
    if summary.redacted:
        preview_warnings.append("Some context fields were omitted or redacted before planning.")

    return ContextPreview(
        context=context,
        context_sources=context_sources or _default_context_sources(context),
        warnings=_dedupe(preview_warnings),
        omitted_sensitive_fields=omitted_sensitive_fields,
        omitted_row_level_fields=omitted_row_level_fields,
        context_char_count=_context_char_count(context),
    )


def inspect_context_omissions(value: Any, root: str = "context") -> RedactionSummary:
    omitted_fields: list[str] = []
    redacted_fields: list[str] = []

    def visit(current: Any, current_path: str) -> None:
        if isinstance(current, dict):
            for key, item in current.items():
                child_path = f"{current_path}.{key}"
                if is_unsafe_key(str(key)):
                    omitted_fields.append(child_path)
                    continue
                visit(item, child_path)
        elif isinstance(current, list):
            for item in current:
                visit(item, f"{current_path}[]")
        elif isinstance(current, str):
            _, changed = redact_text(current)
            if changed:
                redacted_fields.append(current_path)

    visit(value, root)
    return RedactionSummary(
        redacted=bool(omitted_fields or redacted_fields),
        omitted_fields=sorted(set(omitted_fields)),
        redacted_fields=sorted(set(redacted_fields)),
    )


def _partition_omissions(summary: RedactionSummary) -> tuple[list[str], list[str]]:
    sensitive_fields: list[str] = []
    row_level_fields: list[str] = []

    for path in summary.omitted_fields:
        label = _field_label(path)
        if _is_row_level_label(label):
            row_level_fields.append(label)
        else:
            sensitive_fields.append(label)

    for path in summary.redacted_fields:
        sensitive_fields.append(_field_label(path))

    return sorted(set(sensitive_fields)), sorted(set(row_level_fields))


def _field_label(path: str) -> str:
    label = path
    for prefix in ("context_summary.", "lifecycle_manifest.", "lifecycle_context."):
        if label.startswith(prefix):
            label = label.removeprefix(prefix)
    return label.replace("[]", "").strip(".")


def _is_row_level_label(label: str) -> bool:
    parts = [normalize_key(part) for part in label.replace("[]", "").split(".")]
    return any(part in ROW_LEVEL_KEYS for part in parts)


def _default_context_sources(context: dict[str, Any]) -> list[str]:
    if not context:
        return []
    return [f"context_summary.{key}" for key in sorted(context)]


def _context_char_count(context: dict[str, Any]) -> int:
    encoded = json.dumps(context, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return len(encoded)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped
