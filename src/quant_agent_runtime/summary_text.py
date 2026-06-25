from __future__ import annotations

from typing import Any

from quant_agent_runtime.redaction import is_unsafe_key, redact_text


_PREFERRED_SUMMARY_KEYS = (
    "label",
    "summary",
    "state",
    "status",
    "reference_type",
    "reference_id",
    "app_id",
    "source_app",
    "lifecycle_id",
)


def compact_safe_summary_text(value: Any, *, label: str, max_chars: int = 240) -> str | None:
    """Convert structured safe context into compact text for app APIs that require strings."""

    if isinstance(value, str):
        clean = " ".join(value.strip().split())
        if clean.lower().startswith(f"{label.lower()}:"):
            redacted, _changed = redact_text(clean)
            return redacted[:max_chars] if redacted else None

    parts: list[str] = []
    _collect_summary_parts(value, parts)
    if not parts:
        return None
    body = "; ".join(_unique(parts))
    if not body:
        return None
    text = f"{label}: {body}"
    return text[:max_chars]


def has_meaningful_summary(value: Any) -> bool:
    return compact_safe_summary_text(value, label="summary") is not None


def _collect_summary_parts(value: Any, parts: list[str]) -> None:
    if isinstance(value, str):
        clean = " ".join(value.strip().split())
        if not clean or clean == "[missing]":
            return
        redacted, _changed = redact_text(clean)
        if redacted:
            parts.append(redacted[:120])
        return
    if isinstance(value, list):
        for item in value[:8]:
            _collect_summary_parts(item, parts)
        return
    if isinstance(value, dict):
        for key in _PREFERRED_SUMMARY_KEYS:
            if key in value and not is_unsafe_key(key):
                _collect_summary_parts(value.get(key), parts)
        for key, item in value.items():
            if key in _PREFERRED_SUMMARY_KEYS or is_unsafe_key(str(key)):
                continue
            if isinstance(item, (dict, list)):
                _collect_summary_parts(item, parts)
            elif isinstance(item, int) and item > 0 and str(key).endswith("_count"):
                parts.append(f"{key}: {item}")
        return


def _unique(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result
