from __future__ import annotations

from typing import Any

from quant_agent_runtime.models import LedgerEntry
from quant_agent_runtime.redaction import find_unsafe_payload_issues, sanitize_value


class InMemoryLedger:
    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []

    def append(self, entry: LedgerEntry) -> LedgerEntry:
        sanitized_entry = self._sanitize_entry(entry)
        issues = find_unsafe_payload_issues(
            sanitized_entry.model_dump(mode="json"),
            root="ledger",
        )
        if issues:
            raise ValueError("Ledger entry contains unsafe fields.")
        self._entries.append(sanitized_entry)
        return sanitized_entry

    def list_entries(self) -> list[LedgerEntry]:
        return list(self._entries)

    def _sanitize_entry(self, entry: LedgerEntry) -> LedgerEntry:
        payload: dict[str, Any] = entry.model_dump(mode="json")
        sanitized, _ = sanitize_value(payload, path="ledger")
        return LedgerEntry.model_validate(sanitized)
