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

    def get(self, run_id: str) -> LedgerEntry | None:
        for entry in reversed(self._entries):
            if entry.run_id == run_id:
                return entry
        return None

    def append_preflight_record(self, run_id: str, record: dict[str, Any]) -> LedgerEntry:
        for index in range(len(self._entries) - 1, -1, -1):
            entry = self._entries[index]
            if entry.run_id != run_id:
                continue
            updated = entry.model_copy(
                update={
                    "preflight_records": [
                        *entry.preflight_records,
                        record,
                    ],
                },
                deep=True,
            )
            sanitized_entry = self._sanitize_entry(updated)
            issues = find_unsafe_payload_issues(
                sanitized_entry.model_dump(mode="json"),
                root="ledger",
            )
            if issues:
                raise ValueError("Ledger entry contains unsafe fields.")
            self._entries[index] = sanitized_entry
            return sanitized_entry
        raise KeyError(f"Unknown run_id: {run_id}")

    def append_confirmation_record(self, run_id: str, record: dict[str, Any]) -> LedgerEntry:
        for index in range(len(self._entries) - 1, -1, -1):
            entry = self._entries[index]
            if entry.run_id != run_id:
                continue
            updated = entry.model_copy(
                update={
                    "confirmation_records": [
                        *entry.confirmation_records,
                        record,
                    ],
                },
                deep=True,
            )
            sanitized_entry = self._sanitize_entry(updated)
            issues = find_unsafe_payload_issues(
                sanitized_entry.model_dump(mode="json"),
                root="ledger",
            )
            if issues:
                raise ValueError("Ledger entry contains unsafe fields.")
            self._entries[index] = sanitized_entry
            return sanitized_entry
        raise KeyError(f"Unknown run_id: {run_id}")

    def append_action_request(self, run_id: str, record: dict[str, Any]) -> LedgerEntry:
        for index in range(len(self._entries) - 1, -1, -1):
            entry = self._entries[index]
            if entry.run_id != run_id:
                continue
            updated = entry.model_copy(
                update={
                    "action_requests": [
                        *entry.action_requests,
                        record,
                    ],
                },
                deep=True,
            )
            sanitized_entry = self._sanitize_entry(updated)
            issues = find_unsafe_payload_issues(
                sanitized_entry.model_dump(mode="json"),
                root="ledger",
            )
            if issues:
                raise ValueError("Ledger entry contains unsafe fields.")
            self._entries[index] = sanitized_entry
            return sanitized_entry
        raise KeyError(f"Unknown run_id: {run_id}")

    def append_action_result(self, run_id: str, record: dict[str, Any]) -> LedgerEntry:
        for index in range(len(self._entries) - 1, -1, -1):
            entry = self._entries[index]
            if entry.run_id != run_id:
                continue
            updated = entry.model_copy(
                update={
                    "action_results": [
                        *entry.action_results,
                        record,
                    ],
                },
                deep=True,
            )
            sanitized_entry = self._sanitize_entry(updated)
            issues = find_unsafe_payload_issues(
                sanitized_entry.model_dump(mode="json"),
                root="ledger",
            )
            if issues:
                raise ValueError("Ledger entry contains unsafe fields.")
            self._entries[index] = sanitized_entry
            return sanitized_entry
        raise KeyError(f"Unknown run_id: {run_id}")

    def list_entries(self) -> list[LedgerEntry]:
        return list(self._entries)

    def _sanitize_entry(self, entry: LedgerEntry) -> LedgerEntry:
        payload: dict[str, Any] = entry.model_dump(mode="json")
        sanitized, _ = sanitize_value(payload, path="ledger")
        return LedgerEntry.model_validate(sanitized)
