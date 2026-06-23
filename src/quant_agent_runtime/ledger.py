from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from quant_agent_runtime.models import LedgerEntry
from quant_agent_runtime.redaction import find_unsafe_payload_issues, sanitize_value

LEDGER_CONTRACT_SCHEMA = "agent_execution_ledger.v1.schema.json"


class InMemoryLedger:
    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []

    def append(self, entry: LedgerEntry) -> LedgerEntry:
        sanitized_entry = self._validated_entry(entry)
        self._after_write(sanitized_entry)
        self._entries.append(sanitized_entry)
        return sanitized_entry

    def get(self, run_id: str) -> LedgerEntry | None:
        for entry in reversed(self._entries):
            if entry.run_id == run_id:
                return entry
        return None

    def append_preflight_record(self, run_id: str, record: dict[str, Any]) -> LedgerEntry:
        return self._update_entry(
            run_id,
            lambda entry: entry.model_copy(
                update={
                    "preflight_records": [
                        *entry.preflight_records,
                        record,
                    ],
                },
                deep=True,
            ),
        )

    def append_confirmation_record(self, run_id: str, record: dict[str, Any]) -> LedgerEntry:
        return self._update_entry(
            run_id,
            lambda entry: entry.model_copy(
                update={
                    "confirmation_records": [
                        *entry.confirmation_records,
                        record,
                    ],
                },
                deep=True,
            ),
        )

    def append_action_request(self, run_id: str, record: dict[str, Any]) -> LedgerEntry:
        return self._update_entry(
            run_id,
            lambda entry: entry.model_copy(
                update={
                    "action_requests": [
                        *entry.action_requests,
                        record,
                    ],
                },
                deep=True,
            ),
        )

    def append_action_result(self, run_id: str, record: dict[str, Any]) -> LedgerEntry:
        return self._update_entry(
            run_id,
            lambda entry: entry.model_copy(
                update={
                    "action_results": [
                        *entry.action_results,
                        record,
                    ],
                },
                deep=True,
            ),
        )

    def append_cancellation_event(self, run_id: str, record: dict[str, Any]) -> LedgerEntry:
        return self._update_entry(
            run_id,
            lambda entry: entry.model_copy(
                update={
                    "cancellation_events": [
                        *entry.cancellation_events,
                        record,
                    ],
                    "final_status": "cancelled",
                },
                deep=True,
            ),
        )

    def append_recovery_event(self, run_id: str, record: dict[str, Any]) -> LedgerEntry:
        return self._update_entry(
            run_id,
            lambda entry: entry.model_copy(
                update={
                    "recovery_events": [
                        *entry.recovery_events,
                        record,
                    ],
                },
                deep=True,
            ),
        )

    def append_sample_reset_event(self, run_id: str, record: dict[str, Any]) -> LedgerEntry:
        return self._update_entry(
            run_id,
            lambda entry: entry.model_copy(
                update={
                    "recovery_events": [
                        *entry.recovery_events,
                        record,
                    ],
                    "final_status": "sample_reset",
                },
                deep=True,
            ),
        )

    def append_plan_revision_activation(
        self,
        run_id: str,
        record: dict[str, Any],
        *,
        child_run_id: str,
    ) -> LedgerEntry:
        return self._update_entry(
            run_id,
            lambda entry: entry.model_copy(
                update={
                    "recovery_events": [
                        *entry.recovery_events,
                        record,
                    ],
                    "child_run_ids": [
                        *entry.child_run_ids,
                        *([] if child_run_id in entry.child_run_ids else [child_run_id]),
                    ],
                },
                deep=True,
            ),
        )

    def list_entries(self) -> list[LedgerEntry]:
        return list(self._entries)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "storage_mode": "memory",
            "loaded_entry_count": len(self._entries),
            "invalid_entry_count": 0,
            "write_error_count": 0,
        }

    def _update_entry(
        self,
        run_id: str,
        update_entry: Callable[[LedgerEntry], LedgerEntry],
    ) -> LedgerEntry:
        for index in range(len(self._entries) - 1, -1, -1):
            entry = self._entries[index]
            if entry.run_id != run_id:
                continue
            sanitized_entry = self._validated_entry(update_entry(entry))
            self._after_write(sanitized_entry)
            self._entries[index] = sanitized_entry
            return sanitized_entry
        raise KeyError(f"Unknown run_id: {run_id}")

    def _sanitize_entry(self, entry: LedgerEntry) -> LedgerEntry:
        payload: dict[str, Any] = entry.model_dump(mode="json")
        sanitized, _ = sanitize_value(payload, path="ledger")
        return LedgerEntry.model_validate(sanitized)

    def _validated_entry(self, entry: LedgerEntry) -> LedgerEntry:
        sanitized_entry = self._sanitize_entry(entry)
        issues = find_unsafe_payload_issues(
            sanitized_entry.model_dump(mode="json"),
            root="ledger",
        )
        if issues:
            raise ValueError("Ledger entry contains unsafe fields.")
        return sanitized_entry

    def _after_write(self, entry: LedgerEntry) -> None:
        return None


class FileBackedLedger(InMemoryLedger):
    def __init__(
        self,
        ledger_dir: Path | None = None,
        *,
        validate_contract: Callable[[Any, str], None] | None = None,
    ) -> None:
        super().__init__()
        self._ledger_dir = ledger_dir or default_ledger_dir()
        self._validate_contract = validate_contract
        self._invalid_entry_count = 0
        self._write_error_count = 0
        self._loaded_entry_count = 0
        self._ledger_dir.mkdir(parents=True, exist_ok=True)
        self._load_existing_entries()

    def diagnostics(self) -> dict[str, Any]:
        return {
            "storage_mode": "local_json_file_backed",
            "loaded_entry_count": self._loaded_entry_count,
            "invalid_entry_count": self._invalid_entry_count,
            "write_error_count": self._write_error_count,
        }

    def _after_write(self, entry: LedgerEntry) -> None:
        payload = entry.model_dump(mode="json")
        self._validate_payload(payload)
        path = self._path_for_run(entry.run_id)
        temp_path = path.with_suffix(".tmp")
        try:
            with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            temp_path.replace(path)
        except OSError as exc:
            self._write_error_count += 1
            raise ValueError("Ledger entry could not be persisted.") from exc

    def _load_existing_entries(self) -> None:
        entries: list[LedgerEntry] = []
        for path in sorted(self._ledger_dir.glob("*.json")):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if not isinstance(payload, dict):
                    raise ValueError("Ledger file must contain a JSON object.")
                self._validate_payload(payload)
                entry = self._validated_entry(LedgerEntry.model_validate(payload))
            except Exception:
                self._invalid_entry_count += 1
                continue
            entries.append(entry)
        self._entries = entries
        self._loaded_entry_count = len(entries)

    def _validate_payload(self, payload: dict[str, Any]) -> None:
        if self._validate_contract is not None:
            self._validate_contract(payload, LEDGER_CONTRACT_SCHEMA)

    def _path_for_run(self, run_id: str) -> Path:
        safe_run_id = "".join(
            character if character.isalnum() or character in {"_", "-"} else "_"
            for character in run_id
        ).strip("_")
        if not safe_run_id:
            safe_run_id = "run"
        return self._ledger_dir / f"{safe_run_id}.json"


def default_ledger_dir() -> Path:
    configured = os.environ.get("QUANT_AGENT_LEDGER_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".quant_agent" / "ledgers"
