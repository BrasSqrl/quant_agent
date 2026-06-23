from __future__ import annotations

import json
import os
import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_agent_runtime.models import LedgerEntry, LedgerIntegrity, LedgerIntegritySummary
from quant_agent_runtime.redaction import find_unsafe_payload_issues, sanitize_value

LEDGER_CONTRACT_SCHEMA = "agent_execution_ledger.v1.schema.json"
LEDGER_INTEGRITY_ALGORITHM = "sha256"
LEDGER_INTEGRITY_SUPPORT_LEVEL = "local_json_latest_with_append_only_integrity_journal"


class InMemoryLedger:
    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []

    def append(self, entry: LedgerEntry) -> LedgerEntry:
        sanitized_entry = self._validated_entry(entry)
        recorded_entry = self._after_write(sanitized_entry)
        self._entries.append(recorded_entry)
        return recorded_entry

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
            "integrity_support_level": "not_available",
            "legacy_unverified_entry_count": 0,
            "tampered_entry_count": 0,
            "journal_error_count": 0,
        }

    def integrity_summary(self, run_id: str | None = None) -> LedgerIntegritySummary:
        entry = self.get(run_id) if run_id else None
        if entry is not None and entry.ledger_integrity is not None:
            return LedgerIntegritySummary.model_validate(entry.ledger_integrity.model_dump(mode="json"))
        return LedgerIntegritySummary(
            status="not_available",
            diagnostics=[
                {
                    "code": "ledger_integrity_not_available",
                    "message": "The current ledger store does not provide file-backed integrity metadata.",
                }
            ],
        )

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
            recorded_entry = self._after_write(sanitized_entry)
            self._entries[index] = recorded_entry
            return recorded_entry
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

    def _after_write(self, entry: LedgerEntry) -> LedgerEntry:
        return entry


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
        self._legacy_unverified_entry_count = 0
        self._tampered_entry_count = 0
        self._journal_error_count = 0
        self._journal_dir = self._ledger_dir / "integrity_journals"
        self._ledger_dir.mkdir(parents=True, exist_ok=True)
        self._journal_dir.mkdir(parents=True, exist_ok=True)
        self._load_existing_entries()

    def diagnostics(self) -> dict[str, Any]:
        return {
            "storage_mode": "local_json_file_backed",
            "loaded_entry_count": self._loaded_entry_count,
            "invalid_entry_count": self._invalid_entry_count,
            "write_error_count": self._write_error_count,
            "integrity_support_level": LEDGER_INTEGRITY_SUPPORT_LEVEL,
            "legacy_unverified_entry_count": self._legacy_unverified_entry_count,
            "tampered_entry_count": self._tampered_entry_count,
            "journal_error_count": self._journal_error_count,
        }

    def _after_write(self, entry: LedgerEntry) -> LedgerEntry:
        stamped_entry = self._entry_with_integrity(entry)
        payload = stamped_entry.model_dump(mode="json")
        self._validate_payload(payload)
        path = self._path_for_run(entry.run_id)
        temp_path = path.with_suffix(".tmp")
        try:
            with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            temp_path.replace(path)
            self._append_integrity_journal(stamped_entry)
        except OSError as exc:
            self._write_error_count += 1
            raise ValueError("Ledger entry could not be persisted.") from exc
        return stamped_entry

    def _load_existing_entries(self) -> None:
        entries: list[LedgerEntry] = []
        for path in sorted(self._ledger_dir.glob("*.json")):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if not isinstance(payload, dict):
                    raise ValueError("Ledger file must contain a JSON object.")
                self._validate_payload(payload)
                entry = self._entry_loaded_with_integrity(payload)
            except Exception:
                self._invalid_entry_count += 1
                continue
            entries.append(entry)
        self._entries = entries
        self._loaded_entry_count = len(entries)

    def _entry_loaded_with_integrity(self, payload: dict[str, Any]) -> LedgerEntry:
        base_entry = self._validated_entry(LedgerEntry.model_validate(payload))
        integrity = payload.get("ledger_integrity")
        if not isinstance(integrity, dict):
            self._legacy_unverified_entry_count += 1
            legacy_integrity = {
                "status": "legacy_unverified",
                "algorithm": LEDGER_INTEGRITY_ALGORITHM,
                "sequence_number": 0,
                "previous_hash": None,
                "payload_hash": _ledger_payload_hash(payload),
                "recorded_at_utc": None,
                "contract_schema": LEDGER_CONTRACT_SCHEMA,
                "journal_consistent": False,
                "diagnostics": [
                    {
                        "code": "legacy_ledger_without_integrity",
                        "message": "The ledger predates local integrity metadata.",
                    }
                ],
            }
            return base_entry.model_copy(
                update={"ledger_integrity": LedgerIntegrity.model_validate(legacy_integrity)},
                deep=True,
            )

        expected_hash = integrity.get("payload_hash")
        actual_hash = _ledger_payload_hash(payload)
        if not isinstance(expected_hash, str) or expected_hash != actual_hash:
            self._tampered_entry_count += 1
            raise ValueError("Ledger integrity hash mismatch.")
        if not self._journal_matches(base_entry.run_id, integrity):
            self._journal_error_count += 1
            raise ValueError("Ledger integrity journal mismatch.")
        verified = {
            **integrity,
            "status": "verified",
            "journal_consistent": True,
            "diagnostics": list(integrity.get("diagnostics") or []),
        }
        return base_entry.model_copy(
            update={"ledger_integrity": LedgerIntegrity.model_validate(verified)},
            deep=True,
        )

    def _validate_payload(self, payload: dict[str, Any]) -> None:
        if self._validate_contract is not None:
            self._validate_contract(payload, LEDGER_CONTRACT_SCHEMA)

    def _path_for_run(self, run_id: str) -> Path:
        return self._ledger_dir / f"{_safe_run_id(run_id)}.json"

    def _journal_path_for_run(self, run_id: str) -> Path:
        return self._journal_dir / f"{_safe_run_id(run_id)}.jsonl"

    def _entry_with_integrity(self, entry: LedgerEntry) -> LedgerEntry:
        previous = self.get(entry.run_id)
        previous_integrity = previous.ledger_integrity if previous is not None else None
        previous_hash = (
            previous_integrity.payload_hash
            if previous_integrity is not None and previous_integrity.payload_hash
            else None
        )
        previous_sequence = (
            previous_integrity.sequence_number
            if previous_integrity is not None and previous_integrity.sequence_number > 0
            else 0
        )
        payload = entry.model_dump(mode="json")
        payload_hash = _ledger_payload_hash(payload)
        integrity = {
            "status": "verified",
            "algorithm": LEDGER_INTEGRITY_ALGORITHM,
            "sequence_number": previous_sequence + 1,
            "previous_hash": previous_hash,
            "payload_hash": payload_hash,
            "recorded_at_utc": _utc_now_label(),
            "contract_schema": LEDGER_CONTRACT_SCHEMA,
            "journal_consistent": True,
            "diagnostics": [],
        }
        return entry.model_copy(
            update={"ledger_integrity": LedgerIntegrity.model_validate(integrity)},
            deep=True,
        )

    def _append_integrity_journal(self, entry: LedgerEntry) -> None:
        if entry.ledger_integrity is None:
            return
        record = {
            "run_id": entry.run_id,
            **entry.ledger_integrity.model_dump(mode="json"),
        }
        path = self._journal_path_for_run(entry.run_id)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")

    def _journal_matches(self, run_id: str, integrity: dict[str, Any]) -> bool:
        path = self._journal_path_for_run(run_id)
        if not path.is_file():
            return False
        try:
            with path.open("r", encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
        except (OSError, json.JSONDecodeError):
            return False
        if not records:
            return False
        tail = records[-1]
        return (
            tail.get("run_id") == run_id
            and tail.get("payload_hash") == integrity.get("payload_hash")
            and tail.get("sequence_number") == integrity.get("sequence_number")
        )


def _safe_run_id(run_id: str) -> str:
    safe_run_id = "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in run_id
    ).strip("_")
    if not safe_run_id:
        safe_run_id = "run"
    return safe_run_id


def _ledger_payload_hash(payload: dict[str, Any]) -> str:
    canonical_payload = dict(payload)
    canonical_payload.pop("ledger_integrity", None)
    serialized = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _utc_now_label() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def default_ledger_dir() -> Path:
    configured = os.environ.get("QUANT_AGENT_LEDGER_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".quant_agent" / "ledgers"
