from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quant_agent_runtime.redaction import sanitize_value


class LifecycleContextBuilder:
    """Build safe planner context from lifecycle manifests."""

    def build_from_path(self, manifest_path: str | Path) -> dict[str, object]:
        path = Path(manifest_path)
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict):
            raise ValueError("Lifecycle manifest must be a JSON object.")
        return self.build_from_manifest(manifest)

    def build_from_manifest(self, manifest: Mapping[str, Any]) -> dict[str, object]:
        context: dict[str, object] = {
            "lifecycle_summary": self._build_lifecycle_summary(manifest),
            "source_summary": self._build_collection_summary(
                manifest,
                ("source_references", "eda_packages"),
            ),
            "target_summary": self._build_collection_summary(
                manifest,
                ("studio_runs",),
                singletons=("champion_model",),
            ),
            "package_summary": self._build_collection_summary(
                manifest,
                ("documentation_packages", "documentation_drafts"),
            ),
            "bundle_summary": self._build_collection_summary(
                manifest,
                ("monitoring_bundles", "monitoring_runs", "feedback_signals"),
            ),
            "app_availability": self._build_app_availability(manifest),
        }
        sanitized, _ = sanitize_value(context, path="lifecycle_context")
        if not isinstance(sanitized, dict):
            raise ValueError("Lifecycle context builder produced invalid context.")
        return sanitized

    def _build_lifecycle_summary(self, manifest: Mapping[str, Any]) -> str:
        parts: list[str] = []
        label = self._safe_scalar(manifest.get("lifecycle_label"))
        state = self._safe_scalar(manifest.get("state"))
        summary = self._safe_scalar(manifest.get("summary"))
        blocking_app = self._safe_scalar(manifest.get("blocking_app"))
        next_action = self._safe_scalar(manifest.get("next_action_label"))
        next_action_app = self._safe_scalar(manifest.get("next_action_app"))

        if label and state:
            parts.append(f"{label} is in {state} state.")
        elif label:
            parts.append(label)
        elif state:
            parts.append(f"Lifecycle is in {state} state.")

        if summary:
            parts.append(summary)

        sample_workspace = manifest.get("sample_workspace")
        if isinstance(sample_workspace, Mapping):
            workspace_id = self._safe_scalar(sample_workspace.get("sample_workspace_id"))
            sample_owned = sample_workspace.get("sample_owned") is True
            if sample_workspace.get("sample_workspace") is True and workspace_id and sample_owned:
                parts.append(f"Sample workspace {workspace_id} is sample-owned.")

        if blocking_app:
            parts.append(f"Blocking app: {blocking_app}.")
        if next_action and next_action_app:
            parts.append(f"Next action: {next_action} in {next_action_app}.")
        elif next_action:
            parts.append(f"Next action: {next_action}.")

        return " ".join(parts)

    def _build_collection_summary(
        self,
        manifest: Mapping[str, Any],
        collection_names: tuple[str, ...],
        singletons: tuple[str, ...] = (),
    ) -> str:
        item_summaries: list[str] = []
        for collection_name in collection_names:
            collection = manifest.get(collection_name)
            if not isinstance(collection, list):
                continue
            for item in collection:
                if isinstance(item, Mapping):
                    summary = self._build_item_summary(item)
                    if summary:
                        item_summaries.append(summary)

        for singleton_name in singletons:
            item = manifest.get(singleton_name)
            if isinstance(item, Mapping):
                summary = self._build_item_summary(item)
                if summary:
                    item_summaries.append(summary)

        return " ".join(item_summaries)

    def _build_item_summary(self, item: Mapping[str, Any]) -> str:
        label = self._safe_scalar(item.get("label"))
        status = self._safe_scalar(item.get("status"))
        summary = self._safe_scalar(item.get("summary"))
        app = self._safe_scalar(item.get("app"))
        reference_type = self._safe_scalar(item.get("reference_type"))
        run_type = self._safe_scalar(item.get("run_type"))
        signal_type = self._safe_scalar(item.get("signal_type"))
        severity = self._safe_scalar(item.get("severity"))

        heading_parts = [part for part in (app, reference_type or run_type or signal_type) if part]
        heading = " ".join(heading_parts)

        details: list[str] = []
        if label and status:
            details.append(f"{label} ({status})")
        elif label:
            details.append(label)
        elif status:
            details.append(f"Status {status}")

        if summary:
            details.append(summary)
        if severity:
            details.append(f"Severity {severity}.")

        if not details:
            return heading
        if heading:
            return f"{heading}: {' '.join(details)}"
        return " ".join(details)

    def _build_app_availability(self, manifest: Mapping[str, Any]) -> dict[str, object]:
        source_count = self._count_items(manifest, ("source_references", "eda_packages"))
        target_count = self._count_items(manifest, ("studio_runs",)) + int(
            isinstance(manifest.get("champion_model"), Mapping)
        )
        package_count = self._count_items(
            manifest,
            ("documentation_packages", "documentation_drafts"),
        )
        bundle_count = self._count_items(
            manifest,
            ("monitoring_bundles", "monitoring_runs", "feedback_signals"),
        )
        return {
            "quant_data": {
                "available": source_count > 0,
                "summary_count": source_count,
            },
            "quant_studio": {
                "available": target_count > 0,
                "summary_count": target_count,
            },
            "quant_documentation": {
                "available": package_count > 0,
                "summary_count": package_count,
            },
            "quant_monitoring": {
                "available": bundle_count > 0,
                "summary_count": bundle_count,
            },
        }

    def _count_items(self, manifest: Mapping[str, Any], collection_names: tuple[str, ...]) -> int:
        count = 0
        for collection_name in collection_names:
            collection = manifest.get(collection_name)
            if isinstance(collection, list):
                count += sum(1 for item in collection if isinstance(item, Mapping))
        return count

    def _safe_scalar(self, value: Any) -> str:
        if isinstance(value, str):
            return " ".join(value.strip().split())
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        return ""


def build_lifecycle_context_summary(manifest: Mapping[str, Any]) -> dict[str, object]:
    return LifecycleContextBuilder().build_from_manifest(manifest)


def build_lifecycle_context_summary_from_path(manifest_path: str | Path) -> dict[str, object]:
    return LifecycleContextBuilder().build_from_path(manifest_path)
