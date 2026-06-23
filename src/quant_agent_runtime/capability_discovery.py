from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from quant_agent_runtime.app_clients import AgentAppClient, AppClientError
from quant_agent_runtime.contracts import QuantSuiteContractLoader
from quant_agent_runtime.models import CapabilityDefinition
from quant_agent_runtime.redaction import find_unsafe_payload_issues


DISCOVERABLE_AGENT_APPS = (
    "quant_data",
    "quant_studio",
    "quant_documentation",
    "quant_monitoring",
)
CAPABILITY_DISCOVERY_DATA_POLICY = "summaries_and_references_only"
SUPPORTED_EXECUTION_CAPABILITIES = frozenset(
    {
        "quant_studio.prepare_model_config_draft",
        "quant_documentation.create_draft_workspace",
    }
)


@dataclass(frozen=True)
class CapabilityDiscoveryResult:
    supported_preflight_capabilities: list[str]
    supported_execution_capabilities: list[str]
    diagnostics: dict[str, Any]

    def supports_preflight(self, capability_id: str) -> bool:
        return capability_id in set(self.supported_preflight_capabilities)

    def supports_execution(self, capability_id: str) -> bool:
        return capability_id in set(self.supported_execution_capabilities)

    def app_is_unavailable(self, app_id: str) -> bool:
        return app_id in set(_string_list(self.diagnostics.get("unavailable_apps")))


class CapabilityDiscoveryService:
    def __init__(
        self,
        *,
        contract_loader: QuantSuiteContractLoader,
        app_client: AgentAppClient,
        app_ids: Iterable[str] = DISCOVERABLE_AGENT_APPS,
    ) -> None:
        self._contract_loader = contract_loader
        self._app_client = app_client
        self._app_ids = tuple(app_ids)

    def discover(
        self,
        canonical_capabilities: list[CapabilityDefinition] | None = None,
    ) -> CapabilityDiscoveryResult:
        capabilities = canonical_capabilities
        if capabilities is None:
            capabilities = self._contract_loader.load_agent_capabilities()
        canonical_by_id = {capability.capability_id: capability for capability in capabilities}

        app_statuses: list[dict[str, Any]] = []
        top_level_warnings: list[dict[str, str]] = []
        supported_ids: list[str] = []
        supported_execution_ids: list[str] = []
        unsupported_ids: list[str] = []
        discovered_ids: list[str] = []
        discovered_apps: list[str] = []
        unavailable_apps: list[str] = []

        for app_id in self._app_ids:
            status = _empty_app_status(app_id)
            app_statuses.append(status)
            try:
                payload = self._app_client.discover_capabilities(app_id=app_id)
            except AppClientError as exc:
                unavailable_apps.append(app_id)
                warning = _warning(
                    "app_capability_discovery_unavailable",
                    str(exc),
                    app_id=app_id,
                )
                status["warnings"].append(warning)
                top_level_warnings.append(warning)
                continue
            except Exception:
                unavailable_apps.append(app_id)
                warning = _warning(
                    "app_capability_discovery_failed",
                    "Capability discovery failed before a safe response could be read.",
                    app_id=app_id,
                )
                status["warnings"].append(warning)
                top_level_warnings.append(warning)
                continue

            payload_issues = find_unsafe_payload_issues(
                payload,
                root=f"capability_discovery.{app_id}",
            )
            if payload_issues:
                unavailable_apps.append(app_id)
                warning = _warning(
                    "unsafe_capability_discovery_payload",
                    "Capability discovery payload contained unsafe fields or values and was ignored.",
                    app_id=app_id,
                )
                status["warnings"].append(warning)
                top_level_warnings.append(warning)
                continue

            validation_warning = _validate_discovery_payload(payload, app_id)
            if validation_warning is not None:
                unavailable_apps.append(app_id)
                status["warnings"].append(validation_warning)
                top_level_warnings.append(validation_warning)
                continue

            status["available"] = True
            discovered_apps.append(app_id)
            capabilities_payload = payload.get("capabilities", [])
            assert isinstance(capabilities_payload, list)

            for raw_capability in capabilities_payload:
                warning: dict[str, str] | None = None
                if not isinstance(raw_capability, dict):
                    warning = _warning(
                        "malformed_capability_entry",
                        "A discovered capability entry was not an object and was ignored.",
                        app_id=app_id,
                    )
                    status["warnings"].append(warning)
                    top_level_warnings.append(warning)
                    continue

                capability_id = _safe_text(raw_capability.get("capability_id"))
                if capability_id:
                    _append_unique(discovered_ids, capability_id)
                    _append_unique(status["discovered_capability_ids"], capability_id)
                else:
                    warning = _warning(
                        "malformed_capability_entry",
                        "A discovered capability entry did not include a capability_id and was ignored.",
                        app_id=app_id,
                    )
                    status["warnings"].append(warning)
                    top_level_warnings.append(warning)
                    continue

                canonical = canonical_by_id.get(capability_id)
                warning = _capability_reconciliation_warning(
                    raw_capability,
                    canonical,
                    expected_app_id=app_id,
                )
                if warning is not None:
                    _append_unique(unsupported_ids, capability_id)
                    _append_unique(status["unsupported_capability_ids"], capability_id)
                    status["warnings"].append(warning)
                    top_level_warnings.append(warning)
                    continue

                assert canonical is not None
                _append_unique(status["supported_capability_ids"], capability_id)
                if canonical.preflight_required:
                    _append_unique(supported_ids, capability_id)
                if _execution_capability_supported(raw_capability, canonical):
                    _append_unique(supported_execution_ids, capability_id)

            discovered_for_app = set(_string_list(status["discovered_capability_ids"]))
            for canonical in capabilities:
                if canonical.app_id != app_id:
                    continue
                if not canonical.enabled:
                    continue
                if canonical.capability_id in discovered_for_app:
                    continue
                if canonical.preflight_required:
                    warning = _warning(
                        "canonical_capability_not_advertised",
                        "A canonical app-owned preflight capability is not currently advertised by its app.",
                        app_id=app_id,
                        capability_id=canonical.capability_id,
                    )
                elif canonical.capability_id in SUPPORTED_EXECUTION_CAPABILITIES:
                    warning = _warning(
                        "canonical_execution_capability_not_advertised",
                        "A canonical app-owned execution capability is not currently advertised by its app.",
                        app_id=app_id,
                        capability_id=canonical.capability_id,
                    )
                else:
                    continue
                _append_unique(unsupported_ids, canonical.capability_id)
                _append_unique(status["unsupported_capability_ids"], canonical.capability_id)
                status["warnings"].append(warning)
                top_level_warnings.append(warning)

        canonical_order = [capability.capability_id for capability in capabilities]
        ordered_supported_ids = [
            capability_id for capability_id in canonical_order if capability_id in set(supported_ids)
        ]
        ordered_supported_execution_ids = [
            capability_id
            for capability_id in canonical_order
            if capability_id in set(supported_execution_ids)
        ]
        diagnostics = {
            "schema_version": "1.0",
            "data_policy": CAPABILITY_DISCOVERY_DATA_POLICY,
            "discovery_mode": "app_owned_capability_intersection",
            "discovered_apps": discovered_apps,
            "unavailable_apps": unavailable_apps,
            "discovered_capability_ids": discovered_ids,
            "unsupported_capability_ids": unsupported_ids,
            "supported_preflight_capabilities": ordered_supported_ids,
            "supported_execution_capabilities": ordered_supported_execution_ids,
            "reconciliation_warnings": top_level_warnings,
            "app_statuses": app_statuses,
        }
        return CapabilityDiscoveryResult(
            supported_preflight_capabilities=ordered_supported_ids,
            supported_execution_capabilities=ordered_supported_execution_ids,
            diagnostics=diagnostics,
        )


def _empty_app_status(app_id: str) -> dict[str, Any]:
    return {
        "app_id": app_id,
        "endpoint": f"{app_id}:/api/agent/capabilities",
        "available": False,
        "discovered_capability_ids": [],
        "supported_capability_ids": [],
        "unsupported_capability_ids": [],
        "warnings": [],
    }


def _validate_discovery_payload(payload: dict[str, Any], app_id: str) -> dict[str, str] | None:
    if payload.get("data_policy") != CAPABILITY_DISCOVERY_DATA_POLICY:
        return _warning(
            "invalid_capability_discovery_policy",
            "Capability discovery must use summaries-and-references-only data policy.",
            app_id=app_id,
        )
    if payload.get("app_id") != app_id:
        return _warning(
            "capability_discovery_app_mismatch",
            "Capability discovery app_id did not match the configured app client.",
            app_id=app_id,
        )
    if not isinstance(payload.get("capabilities"), list):
        return _warning(
            "malformed_capability_discovery_payload",
            "Capability discovery payload must include a capabilities list.",
            app_id=app_id,
        )
    return None


def _capability_reconciliation_warning(
    discovered: dict[str, Any],
    canonical: CapabilityDefinition | None,
    *,
    expected_app_id: str,
) -> dict[str, str] | None:
    capability_id = _safe_text(discovered.get("capability_id"))
    discovered_app_id = _safe_text(discovered.get("app_id"))
    if canonical is None:
        return _warning(
            "missing_canonical_capability",
            "Discovered capability is not present in the canonical Quant Suite registry.",
            app_id=expected_app_id,
            capability_id=capability_id,
        )
    if discovered_app_id != expected_app_id:
        return _warning(
            "capability_app_mismatch",
            "Discovered capability app_id did not match the app discovery endpoint.",
            app_id=expected_app_id,
            capability_id=capability_id,
        )
    if canonical.app_id != discovered_app_id:
        return _warning(
            "canonical_app_mismatch",
            "Discovered capability app_id did not match the canonical registry.",
            app_id=expected_app_id,
            capability_id=capability_id,
        )
    if not canonical.enabled:
        return _warning(
            "canonical_capability_disabled",
            "Canonical capability is disabled and cannot be advertised for preflight.",
            app_id=expected_app_id,
            capability_id=capability_id,
        )
    if discovered.get("enabled", True) is not True:
        return _warning(
            "discovered_capability_disabled",
            "Discovered capability is disabled by the owning app.",
            app_id=expected_app_id,
            capability_id=capability_id,
        )
    if _safe_text(discovered.get("risk_tier")) != canonical.risk_tier.value:
        return _warning(
            "capability_risk_tier_mismatch",
            "Discovered capability risk tier did not match the canonical registry.",
            app_id=expected_app_id,
            capability_id=capability_id,
        )
    if bool(discovered.get("preflight_required")) != canonical.preflight_required:
        return _warning(
            "capability_preflight_policy_mismatch",
            "Discovered capability preflight policy did not match the canonical registry.",
            app_id=expected_app_id,
            capability_id=capability_id,
        )
    if bool(discovered.get("confirmation_required", canonical.confirmation_required)) != canonical.confirmation_required:
        return _warning(
            "capability_confirmation_policy_mismatch",
            "Discovered capability confirmation policy did not match the canonical registry.",
            app_id=expected_app_id,
            capability_id=capability_id,
        )
    if not canonical.preflight_required and not _execution_capability_supported(discovered, canonical):
        return _warning(
            "capability_not_preflight_or_execution_capable",
            "Discovered capability is canonical but is not a supported app-owned preflight or execution capability.",
            app_id=expected_app_id,
            capability_id=capability_id,
        )
    return None


def _execution_capability_supported(
    discovered: dict[str, Any],
    canonical: CapabilityDefinition,
) -> bool:
    return (
        canonical.capability_id in SUPPORTED_EXECUTION_CAPABILITIES
        and canonical.confirmation_required
        and not canonical.preflight_required
        and discovered.get("execution_supported") is True
    )


def _warning(
    code: str,
    message: str,
    *,
    app_id: str,
    capability_id: str | None = None,
) -> dict[str, str]:
    payload = {
        "code": code,
        "message": message,
        "app_id": app_id,
    }
    if capability_id:
        payload["capability_id"] = capability_id
    return payload


def _append_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def _string_list(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _safe_text(value: Any) -> str:
    return value if isinstance(value, str) else ""
