from __future__ import annotations

from quant_agent_runtime.models import CapabilityDefinition, RiskTier


def default_capabilities() -> list[CapabilityDefinition]:
    return [
        CapabilityDefinition(
            capability_id="quant_suite.inspect_lifecycle_context",
            app_id="quant_suite",
            display_name="Inspect lifecycle context",
            risk_tier=RiskTier.read_only,
            required_fields=["lifecycle_summary"],
        ),
        CapabilityDefinition(
            capability_id="quant_data.run_source_preflight",
            app_id="quant_data",
            display_name="Run source preflight",
            risk_tier=RiskTier.workflow_preflight,
            required_fields=["source_summary"],
            preflight_required=True,
        ),
        CapabilityDefinition(
            capability_id="quant_studio.prepare_model_config_draft",
            app_id="quant_studio",
            display_name="Prepare model configuration draft",
            risk_tier=RiskTier.draft_only,
            required_fields=["target_summary"],
            confirmation_required=True,
        ),
        CapabilityDefinition(
            capability_id="quant_documentation.inspect_package",
            app_id="quant_documentation",
            display_name="Inspect documentation package",
            risk_tier=RiskTier.read_only,
            required_fields=["package_summary"],
        ),
        CapabilityDefinition(
            capability_id="quant_monitoring.validate_bundle",
            app_id="quant_monitoring",
            display_name="Validate monitoring bundle",
            risk_tier=RiskTier.workflow_preflight,
            required_fields=["bundle_summary"],
            preflight_required=True,
        ),
    ]


class CapabilityRegistry:
    def __init__(self, capabilities: list[CapabilityDefinition]) -> None:
        self._capabilities = {item.capability_id: item for item in capabilities}

    @classmethod
    def from_request(
        cls,
        capabilities: list[CapabilityDefinition] | None,
        default_registry: list[CapabilityDefinition] | None = None,
    ) -> "CapabilityRegistry":
        return cls(capabilities or default_registry or default_capabilities())

    def all(self) -> list[CapabilityDefinition]:
        return list(self._capabilities.values())

    def enabled(self) -> list[CapabilityDefinition]:
        return [item for item in self._capabilities.values() if item.enabled]

    def get(self, capability_id: str) -> CapabilityDefinition | None:
        return self._capabilities.get(capability_id)
