from __future__ import annotations

import json
import os
import importlib.util
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from quant_agent_runtime.models import CapabilityDefinition, ProviderRuntimeStatus, RiskTier
from quant_agent_runtime.provider_config import (
    internal_provider_status,
    provider_status_from_contract_payload,
)


@dataclass(frozen=True)
class ContractLoadResult:
    canonical_agent_contracts_loaded: bool
    loaded_agent_contracts: list[str]
    source_label: str


class QuantSuiteContractLoader:
    def __init__(self, quant_suite_root: Path | None = None) -> None:
        env_root = os.environ.get("QUANT_SUITE_ROOT")
        if quant_suite_root is not None:
            self._root = quant_suite_root
            self._source_label = "configured_path"
        elif env_root:
            self._root = Path(env_root)
            self._source_label = "QUANT_SUITE_ROOT"
        else:
            self._root = Path.cwd().parent / "quant_suite"
            self._source_label = "sibling_quant_suite"

    def load_agent_contracts(self) -> ContractLoadResult:
        contracts_dir = self._root / "contracts"
        if not contracts_dir.exists():
            return ContractLoadResult(
                canonical_agent_contracts_loaded=False,
                loaded_agent_contracts=[],
                source_label=self._source_label,
            )

        contract_names = sorted(
            item.name for item in contracts_dir.glob("agent_*.v1.schema.json")
        )
        return ContractLoadResult(
            canonical_agent_contracts_loaded=bool(contract_names),
            loaded_agent_contracts=contract_names,
            source_label=self._source_label,
        )

    @property
    def source_label(self) -> str:
        return self._source_label

    def contract_file(self, name: str) -> Path:
        return self._root / "contracts" / name

    def fixture_file(self, *parts: str) -> Path:
        return self._root / "fixtures" / Path(*parts)

    def load_agent_capabilities(self) -> list[CapabilityDefinition]:
        capability_path = self._root / "contracts" / "agent_capability.v1.example.json"
        if not capability_path.is_file():
            return []

        with capability_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        capabilities: list[CapabilityDefinition] = []
        for item in payload.get("capabilities", []):
            input_schema = item.get("input_schema", {})
            required_fields = input_schema.get("required_fields", [])
            capabilities.append(
                CapabilityDefinition(
                    capability_id=item["capability_id"],
                    app_id=item["app_id"],
                    display_name=item["display_name"],
                    risk_tier=RiskTier(item["risk_tier"]),
                    enabled=item.get("enabled", True),
                    required_fields=[
                        field for field in required_fields if isinstance(field, str)
                    ],
                    preflight_required=item.get("preflight_required", False),
                    confirmation_required=item.get("confirmation_required", False),
                    execution_supported=(
                        item.get("execution_supported")
                        if "execution_supported" in item
                        else None
                    ),
                )
            )
        return capabilities

    def load_agent_workflow_templates(self) -> dict[str, Any]:
        template_path = self._root / "contracts" / "agent_workflow_template.v1.example.json"
        if not template_path.is_file():
            return {}

        try:
            with template_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, JSONDecodeError):
            return {}

        if not isinstance(payload, dict):
            return {}
        return payload

    def load_agent_provider_status(self) -> ProviderRuntimeStatus:
        config_path = self._root / "contracts" / "agent_provider_config.v1.example.json"
        if not config_path.is_file():
            return internal_provider_status()

        try:
            with config_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, JSONDecodeError):
            return provider_status_from_contract_payload(
                {},
                config_source=self._source_label,
                load_errors=["Canonical provider config could not be read."],
            )

        if not isinstance(payload, dict):
            return provider_status_from_contract_payload(
                {},
                config_source=self._source_label,
                load_errors=["Canonical provider config must be a JSON object."],
            )

        return provider_status_from_contract_payload(
            payload,
            config_source=self._source_label,
        )

    def validate_agent_contract_payload(self, payload: Any, schema_name: str) -> None:
        schema_path = self._root / "contracts" / schema_name
        validator_path = self._root / "scripts" / "validate_contracts.py"
        if not schema_path.is_file():
            raise ValueError(f"Canonical contract schema was not found: {schema_name}")
        if not validator_path.is_file():
            raise ValueError("Quant Suite contract validator was not found.")

        spec = importlib.util.spec_from_file_location("quant_suite_contract_validator", validator_path)
        if spec is None or spec.loader is None:
            raise ValueError("Quant Suite contract validator could not be loaded.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        schema = module.load_json(schema_path)
        module.validate_schema(payload, schema)
