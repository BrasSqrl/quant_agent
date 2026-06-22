from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from quant_agent_runtime.models import CapabilityDefinition, RiskTier


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
                )
            )
        return capabilities
