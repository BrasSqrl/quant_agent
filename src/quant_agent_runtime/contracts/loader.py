from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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
