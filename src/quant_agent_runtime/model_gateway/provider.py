from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

from quant_agent_runtime.models import CapabilityDefinition, PolicySettings, ProviderMetadata


@dataclass(frozen=True)
class ProviderPlanRequest:
    user_goal: str
    context_summary: dict[str, Any]
    capabilities: list[CapabilityDefinition]
    policy: PolicySettings


@dataclass(frozen=True)
class ProviderResult:
    raw_output: Mapping[str, Any]
    metadata: ProviderMetadata


class ModelProvider(ABC):
    @abstractmethod
    def generate_plan(self, request: ProviderPlanRequest) -> ProviderResult:
        raise NotImplementedError
