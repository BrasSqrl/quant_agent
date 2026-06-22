from __future__ import annotations

from quant_agent_runtime.models import PlanValidationResult, ValidationIssue


class RuntimeValidationError(Exception):
    def __init__(self, validation: PlanValidationResult) -> None:
        super().__init__("Runtime validation rejected the provider plan.")
        self.validation = validation

    def to_problem(self) -> dict[str, object]:
        return self.validation.model_dump(mode="json")


class MalformedProviderOutputError(RuntimeValidationError):
    @classmethod
    def from_message(cls, message: str) -> "MalformedProviderOutputError":
        validation = PlanValidationResult(
            status="rejected",
            errors=[
                ValidationIssue(
                    code="malformed_provider_output",
                    message=message,
                )
            ],
        )
        return cls(validation)
