from quant_agent_runtime.model_gateway.fake import FakePlanProvider
from quant_agent_runtime.models import PlanRequest
from quant_agent_runtime.planner import PlannerService


def test_ledger_entries_are_safe_and_omit_raw_unsafe_fields() -> None:
    planner = PlannerService(provider=FakePlanProvider())

    result = planner.create_plan(
        PlanRequest(
            user_goal="Use C:\\Users\\me\\secret.csv for the plan.",
            context_summary={
                "lifecycle_summary": "Lifecycle exists.",
                "source_summary": "Source summary only.",
                "target_summary": "Target summary only.",
                "package_summary": "Package summary only.",
                "bundle_summary": "Bundle summary only.",
                "credentials": {"password": "super-secret"},
                "bucket_name": "private-bucket",
                "hidden_commands": ["rm -rf ."],
            },
        )
    )

    entry = planner.ledger.list_entries()[0]
    dumped = str(entry.model_dump(mode="json"))

    assert result.ledger_recorded is True
    assert entry.redaction_summary.redacted is True
    assert "super-secret" not in dumped
    assert "private-bucket" not in dumped
    assert "rm -rf" not in dumped
    assert "C:\\Users\\me\\secret.csv" not in dumped
    assert "hidden_commands" in entry.context_preview.omitted_sensitive_fields
    assert "provider_prompt" not in dumped
