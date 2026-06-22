from __future__ import annotations

from quant_agent_runtime.models import LedgerEntry, RunState
from quant_agent_runtime.orchestration import run_state_from_orchestration


def run_state_for_entry(entry: LedgerEntry) -> RunState:
    return run_state_from_orchestration(entry)
