# Agent Instructions

## Repository Purpose

`quant_agent` is the hosted runtime service for Quant Suite's governed agent
workflow. It owns runtime orchestration code, model gateway boundaries,
provider adapters, planner loop, policy enforcement, runtime state, execution
ledger implementation, app action clients, and runtime observability.

The shared contract and policy source of truth remains the sibling
`../quant_suite` repository.

## Required Source Context

Before changing runtime behavior, read the smallest relevant source context in
`../quant_suite`, starting with:

- `../quant_suite/AGENTS.md`
- `../quant_suite/docs/GOVERNED_AGENT_WORKFLOW_ROADMAP.md`
- `../quant_suite/docs/ASSISTANT_SPEC.md`
- `../quant_suite/docs/HANDOFF_CONTRACTS.md`

Validate against `quant_suite` contracts when available. Do not create
canonical agent contracts in this repository.

## Boundaries

- Do not add Quant Data EDA business logic.
- Do not add Quant Studio model fitting business logic.
- Do not add Quant Documentation drafting internals.
- Do not add Quant Monitoring run logic.
- Do not expose provider keys to browser clients.
- Do not call real LLM providers until provider configuration and secret
  handling are designed.
- Do not mutate app state in plan-only mode.
- Do not add execution routes before policy, preflight, confirmation, and
  ledger contracts exist.
- Do not run shell commands from model output.
- Keep all agent actions bounded by capabilities, policy, validation, and
  ledgering.

## Data Safety

Provider context, plans, and ledgers must not include raw row-level data,
secrets, credentials, raw local paths, bucket names, hidden workflow commands,
raw provider prompts, raw provider responses, or full artifact payloads.

Use summaries, safe labels, references, counts, and validation evidence.

## Validation

Use the narrowest relevant checks first:

```powershell
python -m pytest
python -m compileall src tests
```

When canonical agent contracts exist in `../quant_suite`, add runtime contract
tests that load those files instead of relying on internal test fixtures.
