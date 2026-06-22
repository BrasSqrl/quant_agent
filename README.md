# Quant Agent Runtime

Hosted orchestration service for the future Quant Suite governed agent workflow.
This repository owns runtime code only. Shared contracts, policy source of
truth, examples, and suite-level validation remain in the sibling
`../quant_suite` repository.

## Current Slice

This first slice is plan-only. It provides:

- `GET /health`
- `GET /runtime/manifest`
- `POST /plans`
- deterministic fake-provider planning for local development and tests
- code-enforced policy and plan validation
- safe in-memory ledger entries
- a contract loader boundary for future `quant_suite` agent contracts

It does not provide execution endpoints, real hosted model calls, app action
execution, browser provider keys, or app business logic.

## Setup

```powershell
python -m pip install -e .[dev]
```

## Run

```powershell
python -m uvicorn quant_agent_runtime.api.app:app --reload
```

## Test

```powershell
python -m pytest
python -m compileall src tests
```

## Contract Boundary

The runtime loader looks for future canonical contract drafts under:

1. `QUANT_SUITE_ROOT\contracts`
2. `..\quant_suite\contracts`

If `agent_*.v1.schema.json` files are not present, the runtime reports that it
is using temporary internal test fixtures. Those fixtures are only for this
first implementation slice and must not become the canonical contract source.

## Safety Boundary

The runtime sends only sanitized user goals and context summaries to the model
gateway. It rejects malformed provider output, unknown capabilities, forbidden
actions, missing action inputs, missing confirmation flags, attempted execution,
and unsafe raw context fields in plans.

Ledgers record provider metadata, redaction summaries, plan snapshots,
validation results, and policy rejections without raw secrets, row-level data,
raw local paths, bucket names, hidden commands, raw prompts, or raw provider
responses.
