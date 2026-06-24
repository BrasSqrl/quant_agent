# Quant Agent Runtime

`quant_agent` is the hosted runtime service for Quant Suite's governed agent
workflow. It owns planning, orchestration, app action routing, policy
enforcement, durable ledgers, provider boundaries, recovery, and runtime
observability.

The canonical contracts, workflow templates, governance examples, and roadmap
remain in the sibling `../quant_suite` repository. This repository should load
and validate against those contracts instead of defining canonical schemas of
its own.

## Current Runtime Shape

The runtime is no longer plan-only. It supports a governed modular workflow
runner that can operate:

- the full Quant Suite lifecycle;
- one app's 1-5 workflow;
- a stage range inside one app;
- a selected capability set.

Execution is still bounded. The browser calls `quant_agent`; app work happens
only through app-owned typed agent APIs. The runtime does not implement Quant
Data EDA logic, Quant Studio model fitting logic, Quant Documentation drafting
internals, or Quant Monitoring review logic.

The runtime also includes:

- shared OpenAI/Ollama/fallback planning provider configuration;
- live app capability discovery and canonical capability reconciliation;
- contract-validated preflight, confirmation, action-request, execution, and
  result records;
- local JSON file-backed ledgers with integrity metadata;
- run status, orchestration, progress, history, ledger, and support-bundle
  read APIs;
- pause, resume, cancellation, retry, revalidation, plan revision, and child-run
  activation flows;
- sample-owned demo autopilot preview/one-step advance/reset/narrative flows;
- user-owned plan review, plan approval, readiness, and consent gates;
- role-aware governance, separation-of-duties checks, and optional external
  approval evidence/adapters.

Generic `/execute` and `POST /runs` routes are intentionally not provided.

## Setup

```powershell
python -m pip install -e .[dev]
```

Python 3.11 or newer is required.

## Run

```powershell
python -m uvicorn quant_agent_runtime.api.app:app --host 127.0.0.1 --port 8010 --reload
```

The default app-owned API targets are:

| App | Env var | Default |
| --- | --- | --- |
| Quant Data | `QUANT_DATA_AGENT_API_BASE_URL` | `http://127.0.0.1:8830` |
| Quant Studio | `QUANT_STUDIO_AGENT_API_BASE_URL` | `http://127.0.0.1:8810` |
| Quant Documentation | `QUANT_DOCUMENTATION_AGENT_API_BASE_URL` | `http://127.0.0.1:8840` |
| Quant Monitoring | `QUANT_MONITORING_AGENT_API_BASE_URL` | `http://127.0.0.1:8820` |

When launched through the local Quant Suite stack, these values should line up
with the API service ports used by the suite launcher.

## Shared LLM Provider Configuration

Planning can use the same server-side provider configuration used by other
Quant Suite API services. Agent-specific env vars take precedence over shared
suite env vars.

Common shared env vars:

```powershell
$env:QUANT_LLM_PROVIDER = "openai" # openai, ollama, or disabled
$env:QUANT_LLM_MODEL = "gpt-5.4-nano"
$env:OPENAI_API_KEY = "paste-key-for-this-shell-only"
```

Ollama example:

```powershell
$env:QUANT_LLM_PROVIDER = "ollama"
$env:QUANT_LLM_MODEL = "gemma4:e2b"
$env:QUANT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
```

Agent-specific overrides:

- `QUANT_AGENT_LLM_PROVIDER`
- `QUANT_AGENT_LLM_MODEL`
- `QUANT_AGENT_OPENAI_API_KEY`
- `QUANT_AGENT_OPENAI_BASE_URL`
- `QUANT_AGENT_OLLAMA_BASE_URL`

Additional shared controls:

- `QUANT_LLM_TIMEOUT_SECONDS`
- `QUANT_LLM_MAX_CONTEXT_CHARS`
- `OPENAI_BASE_URL`

If provider configuration is disabled, missing, unsupported, or invalid, the
runtime falls back to deterministic fake planning. Provider keys must remain
server-side. Do not add `VITE_OPENAI_API_KEY` or send provider keys to browser
clients, manifests, ledgers, logs, support bundles, or smoke fixtures.

## Contracts And Templates

The runtime loads canonical agent contracts from:

1. `QUANT_SUITE_ROOT\contracts`
2. `..\quant_suite\contracts`

If canonical contracts are unavailable, internal fixtures may be used by tests,
but those fixtures are not the source of truth.

Important contract families currently used by the runtime include:

- `agent_capability.v1`
- `agent_workflow_template.v1`
- `agent_plan.v1`
- `agent_action_preflight.v1`
- `agent_action_request.v1`
- `agent_action_result.v1`
- `agent_execution_ledger.v1`
- `agent_runtime_manifest.v1`
- `agent_support_bundle.v1`
- governance and external-approval evidence contracts

## Runtime APIs

The runtime manifest at `GET /runtime/manifest` is the safest source for the
currently advertised support levels and route list.

Core read routes:

- `GET /health`
- `GET /runtime/manifest`
- `GET /workflow-runs/{run_id}`
- `GET /runs`
- `GET /runs/{run_id}`
- `GET /runs/{run_id}/orchestration`
- `GET /runs/{run_id}/ledger`
- `GET /runs/{run_id}/demo-narrative`
- `GET /runs/{run_id}/support-bundle`
- `GET /runs/{run_id}/external-approval-submissions`

Core mutating routes:

- `POST /plans`
- `POST /workflow-scope-resolutions`
- `POST /workflow-runs`
- `POST /workflow-runs/{run_id}/advance`
- `POST /workflow-runs/{run_id}/advance-until-blocked`
- `POST /preflights`
- `POST /confirmations`
- `POST /action-requests`
- `POST /executions`
- `POST /retries`
- `POST /cancellations`
- `POST /pauses`
- `POST /resumptions`
- `POST /plan-revisions`
- `POST /plan-revision-activations`
- `POST /run-revalidations`
- `POST /autopilot-previews`
- `POST /autopilot-steps`
- `POST /sample-reset-previews`
- `POST /sample-resets`
- `POST /user-plan-reviews`
- `POST /user-plan-approvals`
- `POST /user-workflow-readiness`
- `POST /user-workflow-consents`
- `POST /external-approval-requests`
- `POST /external-approval-submissions`
- `POST /external-approval-decisions`
- `POST /external-approval-decision-refreshes`

All run-bound mutating routes are subject to governance, safety scanning,
contract validation, and ledgering.

## Workflow Runner

Workflow runs are created through `POST /workflow-runs` or inferred from a goal
through `POST /workflow-scope-resolutions`.

Supported workflow scopes:

- `full_lifecycle`
- `app_workflow`
- `stage_range`
- `capability_set`

The runner resolves action inputs from durable ledger evidence, canonical
workflow templates, live capability discovery, and safe handoff summaries. The
browser must not supply action payloads, policy overrides, execution flags,
preflight records, confirmation records, provider responses, or raw artifact
content.

Advancement modes:

- `POST /workflow-runs/{run_id}/advance` advances one eligible current action.
- `POST /workflow-runs/{run_id}/advance-until-blocked` advances until a gate,
  blocker, running action, app error, or completed scope.

Manual confirmation remains required for guarded steps. Long-running app-owned
actions are represented as ledgered `running` action results with safe app-run
references and idempotent retry/advance behavior.

## Ledger And Recovery

Ledgers are stored as local JSON files. The default location is:

```text
%USERPROFILE%\.quant_agent\ledgers
```

Override it with:

```powershell
$env:QUANT_AGENT_LEDGER_DIR = "C:\path\to\ledgers"
```

Ledger writes are contract-validated and integrity-stamped. API responses do
not expose raw ledger directory paths. Support bundles are read-only,
contract-backed, redacted JSON summaries for a run.

Recovery routes support pause/resume, cancellation, retry, run revalidation,
plan revision preview, and revised-plan activation as a child run. Child runs
start with fresh gates; they do not inherit parent approval, readiness, consent,
confirmation, preflight, preview, execution, retry, pause, cancellation, or
stale-state results.

## Governance

Governance policy packs are loaded from `../quant_suite` by default, with these
overrides:

- `QUANT_AGENT_GOVERNANCE_POLICY_PACK_PATH`
- `QUANT_AGENT_GOVERNANCE_ENVIRONMENT`
- `QUANT_AGENT_ACTOR_ROLE`
- `QUANT_AGENT_ACTOR_ID`

The default local role is `local_developer_operator`. Restricted roles and
environment policy packs are useful for denial tests and governance smoke
coverage. Separation-of-duties and external approval enforcement are policy
controlled. Local workflow operation does not require external approval unless
the selected policy pack enforces it.

External approval support is an evidence/adaptor boundary:

- request package preview;
- local outbox or mock HTTP submission;
- manual decision import or mock decision refresh;
- policy enforcement when configured.

No real external approval vendor client, browser-side approval integration, or
external approval provider credentials are supported here.

## Safety Boundary

The runtime must not include raw row-level data, secrets, credentials, raw local
paths, bucket names, hidden workflow commands, raw provider prompts, raw
provider responses, external adapter payload dumps, or full artifact payloads in
plans, ledgers, manifests, validation errors, support bundles, or exports.

Use summaries, safe labels, references, counts, hashes, validation evidence, and
redaction reports.

The model provider is for planning and explanation only. App APIs own workflow
execution. The runtime must never run shell commands from model output.

## Test

Use the narrowest relevant checks first:

```powershell
python -m pytest
python -m compileall src tests
```

For contract changes, also run the sibling suite validation:

```powershell
cd ..\quant_suite
powershell -ExecutionPolicy Bypass -File scripts\validate_contracts.ps1
```

For live local certification, use the suite script against running app APIs and
the agent runtime:

```powershell
cd ..\quant_suite
powershell -ExecutionPolicy Bypass -File scripts\certify_agent_live_workflows.ps1 `
  -AgentApiBaseUrl http://127.0.0.1:8010 `
  -DataApiBaseUrl http://127.0.0.1:8830 `
  -StudioApiBaseUrl http://127.0.0.1:8810 `
  -DocumentationApiBaseUrl http://127.0.0.1:8840 `
  -MonitoringApiBaseUrl http://127.0.0.1:8820
```

Adjust ports to match the local stack when services are launched on alternate
ports.
