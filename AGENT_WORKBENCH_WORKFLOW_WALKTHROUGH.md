# Agent Workbench Workflow Walkthrough

## Executive Summary

The Agent Workbench can run the Quant Suite workflow through `quant_agent` when
the application stack is running and the selected lifecycle contains safe
context for the data you loaded in Quant Data. The user provides data to Quant
Data first. After that, the Workbench can create a governed workflow run, infer
or apply a workflow scope, advance through app-owned capabilities, stop at
required manual gates, ledger every action, and continue after approval.

The important operating model is:

1. Load or register your data in Quant Data.
2. Start the Quant Suite stack with the desired LLM profile.
3. Confirm the agent runtime and app capability discovery are healthy.
4. Open Agent Workbench from Suite Home or directly in Quant Studio.
5. Select or confirm the lifecycle that contains your loaded data evidence.
6. Prompt the agent for the scope you want, such as full lifecycle, Quant Data
   only, Quant Studio steps 1-5, a stage range, or one specific capability.
7. Click `Create agent`.
8. For user-owned data, complete user-owned gates: readiness, plan review,
   plan approval, and guided workflow consent.
9. Use `Advance until blocked` to let the runner perform every currently
   allowed non-manual action.
10. When it stops for `Confirm`, choose the listed confirmation step and click
    `Confirm`.
11. Continue with `Advance one step` or `Advance until blocked` until the
    selected workflow scope completes or reports a blocker.
12. Review run status, orchestration, run history, ledger, and support bundle
    output to understand what happened.

The LLM does not execute code, mutate app state directly, read raw rows, or
call providers from the browser. The LLM is used for planning, scope
resolution, and explanation. Actual work is done by app-owned APIs through
typed capabilities, and `quant_agent` enforces policy, preflight,
confirmation, contract validation, handoff resolution, and ledgering.

For a full lifecycle run, the intended 20-step path is:

1. Quant Data source intake
2. Quant Data source readiness preflight
3. Quant Data EDA plan
4. Quant Data EDA review
5. Quant Data EDA handoff export
6. Quant Studio model readiness check
7. Quant Studio model config draft
8. Quant Studio candidate model fit
9. Quant Studio candidate comparison
10. Quant Studio documentation and monitoring package handoff
11. Quant Documentation package intake
12. Quant Documentation draft workspace
13. Quant Documentation section drafting
14. Quant Documentation citation and claim review
15. Quant Documentation review export package
16. Quant Monitoring bundle intake
17. Quant Monitoring profile draft
18. Quant Monitoring bundle validation
19. Quant Monitoring run review
20. Quant Monitoring feedback signal

If you only want part of the workflow, prompt for that part. Examples:

- `run the full Quant Suite workflow`
- `run Quant Data steps 1-5`
- `run Quant Studio steps 1-5`
- `run Quant Studio steps 2-4`
- `fit a candidate model in Quant Studio`
- `create the Documentation draft workspace only`
- `just validate the Monitoring bundle`

The runner should not invent missing data, missing handoffs, unsupported
capabilities, or skipped gates. If the loaded data or upstream evidence is not
available in the selected lifecycle, it should stop with a clear blocker.

## Detailed Walkthrough

### 1. Understand The Pieces

The workflow is split across five local services:

| Service | Role | Default Docker URL |
| --- | --- | --- |
| Quant Studio web | Browser UI and Agent Workbench surface | `http://127.0.0.1:5810` |
| Quant Agent API | Governs planning, orchestration, gates, ledgers, app routing | `http://127.0.0.1:8000` |
| Quant Data API | Owns source intake, preflight, EDA, handoff export | `http://127.0.0.1:8830` |
| Quant Studio API | Owns readiness, draft config, model fit, comparison, handoff | `http://127.0.0.1:8810` |
| Quant Documentation API | Owns package inspection, drafting, claim review, export | `http://127.0.0.1:8840` |
| Quant Monitoring API | Owns bundle inspection, validation, monitoring review, feedback | `http://127.0.0.1:8820` |

When launched through Docker, Quant Studio web is built with
`VITE_QUANT_AGENT_API_BASE_URL=http://localhost:8000` by default, so the
Workbench calls `quant_agent` at port `8000`.

When running `quant_agent` manually from this repo, the README example uses
port `8010`. If you use the manual source-service mode, make sure the Studio
frontend was built or launched with `VITE_QUANT_AGENT_API_BASE_URL` pointing to
the same agent URL.

### 2. Choose The LLM Mode

The same profile selector is used by the right-drawer assistants, Quant
Documentation drafting, and the agent planning gateway when launched through
the suite Docker script.

#### Option A: Deterministic Local Fallback

Use this when you want predictable local behavior without an OpenAI key or a
local Ollama model.

```powershell
cd C:\Users\matth\Desktop\quant_suite
$env:QUANT_LLM_MODEL_PROFILE = "disabled_deterministic"
Remove-Item Env:\OPENAI_API_KEY -ErrorAction SilentlyContinue
powershell -ExecutionPolicy Bypass -File .\Start-QuantSuiteDocker.ps1 -Build
```

#### Option B: OpenAI

Use this when the right drawer, Quant Documentation, and the Agent Workbench
should all use the server-side OpenAI configuration.

```powershell
cd C:\Users\matth\Desktop\quant_suite
$env:QUANT_LLM_MODEL_PROFILE = "openai_gpt_5_4_mini"
$env:OPENAI_API_KEY = Read-Host "OpenAI API key for this session"
powershell -ExecutionPolicy Bypass -File .\Start-QuantSuiteDocker.ps1 -Build -CheckLlm
```

Do not create `VITE_OPENAI_API_KEY`. Browser containers should never receive
provider keys.

#### Option C: Host Ollama

Use this when the shared local provider should be host Ollama.

```powershell
cd C:\Users\matth\Desktop\quant_suite
$env:QUANT_LLM_MODEL_PROFILE = "ollama_gemma4_e2b_local"
$env:QUANT_OLLAMA_BASE_URL = "http://host.docker.internal:11434"
Remove-Item Env:\OPENAI_API_KEY -ErrorAction SilentlyContinue
powershell -ExecutionPolicy Bypass -File .\Start-QuantSuiteDocker.ps1 -Build -CheckLlm
```

### 3. Confirm The Stack Is Running

After startup, check the API services. These checks should return JSON without
showing any provider key.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/runtime/manifest
Invoke-RestMethod http://127.0.0.1:8830/api/agent/capabilities
Invoke-RestMethod http://127.0.0.1:8810/api/agent/capabilities
Invoke-RestMethod http://127.0.0.1:8840/api/agent/capabilities
Invoke-RestMethod http://127.0.0.1:8820/api/agent/capabilities
```

In the agent manifest, verify these fields conceptually:

- `workflow_run_support_level` is present.
- `workflow_scope_resolution_support_level` is present.
- `supported_workflow_scopes` includes `full_lifecycle`, `app_workflow`,
  `stage_range`, and `capability_set`.
- `supported_preflight_capabilities` includes the preflight-capable Data,
  Studio, and Monitoring actions.
- `supported_execution_capabilities` includes the app-owned executable
  capabilities.
- `provider_status` reports the selected profile, provider, model, and fallback
  state without exposing keys.

If any app capability route fails, the agent can still start, but workflows
that need that app should report capability gaps or app-unavailable blockers.

### 4. Load Data Into Quant Data

The only thing the user must bring is data. Any supported Quant Data loading or
source-registration path is acceptable.

Before prompting the agent, make sure the data is represented in the selected
suite lifecycle as safe context. The agent does not need raw rows, but it does
need enough sanitized evidence to plan and route work. Look for evidence such
as:

- a source reference;
- a source summary;
- counts or labels for loaded datasets;
- an existing lifecycle id;
- no unsafe raw file paths or raw row payloads in browser-visible context.

If you load data but the selected lifecycle does not show the source context,
the agent may correctly stop with a missing input or missing handoff blocker.
That is expected. The fix is to select or create the lifecycle that contains
the Quant Data source evidence, not to bypass the blocker.

### 5. Open Agent Workbench

Open Quant Studio web:

```text
http://127.0.0.1:5810
```

Use Suite Home or the available app navigation to open Agent Workbench. A
direct Workbench URL has these query parameters:

```text
http://127.0.0.1:5810/?suite=1&agent=1
```

When opening from an app rail, the URL can include launch context such as:

```text
http://127.0.0.1:5810/?suite=1&agent=1&source_app=quant_data&source_stage=source_intake
```

The Workbench uses that launch context to default the target app and stage, but
you can still prompt for a different valid scope.

### 6. Read The Runtime Status Panel

At the top of Agent Workbench, inspect `Runtime status`.

You want to see:

- runtime status available;
- workflow runs route available;
- workflow scope resolver route available;
- run status, orchestration, run history, and ledger export routes available;
- app capability discovery loaded for the apps you intend to use;
- LLM role shown as planning and explanation only;
- workflow execution shown as app-owned guarded actions.

If runtime is offline, the Workbench cannot create or advance an agent run.
Check `http://127.0.0.1:8000/health` and confirm Docker exposed the agent API
port.

### 7. Select The Workflow Scope

The Workbench supports four main scopes.

#### Full Lifecycle

Use this when you want the entire suite run:

```text
run the full Quant Suite workflow
```

Expected template:

```text
full_lifecycle_default
```

Expected selected capabilities:

```text
quant_data.register_source_reference
quant_data.run_source_preflight
quant_data.create_eda_plan
quant_data.run_eda_review
quant_data.export_eda_handoff
quant_studio.run_model_readiness_check
quant_studio.prepare_model_config_draft
quant_studio.fit_candidate_model
quant_studio.compare_candidate_runs
quant_studio.create_documentation_package
quant_documentation.inspect_package
quant_documentation.create_draft_workspace
quant_documentation.draft_section
quant_documentation.find_unsupported_claims
quant_documentation.export_markdown_review_package
quant_monitoring.inspect_bundle
quant_monitoring.prepare_profile_draft
quant_monitoring.validate_bundle
quant_monitoring.run_monitoring_review
quant_monitoring.create_feedback_signal
```

#### App Workflow

Use this when you only want one app's 1-5 workflow.

Examples:

```text
run Quant Data steps 1-5
run Quant Studio steps 1-5
run Quant Documentation steps 1-5
run Quant Monitoring steps 1-5
```

You can also use the Workbench `Workflow scope` controls:

- `Workflow scope`: `app_workflow`
- `Target app`: one of the four apps

#### Stage Range

Use this when you only want part of one app's workflow.

Examples:

```text
run Quant Studio steps 2-4
run Quant Data steps 3-5
run Documentation steps 2-4
run Monitoring step 3 only
```

The runner should select only the capabilities inside the requested stage
range. It should not continue into downstream apps.

#### Capability Set

Use this when you want one or more specific capabilities.

Examples:

```text
just validate the Monitoring bundle
fit a candidate model in Quant Studio
create the Documentation draft workspace only
```

Capability-set runs are useful when you already know the exact task and do not
want a full app workflow.

### 8. Create The Agent Run

In Agent Workbench:

1. Enter the prompt in `Agent prompt`.
2. Confirm or adjust `Target app` and `Workflow scope` if needed.
3. Click `Create agent`.

The Workbench should create a workflow run through:

```text
POST /workflow-runs
```

or first resolve scope through:

```text
POST /workflow-scope-resolutions
```

The browser should not call app action endpoints directly.

After creation, inspect:

- `Selected workflow scope`
- selected template ids;
- selected capability count;
- capability gaps;
- current step;
- run state;
- orchestration status.

If the workflow shows capability gaps, the app service may not be advertising
the expected action, the app may be unavailable, or the canonical contract and
app capability response may not match.

### 9. Complete User-Owned Gates

If the data is your own data rather than a certified sample workspace, the run
is user-owned. User-owned runs must pass these run-level gates before guarded
actions can continue:

1. `Check readiness`
2. Plan assumption review
3. `Approve active plan`
4. `Approve guided user workflow`

The run-level approval is not a substitute for step confirmation. Some
individual steps still require `Confirm`.

If controls stay disabled, read the disabled tooltip or blocker text. Common
causes:

- ownership not classified yet;
- readiness not checked;
- plan assumptions not reviewed;
- active plan not approved;
- guided workflow consent not recorded;
- selected step requires confirmation;
- selected app capability is unavailable;
- missing source or handoff evidence.

### 10. Advance The Workflow

Use the workflow controls in the Agent Workbench blueprint:

- `Advance one step`: runs only the next eligible current action.
- `Advance until blocked`: keeps advancing until it hits a confirmation,
  failed preflight, missing input, running app action, app error, terminal
  state, or completed scope.
- `Refresh workflow run`: reloads workflow status from `quant_agent`.

The runner chooses from the current step's allowed actions. Examples:

- `run_preflight`
- `preview_action_request`
- `execute_step`
- `retry_failed_step`
- `confirm_step`

The runner does not auto-confirm. If the current step allows only
`confirm_step`, `Advance until blocked` should stop and report manual
confirmation required.

### 11. Confirm Required Steps

When the runner stops for confirmation:

1. Find the `Confirmation step` selector.
2. Choose the step shown as requiring confirmation.
3. Click `Confirm`.
4. Click `Advance one step` or `Advance until blocked` again.

Confirmation is ledgered through:

```text
POST /confirmations
```

It records approval evidence, but it does not by itself execute an app action.
The next advance normally previews an action request or executes an app-owned
capability if all gates are satisfied.

### 12. Understand What Each App Workflow Does

#### Quant Data Steps 1-5

| Step | Capability | Gate behavior | Output evidence |
| --- | --- | --- | --- |
| 1 | `quant_data.register_source_reference` | confirmation required | `source_reference` |
| 2 | `quant_data.run_source_preflight` | preflight, no confirmation | `preflight_summary` |
| 3 | `quant_data.create_eda_plan` | confirmation required | `eda_plan` |
| 4 | `quant_data.run_eda_review` | preflight and confirmation required | `eda_package` |
| 5 | `quant_data.export_eda_handoff` | confirmation required | `eda_handoff` |

If you have loaded data first, Data step 1 registers or resolves a safe source
reference, and later steps use that reference rather than raw rows.

#### Quant Studio Steps 1-5

| Step | Capability | Gate behavior | Output evidence |
| --- | --- | --- | --- |
| 1 | `quant_studio.run_model_readiness_check` | preflight, no confirmation | `model_readiness_summary` |
| 2 | `quant_studio.prepare_model_config_draft` | confirmation required | `model_config_draft` |
| 3 | `quant_studio.fit_candidate_model` | preflight and confirmation required | `studio_run` |
| 4 | `quant_studio.compare_candidate_runs` | confirmation required | `champion_recommendation` |
| 5 | `quant_studio.create_documentation_package` | confirmation required | `documentation_package`, `monitoring_bundle` |

For a Studio-only run that starts at step 1, the lifecycle should already have
the handoff or target summary needed by Studio. For a full lifecycle run, Data
step 5 produces `eda_handoff`, which Studio can use.

#### Quant Documentation Steps 1-5

| Step | Capability | Gate behavior | Output evidence |
| --- | --- | --- | --- |
| 1 | `quant_documentation.inspect_package` | no confirmation | `documentation_package_summary` |
| 2 | `quant_documentation.create_draft_workspace` | confirmation required | `documentation_draft` |
| 3 | `quant_documentation.draft_section` | confirmation required | `draft_section` |
| 4 | `quant_documentation.find_unsupported_claims` | no confirmation | `claim_review_summary` |
| 5 | `quant_documentation.export_markdown_review_package` | confirmation required | `documentation_review_package` |

Documentation actions return reviewable draft and review-package references.
They should not expose raw provider prompts or raw document payloads in the
agent ledger.

#### Quant Monitoring Steps 1-5

| Step | Capability | Gate behavior | Output evidence |
| --- | --- | --- | --- |
| 1 | `quant_monitoring.inspect_bundle` | no confirmation | `bundle_summary` |
| 2 | `quant_monitoring.prepare_profile_draft` | confirmation required | `monitoring_profile_draft` |
| 3 | `quant_monitoring.validate_bundle` | preflight, no confirmation | `bundle_validation_summary` |
| 4 | `quant_monitoring.run_monitoring_review` | preflight and confirmation required | `monitoring_run` |
| 5 | `quant_monitoring.create_feedback_signal` | confirmation required | `feedback_signal` |

Feedback signal creation is advisory/reviewable. It should not automatically
start retraining.

### 13. How Handoffs Work

The runner passes only ledgered safe references forward. It does not pass raw
datasets, raw model artifacts, raw documents, or raw monitoring payloads.

Key handoffs:

| Producer | Output | Consumer |
| --- | --- | --- |
| Data handoff export | `eda_handoff` | Studio readiness and config |
| Studio model config draft | `model_config_draft` | Studio candidate fit |
| Studio candidate fit | `studio_run` | Studio comparison |
| Studio comparison | `champion_recommendation` | Studio handoff package |
| Studio package handoff | `documentation_package` | Documentation package intake |
| Studio package handoff | `monitoring_bundle` | Monitoring bundle intake |
| Documentation draft workspace | `documentation_draft` | Documentation section drafting |
| Documentation section drafting | `draft_section` | Claim review |
| Documentation claim review | `claim_review_summary` | Review export |
| Monitoring profile draft | `monitoring_profile_draft` | Bundle validation |
| Monitoring bundle validation | `bundle_validation_summary` | Monitoring review |
| Monitoring review | `monitoring_run` | Feedback signal |

If a run starts in the middle, the selected lifecycle must already contain the
required upstream reference, or the runner should stop with a missing handoff
blocker.

### 14. Monitor Progress

Use these panels while the run advances:

- `Selected workflow scope`: confirms the selected templates and capabilities.
- `Run status`: shows run state, ownership, readiness, consent, and final
  status.
- `Run progress`: shows completed, blocked, failed, and current-step counts.
- `Orchestration`: shows ordered steps, current step, required gate, latest
  references, and allowed actions.
- `Run history`: lists prior durable runs.
- `Ledger audit`: lets you view or download the safe ledger JSON.
- `Support bundle`: packages safe evidence and diagnostics for inspection.

The ledger is the source of truth for what happened. It should contain
summaries, counts, references, validation evidence, and redaction summaries.

### 15. Common Successful Flow

For a full lifecycle run after loading data:

1. Open Workbench.
2. Prompt:

   ```text
   run the full Quant Suite workflow for the selected lifecycle
   ```

3. Click `Create agent`.
4. If user-owned gates appear:
   - click `Check readiness`;
   - review assumptions;
   - click `Approve active plan`;
   - click `Approve guided user workflow`.
5. Click `Advance until blocked`.
6. If it stops at confirmation:
   - select the confirmation step;
   - click `Confirm`;
   - click `Advance until blocked` again.
7. Repeat confirmation and advance cycles until the run state is completed or a
   clear blocker appears.
8. Review `Orchestration`, `Run progress`, `Run history`, and `Ledger audit`.

For an app-only workflow after loading data:

1. Prompt:

   ```text
   run Quant Data steps 1-5
   ```

   or:

   ```text
   run Quant Studio steps 1-5
   ```

2. Click `Create agent`.
3. Complete user-owned gates if required.
4. Use `Advance until blocked`.
5. Confirm when required.
6. Continue until the app-scoped workflow completes.

For a single task:

1. Prompt:

   ```text
   just validate the Monitoring bundle
   ```

2. Click `Create agent`.
3. Complete any required readiness gates.
4. Use `Advance one step` or `Advance until blocked`.

### 16. Programmatic API Walkthrough

The UI is the normal path, but the runtime can be checked through HTTP.

#### Resolve A Natural Language Scope

```powershell
$body = @{
  goal = "run Quant Studio steps 2-4"
  current_context_summary = @{}
} | ConvertTo-Json -Depth 20

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/workflow-scope-resolutions `
  -ContentType "application/json" `
  -Body $body
```

#### Create A Full Lifecycle Workflow Run

```powershell
$body = @{
  goal = "run the full Quant Suite workflow"
  workflow_scope = "full_lifecycle"
  context_summary = @{}
} | ConvertTo-Json -Depth 20

$run = Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/workflow-runs `
  -ContentType "application/json" `
  -Body $body

$run.run_id
```

In real Workbench operation, `context_summary` is built from the selected
lifecycle and launch context. Do not paste raw rows, raw files, provider
responses, or secrets into this payload.

#### Advance Until Blocked

```powershell
$body = @{
  advance_intent = "advance_workflow_until_blocked"
} | ConvertTo-Json -Depth 20

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/workflow-runs/$($run.run_id)/advance-until-blocked" `
  -ContentType "application/json" `
  -Body $body
```

If the response says `manual_confirmation_required`, use the Workbench confirm
control or call `POST /confirmations` with the selected step id. The browser
and user should never supply app action payloads directly.

#### Refresh Status

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/workflow-runs/$($run.run_id)"
Invoke-RestMethod "http://127.0.0.1:8000/runs/$($run.run_id)"
Invoke-RestMethod "http://127.0.0.1:8000/runs/$($run.run_id)/orchestration"
Invoke-RestMethod "http://127.0.0.1:8000/runs/$($run.run_id)/ledger"
```

### 17. Troubleshooting

#### Runtime Offline

Symptoms:

- Workbench says runtime offline.
- `Create agent` is disabled.

Checks:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/runtime/manifest
```

Fixes:

- Start the Docker stack.
- Confirm port `8000` is not blocked.
- If running from source on port `8010`, rebuild or relaunch the Studio
  frontend with the matching `VITE_QUANT_AGENT_API_BASE_URL`.

#### Capability Gaps

Symptoms:

- Selected workflow shows capability gaps.
- Advance reports capability unavailable.

Checks:

```powershell
Invoke-RestMethod http://127.0.0.1:8830/api/agent/capabilities
Invoke-RestMethod http://127.0.0.1:8810/api/agent/capabilities
Invoke-RestMethod http://127.0.0.1:8840/api/agent/capabilities
Invoke-RestMethod http://127.0.0.1:8820/api/agent/capabilities
```

Fixes:

- Start the missing app API.
- Rebuild the app if capability code changed.
- Confirm the app capability id, app id, preflight policy, confirmation policy,
  and execution support match the canonical suite contracts.

#### Missing Source Context

Symptoms:

- Data step blocks with missing source summary or source reference.
- Full lifecycle cannot start from loaded data.

Fixes:

- Return to Quant Data.
- Confirm the data has been loaded or registered.
- Confirm the selected lifecycle contains the source evidence.
- Reopen Workbench from the lifecycle that contains that source evidence.

#### Missing Handoff Reference

Symptoms:

- A mid-workflow stage blocks with `missing_workflow_handoff_reference`.

Explanation:

The selected scope requires an upstream reference that is not present in the
ledger or lifecycle context. For example, Studio candidate fit needs a
`model_config_draft`, and Documentation section drafting needs a
`documentation_draft`.

Fixes:

- Start earlier in the workflow.
- Run the producer step first.
- Select a lifecycle that already has the needed safe reference.

#### Manual Confirmation Required

Symptoms:

- `Advance until blocked` stops with `manual_confirmation_required`.

Fix:

- Select the confirmation step.
- Click `Confirm`.
- Continue with `Advance one step` or `Advance until blocked`.

This is expected for guarded steps and should not be bypassed.

#### User-Owned Gate Required

Symptoms:

- Confirm, Run, or Retry buttons are disabled.
- Workbench mentions readiness, plan approval, consent, or user-owned gates.

Fix:

1. Click `Check readiness`.
2. Review assumptions.
3. Click `Approve active plan`.
4. Click `Approve guided user workflow`.
5. Then confirm individual steps as needed.

#### OpenAI Or Ollama Not Used

Symptoms:

- Provider status shows deterministic fallback.
- Prompted scope still works but model-backed planning is not active.

Checks:

```powershell
$manifest = Invoke-RestMethod http://127.0.0.1:8000/runtime/manifest
$manifest.provider_status
```

Fixes:

- For OpenAI, confirm `QUANT_LLM_MODEL_PROFILE` is an OpenAI profile and
  `OPENAI_API_KEY` was set before Docker startup.
- For Ollama, confirm the host model is running and
  `QUANT_OLLAMA_BASE_URL=http://host.docker.internal:11434`.
- Rebuild or restart the stack after changing provider env vars.

#### Long-Running App Action

Symptoms:

- A step reports `running`.

Explanation:

The owning app accepted the action and returned a safe app-run reference. The
ledger records the running state instead of blocking the browser.

Fix:

- Refresh workflow run status.
- Check the owning app for app-run completion details.
- Continue after the app-owned action result is available.

### 18. Safety Rules To Keep In Mind

Do:

- Load data through Quant Data.
- Use the selected lifecycle as the safe context boundary.
- Let `quant_agent` call app-owned agent APIs.
- Use manual confirmation where required.
- Inspect ledgers and support bundles for audit evidence.

Do not:

- Paste raw row-level data into the prompt.
- Paste provider keys into the browser.
- Create `VITE_OPENAI_API_KEY`.
- Try to make the LLM execute shell commands.
- Bypass missing handoff blockers.
- Call app `/api/agent/actions/...` endpoints directly from browser code.
- Add generic `/execute` or `POST /runs` routes.

### 19. Validation Commands

Run these when you want to verify the implementation state from source.

Contracts:

```powershell
cd C:\Users\matth\Desktop\quant_suite
powershell -ExecutionPolicy Bypass -File scripts\validate_contracts.ps1
```

Agent focused workflow tests:

```powershell
cd C:\Users\matth\Desktop\quant_agent
python -m pytest tests\test_api.py -k "workflow_scope_resolution or workflow_run or full_lifecycle_workflow_advances or app_scoped_workflow" --basetemp .pytest-tmp\workflow-check
python -m compileall src tests
```

Agent full tests:

```powershell
cd C:\Users\matth\Desktop\quant_agent
python -m pytest --basetemp .pytest-tmp
```

Workbench frontend:

```powershell
cd C:\Users\matth\Desktop\quant_studio\frontend-react
npm test
npm run build
```

Live workflow certification:

```powershell
cd C:\Users\matth\Desktop\quant_suite
powershell -ExecutionPolicy Bypass -File scripts\certify_agent_live_workflows.ps1 `
  -AgentApiBaseUrl http://127.0.0.1:8000 `
  -DataApiBaseUrl http://127.0.0.1:8830 `
  -StudioApiBaseUrl http://127.0.0.1:8810 `
  -DocumentationApiBaseUrl http://127.0.0.1:8840 `
  -MonitoringApiBaseUrl http://127.0.0.1:8820
```

### 20. What Completion Looks Like

A completed scoped run should have:

- no unresolved capability gaps;
- no current orchestration blocker;
- completed or completed-with-warnings status for the selected scope;
- ledgered preflight, confirmation, action request, and action result records
  where required;
- safe handoff references between dependent steps;
- no raw rows, raw paths, provider keys, raw prompts, raw provider responses,
  or raw app payloads in exported ledgers or support bundles.

If the run stops early with a blocker, that is still a valid governed outcome.
The blocker tells you which context, confirmation, preflight, app capability,
or handoff must be resolved before continuing.
