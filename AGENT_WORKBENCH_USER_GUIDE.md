# Agent Workbench User Guide

## Executive Summary

Use Agent Workbench after you have loaded or registered a dataset in Quant
Data. The Workbench lets you ask the agent to run the full Quant Suite workflow
or only a selected portion of it, such as `Quant Studio steps 1-5` or `just
validate the Monitoring bundle`.

The simplest successful path is:

1. Load your dataset in Quant Data.
2. Open Agent Workbench from the suite UI.
3. Confirm the runtime status says the governed agent runtime is available.
4. Confirm the Workbench is using the lifecycle or context that contains your
   loaded dataset.
5. Choose one of the guided workflow buttons:
   `Run Quant Workflow`, `Run Quant Studio`, or `Run Quant Monitoring`.
6. If you need extra instructions, type them in the prompt box first, then
   click the guided button. For example: `Use an XGBoost model that is
   conservative`, then click `Run Quant Studio`.
7. Complete the user-owned gates when they appear:
   `Check readiness`, plan assumption review, `Approve active plan`, and
   `Approve guided user workflow`.
8. Click `Advance until blocked`.
9. When the runner stops at a manual confirmation, review the step and click
   `Confirm`.
10. Click `Advance until blocked` again.
11. Repeat confirmation and advance until the selected workflow completes or
    the Workbench shows a real blocker.
12. Use `Run progress`, `Orchestration`, `Run history`, `Ledger audit`, and
    `Support bundle` to inspect what happened.

The agent does not run arbitrary code and does not use browser-supplied action
payloads. The LLM is used for planning, scoping, and explanation. Actual work is
performed by app-owned APIs through typed capabilities, and `quant_agent`
enforces workflow scope, preflights, confirmations, user-owned gates, policy,
redaction, and ledgering.

If you only remember one thing: click a guided workflow button, use `Advance
until blocked`, and when it blocks for a manual gate, complete the named gate
and advance again.

## Starting Assumption

This guide assumes:

- you have already loaded or registered a dataset in Quant Data;
- you have opened Agent Workbench from the Quant Suite UI;
- the local Quant Suite stack is running;
- you want the agent to act on safe summaries and references for the loaded
  dataset, not on raw rows or raw file paths.

If the agent says it cannot find source context, source reference, lifecycle
context, or an upstream handoff, the likely issue is that the Workbench is not
pointed at the lifecycle that contains the loaded dataset evidence.

## What The Workbench Can Run

The current runner supports these workflow scopes.

| Scope | Use it when | Example prompt |
| --- | --- | --- |
| Full lifecycle | You want the whole suite path from Data through Monitoring. | `Run the full Quant Suite workflow for this dataset.` |
| App workflow | You only want one app's 1-5 workflow. | `Run Quant Studio steps 1-5.` |
| Stage range | You want a portion of one app's workflow. | `Run Quant Studio steps 2-4.` |
| Capability set | You want one or more specific safe actions. | `Just validate the Monitoring bundle.` |

The full lifecycle is the 20-step path:

| Range | App | Steps |
| --- | --- | --- |
| 1-5 | Quant Data | source reference, source readiness preflight, EDA plan, EDA review, EDA handoff |
| 6-10 | Quant Studio | model readiness, model config draft, candidate fit, candidate comparison, documentation/monitoring package |
| 11-15 | Quant Documentation | package inspection, draft workspace, section draft, claim review, review export |
| 16-20 | Quant Monitoring | bundle inspection, profile draft, bundle validation, monitoring review, feedback signal |

You do not have to run the full path. If you are already in Quant Studio and
only want model-related work, ask for the Studio stages you need. If you are in
Documentation, ask only for Documentation steps. The selected scope is supposed
to constrain the agent; it should not silently run outside that requested
scope.

## Recommended Prompts

Use direct prompts when the guided buttons are not specific enough. The agent
is better at choosing a scope when your request names the app, stage range, or
capability.

The three primary guided buttons submit fixed app-scoped workflow runs:

| Guided button | What it starts | Required starting point |
| --- | --- | --- |
| `Run Quant Workflow` | Quant Data steps 1-5 | data loaded in Quant Data step 1 |
| `Run Quant Studio` | Quant Studio steps 1-5 | data or a handoff loaded in Studio step 1 |
| `Run Quant Monitoring` | Quant Monitoring steps 1-5 | monitoring bundle loaded in Monitoring step 1 |

If you type extra instructions before clicking a guided button, the Workbench
adds them to the workflow goal. For example, type:

```text
Use an XGBoost model that is conservative
```

Then click `Run Quant Studio`. The resulting run stays scoped to Quant Studio
steps 1-5, but the agent receives your additional intent text.

Full lifecycle:

```text
Run the full Quant Suite workflow for this loaded dataset.
```

Quant Data only:

```text
Run Quant Data steps 1-5 for the loaded dataset.
```

Quant Studio only:

```text
Run Quant Studio steps 1-5 using the current lifecycle context.
```

Quant Studio partial workflow:

```text
Run Quant Studio steps 2-4.
```

Quant Documentation only:

```text
Run Quant Documentation steps 1-5 for the current documentation package.
```

Quant Monitoring only:

```text
Run Quant Monitoring steps 1-5 for the current monitoring bundle.
```

One capability:

```text
Just validate the Monitoring bundle.
```

Model fit only:

```text
Run the Quant Studio model fit step for the current model configuration.
```

Avoid prompts that ask the agent to skip gates, bypass validation, use raw
rows, expose file paths, or ignore policy. The runtime should reject those
requests.

## First Screen Checklist

Before clicking anything, scan these areas.

### Runtime Status

Look for the governed runtime status near the top of the Workbench.

Expected:

- runtime available;
- LLM role is planning and explanation only;
- workflow execution is app-owned guarded actions;
- workflow run routes are available;
- app capability discovery is available for the apps you intend to use.

If runtime status is unavailable, the Workbench cannot create or advance runs.
Check that the suite stack is running and that Quant Studio web can reach the
Quant Agent API.

### Context And Policy

This panel explains what the browser is allowed to send.

Expected concepts:

- context policy uses sanitized summaries and safe references;
- rows included is `no`;
- execution policy is typed app-owned capabilities only;
- browser payloads are goal, scope, and sanitized context.

This section is not where you choose the model or give the agent data. It is a
safety summary. The agent should not need raw row data in the browser.

### Guided Buttons And Agent Prompt

The guided buttons are the primary way to start a common run. Use the prompt
box when you need extra specifications or a custom scope.

Use a prompt that includes the scope you actually want when you are not using a
guided button. For example, if you only want Studio, say `Run Quant Studio
steps 1-5`, not `run the workflow`.

### Target App And Workflow Scope

If the Workbench was opened from an app rail, the target app may already be set.
For example, opening from Quant Data should bias the UI toward Quant Data. You
can still prompt for a different valid scope.

If the UI exposes scope controls, set them consistently with your prompt:

- `full_lifecycle` for the full suite;
- `app_workflow` plus the app for one app's 1-5 workflow;
- `stage_range` plus app, start stage, and end stage;
- `capability_set` plus selected capability ids.

### Evidence-Bound Indicator

If the Workbench shows an evidence-bound or summaries-only policy indicator,
that is expected. It means actions should be driven by ledgered evidence and
safe references, not raw browser payloads.

## Button Guide

This section explains the buttons you are most likely to use.

| Button | Use it when | What it does |
| --- | --- | --- |
| `Run Quant Workflow` | Your data is loaded in Quant Data and you want Data steps 1-5. | Creates a Quant Data app-scoped workflow run. |
| `Run Quant Studio` | You want Studio steps 1-5, with any prompt text used as extra intent. | Creates a Quant Studio app-scoped workflow run. |
| `Run Quant Monitoring` | You want Monitoring steps 1-5 for a loaded monitoring bundle. | Creates a Quant Monitoring app-scoped workflow run. |
| prompt send button | You have typed a custom scoped request. | Creates the governed plan/workflow run from the prompt. |
| `Check readiness` | The run is user-owned and readiness has not been checked. | Records whether the run has enough safe context to proceed. |
| `Approve active plan` | Assumptions were reviewed and accepted. | Approves the current active plan. |
| `Approve guided user workflow` | Readiness and plan approval are complete. | Gives run-level consent for guided draft workflow actions. |
| `Run preflight` | You want to manually run the selected step's preflight. | Runs the app-owned preflight for the selected step. |
| `Confirm` | The current step requires manual confirmation. | Records step-level confirmation; it does not execute by itself. |
| `Preview Readiness` | You want to build the action request preview for a selected step. | Creates a non-browser action request from ledgered evidence. |
| `Advance one step` | You want the runner to move one allowed action only. | Advances the current workflow by one permitted action. |
| `Advance until blocked` | You want the runner to continue until it hits a gate or blocker. | Runs permitted actions until confirmation, missing input, app error, running state, or completion. |
| `Refresh workflow run` | You want current state after a run, app action, or page pause. | Reloads the workflow run state from `quant_agent`. |
| `Pause Run` | You need to temporarily freeze a run. | Blocks gated actions until resumed. |
| `Resume Run` | A paused run should continue. | Revalidates the current step before allowing gated actions again. |
| `Cancel Run` | You want to stop the run permanently. | Marks the run cancelled and blocks future gated actions. |
| `View ledger` | You want the audit record. | Loads the sanitized ledger. |

For normal use, prefer:

```text
Guided button -> Check readiness / review / approve / consent if needed -> Advance until blocked -> Confirm when asked -> Advance until blocked
```

The lower-level buttons are useful when debugging or when you want to move one
gate at a time.

## The Normal Full Workflow Walkthrough

Use this path when your data is already loaded in Quant Data and you want the
entire suite to run.

### 1. Prompt For The Full Workflow

In the agent prompt box, type:

```text
Run the full Quant Suite workflow for this loaded dataset.
```

Click the prompt send button. The guided `Run Quant Workflow` button is for
Quant Data steps 1-5, not the full 20-step lifecycle.

Expected result:

- the agent creates a plan or workflow run;
- the selected scope is `full_lifecycle`;
- the run sequence shows Data, Studio, Documentation, and Monitoring steps;
- the Workbench shows the current step near the beginning of the Data workflow.

If the scope is wrong, create a new run with a clearer prompt or adjust the
scope controls before creating the run.

### 2. Complete User-Owned Gates

For user-owned data, the Workbench may block before any action can run.

Complete these gates when shown:

1. `Check readiness`
2. plan assumption review
3. `Approve active plan`
4. `Approve guided user workflow`

Plan review matters. If an assumption is wrong, mark it for revision instead of
approving it. A revision request should route into the plan revision and child
run flow. Do not approve a plan that describes the wrong dataset, wrong target,
or wrong app scope.

### 3. Advance Until The First Blocker

Click `Advance until blocked`.

The runner should perform any currently allowed action and then stop at the
next required gate. Typical blockers are:

- manual confirmation required;
- preflight blocked;
- missing source or handoff evidence;
- app capability unavailable;
- action running or pending;
- app error.

This is normal. The runner is designed to stop rather than skip a gate.

### 4. Confirm A Manual Step

When the current blocker says a step requires confirmation:

1. Read the current step label and capability id.
2. Check the plan and the latest preflight/result evidence.
3. If it is the step you intend to allow, click `Confirm`.
4. Confirm should record approval for that step only.

`Confirm` is not the same as `Run`. It only satisfies a gate. After confirming,
click `Advance until blocked` again.

### 5. Continue App By App

The full lifecycle crosses app boundaries by passing ledgered safe references:

| Boundary | Expected handoff |
| --- | --- |
| Data -> Studio | `eda_handoff` |
| Studio -> Documentation | `documentation_package` |
| Studio -> Monitoring | `monitoring_bundle` |
| Documentation internal | `documentation_draft`, `draft_section`, `claim_review_summary`, `documentation_review_package` |
| Monitoring internal | `bundle_summary`, `monitoring_profile_draft`, `bundle_validation_summary`, `monitoring_run`, `feedback_signal` |

If a boundary handoff is missing, the agent should block with a missing-handoff
message. That means an upstream step did not produce the reference needed by a
downstream step, or the selected workflow started in the middle without the
required context.

### 6. Finish And Inspect Evidence

When the run completes the selected scope, review:

- `Run progress` for counts and current/final status;
- `Orchestration` for each step status;
- `Run history` for prior runs;
- `Ledger audit` for the durable run record;
- `Support bundle` for a redacted exportable summary.

The ledger and support bundle should contain summaries and references only. If
you see raw rows, raw local paths, provider keys, raw prompts, raw provider
responses, or raw app payloads, treat that as a bug.

## Running Only One App

Use app-scoped workflows when you do not want the full suite.

### Quant Data Steps 1-5

Use when the dataset is loaded and you want Data to create source and EDA
handoff evidence.

Prompt:

```text
Run Quant Data steps 1-5 for this loaded dataset.
```

Expected outputs include:

- `source_reference`;
- source readiness summary;
- `eda_plan`;
- `eda_package`;
- `eda_handoff`.

The EDA handoff is the main upstream evidence for later Studio work.

### Quant Studio Steps 1-5

Use when the current lifecycle already has a Data handoff or Studio-ready input
and you want model workflow evidence.

Prompt:

```text
Run Quant Studio steps 1-5.
```

Expected outputs include:

- model readiness summary;
- `model_config_draft`;
- `studio_run`;
- `champion_recommendation`;
- `documentation_package`;
- `monitoring_bundle`.

Candidate comparison should create a reviewable recommendation. It should not
silently promote an unreviewed champion.

### Quant Documentation Steps 1-5

Use when you already have a documentation package or want Documentation-only
work.

Prompt:

```text
Run Quant Documentation steps 1-5 for the current package.
```

Expected outputs include:

- package inspection summary;
- `documentation_draft`;
- `draft_section`;
- claim review summary;
- `documentation_review_package`.

Drafting should create reviewable references. It should not silently publish a
final approved document.

### Quant Monitoring Steps 1-5

Use when you already have a monitoring bundle or want Monitoring-only work.

Prompt:

```text
Run Quant Monitoring steps 1-5 for the current bundle.
```

Expected outputs include:

- bundle inspection summary;
- `monitoring_profile_draft`;
- bundle validation summary;
- `monitoring_run`;
- `feedback_signal`.

The feedback signal is advisory. It should not automatically start retraining.

## Running A Stage Range

Use a stage range when you want a portion of one app's workflow.

Examples:

```text
Run Quant Studio steps 2-4.
```

```text
Run Quant Data steps 3-5.
```

```text
Run Quant Documentation steps 2-4.
```

```text
Run Quant Monitoring step 3 only.
```

Important: a stage range that starts in the middle still needs upstream safe
references. For example, Studio step 3 needs a model config reference. If that
reference does not exist in the selected lifecycle or ledger, the runner should
block rather than invent it.

## Running A Capability Set

Use a capability set for a single task or a small selected group.

Examples:

```text
Just validate the Monitoring bundle.
```

```text
Only run the Quant Data source readiness preflight.
```

```text
Create the Documentation draft workspace only.
```

The agent may infer a capability set from the prompt. If the inferred selection
is not what you intended, stop and create a clearer scoped run.

## What Each Status Means

| Status or blocker | Meaning | What to do |
| --- | --- | --- |
| `planned` | A run exists but no gated action has completed yet. | Start with readiness/review/consent gates or advance. |
| `waiting_for_input` | Required safe context or handoff evidence is missing. | Select the right lifecycle or run the upstream step that creates the handoff. |
| `waiting_for_confirmation` | The current step requires manual confirmation. | Review the step, then click `Confirm` if appropriate. |
| `needs_preflight` | A preflight-capable step has not been checked. | Click `Advance until blocked` or `Run preflight`. |
| `preflight_blocked` | The app-owned preflight found a blocker. | Fix the source/app issue, then rerun or revise. |
| `ready_for_action_request` | Gates are satisfied and an action request preview can be built. | Click `Advance one step` or `Advance until blocked`. |
| `ready_for_execution` | The step can be executed through the app-owned API. | Click `Advance one step` or `Advance until blocked`. |
| `running` | An app-owned long-running action has started or returned a running reference. | Refresh status later; do not start duplicate work manually. |
| `completed` | The selected step or scope completed successfully. | Inspect outputs and ledger evidence. |
| `failed_recoverable` | The action failed in a way that may support retry. | Review error, use retry only when the UI says it is allowed. |
| `failed_terminal` | The action failed terminally. | Cancel, revise plan, or create a new run after fixing inputs. |
| `paused` | The run was paused. | Resume or cancel. Gated actions stay blocked while paused. |
| `cancelled` | The run was cancelled. | Create a new run if you need to continue. |

## Common Problems And Fixes

### The Workbench Cannot Create An Agent

Likely causes:

- Quant Agent API is not running;
- Studio web is pointing at the wrong agent URL;
- runtime manifest is unavailable;
- governance policy denies the route.

What to do:

- check the runtime status panel;
- verify the agent API health route outside the browser;
- refresh the Workbench after the stack is healthy.

### The Agent Says It Cannot Find My Data

Likely causes:

- the dataset is loaded in Quant Data but not attached to the selected lifecycle;
- the Workbench launch context points to a different app or lifecycle;
- the prompt starts at a downstream step without upstream handoff evidence.

What to do:

- return to Quant Data and verify the loaded source has a safe source summary or
  source reference;
- select the correct lifecycle;
- run Quant Data steps 1-5 first to produce an `eda_handoff`.

### The Agent Keeps Asking For Confirmation

This is expected for guarded steps. Confirmation is a deliberate manual gate.

What to do:

- review the current step;
- click `Confirm`;
- then click `Advance until blocked` again.

### Advance Until Blocked Stops Too Early

This is usually correct. It stops at gates, missing handoffs, failed preflights,
app errors, running actions, or completion.

What to do:

- read the current blocker in `Run progress` and `Orchestration`;
- complete the named gate or fix the missing upstream evidence;
- refresh the run and advance again.

### A Button Is Disabled

Disabled buttons usually mean a gate is missing, the route is not advertised by
the runtime manifest, governance denies the action, or the current step is not
eligible for that action.

Check:

- runtime status;
- selected workflow scope;
- current step in `Orchestration`;
- user-owned readiness, plan approval, and consent;
- step-level confirmation;
- governance and separation-of-duties summaries if present.

### External Approval Panels Appear

External approval is optional enterprise governance. It is not supposed to be
the default local critical path. If your local policy enables external approval
enforcement, execution may be blocked until approval evidence exists.

For normal local testing, use the default local developer configuration unless
you are specifically testing governance behavior.

### The Agent Uses Fallback Planning Instead Of OpenAI Or Ollama

The runner can still operate with deterministic fallback planning. If you
expected OpenAI or Ollama, check the runtime status provider diagnostics. The
provider key or Ollama URL should never appear in the browser, ledger, manifest,
or support bundle.

## What Not To Do

Do not paste raw datasets into the agent prompt.

Do not paste API keys into the Workbench.

Do not create browser environment variables such as `VITE_OPENAI_API_KEY`.

Do not call app-owned agent endpoints directly from the browser to bypass
`quant_agent`.

Do not approve a plan whose assumptions do not match your intended dataset,
target, or workflow scope.

Do not expect the agent to skip confirmation, preflight, readiness, policy, or
capability checks.

## Behind The Scenes

The Workbench talks to `quant_agent`, not directly to Quant Data, Quant Studio,
Quant Documentation, or Quant Monitoring action endpoints.

Typical route sequence:

1. `GET /runtime/manifest`
2. `POST /workflow-scope-resolutions` when natural-language scope inference is
   needed
3. `POST /workflow-runs`
4. `POST /user-workflow-readiness` for user-owned data
5. `POST /user-plan-reviews`
6. `POST /user-plan-approvals`
7. `POST /user-workflow-consents`
8. `POST /workflow-runs/{run_id}/advance-until-blocked`
9. `POST /confirmations` when required
10. more workflow advance calls until blocked or complete
11. `GET /workflow-runs/{run_id}`
12. `GET /runs/{run_id}/orchestration`
13. `GET /runs/{run_id}/ledger`
14. `GET /runs/{run_id}/support-bundle`

During advancement, `quant_agent` resolves action inputs from:

- the canonical workflow template;
- the live capability registry;
- the recorded plan;
- prior ledgered preflight/result records;
- safe app-owned handoff references.

The browser should not supply action inputs, execution flags, preflight records,
confirmation records, or provider outputs.

## Quick Decision Tree

Use this when you are unsure what to click.

| You see | Next action |
| --- | --- |
| No plan or workflow run exists | Click a guided workflow button, or type a custom prompt and click the prompt send button. |
| Runtime unavailable | Fix the stack, then refresh Workbench. |
| User-owned readiness not checked | Click `Check readiness`. |
| Plan assumptions not reviewed | Review assumptions. Accept or mark revise. |
| Plan reviewed with all assumptions accepted | Click `Approve active plan`. |
| User-owned consent missing | Click `Approve guided user workflow`. |
| Current step needs preflight | Click `Advance until blocked` or `Run preflight`. |
| Current step needs confirmation | Review the step and click `Confirm`. |
| Current step is ready for action request or execution | Click `Advance until blocked`. |
| Current step is running | Refresh workflow/run status later. |
| Current step is blocked by missing handoff | Run the upstream step or select the lifecycle with the needed reference. |
| Run completed selected scope | Inspect progress, ledger, and support bundle. |

## Recommended First Test

After loading a dataset in Quant Data, test the smallest useful path first:

```text
Run Quant Data steps 1-5 for this loaded dataset.
```

Why this is a good first test:

- it verifies the Workbench can see the dataset context;
- it exercises source readiness and Data handoff creation;
- it produces `eda_handoff`, which is the main evidence needed by Studio;
- it avoids debugging the entire 20-step lifecycle at once.

After Quant Data steps 1-5 complete, test Studio:

```text
Run Quant Studio steps 1-5 using the current Data handoff.
```

Then test the full lifecycle:

```text
Run the full Quant Suite workflow for this loaded dataset.
```

## Expected Evidence From A Healthy Run

A healthy run should leave you with:

- a visible workflow run id;
- an ordered orchestration list;
- current and final run state;
- ledgered preflight records;
- ledgered confirmation records for guarded steps;
- ledgered action request previews;
- ledgered action results from app-owned APIs;
- safe output references such as `eda_handoff`, `studio_run`,
  `documentation_package`, `monitoring_bundle`, or `feedback_signal`;
- a sanitized ledger export;
- a sanitized support bundle.

It should not expose:

- raw rows;
- raw local file paths;
- bucket names;
- credentials or API keys;
- raw prompts;
- raw provider responses;
- raw app payloads;
- hidden shell commands.

## If You Need To Explain A Run Later

Use these panels:

- `Run progress` for the short status summary;
- `Orchestration` for step-by-step status and blockers;
- `Run history` to find prior runs;
- `Ledger audit` for the durable audit record;
- `Support bundle` for a redacted export package.

For sample demo runs, the `Demo narrative` panel may also explain the run. For
normal user-owned data, rely on progress, orchestration, ledger, and support
bundle.
