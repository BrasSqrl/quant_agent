from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from quant_agent_runtime import __version__
from quant_agent_runtime.app_clients import AppClientError
from quant_agent_runtime.models import (
    ActionRequestPreviewRequest,
    ActionRequestPreviewResult,
    CancellationRequest,
    CancellationResult,
    ConfirmationRequest,
    ConfirmationResult,
    DemoNarrativeResult,
    ExecutionRequest,
    ExecutionResult,
    LedgerEntry,
    PauseRequest,
    PauseResult,
    PlanRevisionActivationRequest,
    PlanRevisionActivationResult,
    PlanRevisionRequest,
    PlanRevisionResult,
    PreflightRequest,
    PreflightResult,
    PlanRequest,
    PlanResult,
    ResumptionRequest,
    ResumptionResult,
    RetryRequest,
    RetryResult,
    RunRevalidationRequest,
    RunRevalidationResult,
    RunListResult,
    RunOrchestrationResult,
    RunStatusResult,
    RuntimeManifest,
    SampleAutopilotPreviewRequest,
    SampleAutopilotPreviewResult,
    SampleAutopilotStepRequest,
    SampleAutopilotStepResult,
    SampleResetPreviewRequest,
    SampleResetPreviewResult,
    SampleResetRequest,
    SampleResetResult,
    UserPlanApprovalRequest,
    UserPlanApprovalResult,
    UserPlanReviewRequest,
    UserPlanReviewResult,
    UserWorkflowConsentRequest,
    UserWorkflowConsentResult,
    UserWorkflowReadinessRequest,
    UserWorkflowReadinessResult,
)
from quant_agent_runtime.runtime import RuntimeContainer, build_runtime
from quant_agent_runtime.validation.errors import RuntimeValidationError


def create_app(runtime: RuntimeContainer | None = None) -> FastAPI:
    runtime_container = runtime or build_runtime()
    api = FastAPI(
        title="Quant Agent Runtime",
        version=__version__,
        description="Plan-only hosted runtime slice for Quant Suite governed agents.",
    )
    api.state.runtime = runtime_container
    api.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5810",
            "http://127.0.0.1:5810",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
        ],
        allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "service": "quant-agent-runtime",
            "version": __version__,
            "plan_only_mode": False,
            "execution_supported": True,
            "execution_support_level": "single_step_review_draft_actions_only",
        }

    @api.get("/runtime/manifest", response_model=RuntimeManifest)
    def runtime_manifest() -> RuntimeManifest:
        return runtime_container.manifest()

    @api.post("/plans", response_model=PlanResult)
    def create_plan(request: PlanRequest) -> PlanResult:
        try:
            return runtime_container.planner.create_plan(request)
        except RuntimeValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.to_problem()) from exc

    @api.post("/preflights", response_model=PreflightResult)
    def create_preflight(request: PreflightRequest) -> PreflightResult:
        try:
            return runtime_container.preflight.create_preflight(request)
        except RuntimeValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.to_problem()) from exc
        except AppClientError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={
                    "code": "app_unavailable" if exc.status_code == 503 else "app_preflight_error",
                    "message": str(exc),
                },
            ) from exc

    @api.post("/confirmations", response_model=ConfirmationResult)
    def create_confirmation(request: ConfirmationRequest) -> ConfirmationResult:
        try:
            return runtime_container.confirmation.create_confirmation(request)
        except RuntimeValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.to_problem()) from exc

    @api.post("/action-requests", response_model=ActionRequestPreviewResult)
    def create_action_request(request: ActionRequestPreviewRequest) -> ActionRequestPreviewResult:
        try:
            return runtime_container.action_request.create_action_request(request)
        except RuntimeValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.to_problem()) from exc

    @api.post("/executions", response_model=ExecutionResult)
    def create_execution(request: ExecutionRequest) -> ExecutionResult:
        try:
            return runtime_container.execution.execute_step(request)
        except RuntimeValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.to_problem()) from exc

    @api.post("/retries", response_model=RetryResult)
    def retry_execution(request: RetryRequest) -> RetryResult:
        try:
            return runtime_container.retry.retry_step(request)
        except RuntimeValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.to_problem()) from exc

    @api.get("/runs", response_model=RunListResult)
    def list_runs(
        lifecycle_id: str | None = None,
        app_id: str | None = None,
        capability_id: str | None = None,
        final_status: str | None = None,
        limit: int = 50,
    ) -> RunListResult:
        try:
            return runtime_container.run_status.list_runs(
                lifecycle_id=lifecycle_id,
                app_id=app_id,
                capability_id=capability_id,
                final_status=final_status,
                limit=limit,
            )
        except RuntimeValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.to_problem()) from exc

    @api.get("/runs/{run_id}", response_model=RunStatusResult)
    def get_run_status(run_id: str) -> RunStatusResult:
        try:
            return runtime_container.run_status.get_run_status(run_id)
        except RuntimeValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.to_problem()) from exc

    @api.get("/runs/{run_id}/orchestration", response_model=RunOrchestrationResult)
    def get_run_orchestration(run_id: str) -> RunOrchestrationResult:
        try:
            return runtime_container.orchestration.get_run_orchestration(run_id)
        except RuntimeValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.to_problem()) from exc

    @api.get("/runs/{run_id}/ledger", response_model=LedgerEntry)
    def get_run_ledger(run_id: str) -> LedgerEntry:
        try:
            return runtime_container.run_status.get_ledger_entry(run_id)
        except RuntimeValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.to_problem()) from exc

    @api.get("/runs/{run_id}/demo-narrative", response_model=DemoNarrativeResult)
    def get_run_demo_narrative(run_id: str) -> DemoNarrativeResult:
        try:
            return runtime_container.demo_narrative.get_demo_narrative(run_id)
        except RuntimeValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.to_problem()) from exc

    @api.post("/cancellations", response_model=CancellationResult)
    def cancel_run(request: CancellationRequest) -> CancellationResult:
        try:
            return runtime_container.run_status.cancel_run(request)
        except RuntimeValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.to_problem()) from exc

    @api.post("/pauses", response_model=PauseResult)
    def pause_run(request: PauseRequest) -> PauseResult:
        try:
            return runtime_container.run_status.pause_run(request)
        except RuntimeValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.to_problem()) from exc

    @api.post("/resumptions", response_model=ResumptionResult)
    def resume_run(request: ResumptionRequest) -> ResumptionResult:
        try:
            return runtime_container.run_status.resume_run(request)
        except RuntimeValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.to_problem()) from exc

    @api.post("/plan-revisions", response_model=PlanRevisionResult)
    def preview_plan_revision(request: PlanRevisionRequest) -> PlanRevisionResult:
        try:
            return runtime_container.plan_revision.preview_revision(request)
        except RuntimeValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.to_problem()) from exc

    @api.post("/plan-revision-activations", response_model=PlanRevisionActivationResult)
    def activate_plan_revision(request: PlanRevisionActivationRequest) -> PlanRevisionActivationResult:
        try:
            return runtime_container.plan_revision_activation.activate_revision(request)
        except RuntimeValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.to_problem()) from exc

    @api.post("/run-revalidations", response_model=RunRevalidationResult)
    def revalidate_run(request: RunRevalidationRequest) -> RunRevalidationResult:
        try:
            return runtime_container.revalidation.check_current_context(request)
        except RuntimeValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.to_problem()) from exc

    @api.post("/autopilot-previews", response_model=SampleAutopilotPreviewResult)
    def preview_sample_autopilot(request: SampleAutopilotPreviewRequest) -> SampleAutopilotPreviewResult:
        try:
            return runtime_container.sample_autopilot.preview_autopilot(request)
        except RuntimeValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.to_problem()) from exc

    @api.post("/autopilot-steps", response_model=SampleAutopilotStepResult)
    def advance_sample_autopilot_step(request: SampleAutopilotStepRequest) -> SampleAutopilotStepResult:
        try:
            return runtime_container.sample_autopilot_step.advance_one_step(request)
        except RuntimeValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.to_problem()) from exc

    @api.post("/sample-reset-previews", response_model=SampleResetPreviewResult)
    def preview_sample_reset(request: SampleResetPreviewRequest) -> SampleResetPreviewResult:
        try:
            return runtime_container.sample_reset.preview_reset(request)
        except RuntimeValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.to_problem()) from exc

    @api.post("/sample-resets", response_model=SampleResetResult)
    def reset_sample_demo(request: SampleResetRequest) -> SampleResetResult:
        try:
            return runtime_container.sample_reset.reset_sample(request)
        except RuntimeValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.to_problem()) from exc

    @api.post("/user-plan-reviews", response_model=UserPlanReviewResult)
    def review_user_plan(request: UserPlanReviewRequest) -> UserPlanReviewResult:
        try:
            return runtime_container.user_workflow.review_plan(request)
        except RuntimeValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.to_problem()) from exc

    @api.post("/user-plan-approvals", response_model=UserPlanApprovalResult)
    def approve_user_plan(request: UserPlanApprovalRequest) -> UserPlanApprovalResult:
        try:
            return runtime_container.user_workflow.approve_plan(request)
        except RuntimeValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.to_problem()) from exc

    @api.post("/user-workflow-readiness", response_model=UserWorkflowReadinessResult)
    def check_user_workflow_readiness(
        request: UserWorkflowReadinessRequest,
    ) -> UserWorkflowReadinessResult:
        try:
            return runtime_container.user_workflow.check_readiness(request)
        except RuntimeValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.to_problem()) from exc

    @api.post("/user-workflow-consents", response_model=UserWorkflowConsentResult)
    def approve_user_workflow_consent(
        request: UserWorkflowConsentRequest,
    ) -> UserWorkflowConsentResult:
        try:
            return runtime_container.user_workflow.approve_consent(request)
        except RuntimeValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.to_problem()) from exc

    return api


app = create_app()
