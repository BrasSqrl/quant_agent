from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from quant_agent_runtime import __version__
from quant_agent_runtime.models import PlanRequest, PlanResult, RuntimeManifest
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
            "plan_only_mode": True,
            "execution_supported": False,
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

    return api


app = create_app()
