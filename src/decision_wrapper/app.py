"""FastAPI app exposing a Jev-compatible decision endpoint, backed by the
Copilot CLI.

POST /v1/decisions  -- same request/response shape as DecisionRequest/Response.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

from .backend import BackendError, CopilotCliBackend
from .engine import decide
from .schema import DecisionRequest, DecisionResponse

app = FastAPI(
    title="decision-wrapper",
    description=(
        "A Jev-style System One decision API, backed by the GitHub Copilot "
        "CLI (gpt-6-luna, gpt-5.4-mini, ...). POST a `state` plus typed "
        "`questions` (noul/choice/score) and get calibrated answers back."
    ),
    version="0.1.0",
)

_backend = CopilotCliBackend(
    max_concurrency=int(os.environ.get("DECISION_WRAPPER_MAX_CONCURRENCY", "4")),
    timeout_s=float(os.environ.get("DECISION_WRAPPER_TIMEOUT_S", "90")),
)
_default_samples = int(os.environ.get("DECISION_WRAPPER_SAMPLES", "3"))


@app.post("/v1/decisions", response_model=DecisionResponse)
async def post_decision(request: DecisionRequest) -> DecisionResponse:
    try:
        return await decide(request, _backend, samples=_default_samples)
    except BackendError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}
