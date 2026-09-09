"""
FastAPI application exposing the LangGraph agent over HTTP.

Agent invocations can take 2-3+ minutes under free-tier LLM load, so all
work is dispatched to a background thread (asyncio.to_thread) and clients
poll via GET /api/recommend/{job_id} for progress. The agent's
multi-turn requirement-collection loop is surfaced via the
awaiting_input status + POST /api/recommend/{job_id}/answer.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.agent.graph import build_graph
from app.agent.state import AgentState
from app.api.jobs import Job, JobStore, JobStatus, label_for_node
from app.config import get_api_settings
from app.models.schemas import SystemDesignRecommendation, UserRequirements


class RecommendRequest(BaseModel):
    message: str


class AnswerRequest(BaseModel):
    answer: str


api_settings = get_api_settings()

app = FastAPI(title="AWS Instance Advisor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[api_settings.cors_allowed_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = JobStore()


def _empty_state() -> AgentState:
    return {
        "requirements": UserRequirements(),
        "latest_user_message": None,
        "next_question": None,
        "pending_field": None,
        "technical_needs": None,
        "instance_candidates": None,
        "database_candidates": None,
        "cache_candidates": None,
        "recommendation": None,
        "system_design_recommendation": None,
        "terraform_files": None,
    }


def _run_graph_with_streaming(
    job_id: str,
    state: AgentState,
    collecting: bool = False,
) -> AgentState:
    """
    Run one graph pass using .stream() so we surface per-node stage
    labels to the job. Returns the final state after the full pass.

    *collecting* marks whether this pass is part of the initial
    requirement-collection phase — used so we can report
    "collecting" / "awaiting_input" statuses distinct from the
    downstream "running" research/recommendation phase.
    """
    graph = build_graph()
    final_state: AgentState = state

    for chunk in graph.stream(state):
        node_name = next(iter(chunk.keys()))
        stage_label = label_for_node(node_name)
        jobs.update_stage(job_id, stage_label)
        final_state = chunk[node_name]
        jobs.update_state(job_id, final_state)

    return final_state


def _graph_loop_sync(job_id: str, initial_state: AgentState) -> None:
    """
    Synchronous (thread-bound) driver that mirrors the CLI loop in
    app/main.py but writes progress into the shared JobStore.

    Runs passes of graph.stream(state) until either a final
    recommendation is produced or the agent asks a follow-up question.
    On any unhandled exception the job is marked errored.
    """
    state = initial_state
    try:
        while True:
            state = _run_graph_with_streaming(job_id, state, collecting=True)

            if (
                state.get("system_design_recommendation") is not None
                or state.get("recommendation") is not None
            ):
                final = _serialize_result(state)
                jobs.update_status(job_id, "done", result=final)
                return

            if state.get("next_question"):
                jobs.update_status(
                    job_id,
                    "awaiting_input",
                    next_question=state["next_question"],
                )
                return

            jobs.update_status(
                job_id,
                "error",
                error="No recommendation or follow-up question was produced.",
            )
            return

    except Exception as exc:  # pragma: no cover - defensive
        jobs.update_status(job_id, "error", error=f"{type(exc).__name__}: {exc}")


def _resume_with_answer_sync(job_id: str, answer: str) -> None:
    """Resume a job in awaiting_input status after the user replies."""
    job = jobs.get(job_id)
    if job is None:
        return
    state = job.state
    state["latest_user_message"] = answer
    jobs.update_status(job_id, "running")
    try:
        while True:
            state = _run_graph_with_streaming(job_id, state, collecting=True)

            if (
                state.get("system_design_recommendation") is not None
                or state.get("recommendation") is not None
            ):
                final = _serialize_result(state)
                jobs.update_status(job_id, "done", result=final)
                return

            if state.get("next_question"):
                jobs.update_status(
                    job_id,
                    "awaiting_input",
                    next_question=state["next_question"],
                )
                return

            jobs.update_status(
                job_id,
                "error",
                error="No recommendation or follow-up question was produced.",
            )
            return

    except Exception as exc:  # pragma: no cover - defensive
        jobs.update_status(job_id, "error", error=f"{type(exc).__name__}: {exc}")


def _serialize_result(state: AgentState) -> dict[str, Any]:
    rec = state.get("system_design_recommendation")
    v1_rec = state.get("recommendation")
    tf_files = state.get("terraform_files")
    result: dict[str, Any] = {}
    if rec is not None:
        if isinstance(rec, SystemDesignRecommendation):
            result["system_design_recommendation"] = rec.model_dump(mode="json")
        else:
            result["system_design_recommendation"] = rec
    if v1_rec is not None:
        if hasattr(v1_rec, "model_dump"):
            result["recommendation"] = v1_rec.model_dump(mode="json")
        else:
            result["recommendation"] = v1_rec
    if tf_files is not None:
        result["terraform_files"] = tf_files
    return result


def _job_response(job: Job) -> dict[str, Any]:
    resp: dict[str, Any] = {
        "job_id": job.job_id,
        "status": job.status,
        "current_stage": job.current_stage,
        "created_at": job.created_at,
    }
    if job.status == "awaiting_input" and job.next_question is not None:
        resp["next_question"] = job.next_question
    if job.status == "done" and job.result is not None:
        resp["result"] = job.result
    if job.status == "error" and job.error is not None:
        resp["error"] = job.error
    return resp


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Liveness probe. No LLM or live-data calls."""
    return {"status": "ok", "time": time.time()}


@app.post("/api/recommend")
def create_recommend_job(req: RecommendRequest) -> dict[str, Any]:
    """Kick off a new agent run. Returns immediately with a job_id to poll."""
    job_id = str(uuid.uuid4())
    state = _empty_state()
    state["latest_user_message"] = req.message

    job = Job(
        job_id=job_id,
        status="collecting",
        current_stage="Initializing",
        state=state,
    )
    jobs.put(job)

    thread = threading.Thread(
        target=_graph_loop_sync,
        args=(job_id, state),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


@app.get("/api/recommend/{job_id}")
def get_job_status(job_id: str) -> dict[str, Any]:
    """Poll the current state / progress / result of a job."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _job_response(job)


@app.post("/api/recommend/{job_id}/answer")
def answer_question(job_id: str, req: AnswerRequest) -> dict[str, Any]:
    """
    Provide the user's reply to a follow-up question. Only valid when
    the job is in 'awaiting_input' status. Resumes execution in the
    background.
    """
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.status != "awaiting_input":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not awaiting input (current status: {job.status})",
        )

    thread = threading.Thread(
        target=_resume_with_answer_sync,
        args=(job_id, req.answer),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "running"}
